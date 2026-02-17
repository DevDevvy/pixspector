from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Protocol, Tuple

import cv2
import numpy as np

from ..utils.logging import get_logger, log_analysis_step

_logger = get_logger("analysis.watermark")


@dataclass
class WatermarkHit:
    scheme_id: str
    confidence: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WatermarkDetectorResult:
    scheme_id: str
    display_name: str
    status: str
    hits: List[WatermarkHit]
    confidence: float
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scheme_id": self.scheme_id,
            "display_name": self.display_name,
            "status": self.status,
            "confidence": self.confidence,
            "hits": [hit.to_dict() for hit in self.hits],
            "notes": self.notes,
        }


@dataclass
class WatermarkReport:
    hits: List[WatermarkHit]
    max_confidence: float
    schemes_checked: List[str]
    detectors: List[WatermarkDetectorResult]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": [hit.to_dict() for hit in self.hits],
            "max_confidence": self.max_confidence,
            "schemes_checked": self.schemes_checked,
            "detectors": [detector.to_dict() for detector in self.detectors],
            "notes": self.notes,
        }


class WatermarkDetector(Protocol):
    scheme_id: str
    display_name: str

    def detect(self, rgb_u8: np.ndarray) -> WatermarkDetectorResult:
        ...


def _compute_spectral_features(rgb_u8: np.ndarray) -> Dict[str, float]:
    """Compute lightweight periodicity features from the luminance channel."""
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape

    # Keep analysis stable and fast across input sizes.
    target = 256
    scale = min(target / max(h, 1), target / max(w, 1), 1.0)
    if scale < 1.0:
        gray = cv2.resize(gray, (max(32, int(w * scale)), max(32, int(h * scale))), interpolation=cv2.INTER_AREA)

    gray = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)

    win_y = np.hanning(gray.shape[0]).astype(np.float32)
    win_x = np.hanning(gray.shape[1]).astype(np.float32)
    window = np.outer(win_y, win_x)
    fft = np.fft.fftshift(np.fft.fft2(gray * window))
    power = np.abs(fft) ** 2

    cy, cx = power.shape[0] // 2, power.shape[1] // 2
    yy, xx = np.indices(power.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = max(8, min(cy, cx) - 2)

    # Suppress low-frequency bias around DC.
    dc_mask = rr <= 4
    power = power.copy()
    power[dc_mask] = 0.0

    radius_bins = np.clip(rr.astype(np.int32), 0, max_radius)
    radial_energy = np.bincount(radius_bins.ravel(), weights=power.ravel(), minlength=max_radius + 1)
    radial_counts = np.bincount(radius_bins.ravel(), minlength=max_radius + 1)
    radial_profile = radial_energy / np.maximum(radial_counts, 1)

    total_energy = float(radial_energy.sum() + 1e-9)

    def band_ratio(start_frac: float, end_frac: float) -> float:
        start = int(max_radius * start_frac)
        end = max(start + 1, int(max_radius * end_frac))
        return float(radial_energy[start:end].sum() / total_energy)

    # 2D autocorrelation peak strength: periodic watermarks tend to produce regular peaks.
    autocorr = np.fft.ifft2(np.abs(np.fft.fft2(gray)) ** 2).real
    autocorr = np.fft.fftshift(autocorr)
    autocorr = autocorr - autocorr.min()
    center_val = float(autocorr[autocorr.shape[0] // 2, autocorr.shape[1] // 2] + 1e-9)

    autocorr_radius = np.sqrt((yy - autocorr.shape[0] // 2) ** 2 + (xx - autocorr.shape[1] // 2) ** 2)
    peak_zone = (autocorr_radius >= 6) & (autocorr_radius <= min(40, max_radius))
    side_peak = float(np.max(autocorr[peak_zone])) if np.any(peak_zone) else 0.0
    periodic_peak_ratio = side_peak / center_val

    highpass_std = float(np.std(gray))

    return {
        "low_band_ratio": band_ratio(0.06, 0.2),
        "mid_band_ratio": band_ratio(0.2, 0.45),
        "high_band_ratio": band_ratio(0.45, 0.8),
        "periodic_peak_ratio": periodic_peak_ratio,
        "highpass_std": highpass_std,
        "profile_length": float(len(radial_profile)),
    }


class _HeuristicDetector:
    scheme_id = "unknown"
    display_name = "Unknown"
    expected_band = "mid"

    def _score(self, features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        band_lookup = {
            "low": features["low_band_ratio"],
            "mid": features["mid_band_ratio"],
            "high": features["high_band_ratio"],
        }
        band_energy = band_lookup.get(self.expected_band, features["mid_band_ratio"])

        periodicity = np.clip((features["periodic_peak_ratio"] - 0.015) / 0.09, 0.0, 1.0)
        band_signal = np.clip((band_energy - 0.16) / 0.32, 0.0, 1.0)
        texture_gate = np.clip((features["highpass_std"] - 0.015) / 0.08, 0.0, 1.0)
        confidence = float(np.clip(0.5 * periodicity + 0.35 * band_signal + 0.15 * texture_gate, 0.0, 1.0))

        diagnostics = {
            "band_energy": float(band_energy),
            "periodicity": float(periodicity),
            "texture_gate": float(texture_gate),
            "periodic_peak_ratio": float(features["periodic_peak_ratio"]),
            "highpass_std": float(features["highpass_std"]),
        }
        return confidence, diagnostics

    def detect(self, rgb_u8: np.ndarray) -> WatermarkDetectorResult:
        features = _compute_spectral_features(rgb_u8)
        confidence, diagnostics = self._score(features)

        hits: List[WatermarkHit] = []
        if confidence >= 0.55:
            hits.append(
                WatermarkHit(
                    scheme_id=self.scheme_id,
                    confidence=confidence,
                    details={
                        "method": "spectral_periodicity_heuristic",
                        **diagnostics,
                    },
                )
            )

        status = "hit" if hits else "no_hit"
        notes = [
            "Heuristic detector based on spectral periodicity and autocorrelation.",
            "Result is probabilistic and should be interpreted with other forensic modules.",
        ]

        return WatermarkDetectorResult(
            scheme_id=self.scheme_id,
            display_name=self.display_name,
            status=status,
            hits=hits,
            confidence=confidence,
            notes=notes,
        )


class SynthIDDetector(_HeuristicDetector):
    scheme_id = "synthid"
    display_name = "Google SynthID"
    expected_band = "mid"


class StableSignatureDetector(_HeuristicDetector):
    scheme_id = "stable_signature"
    display_name = "Stable Signature"
    expected_band = "high"


class AdobeCCDetector(_HeuristicDetector):
    scheme_id = "adobe_cc"
    display_name = "Adobe Content Credentials"
    expected_band = "low"


def run_watermark_detection(rgb_u8: np.ndarray) -> WatermarkReport:
    detectors: List[WatermarkDetector] = [
        SynthIDDetector(),
        StableSignatureDetector(),
        AdobeCCDetector(),
    ]

    detector_results: List[WatermarkDetectorResult] = []
    hits: List[WatermarkHit] = []

    for detector in detectors:
        log_analysis_step(_logger, "watermark", f"Checking watermark scheme: {detector.scheme_id}")
        result = detector.detect(rgb_u8)
        detector_results.append(result)
        hits.extend(result.hits)

    max_confidence = 0.0
    for result in detector_results:
        max_confidence = max(max_confidence, float(result.confidence))
    for hit in hits:
        max_confidence = max(max_confidence, float(hit.confidence))

    report = WatermarkReport(
        hits=hits,
        max_confidence=max_confidence,
        schemes_checked=[detector.scheme_id for detector in detector_results],
        detectors=detector_results,
        notes=[
            "Watermark checks use non-destructive heuristics and may produce false positives/negatives.",
            "Treat hits as supporting provenance evidence, not standalone proof.",
        ],
    )

    log_analysis_step(
        _logger,
        "watermark",
        f"Watermark detection completed - hits: {len(hits)}",
        details={"max_confidence": max_confidence},
    )

    return report
