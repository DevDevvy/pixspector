from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Protocol

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


class _StubDetector:
    scheme_id = "unknown"
    display_name = "Unknown"

    def detect(self, rgb_u8: np.ndarray) -> WatermarkDetectorResult:
        return WatermarkDetectorResult(
            scheme_id=self.scheme_id,
            display_name=self.display_name,
            status="stub",
            hits=[],
            confidence=0.0,
            notes=["Detector is a stub and does not perform real extraction yet."],
        )


class SynthIDDetector(_StubDetector):
    scheme_id = "synthid"
    display_name = "Google SynthID"


class StableSignatureDetector(_StubDetector):
    scheme_id = "stable_signature"
    display_name = "Stable Signature"


class AdobeCCDetector(_StubDetector):
    scheme_id = "adobe_cc"
    display_name = "Adobe Content Credentials"


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
            "Watermark detectors are currently stubs; results will be empty until extraction is implemented.",
        ],
    )

    log_analysis_step(
        _logger,
        "watermark",
        f"Watermark detection completed - hits: {len(hits)}",
        details={"max_confidence": max_confidence},
    )

    return report
