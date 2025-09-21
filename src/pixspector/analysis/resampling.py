from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

import cv2
import numpy as np


@dataclass
class ResamplingResult:
    heatmap_u8: np.ndarray              # visualization (uint8) of periodicity strength per patch
    strong_ratio: float                 # fraction of patches >= strong threshold
    moderate_ratio: float               # fraction of patches >= moderate threshold (but < strong)
    strong_threshold: float
    moderate_threshold: float
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self).copy()
        d["heatmap_u8"] = {"shape": list(self.heatmap_u8.shape), "dtype": "uint8"}
        return d


def _periodicity_score(patch_gray: np.ndarray) -> float:
    """
    Heuristic resampling detector:
    - use second derivatives to emphasize interpolation traces
    - take 2D DFT magnitude
    - collapse to 1D radial profile and measure the strength of periodic peaks
    Returns [0..1] score.
    """
    g = patch_gray.astype(np.float32)
    # Second derivatives (Laplacian) -> absolute
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    lap = np.abs(lap)

    # Window to reduce boundary effects
    h, w = lap.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    win = (wy * wx).astype(np.float32)
    lap *= win

    # 2D FFT magnitude (log)
    fft = np.fft.fft2(lap)
    mag = np.abs(np.fft.fftshift(fft))
    mag = np.log1p(mag)

    # Radial profile
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(mag.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = rr.astype(np.int32)
    max_r = min(cy, cx) - 1
    radial = np.array([mag[r == i].mean() if np.any(r == i) else 0.0 for i in range(0, max_r)], dtype=np.float32)

    # Normalize radial curve
    if radial.size < 8 or float(radial.max()) == 0.0:
        return 0.0
    rc = (radial - radial.min()) / (np.ptp(radial) + 1e-6)

    # Peakiness: take 1D FFT of the radial profile and measure spectral concentration at non-zero bins
    spec = np.abs(np.fft.rfft(rc))
    if spec.size <= 2:
        return 0.0
    # ignore DC (index 0)
    non_dc = spec[1:]
    # score: prominence of the top peak vs. median
    top = float(np.max(non_dc))
    med = float(np.median(non_dc) + 1e-6)
    score = (top / (top + med))  # ~0.5..1; map to [0..1]
    return float(np.clip(score, 0.0, 1.0))


def run_resampling_map(
    gray_u8: np.ndarray,
    patch: int = 128,
    stride: int = 64,
    thr_strong: float = 0.35,
    thr_moderate: float = 0.25,
) -> ResamplingResult:
    """
    Slide over the image; compute periodicity score per patch.
    Produce a downsampled heatmap aligned to patch grid.
    """
    h, w = gray_u8.shape[:2]
    if h < patch or w < patch:
        # Upscale small images slightly to stabilize FFT features
        scale = max(patch / h, patch / w)
        gray_u8 = cv2.resize(gray_u8, (int(w * scale) + 1, int(h * scale) + 1), interpolation=cv2.INTER_CUBIC)
        h, w = gray_u8.shape[:2]

    ys = list(range(0, h - patch + 1, stride))
    xs = list(range(0, w - patch + 1, stride))
    heat = np.zeros((len(ys), len(xs)), dtype=np.float32)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            p = gray_u8[y:y + patch, x:x + patch]
            heat[iy, ix] = _periodicity_score(p)

    strong_mask = heat >= thr_strong
    mod_mask = (heat >= thr_moderate) & (~strong_mask)

    # Visualize heatmap -> upscale to image size for nicer overlays later
    heat_norm = (255.0 * (heat - heat.min()) / (np.ptp(heat) + 1e-6)).astype(np.uint8)
    vis = cv2.resize(heat_norm, (w, h), interpolation=cv2.INTER_NEAREST)

    return ResamplingResult(
        heatmap_u8=vis,
        strong_ratio=float(np.mean(strong_mask)),
        moderate_ratio=float(np.mean(mod_mask)),
        strong_threshold=float(thr_strong),
        moderate_threshold=float(thr_moderate),
        meta={
            "patch": patch,
            "stride": stride,
            "note": "High periodicity may indicate resampling (scale/rotate/splice). Inspect localized peaks."
        },
    )
