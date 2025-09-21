from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import cv2
import numpy as np


@dataclass
class PRNUResult:
    residual_u8: np.ndarray               # visualization of noise residual (uint8)
    mean_abs_residual: float              # global mean absolute residual (0..255)
    p95_abs_residual: float               # 95th percentile residual
    correlation_with_ref: Optional[float] # optional correlation vs. provided ref pattern
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self).copy()
        d["residual_u8"] = {"shape": list(self.residual_u8.shape), "dtype": "uint8"}
        return d


def _simple_denoise(gray: np.ndarray, sigma: float) -> np.ndarray:
    """
    Simple denoising to estimate content; residual = gray - denoised.
    Using Gaussian for portability (wavelet would require extra deps).
    """
    k = int(max(3, round(sigma * 3) | 1))
    den = cv2.GaussianBlur(gray, (k, k), sigmaX=sigma, sigmaY=sigma)
    return den


def _norm_cross_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) - float(np.mean(a))
    b = b.astype(np.float32) - float(np.mean(b))
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.clip(np.sum(a * b) / denom, -1.0, 1.0))


def run_prnu(
    gray_u8: np.ndarray,
    wavelet_denoise_sigma: float = 3.0,
    ref_pattern: Optional[np.ndarray] = None,
    correlate_min_nz: float = 0.05,
) -> PRNUResult:
    """
    Blind residual estimate (PRNU-like). If a reference pattern is provided, compute
    normalized cross-correlation as a loose 'camera match' indicator.
    """
    g = gray_u8.astype(np.float32)
    den = _simple_denoise(g, sigma=wavelet_denoise_sigma)
    residual = g - den

    # Visualization: normalize to 0..255
    r = residual.copy()
    r = (r - r.min()) / (np.ptp(r) + 1e-6)
    vis = (255.0 * r).astype(np.uint8)

    mean_abs = float(np.mean(np.abs(residual)))
    p95 = float(np.percentile(np.abs(residual), 95))

    corr = None
    if ref_pattern is not None:
        rp = ref_pattern
        # Resize/crop to match
        h, w = residual.shape
        rh, rw = rp.shape[:2]
        if rh != h or rw != w:
            rp = cv2.resize(rp.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        # Check non-zero coverage to avoid degenerate refs
        if float(np.mean(np.abs(rp) > 1e-6)) >= correlate_min_nz:
            corr = _norm_cross_corr(residual, rp)

    return PRNUResult(
        residual_u8=vis,
        mean_abs_residual=mean_abs,
        p95_abs_residual=p95,
        correlation_with_ref=corr,
        meta={
            "wavelet_denoise_sigma": wavelet_denoise_sigma,
            "note": "Residual approximates sensor PRNU; correlations can support camera attribution when a reference exists."
        },
    )
