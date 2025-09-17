from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np
from scipy.signal import find_peaks


@dataclass
class FFTChecksResult:
    fft_logmag_u8: np.ndarray           # 2D FFT log-magnitude visualization (uint8)
    radial_profile: List[float]         # normalized radial spectrum (0..1)
    peak_indices: List[int]             # indices of prominent peaks in radial spectrum
    highfreq_rolloff: float             # fraction of energy in top radial quartile (0..1)
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self).copy()
        d["fft_logmag_u8"] = {"shape": list(self.fft_logmag_u8.shape), "dtype": "uint8"}
        return d


def run_fft_checks(gray_u8: np.ndarray, radial_bins: int = 64, peak_prominence: float = 6.0,
                   highfreq_rolloff_warn: float = 0.75) -> FFTChecksResult:
    """
    Compute 2D FFT log-magnitude (visualization), radial profile, detect periodic peaks,
    and estimate high-frequency rolloff.
    """
    g = gray_u8.astype(np.float32)
    # Window to reduce border effects
    h, w = g.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    g = g * (wy * wx).astype(np.float32)

    fft = np.fft.fft2(g)
    mag = np.abs(np.fft.fftshift(fft))
    logmag = np.log1p(mag)
    # Normalize to 0..255 u8 for visualization
    vis = (255.0 * (logmag - logmag.min()) / (logmag.ptp() + 1e-6)).astype(np.uint8)

    # Radial profile (bin to fixed length)
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(logmag.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = rr / (rr.max() + 1e-6)  # [0..1]
    bins = np.linspace(0, 1.0, radial_bins + 1)
    radial = np.zeros(radial_bins, dtype=np.float32)
    for i in range(radial_bins):
        m = (r >= bins[i]) & (r < bins[i + 1])
        if np.any(m):
            radial[i] = float(logmag[m].mean())

    # normalize radial
    if float(radial.max()) > 0:
        radial = (radial - radial.min()) / (radial.ptp() + 1e-6)

    # Peak detection on radial curve (ignore bin 0)
    rp = radial.copy()
    rp[0] = 0.0
    peaks, _ = find_peaks(rp, prominence=peak_prominence / 100.0)  # scale a bit
    peak_list = [int(i) for i in peaks.tolist()]

    # High-frequency rolloff: energy share in top quartile of radii
    q = int(0.75 * radial_bins)
    highfreq_energy = float(np.mean(radial[q:])) if q < radial_bins else 0.0

    return FFTChecksResult(
        fft_logmag_u8=vis,
        radial_profile=[float(x) for x in radial.tolist()],
        peak_indices=peak_list,
        highfreq_rolloff=highfreq_energy,
        meta={
            "radial_bins": radial_bins,
            "peak_prominence": peak_prominence,
            "highfreq_rolloff_warn": highfreq_rolloff_warn,
            "note": "Spikes suggest periodic artifacts; extreme rolloff may indicate over-smoothing or synthesis."
        },
    )
