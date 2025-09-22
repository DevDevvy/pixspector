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


def _is_natural_texture(patch_gray: np.ndarray) -> bool:
    """
    Determine if a patch contains natural texture vs artificial patterns.
    Returns True if the texture appears natural.
    """
    # Calculate local contrast and edge patterns
    sobel_x = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Natural textures have more random edge orientations
    gradient_angle = np.arctan2(sobel_y, sobel_x)
    
    # Histogram of gradient orientations (8 bins)
    hist, _ = np.histogram(gradient_angle.flatten(), bins=8, range=(-np.pi, np.pi))
    hist = hist.astype(np.float32)
    hist = hist / (np.sum(hist) + 1e-6)
    
    # Natural textures have more uniform orientation distribution
    orientation_uniformity = 1.0 - np.var(hist)
    
    # Check for repetitive patterns using autocorrelation
    patch_float = patch_gray.astype(np.float32)
    patch_norm = patch_float - np.mean(patch_float)
    
    # Compute autocorrelation via FFT
    fft = np.fft.fft2(patch_norm)
    autocorr = np.fft.ifft2(fft * np.conj(fft)).real
    autocorr = np.fft.fftshift(autocorr)
    
    # Look for secondary peaks (indicating repetition)
    center = autocorr.shape[0] // 2
    autocorr[center-2:center+3, center-2:center+3] = 0  # Remove center peak
    
    max_secondary = np.max(autocorr)
    center_peak = np.max(patch_norm**2) * patch_norm.size  # Approximate center peak value
    
    repetition_ratio = max_secondary / (center_peak + 1e-6)
    
    # Natural texture: high orientation uniformity, low repetition
    return orientation_uniformity > 0.6 and repetition_ratio < 0.3


def _detect_interpolation_artifacts(patch_gray: np.ndarray) -> float:
    """
    Specifically look for interpolation artifacts typical of AI upscaling.
    Returns score [0..1] where higher indicates more artificial interpolation.
    """
    # Convert to float
    g = patch_gray.astype(np.float32)
    
    # Look for checkerboard patterns (common in bicubic interpolation)
    kernel_checkerboard = np.array([[-1, 1, -1],
                                    [1, -4, 1], 
                                    [-1, 1, -1]], dtype=np.float32)
    
    checkerboard_response = cv2.filter2D(g, -1, kernel_checkerboard)
    checkerboard_energy = np.mean(np.abs(checkerboard_response))
    
    # Look for regular grid patterns in second derivatives
    laplacian = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    
    # Check for grid-aligned periodicity in Laplacian
    h, w = laplacian.shape
    
    # Sample at regular grid positions vs offset positions
    grid_positions = []
    offset_positions = []
    
    for y in range(2, h-2, 4):  # Every 4 pixels
        for x in range(2, w-2, 4):
            grid_positions.append(abs(laplacian[y, x]))
            # Offset by 2 pixels
            if y+2 < h and x+2 < w:
                offset_positions.append(abs(laplacian[y+2, x+2]))
    
    if len(grid_positions) > 10 and len(offset_positions) > 10:
        grid_energy = np.mean(grid_positions)
        offset_energy = np.mean(offset_positions)
        
        # Artificial interpolation often has stronger responses at grid positions
        grid_bias = (grid_energy - offset_energy) / (grid_energy + offset_energy + 1e-6)
        grid_bias = max(0, grid_bias)  # Only positive bias matters
    else:
        grid_bias = 0
    
    # Combine checkerboard and grid bias scores
    artifact_score = 0.6 * (checkerboard_energy / 50.0) + 0.4 * grid_bias
    
    return float(np.clip(artifact_score, 0.0, 1.0))


def _periodicity_score(patch_gray: np.ndarray) -> float:
    """
    Enhanced periodicity detection that distinguishes natural textures from artifacts.
    """
    # First check if this looks like natural texture
    if _is_natural_texture(patch_gray):
        # For natural textures, look specifically for interpolation artifacts
        return _detect_interpolation_artifacts(patch_gray)
    
    # For non-natural textures, use the enhanced FFT analysis
    g = patch_gray.astype(np.float32)
    
    # Use second derivatives to emphasize interpolation traces
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

    # Enhanced radial profile analysis
    cy, cx = h // 2, w // 2
    yy, xx = np.indices(mag.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = rr.astype(np.int32)
    max_r = min(cy, cx) - 1
    
    radial = np.array([mag[r == i].mean() if np.any(r == i) else 0.0 for i in range(0, max_r)], dtype=np.float32)

    if radial.size < 8 or float(radial.max()) == 0.0:
        return 0.0
    
    # Normalize radial curve
    rc = (radial - radial.min()) / (np.ptp(radial) + 1e-6)

    # Look for specific AI upscaling signatures
    spec = np.abs(np.fft.rfft(rc))
    if spec.size <= 3:
        return 0.0
    
    # ignore DC (index 0)
    non_dc = spec[1:]
    
    # AI upscaling often creates multiple peaks at specific frequencies
    # Look for peaks at 1/2, 1/3, 1/4 of the spectrum (common upscaling ratios)
    n = len(non_dc)
    suspected_freqs = [n//4, n//3, n//2]  # 4x, 3x, 2x upscaling signatures
    
    peak_strength = 0.0
    for freq_idx in suspected_freqs:
        if 0 < freq_idx < n:
            # Check if there's a significant peak at this frequency
            local_region = max(1, freq_idx - 2), min(n, freq_idx + 3)
            local_max = np.max(non_dc[local_region[0]:local_region[1]])
            local_median = np.median(non_dc)
            
            if local_max > local_median * 2:  # Significant peak
                peak_strength += (local_max - local_median) / (local_max + local_median + 1e-6)
    
    # Normalize by number of checked frequencies
    if len(suspected_freqs) > 0:
        peak_strength /= len(suspected_freqs)
    
    return float(np.clip(peak_strength, 0.0, 1.0))


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
