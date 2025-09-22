from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

from ..core.image_io import ensure_color


@dataclass
class ELAResult:
    diff_u8: np.ndarray           # 3-channel visualization image (uint8)
    mean_abs_diff: float          # global mean absolute diff (0..255)
    p95_abs_diff: float           # 95th percentile abs diff (0..255)
    strong_regions_ratio: float   # fraction of pixels above a heuristic threshold
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self).copy()
        # don't inline the full image array in JSON contexts
        d["diff_u8"] = {"shape": list(self.diff_u8.shape), "dtype": "uint8"}
        return d


def _pil_recompress(rgb: np.ndarray, quality: int) -> np.ndarray:
    """
    Recompress an RGB array with Pillow at given JPEG quality, return RGB uint8.
    """
    h, w = rgb.shape[:2]
    pil_im = Image.fromarray(rgb)
    from io import BytesIO
    buf = BytesIO()
    pil_im.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    arr = np.array(recompressed, dtype=np.uint8)
    if arr.shape[0] != h or arr.shape[1] != w:
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    return arr


def _analyze_compression_consistency(diff: np.ndarray) -> tuple[float, float]:
    """
    Analyze if compression differences are consistent with natural JPEG behavior.
    Returns (consistency_score, uniformity_score) where higher means more natural.
    """
    # Convert to grayscale for analysis
    if diff.ndim == 3:
        gray_diff = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray_diff = diff.astype(np.float32)
    
    # 1. Check for 8x8 block structure (natural JPEG should have some block artifacts)
    h, w = gray_diff.shape
    block_variance = []
    for y in range(0, h-8, 8):
        for x in range(0, w-8, 8):
            block = gray_diff[y:y+8, x:x+8]
            block_variance.append(np.var(block))
    
    if len(block_variance) > 0:
        # Natural JPEG should have relatively uniform block variance
        block_consistency = 1.0 - (np.std(block_variance) / (np.mean(block_variance) + 1e-6))
        block_consistency = np.clip(block_consistency, 0, 1)
    else:
        block_consistency = 0.5
    
    # 2. Check spatial frequency distribution
    fft = np.fft.fft2(gray_diff)
    fft_mag = np.abs(np.fft.fftshift(fft))
    
    # Natural JPEG compression has predictable frequency rolloff
    center_y, center_x = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    
    # Sample at different frequency rings
    rings = [0.1, 0.3, 0.5, 0.7, 0.9]
    max_dist = min(center_y, center_x)
    ring_energies = []
    
    for ring in rings:
        inner_rad = ring * max_dist * 0.8
        outer_rad = ring * max_dist * 1.2
        mask = (distances >= inner_rad) & (distances < outer_rad)
        if np.any(mask):
            ring_energies.append(np.mean(fft_mag[mask]))
        else:
            ring_energies.append(0)
    
    # Natural compression should show gradual frequency rolloff
    if len(ring_energies) >= 3:
        # Check if energy decreases with frequency (mostly)
        decreasing_trend = sum(ring_energies[i] >= ring_energies[i+1] for i in range(len(ring_energies)-1))
        uniformity_score = decreasing_trend / (len(ring_energies) - 1)
    else:
        uniformity_score = 0.5
    
    return block_consistency, uniformity_score


def run_ela(bgr_u8: np.ndarray, recompress_quality: int = 90, blur_sigma: float = 0.0) -> ELAResult:
    """
    Enhanced Error Level Analysis with natural compression pattern detection.
    """
    bgr_u8 = ensure_color(bgr_u8)
    rgb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB)
    rec_rgb = _pil_recompress(rgb, quality=recompress_quality)

    diff = cv2.absdiff(rgb, rec_rgb).astype(np.float32)
    if blur_sigma and blur_sigma > 0:
        k = int(max(3, round(blur_sigma * 4) | 1))
        diff = cv2.GaussianBlur(diff, (k, k), blur_sigma)

    # Analyze compression consistency
    consistency, uniformity = _analyze_compression_consistency(diff)
    
    # Normalize per-channel to full 0..255 for visualization
    vis = diff.copy()
    for c in range(3):
        ch = vis[..., c]
        m, M = ch.min(), ch.max()
        if M > m:
            vis[..., c] = (255.0 * (ch - m) / (M - m))
    vis_u8 = np.clip(vis, 0, 255).astype(np.uint8)
    vis_u8 = cv2.cvtColor(vis_u8, cv2.COLOR_RGB2BGR)

    # Enhanced metrics that account for natural compression patterns
    gray_diff = cv2.cvtColor(vis_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_abs = float(np.mean(gray_diff))
    p95_abs = float(np.percentile(gray_diff, 95))
    
    # Adjust strong ratio based on compression consistency
    base_threshold = 200.0
    # If compression looks natural, raise the threshold significantly
    if consistency > 0.7 and uniformity > 0.6:
        adjusted_threshold = base_threshold * 1.5  # Much higher threshold for natural compression
    elif consistency > 0.5 and uniformity > 0.4:
        adjusted_threshold = base_threshold * 1.2
    else:
        adjusted_threshold = base_threshold
    
    strong_ratio = float(np.mean(gray_diff > adjusted_threshold))

    return ELAResult(
        diff_u8=vis_u8,
        mean_abs_diff=mean_abs,
        p95_abs_diff=p95_abs,
        strong_regions_ratio=strong_ratio,
        meta={
            "recompress_quality": recompress_quality,
            "blur_sigma": blur_sigma,
            "compression_consistency": consistency,
            "frequency_uniformity": uniformity,
            "adjusted_threshold": adjusted_threshold,
            "note": "Enhanced ELA with natural compression pattern detection."
        },
    )
