from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any

import cv2
import numpy as np


@dataclass
class CFAResult:
    inconsistency_map_u8: np.ndarray     # visualization (uint8) of CFA inconsistency per patch
    strong_ratio: float
    moderate_ratio: float
    strong_threshold: float
    moderate_threshold: float
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self).copy()
        d["inconsistency_map_u8"] = {"shape": list(self.inconsistency_map_u8.shape), "dtype": "uint8"}
        return d


def _estimate_camera_processing_level(rgb_u8: np.ndarray) -> float:
    """
    Estimate how much in-camera processing was applied.
    Modern cameras heavily process images, which can mask CFA patterns.
    Returns score [0..1] where higher means more processing.
    """
    rgb = rgb_u8.astype(np.float32)
    
    # Check for noise reduction (smooth areas should be very smooth)
    noise_reduction_score = 0.0
    h, w = rgb.shape[:2]
    
    # Sample smooth regions (low gradient areas)
    for ch in range(3):
        channel = rgb[..., ch]
        grad_mag = np.sqrt(cv2.Sobel(channel, cv2.CV_32F, 1, 0)**2 + 
                          cv2.Sobel(channel, cv2.CV_32F, 0, 1)**2)
        
        # Find smooth regions (bottom 20% of gradient magnitudes)
        smooth_threshold = np.percentile(grad_mag, 20)
        smooth_mask = grad_mag < smooth_threshold
        
        if np.any(smooth_mask):
            smooth_areas = channel[smooth_mask]
            # In heavily processed images, smooth areas are very uniform
            uniformity = 1.0 - (np.std(smooth_areas) / (np.mean(smooth_areas) + 1e-6))
            noise_reduction_score += uniformity
    
    noise_reduction_score /= 3.0  # Average across channels
    
    # Check for sharpening (enhanced edges)
    sharpening_score = 0.0
    for ch in range(3):
        channel = rgb[..., ch]
        laplacian = cv2.Laplacian(channel, cv2.CV_32F, ksize=3)
        
        # Strong Laplacian responses at edges indicate sharpening
        edge_strength = np.std(laplacian)
        # Normalize by typical values (empirically determined)
        sharpening_score += min(1.0, edge_strength / 15.0)
    
    sharpening_score /= 3.0
    
    # Overall processing level
    processing_level = 0.6 * noise_reduction_score + 0.4 * sharpening_score
    return float(np.clip(processing_level, 0.0, 1.0))


def _cfa_periodicity_score(rgb_u8: np.ndarray) -> float:
    """
    Enhanced CFA analysis that accounts for modern camera processing.
    """
    rgb = rgb_u8.astype(np.float32)
    
    # Estimate processing level first
    processing_level = _estimate_camera_processing_level(rgb_u8)
    
    # If heavy processing detected, CFA patterns may be legitimately absent
    if processing_level > 0.8:
        # For heavily processed images, only flag if patterns are completely absent
        # or show clear signs of synthesis
        synthesis_threshold = 0.6
    else:
        # For lightly processed images, use standard threshold
        synthesis_threshold = 0.3
    
    # Focus on green channel (most informative for Bayer CFA)
    g = rgb[..., 1]
    h, w = g.shape
    
    # Enhanced phase analysis with multiple scales
    phase_scores = []
    
    for scale in [2, 4]:  # Check 2x2 and 4x4 periodicities
        phases = np.zeros((scale, scale), dtype=np.float32)
        counts = np.zeros((scale, scale), dtype=np.int64)
        
        # Build prediction error map
        gp = np.pad(g, 1, mode="reflect")
        up = gp[0:h, 1:w + 1]
        down = gp[2:h + 2, 1:w + 1]
        left = gp[1:h + 1, 0:w]
        right = gp[1:h + 1, 2:w + 2]
        pred = 0.25 * (up + down + left + right)
        resid = np.abs(g - pred)
        
        # Aggregate by phase
        for y in range(h):
            for x in range(w):
                py, px = y % scale, x % scale
                phases[py, px] += resid[y, x]
                counts[py, px] += 1
        
        # Normalize by counts
        phases = phases / (counts + 1e-6)
        
        # Calculate phase variance (high variance = inconsistent CFA)
        if np.any(phases > 0):
            phases_norm = (phases - phases.min()) / (np.ptp(phases) + 1e-6)
            phase_variance = float(np.var(phases_norm))
            phase_scores.append(phase_variance)
    
    # Use the scale that shows most inconsistency
    if phase_scores:
        max_inconsistency = max(phase_scores)
        
        # Adjust score based on processing level
        if max_inconsistency > synthesis_threshold:
            # Further analysis: check if inconsistency pattern looks synthetic
            return _check_synthesis_patterns(rgb_u8, max_inconsistency)
        else:
            return 0.0
    
    return 0.0


def _check_synthesis_patterns(rgb_u8: np.ndarray, base_score: float) -> float:
    """
    Check if CFA inconsistencies match patterns typical of AI synthesis.
    """
    # AI-generated images often have:
    # 1. Completely missing CFA patterns
    # 2. Artificially regular patterns
    # 3. Inconsistent patterns across the image
    
    rgb = rgb_u8.astype(np.float32)
    g = rgb[..., 1]  # Green channel
    h, w = g.shape
    
    # Check spatial consistency of CFA pattern strength
    patch_size = 32
    pattern_strengths = []
    
    for y in range(0, h - patch_size, patch_size // 2):
        for x in range(0, w - patch_size, patch_size // 2):
            patch = g[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                # Quick CFA strength estimate for this patch
                phases_2x2 = np.zeros(4)
                for py in range(2):
                    for px in range(2):
                        phase_pixels = patch[py::2, px::2]
                        phases_2x2[py * 2 + px] = np.var(phase_pixels)
                
                pattern_strength = np.var(phases_2x2) if np.any(phases_2x2) else 0
                pattern_strengths.append(pattern_strength)
    
    if len(pattern_strengths) > 4:
        # AI images often have very inconsistent CFA strength across regions
        spatial_consistency = 1.0 - (np.std(pattern_strengths) / (np.mean(pattern_strengths) + 1e-6))
        
        # If spatial consistency is very low, likely synthetic
        if spatial_consistency < 0.3:
            return min(1.0, base_score * 1.5)  # Boost score for likely synthesis
        elif spatial_consistency > 0.8:
            return max(0.0, base_score * 0.5)  # Reduce score for natural variation
    
    return base_score


def run_cfa_map(
    rgb_u8: np.ndarray,
    patch: int = 64,
    stride: int = 32,
    thr_strong: float = 0.35,
    thr_moderate: float = 0.25,
) -> CFAResult:
    """
    Slide over RGB image; compute CFA inconsistency score per patch.
    """
    h, w = rgb_u8.shape[:2]
    ys = list(range(0, max(1, h - patch + 1), stride))
    xs = list(range(0, max(1, w - patch + 1), stride))
    heat = np.zeros((len(ys), len(xs)), dtype=np.float32)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            p = rgb_u8[y:y + patch, x:x + patch]
            if p.shape[0] != patch or p.shape[1] != patch:
                continue
            heat[iy, ix] = _cfa_periodicity_score(p)

    strong_mask = heat >= thr_strong
    mod_mask = (heat >= thr_moderate) & (~strong_mask)

    heat_norm = (255.0 * (heat - heat.min()) / (np.ptp(heat) + 1e-6)).astype(np.uint8)
    vis = cv2.resize(heat_norm, (w, h), interpolation=cv2.INTER_NEAREST)

    return CFAResult(
        inconsistency_map_u8=vis,
        strong_ratio=float(np.mean(strong_mask)),
        moderate_ratio=float(np.mean(mod_mask)),
        strong_threshold=float(thr_strong),
        moderate_threshold=float(thr_moderate),
        meta={
            "patch": patch,
            "stride": stride,
            "note": "CFA/demosaicing inconsistencies can indicate splices or non-camera synthesis."
        },
    )
