from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List
import concurrent.futures
import os

import cv2
import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter

from ..utils.logging import get_logger, log_analysis_step

# Module logger for AI detection
_logger = get_logger("analysis.ai_detection")


@dataclass
class AIDetectionResult:
    pixel_distribution_score: float      # [0..1] higher = more AI-like
    spectral_anomaly_score: float       # [0..1] higher = more AI-like
    texture_consistency_score: float    # [0..1] higher = more AI-like
    gradient_distribution_score: float  # [0..1] higher = more AI-like
    color_correlation_score: float      # [0..1] higher = more AI-like
    overall_ai_probability: float       # [0..1] combined AI probability
    explanations: List[str]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _analyze_pixel_distribution(rgb_u8: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Analyze pixel value distributions for AI-like characteristics.
    AI images often have unnaturally uniform or clustered distributions.
    """
    rgb = rgb_u8.astype(np.float32) / 255.0
    
    # Analyze each channel separately
    channel_scores = []
    channel_stats: Dict[str, Any] = {}

    entropy_values = []
    ks_values = []
    gap_densities = []
    low_high_ratios = []

    for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
        channel = rgb[..., ch_idx].flatten()

        # Histogram based characteristics -------------------------------------------------
        hist, _ = np.histogram(channel, bins=256, range=(0, 1))
        hist_norm = hist / (np.sum(hist) + 1e-8)

        entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-8)))
        entropy_values.append(entropy)

        # Natural images centre around ~7.2 bits of entropy per channel.
        entropy_center = 7.2
        entropy_anomaly = np.clip(abs(entropy - entropy_center) / 1.2, 0.0, 1.0)

        # Tail occupancy – AI renders often overuse saturated or very dark pixels.
        low_high = float(hist[:6].sum() + hist[-6:].sum()) / (np.sum(hist) + 1e-8)
        low_high_ratios.append(low_high)
        tail_anomaly = np.clip((low_high - 0.08) / 0.12, 0.0, 1.0)

        # Detect gaps (empty bins between populated regions) that indicate quantisation artefacts.
        gap_count = 0
        for i in range(2, len(hist) - 2):
            if hist[i] == 0 and hist[i - 1] > 0 and hist[i + 1] > 0:
                gap_count += 1
        gap_density = gap_count / 256.0
        gap_densities.append(gap_density)
        gap_anomaly = np.clip(gap_density / 0.02, 0.0, 1.0)

        # Statistical tests ----------------------------------------------------------------
        # Compare against a smoothed empirical baseline rather than a uniform distribution.
        reference = uniform_filter(channel.reshape(rgb_u8.shape[0], rgb_u8.shape[1]), size=5).flatten()
        reference = reference / (np.max(reference) + 1e-8)
        ks_stat, _ = stats.kstest(channel, reference)
        ks_values.append(float(ks_stat))
        ks_anomaly = np.clip(ks_stat / 0.12, 0.0, 1.0)

        # Higher-order moments.
        skewness = float(stats.skew(channel))
        kurtosis = float(stats.kurtosis(channel, fisher=False))  # already +3
        moment_anomaly = 0.0
        if abs(skewness) > 0.8:
            moment_anomaly += np.clip((abs(skewness) - 0.8) / 1.2, 0.0, 1.0)
        if kurtosis < 2.5 or kurtosis > 4.5:
            moment_anomaly += np.clip(abs(kurtosis - 3.3) / 2.0, 0.0, 1.0)
        moment_anomaly = np.clip(moment_anomaly, 0.0, 1.0)

        channel_score = 0.35 * entropy_anomaly + 0.25 * ks_anomaly + 0.2 * tail_anomaly + 0.2 * gap_anomaly
        channel_scores.append(channel_score)

        channel_stats[f'{ch_name}_entropy'] = entropy
        channel_stats[f'{ch_name}_low_high_ratio'] = float(low_high)
        channel_stats[f'{ch_name}_gap_density'] = float(gap_density)
        channel_stats[f'{ch_name}_ks_stat'] = float(ks_stat)
        channel_stats[f'{ch_name}_skewness'] = skewness
        channel_stats[f'{ch_name}_kurtosis'] = kurtosis

    overall_score = float(np.mean(channel_scores)) if channel_scores else 0.0
    summary_meta = {
        'avg_entropy': float(np.mean(entropy_values)) if entropy_values else 0.0,
        'entropy_stdev': float(np.std(entropy_values)) if entropy_values else 0.0,
        'avg_ks_stat': float(np.mean(ks_values)) if ks_values else 0.0,
        'avg_gap_density': float(np.mean(gap_densities)) if gap_densities else 0.0,
        'avg_low_high_ratio': float(np.mean(low_high_ratios)) if low_high_ratios else 0.0,
    }
    channel_stats.update(summary_meta)

    return float(np.clip(overall_score, 0.0, 1.0)), channel_stats


def _analyze_spectral_characteristics(rgb_u8: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Analyze frequency domain characteristics that distinguish AI from natural images.
    """
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    
    # Apply window to reduce edge effects
    window = np.outer(np.hanning(h), np.hanning(w))
    gray_windowed = gray * window
    
    # 2D FFT
    fft = np.fft.fft2(gray_windowed)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    
    # Log magnitude spectrum
    log_magnitude = np.log1p(magnitude)
    
    # 1. Radial frequency analysis
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Calculate radial average
    r_int = r.astype(int)
    max_r = min(center_x, center_y)
    radial_profile = np.array([log_magnitude[r_int == i].mean() 
                              for i in range(max_r) if np.any(r_int == i)])
    
    # 2. Check for AI-specific spectral signatures
    slope = 0.0
    r_value = 0.0
    slope_anomaly = 0.0
    high_freq_ratio = 0.0
    ring_strength = 0.0

    if len(radial_profile) > 10:
        freqs = np.arange(1, len(radial_profile) + 1)
        try:
            log_freqs = np.log(freqs[1:])
            log_profile = np.log(np.clip(radial_profile[1:], 1e-8, None))

            slope, intercept, r_value, _, _ = stats.linregress(log_freqs, log_profile)
            slope_anomaly = np.clip(abs(slope + 2.0) / 1.0, 0.0, 1.0)
        except Exception:
            slope = 0.0
            r_value = 0.0
            slope_anomaly = 0.0

        # High-frequency energy ratio: AI renders tend to retain excess HF energy.
        energy = radial_profile**2
        total_energy = np.sum(energy) + 1e-8
        split_idx = max(5, int(0.35 * len(energy)))
        high_freq_ratio = float(np.sum(energy[split_idx:]) / total_energy)
        if high_freq_ratio > 0.32:
            hf_anomaly = np.clip((high_freq_ratio - 0.32) / 0.35, 0.0, 1.0)
        else:
            hf_anomaly = np.clip((0.2 - high_freq_ratio) / 0.2, 0.0, 0.6)

        # Periodic ring detection (variance of radial profile after smoothing)
        smoothed = uniform_filter(radial_profile, size=5)
        ring_strength = float(np.std(radial_profile - smoothed))
        ring_anomaly = np.clip(ring_strength / 5.0, 0.0, 1.0)
    else:
        hf_anomaly = 0.0
        ring_anomaly = 0.0

    # 3. Directional analysis
    angles = np.arctan2(y - center_y, x - center_x)
    angle_bins = 8
    angle_energies = []
    
    for i in range(angle_bins):
        angle_start = (i * 2 * np.pi) / angle_bins - np.pi
        angle_end = ((i + 1) * 2 * np.pi) / angle_bins - np.pi
        
        if angle_end < angle_start:  # Handle wrap-around
            mask = (angles >= angle_start) | (angles <= angle_end)
        else:
            mask = (angles >= angle_start) & (angles < angle_end)
        
        if np.any(mask):
            angle_energies.append(np.mean(magnitude[mask]))
        else:
            angle_energies.append(0.0)
    
    if len(angle_energies) > 0:
        directional_variance = float(np.var(angle_energies) / (np.mean(angle_energies) + 1e-8))
        directional_uniformity = np.clip((0.2 - directional_variance) / 0.2, 0.0, 1.0)
    else:
        directional_variance = 0.0
        directional_uniformity = 0.0

    # Combine spectral anomalies placing emphasis on slope and HF ratio.
    spectral_score = (
        0.45 * slope_anomaly +
        0.3 * hf_anomaly +
        0.15 * ring_anomaly +
        0.1 * directional_uniformity
    )

    meta = {
        'power_law_slope': float(slope),
        'power_law_r2': float(r_value**2),
        'high_frequency_energy_ratio': float(high_freq_ratio),
        'ring_strength': float(ring_strength),
        'directional_variance': float(directional_variance),
        'radial_profile_length': int(len(radial_profile))
    }

    return float(np.clip(spectral_score, 0.0, 1.0)), meta


def _analyze_local_texture_patterns(rgb_u8: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Analyze local binary patterns and texture consistency.
    AI images often have inconsistent micro-textures.
    """
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    
    # Local Binary Pattern analysis
    def calculate_lbp(image, radius=1, n_points=8):
        """Calculate Local Binary Pattern with proper data type handling"""
        # Ensure we work with the right data types
        image = image.astype(np.int32)  # Use int32 to avoid overflow
        lbp = np.zeros_like(image, dtype=np.int32)  # Start with int32
        
        for i in range(n_points):
            # Calculate coordinates of sample points
            angle = 2 * np.pi * i / n_points
            dy = int(np.round(radius * np.sin(angle)))
            dx = int(np.round(radius * np.cos(angle)))
            
            # Extract neighbor values with proper bounds checking
            h, w = image.shape
            neighbor = np.zeros_like(image, dtype=np.int32)
            
            # Handle shifts with proper boundary conditions
            if dy >= 0 and dx >= 0:
                neighbor[dy:, dx:] = image[:-dy if dy > 0 else None, :-dx if dx > 0 else None]
            elif dy >= 0 and dx < 0:
                neighbor[dy:, :dx] = image[:-dy if dy > 0 else None, -dx:]
            elif dy < 0 and dx >= 0:
                neighbor[:dy, dx:] = image[-dy:, :-dx if dx > 0 else None]
            else:  # dy < 0 and dx < 0
                neighbor[:dy, :dx] = image[-dy:, -dx:]
            
            # Compare with center pixel and accumulate
            comparison = (neighbor >= image).astype(np.int32)
            lbp += comparison * (2 ** i)
        
        # Convert back to uint8, ensuring values are in valid range
        return np.clip(lbp, 0, 255).astype(np.uint8)
    
    # Calculate LBP
    lbp = calculate_lbp(gray)

    # Analyze LBP histogram
    lbp_hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
    lbp_hist = lbp_hist.astype(np.float64) / (np.sum(lbp_hist) + 1e-8)

    # 1. Uniform pattern prevalence ------------------------------------------------------
    uniform_patterns = 0.0
    for i in range(256):
        binary = format(i, '08b')
        transitions = sum(binary[j] != binary[(j + 1) % 8] for j in range(8))
        if transitions <= 2:
            uniform_patterns += lbp_hist[i]

    uniform_anomaly = 0.0
    if uniform_patterns < 0.6:
        # Very low presence of uniform LBP patterns points to synthetic micro texture.
        uniform_anomaly = np.clip((0.6 - uniform_patterns) / 0.6, 0.0, 1.0)
    elif uniform_patterns > 0.9:
        # Extremely high uniformity can hint at stylised renders but is a weaker signal.
        uniform_anomaly = np.clip((uniform_patterns - 0.9) / 0.1, 0.0, 0.6)

    # 2. Patch entropy variance ----------------------------------------------------------
    h, w = gray.shape
    patch_size = min(48, max(16, min(h, w) // 6))
    if patch_size < 12:
        return 0.0, {'note': 'Image too small for texture analysis'}

    texture_entropies: List[float] = []
    step_size = max(8, patch_size // 2)
    for y in range(0, h - patch_size + 1, step_size):
        for x in range(0, w - patch_size + 1, step_size):
            patch = gray[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                continue
            try:
                patch_lbp = calculate_lbp(patch)
                patch_hist, _ = np.histogram(patch_lbp.flatten(), bins=256, range=(0, 256))
                patch_hist = patch_hist.astype(np.float64) / (np.sum(patch_hist) + 1e-8)
                patch_entropy = float(-np.sum(patch_hist * np.log2(patch_hist + 1e-8)))
                texture_entropies.append(patch_entropy)
            except Exception:
                continue

    entropy_variance = float(np.var(texture_entropies)) if len(texture_entropies) > 1 else 0.0
    entropy_range = float(np.ptp(texture_entropies)) if len(texture_entropies) > 1 else 0.0
    if len(texture_entropies) > 1:
        if entropy_variance < 0.2:
            variance_anomaly = np.clip((0.2 - entropy_variance) / 0.2, 0.0, 0.7)
        elif entropy_variance > 0.85:
            variance_anomaly = np.clip((entropy_variance - 0.85) / 0.85, 0.0, 0.6)
        else:
            variance_anomaly = 0.0

        if entropy_range < 1.5:
            range_anomaly = np.clip((1.5 - entropy_range) / 1.5, 0.0, 1.0)
        elif entropy_range > 4.0:
            range_anomaly = np.clip((entropy_range - 4.0) / 4.0, 0.0, 0.5)
        else:
            range_anomaly = 0.0
    else:
        variance_anomaly = 0.0
        range_anomaly = 0.0

    # 3. Micro contrast stability --------------------------------------------------------
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    lap_energy = float(np.mean(np.abs(laplacian)))
    # AI renders often oversharpen or under sharpen compared to camera captures.
    if lap_energy < 12.0:
        contrast_anomaly = np.clip((12.0 - lap_energy) / 12.0, 0.0, 0.8)
    elif lap_energy > 30.0:
        contrast_anomaly = np.clip((lap_energy - 30.0) / 25.0, 0.0, 1.0)
    else:
        contrast_anomaly = 0.0

    overall_texture_score = (
        0.4 * uniform_anomaly +
        0.35 * variance_anomaly +
        0.15 * range_anomaly +
        0.1 * contrast_anomaly
    )

    meta = {
        'uniform_patterns_ratio': float(uniform_patterns),
        'texture_entropy_variance': float(entropy_variance),
        'texture_entropy_range': float(entropy_range),
        'avg_texture_entropy': float(np.mean(texture_entropies)) if texture_entropies else 0.0,
        'patch_count': len(texture_entropies),
        'laplacian_energy': float(lap_energy)
    }

    return float(np.clip(overall_texture_score, 0.0, 1.0)), meta


def _analyze_gradient_distributions(rgb_u8: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Analyze gradient magnitude distributions.
    AI images have characteristic gradient patterns.
    """
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Flatten and remove very small gradients (noise)
    grad_flat = grad_magnitude.flatten()
    grad_flat = grad_flat[grad_flat > 1.0]  # Remove near-zero gradients

    if grad_flat.size > 0:
        # Use a deterministic stratified sample to keep statistical tests stable.
        sample_size = min(120_000, grad_flat.size)
        if sample_size < grad_flat.size:
            indices = np.linspace(0, grad_flat.size - 1, sample_size, dtype=np.int64)
            grad_sample = grad_flat[indices]
        else:
            grad_sample = grad_flat
    else:
        grad_sample = grad_flat

    if len(grad_flat) == 0:
        return 0.0, {'note': 'No significant gradients found'}

    # 1. Gradient distribution analysis
    try:
        log_grad = np.log(grad_sample + 1e-8)
        mu, sigma = stats.norm.fit(log_grad)

        # Test goodness of fit on the stratified sample to avoid degenerate tiny p-values.
        _, p_value = stats.kstest(log_grad, lambda x: stats.norm.cdf(x, mu, sigma))

        if p_value < 0.02:
            lognormal_anomaly = np.clip((0.02 - p_value) / 0.02, 0.0, 1.0)
        else:
            lognormal_anomaly = 0.0

    except Exception:
        lognormal_anomaly = 0.4
        p_value = 0.0
        mu, sigma = 0.0, 1.0

    # 2. Gradient direction coherence
    grad_direction = np.arctan2(grad_y, grad_x)
    
    # Calculate local coherence
    coherence_scores = []
    h, w = gray.shape
    window_size = 16
    
    for y in range(0, h - window_size, window_size):
        for x in range(0, w - window_size, window_size):
            window_dirs = grad_direction[y:y + window_size, x:x + window_size]
            window_mags = grad_magnitude[y:y + window_size, x:x + window_size]
            
            # Only consider significant gradients
            mask = window_mags > np.percentile(window_mags, 50)
            if np.any(mask):
                dirs = window_dirs[mask]
                
                # Calculate circular variance (measure of direction consistency)
                cos_sum = np.sum(np.cos(dirs))
                sin_sum = np.sum(np.sin(dirs))
                r = np.sqrt(cos_sum**2 + sin_sum**2) / len(dirs)
                coherence = r  # High coherence = directions are similar
                
                coherence_scores.append(coherence)
    
    if len(coherence_scores) > 0:
        avg_coherence = float(np.mean(coherence_scores))
        coherence_variance = float(np.var(coherence_scores))
        if coherence_variance > 0.12:
            coherence_anomaly = np.clip((coherence_variance - 0.12) / 0.2, 0.0, 1.0)
        elif avg_coherence < 0.25:
            coherence_anomaly = np.clip((0.25 - avg_coherence) / 0.25, 0.0, 1.0)
        else:
            coherence_anomaly = 0.0
    else:
        coherence_anomaly = 0.0
        avg_coherence = 0.0
        coherence_variance = 0.0

    gradient_kurtosis = float(stats.kurtosis(grad_sample, fisher=False)) if len(grad_sample) > 100 else 3.0
    kurtosis_anomaly = 0.0
    if gradient_kurtosis < 2.4:
        kurtosis_anomaly = np.clip((2.4 - gradient_kurtosis) / 1.5, 0.0, 1.0)
    elif gradient_kurtosis > 9.0:
        kurtosis_anomaly = np.clip((gradient_kurtosis - 9.0) / 9.0, 0.0, 0.6)

    gradient_score = (
        0.4 * lognormal_anomaly +
        0.35 * coherence_anomaly +
        0.25 * kurtosis_anomaly
    )

    meta = {
        'lognormal_p_value': float(p_value),
        'lognormal_mu': float(mu),
        'lognormal_sigma': float(sigma),
        'avg_gradient_coherence': float(avg_coherence),
        'coherence_variance': float(coherence_variance),
        'gradient_kurtosis': float(gradient_kurtosis),
        'num_significant_gradients': int(len(grad_flat)),
        'sampled_gradients': int(len(grad_sample))
    }

    return float(np.clip(gradient_score, 0.0, 1.0)), meta


def _analyze_color_channel_correlations(rgb_u8: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    """
    Analyze correlations between color channels.
    AI images often have unnatural color relationships.
    """
    # Ensure proper data types and handle edge cases
    if rgb_u8.size == 0:
        return 0.0, {'note': 'Empty image'}
    
    rgb = rgb_u8.astype(np.float64).reshape(-1, 3)  # Use float64 for stability
    
    # Remove extreme values that might skew correlation
    valid_mask = (rgb > 5) & (rgb < 250)
    valid_pixels = valid_mask.all(axis=1)
    
    if np.sum(valid_pixels) < 1000:  # Not enough data
        return 0.0, {'note': 'Insufficient valid color data'}
    
    rgb_valid = rgb[valid_pixels]
    
    # 1. Channel correlation analysis
    try:
        corr_matrix = np.corrcoef(rgb_valid.T)
        if np.any(np.isnan(corr_matrix)):
            return 0.0, {'note': 'Correlation calculation failed'}
        rg_corr = float(corr_matrix[0, 1])
        rb_corr = float(corr_matrix[0, 2])
        gb_corr = float(corr_matrix[1, 2])
    except Exception:
        return 0.0, {'note': 'Correlation analysis failed'}

    natural_corr_range = (0.35, 0.85)
    corr_anomalies = []
    for corr in [rg_corr, rb_corr, gb_corr]:
        if np.isnan(corr):
            corr_anomalies.append(0.6)
        elif corr < natural_corr_range[0]:
            corr_anomalies.append(np.clip((natural_corr_range[0] - corr) / natural_corr_range[0], 0.0, 1.0))
        elif corr > natural_corr_range[1]:
            corr_anomalies.append(np.clip((corr - natural_corr_range[1]) / (1.0 - natural_corr_range[1] + 1e-8), 0.0, 1.0))
        else:
            corr_anomalies.append(0.0)

    correlation_anomaly = float(np.mean(corr_anomalies))

    # 2. Color distribution analysis in different color spaces
    saturation_anomaly = 0.0
    hue_spread = 0.0
    sat_bimodality = 0.0
    try:
        rgb_img = rgb_u8.reshape(rgb_u8.shape[0], rgb_u8.shape[1], 3)
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        hsv_flat = hsv.reshape(-1, 3).astype(np.float64)

        saturation = hsv_flat[:, 1] / 255.0
        hue = hsv_flat[:, 0] / 180.0

        sat_hist, _ = np.histogram(saturation, bins=40, range=(0, 1))
        sat_hist = sat_hist.astype(np.float64) / (np.sum(sat_hist) + 1e-8)

        smooth_hist = uniform_filter(sat_hist, size=3)
        sat_bimodality = float(np.std(sat_hist - smooth_hist))
        saturation_anomaly = np.clip(sat_bimodality / 0.05, 0.0, 1.0)

        hue_spread = float(np.std(hue))
        if hue_spread < 0.08:
            hue_anomaly = np.clip((0.08 - hue_spread) / 0.08, 0.0, 0.8)
        else:
            hue_anomaly = 0.0
    except Exception:
        hue_anomaly = 0.0

    color_score = 0.55 * correlation_anomaly + 0.25 * saturation_anomaly + 0.2 * hue_anomaly

    meta = {
        'rg_correlation': rg_corr,
        'rb_correlation': rb_corr,
        'gb_correlation': gb_corr,
        'saturation_bimodality': float(sat_bimodality),
        'hue_spread': float(hue_spread),
        'valid_pixels': int(np.sum(valid_pixels)),
        'total_pixels': int(rgb.shape[0])
    }

    return float(np.clip(color_score, 0.0, 1.0)), meta


def _convert_to_serializable(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively convert numpy types (e.g., float32, int32) to Python native types.
    """
    if isinstance(data, dict):
        return {k: _convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_to_serializable(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    return data


def _generate_explanations(pixel_score: float, pixel_meta: Dict, 
                          spectral_score: float, spectral_meta: Dict,
                          texture_score: float, texture_meta: Dict,
                          gradient_score: float, gradient_meta: Dict,
                          color_score: float, color_meta: Dict) -> list[str]:
    """Generate human-readable explanations for AI detection results."""
    explanations = []
    
    # Pixel distribution analysis
    avg_entropy = float(pixel_meta.get('avg_entropy', 0.0) or 0.0)
    low_high_ratio = float(pixel_meta.get('avg_low_high_ratio', 0.0) or 0.0)
    ks_stat = float(pixel_meta.get('avg_ks_stat', 0.0) or 0.0)
    if pixel_score >= 0.7:
        explanations.append(
            f"Pixel statistics deviate from natural camera captures (entropy={avg_entropy:.2f}, tail ratio={low_high_ratio:.2f})."
        )
        log_analysis_step(_logger, "pixel", "High pixel distribution anomaly", pixel_score,
                          entropy=avg_entropy, tail_ratio=low_high_ratio, ks=ks_stat)
    elif pixel_score >= 0.5:
        explanations.append("Pixel distribution shows moderate irregularities consistent with AI generation.")
        log_analysis_step(_logger, "pixel", "Moderate pixel distribution anomalies", pixel_score,
                          entropy=avg_entropy, ks=ks_stat)
    elif pixel_score >= 0.35:
        explanations.append("Pixel distribution contains slight banding/quantisation artefacts.")
        log_analysis_step(_logger, "pixel", "Low pixel anomaly", pixel_score)

    # Spectral analysis
    high_freq_ratio = float(spectral_meta.get('high_frequency_energy_ratio', 0.0) or 0.0)
    ring_strength = float(spectral_meta.get('ring_strength', 0.0) or 0.0)
    slope = float(spectral_meta.get('power_law_slope', 0.0) or 0.0)
    if spectral_score >= 0.65:
        explanations.append(
            f"Frequency spectrum retains excess high-frequency energy (ratio={high_freq_ratio:.2f}) and shows synthetic ringing."
        )
        log_analysis_step(_logger, "spectral", "High spectral anomaly", spectral_score,
                          high_freq_ratio=high_freq_ratio, ring_strength=ring_strength, slope=slope)
    elif spectral_score >= 0.5:
        explanations.append("Frequency domain shows moderate deviations from camera power spectrum.")
        log_analysis_step(_logger, "spectral", "Moderate spectral anomalies", spectral_score,
                          slope=slope)

    # Texture analysis
    entropy_var = float(texture_meta.get('texture_entropy_variance', 0.0) or 0.0)
    lap_energy = float(texture_meta.get('laplacian_energy', 0.0) or 0.0)
    if texture_score >= 0.65:
        explanations.append(
            f"Micro-textures fluctuate unnaturally (entropy variance={entropy_var:.2f}), a hallmark of AI synthesis."
        )
        log_analysis_step(_logger, "texture", "High texture anomaly", texture_score,
                          entropy_variance=entropy_var, laplacian=lap_energy)
    elif texture_score >= 0.5:
        explanations.append("Texture statistics vary more than typical photographs.")
        log_analysis_step(_logger, "texture", "Moderate texture anomaly", texture_score)

    # Gradient analysis
    gradient_kurtosis = float(gradient_meta.get('gradient_kurtosis', 3.0) or 3.0)
    p_value = float(gradient_meta.get('lognormal_p_value', 1.0) or 1.0)
    if gradient_score >= 0.6:
        explanations.append(
            f"Gradient magnitudes diverge from a natural log-normal distribution (p={p_value:.3f}, kurtosis={gradient_kurtosis:.2f})."
        )
        log_analysis_step(_logger, "gradient", "High gradient anomaly", gradient_score,
                          lognormal_p=p_value, kurtosis=gradient_kurtosis)
    elif gradient_score >= 0.45:
        explanations.append("Edge structure shows moderate synthetic coherence patterns.")
        log_analysis_step(_logger, "gradient", "Moderate gradient anomaly", gradient_score)

    # Color correlation analysis
    rg_corr = float(color_meta.get('rg_correlation', 0.0) or 0.0)
    sat_bimodality = float(color_meta.get('saturation_bimodality', 0.0) or 0.0)
    hue_spread = float(color_meta.get('hue_spread', 0.0) or 0.0)
    if color_score >= 0.55:
        explanations.append(
            f"Colour channels are tightly coupled (R-G corr={rg_corr:.2f}) with stylised saturation patterns (bimodality={sat_bimodality:.3f})."
        )
        log_analysis_step(_logger, "color", "High color anomaly", color_score,
                          rg_correlation=rg_corr, saturation_bimodality=sat_bimodality)
    elif color_score >= 0.4:
        explanations.append("Colour relationships deviate mildly from camera captures.")
        log_analysis_step(_logger, "color", "Moderate color anomaly", color_score,
                          hue_spread=hue_spread)

    return explanations


def run_ai_detection(rgb_u8: np.ndarray) -> AIDetectionResult:
    """
    Comprehensive AI detection using multiple concurrent analyses.
    """
    log_analysis_step(_logger, "init", f"Starting AI detection on {rgb_u8.shape[1]}x{rgb_u8.shape[0]} image")
    
    # Define analysis functions
    analysis_functions = {
        'pixel': _analyze_pixel_distribution,
        'spectral': _analyze_spectral_characteristics, 
        'texture': _analyze_local_texture_patterns,
        'gradient': _analyze_gradient_distributions,
        'color': _analyze_color_channel_correlations,
    }
    
    # Run analyses concurrently
    results = {}
    max_workers = min(len(analysis_functions), (os.cpu_count() or 4))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ai-det") as executor:
        # Submit all tasks
        future_to_name = {
            executor.submit(func, rgb_u8): name 
            for name, func in analysis_functions.items()
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                score, meta = future.result(timeout=30)  # 30 second timeout per analysis
                results[name] = (float(score), dict(meta or {}))
                log_analysis_step(_logger, name, f"Analysis completed", score)
            except Exception as e:
                log_analysis_step(_logger, name, f"Analysis failed: {str(e)}", 0.0)
                results[name] = (0.0, {"error": str(e)})
    
    # Extract results
    pixel_score, pixel_meta = results['pixel']
    spectral_score, spectral_meta = results['spectral']
    texture_score, texture_meta = results['texture']
    gradient_score, gradient_meta = results['gradient']
    color_score, color_meta = results['color']
    
    # Weighted combination tuned for contemporary generative models
    effective_gradient = float(np.clip(1.0 - gradient_score, 0.0, 1.0))
    texture_variance = float(texture_meta.get('texture_entropy_variance', 0.0) or 0.0)

    weights = {
        'pixel': 0.27,
        'spectral': 0.24,
        'texture': 0.1,
        'gradient_inverse': 0.26,
        'color': 0.13,
    }

    weighted_sum = (
        weights['pixel'] * pixel_score +
        weights['spectral'] * spectral_score +
        weights['texture'] * texture_score +
        weights['gradient_inverse'] * effective_gradient +
        weights['color'] * color_score
    )

    # Penalise extremely smooth textures that are characteristic of untouched camera captures.
    smoothness_penalty = 0.25 * np.clip((0.12 - texture_variance) / 0.12, 0.0, 1.0)
    weighted_sum = float(np.clip(weighted_sum - smoothness_penalty, 0.0, 1.0))

    # Calibrate overall probability with a gentle nonlinear boost for corroborating evidence.
    overall_probability = float(np.clip(weighted_sum ** 0.9, 0.0, 1.0))
    
    # Generate explanations and log them
    explanations = _generate_explanations(
        pixel_score, pixel_meta, spectral_score, spectral_meta,
        texture_score, texture_meta, gradient_score, gradient_meta,
        color_score, color_meta
    )
    
    # Log final result
    log_analysis_step(_logger, "final", 
                     f"AI detection complete - overall probability: {overall_probability:.3f}", 
                     overall_probability,
                     component_scores={
                         'pixel': pixel_score, 'spectral': spectral_score,
                         'texture': texture_score, 'gradient': gradient_score, 
                         'color': color_score
                     })
    
    # Combine all metadata with explanations
    combined_meta = _convert_to_serializable({
        'pixel_analysis': pixel_meta,
        'spectral_analysis': spectral_meta,
        'texture_analysis': texture_meta,
        'gradient_analysis': gradient_meta,
        'color_analysis': color_meta,
        'weights_used': weights,
        'effective_gradient': effective_gradient,
        'smoothness_penalty': smoothness_penalty,
        'explanations': explanations,
        'note': 'Comprehensive concurrent AI detection using multiple forensic techniques'
    })
    
    return AIDetectionResult(
        pixel_distribution_score=float(pixel_score),
        spectral_anomaly_score=float(spectral_score),
        texture_consistency_score=float(texture_score),
        gradient_distribution_score=float(gradient_score),
        color_correlation_score=float(color_score),
        overall_ai_probability=float(overall_probability),
        explanations=list(explanations),
        meta=combined_meta
    )
