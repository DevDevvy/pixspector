from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
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
    channel_stats = {}
    
    for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
        channel = rgb[..., ch_idx].flatten()
        
        # 1. Test for unnaturally uniform distribution
        hist, _ = np.histogram(channel, bins=256, range=(0, 1))
        hist_norm = hist / (np.sum(hist) + 1e-8)
        
        # Calculate entropy - AI images often have either too high or too low entropy
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))
        natural_entropy_range = (6.0, 7.8)  # Typical for natural images
        
        if entropy < natural_entropy_range[0]:
            entropy_anomaly = (natural_entropy_range[0] - entropy) / natural_entropy_range[0]
        elif entropy > natural_entropy_range[1]:
            entropy_anomaly = (entropy - natural_entropy_range[1]) / (8.0 - natural_entropy_range[1])
        else:
            entropy_anomaly = 0.0
        
        # 2. Check for clustering (AI images often have pixel values clustered)
        # Use Kolmogorov-Smirnov test against uniform distribution
        uniform_samples = np.random.uniform(0, 1, len(channel))
        ks_stat, _ = stats.kstest(channel, uniform_samples)
        
        # 3. Check for gaps in distribution (common in AI images)
        gaps_score = 0.0
        for i in range(1, len(hist)):
            if hist[i-1] > 0 and hist[i] == 0 and i < len(hist)-1 and hist[i+1] > 0:
                gaps_score += 1.0
        gaps_score = gaps_score / 256.0
        
        # 4. Statistical moments analysis
        skewness = stats.skew(channel)
        kurtosis = stats.kurtosis(channel) + 3  # Convert to standard kurtosis
        
        # AI images often have unusual higher-order moments
        moment_anomaly = 0.0
        natural_skewness_range = (-0.5, 0.5)
        natural_kurtosis_range = (2.5, 4.0)
        
        if not (natural_skewness_range[0] <= skewness <= natural_skewness_range[1]):
            moment_anomaly += 0.3
        if not (natural_kurtosis_range[0] <= kurtosis <= natural_kurtosis_range[1]):
            moment_anomaly += 0.7
        
        # Combine scores for this channel
        channel_score = 0.3 * entropy_anomaly + 0.3 * ks_stat + 0.2 * gaps_score + 0.2 * moment_anomaly
        channel_scores.append(channel_score)
        
        # Convert numpy types to Python types for JSON serialization
        channel_stats[f'{ch_name}_entropy'] = float(entropy)
        channel_stats[f'{ch_name}_ks_stat'] = float(ks_stat)
        channel_stats[f'{ch_name}_gaps'] = float(gaps_score)
        channel_stats[f'{ch_name}_skewness'] = float(skewness)
        channel_stats[f'{ch_name}_kurtosis'] = float(kurtosis)
    
    overall_score = np.mean(channel_scores)
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
    if len(radial_profile) > 10:
        # Fit power law to radial profile
        freqs = np.arange(1, len(radial_profile) + 1)
        try:
            # Natural images typically follow f^(-2) power law
            log_freqs = np.log(freqs[1:])  # Skip DC
            log_profile = radial_profile[1:]
            
            # Linear regression in log space
            slope, intercept, r_value, _, _ = stats.linregress(log_freqs, log_profile)
            
            # Natural images: slope around -1.5 to -2.5
            natural_slope_range = (-2.5, -1.5)
            if slope < natural_slope_range[0] or slope > natural_slope_range[1]:
                slope_anomaly = min(1.0, abs(slope - np.mean(natural_slope_range)) / 2.0)
            else:
                slope_anomaly = 0.0
                
        except:
            slope_anomaly = 0.0
            slope = 0.0
            r_value = 0.0
    else:
        slope_anomaly = 0.0
        slope = 0.0
        r_value = 0.0
    
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
    
    # Natural images have more varied directional content
    if len(angle_energies) > 0:
        directional_variance = np.var(angle_energies) / (np.mean(angle_energies) + 1e-8)
        # AI images often have too uniform directional content
        if directional_variance < 0.1:  # Too uniform
            directional_anomaly = (0.1 - directional_variance) / 0.1
        else:
            directional_anomaly = 0.0
    else:
        directional_anomaly = 0.0
        directional_variance = 0.0
    
    # Combine spectral anomalies
    spectral_score = 0.6 * slope_anomaly + 0.4 * directional_anomaly
    
    meta = {
        'power_law_slope': float(slope),
        'power_law_r2': float(r_value**2),
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
    
    # 1. Uniform patterns analysis
    # Natural images have more uniform LBP patterns
    uniform_patterns = 0.0
    for i in range(256):
        # Count transitions in binary representation
        binary = format(i, '08b')
        transitions = sum(binary[j] != binary[(j+1) % 8] for j in range(8))
        if transitions <= 2:
            uniform_patterns += lbp_hist[i]
    
    # AI images often have fewer uniform patterns
    if uniform_patterns < 0.6:  # Threshold for natural images
        uniform_anomaly = (0.6 - uniform_patterns) / 0.6
    else:
        uniform_anomaly = 0.0
    
    # 2. Spatial consistency of textures
    h, w = gray.shape
    patch_size = min(32, h//4, w//4)  # Ensure patch fits in image
    if patch_size < 8:  # Too small image
        return 0.0, {'note': 'Image too small for texture analysis'}
    
    texture_consistency_scores = []
    
    step_size = max(1, patch_size // 2)
    for y in range(0, h - patch_size + 1, step_size):
        for x in range(0, w - patch_size + 1, step_size):
            patch = gray[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                try:
                    patch_lbp = calculate_lbp(patch)
                    patch_hist, _ = np.histogram(patch_lbp.flatten(), bins=256, range=(0, 256))
                    patch_hist = patch_hist.astype(np.float64) / (np.sum(patch_hist) + 1e-8)
                    
                    # Calculate entropy of this patch
                    patch_entropy = -np.sum(patch_hist * np.log2(patch_hist + 1e-8))
                    texture_consistency_scores.append(patch_entropy)
                except Exception:
                    # Skip problematic patches
                    continue
    
    if len(texture_consistency_scores) > 1:
        # AI images often have very inconsistent texture entropy across patches
        texture_variance = np.var(texture_consistency_scores)
        # High variance indicates inconsistent AI-generated textures
        texture_anomaly = min(1.0, texture_variance / 2.0)  # Normalize
    else:
        texture_anomaly = 0.0
    
    overall_texture_score = 0.5 * uniform_anomaly + 0.5 * texture_anomaly
    
    meta = {
        'uniform_patterns_ratio': float(uniform_patterns),
        'texture_entropy_variance': float(texture_variance if 'texture_variance' in locals() else 0.0),
        'avg_texture_entropy': float(np.mean(texture_consistency_scores) if texture_consistency_scores else 0.0),
        'patch_count': len(texture_consistency_scores)
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
    
    if len(grad_flat) == 0:
        return 0.0, {'note': 'No significant gradients found'}
    
    # 1. Gradient distribution analysis
    try:
        log_grad = np.log(grad_flat + 1e-8)
        mu, sigma = stats.norm.fit(log_grad)
        
        # Test goodness of fit
        _, p_value = stats.kstest(log_grad, lambda x: stats.norm.cdf(x, mu, sigma))
        
        # Natural images typically have good log-normal fit (p > 0.01)
        if p_value < 0.01:
            lognormal_anomaly = 1.0 - p_value / 0.01
        else:
            lognormal_anomaly = 0.0
            
    except:
        lognormal_anomaly = 0.5  # Moderate suspicion if fitting fails
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
        avg_coherence = np.mean(coherence_scores)
        coherence_variance = np.var(coherence_scores)
        
        # AI images often have either too high or too low coherence variance
        if coherence_variance > 0.15 or avg_coherence < 0.2:
            coherence_anomaly = min(1.0, max(coherence_variance / 0.15, 
                                           (0.2 - avg_coherence) / 0.2))
        else:
            coherence_anomaly = 0.0
    else:
        coherence_anomaly = 0.0
        avg_coherence = 0.0
        coherence_variance = 0.0
    
    # Combine gradient anomalies
    gradient_score = 0.6 * lognormal_anomaly + 0.4 * coherence_anomaly
    
    meta = {
        'lognormal_p_value': float(p_value),
        'lognormal_mu': float(mu),
        'lognormal_sigma': float(sigma),
        'avg_gradient_coherence': float(avg_coherence),
        'coherence_variance': float(coherence_variance),
        'num_significant_gradients': int(len(grad_flat))
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
        
        # Handle potential NaN values
        if np.any(np.isnan(corr_matrix)):
            return 0.0, {'note': 'Correlation calculation failed'}
        
        # Extract correlations
        rg_corr = corr_matrix[0, 1]  # Red-Green
        rb_corr = corr_matrix[0, 2]  # Red-Blue  
        gb_corr = corr_matrix[1, 2]  # Green-Blue
        
    except Exception:
        return 0.0, {'note': 'Correlation analysis failed'}
    
    # Natural images typically have moderate positive correlations
    natural_corr_range = (0.3, 0.8)
    
    corr_anomalies = []
    for corr in [rg_corr, rb_corr, gb_corr]:
        if np.isnan(corr):
            corr_anomalies.append(0.5)  # Moderate suspicion for NaN
        elif corr < natural_corr_range[0]:
            corr_anomalies.append((natural_corr_range[0] - corr) / natural_corr_range[0])
        elif corr > natural_corr_range[1]:
            corr_anomalies.append((corr - natural_corr_range[1]) / (1.0 - natural_corr_range[1]))
        else:
            corr_anomalies.append(0.0)
    
    correlation_anomaly = np.mean(corr_anomalies)
    
    # 2. Color distribution analysis in different color spaces
    try:
        # Convert to HSV for additional analysis
        rgb_img = rgb_u8.reshape(rgb_u8.shape[0], rgb_u8.shape[1], 3)
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        hsv_flat = hsv.reshape(-1, 3).astype(np.float64)
        
        # Analyze saturation distribution
        saturation = hsv_flat[:, 1] / 255.0
        
        # Natural images have a characteristic saturation distribution
        sat_hist, _ = np.histogram(saturation, bins=50, range=(0, 1))
        sat_hist = sat_hist.astype(np.float64) / (np.sum(sat_hist) + 1e-8)
        
        # AI images often have bimodal or too uniform saturation
        # Check for modes in saturation
        sat_peaks = []
        for i in range(1, len(sat_hist) - 1):
            if sat_hist[i] > sat_hist[i-1] and sat_hist[i] > sat_hist[i+1] and sat_hist[i] > 0.01:
                sat_peaks.append((i, sat_hist[i]))
        
        # Sort peaks by height
        sat_peaks.sort(key=lambda x: x[1], reverse=True)
        
        if len(sat_peaks) >= 2:
            # Check if we have strong bimodal distribution (AI characteristic)
            first_peak, second_peak = sat_peaks[0][1], sat_peaks[1][1]
            if second_peak > 0.3 * first_peak:  # Significant second peak
                saturation_anomaly = min(1.0, second_peak / first_peak)
            else:
                saturation_anomaly = 0.0
        else:
            # Too few peaks might also be suspicious
            saturation_anomaly = 0.2
            
    except Exception:
        saturation_anomaly = 0.0
        sat_peaks = []
    
    # Combine color anomalies
    color_score = 0.7 * correlation_anomaly + 0.3 * saturation_anomaly
    
    meta = {
        'rg_correlation': float(rg_corr) if not np.isnan(rg_corr) else None,
        'rb_correlation': float(rb_corr) if not np.isnan(rb_corr) else None,
        'gb_correlation': float(gb_corr) if not np.isnan(gb_corr) else None,
        'saturation_peaks': int(len(sat_peaks)),
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
    if pixel_score >= 0.7:
        avg_entropy = pixel_meta.get('avg_entropy', 0)
        explanations.append(f"Pixel distribution is extremely uniform (entropy={avg_entropy:.2f}), strongly indicating AI generation")
        log_analysis_step(_logger, "pixel", f"High uniformity detected - entropy={avg_entropy:.2f}", pixel_score)
    elif pixel_score >= 0.5:
        explanations.append(f"Pixel distribution shows moderate uniformity patterns typical of AI (score={pixel_score:.2f})")
        log_analysis_step(_logger, "pixel", f"Moderate AI patterns in pixel distribution", pixel_score)
    elif pixel_score >= 0.3:
        explanations.append(f"Some pixel distribution irregularities detected")
        log_analysis_step(_logger, "pixel", f"Minor pixel distribution anomalies", pixel_score)
    
    # Spectral analysis
    if spectral_score >= 0.7:
        periodicity = spectral_meta.get('periodicity_strength', 0)
        explanations.append(f"Frequency spectrum shows strong periodic patterns (strength={periodicity:.2f}) characteristic of AI synthesis")
        log_analysis_step(_logger, "spectral", f"Strong periodic patterns detected", spectral_score, periodicity=periodicity)
    elif spectral_score >= 0.5:
        explanations.append(f"Frequency domain shows moderate AI-like artifacts (score={spectral_score:.2f})")
        log_analysis_step(_logger, "spectral", f"Moderate spectral anomalies", spectral_score)
    
    # Texture analysis
    if texture_score >= 0.7:
        variance_ratio = texture_meta.get('texture_variance_ratio', 0)
        explanations.append(f"Texture patterns are unnaturally consistent (variance ratio={variance_ratio:.2f}), indicating AI generation")
        log_analysis_step(_logger, "texture", f"Unnatural texture consistency", texture_score, variance_ratio=variance_ratio)
    elif texture_score >= 0.5:
        explanations.append(f"Texture analysis reveals moderate AI-like characteristics")
        log_analysis_step(_logger, "texture", f"Moderate texture anomalies", texture_score)
    
    # Gradient analysis  
    if gradient_score >= 0.7:
        kurtosis = gradient_meta.get('gradient_kurtosis', 0)
        explanations.append(f"Gradient distribution is highly atypical (kurtosis={kurtosis:.2f}), suggesting AI synthesis")
        log_analysis_step(_logger, "gradient", f"Atypical gradient distribution", gradient_score, kurtosis=kurtosis)
    elif gradient_score >= 0.5:
        explanations.append(f"Gradient patterns show moderate AI indicators")
        log_analysis_step(_logger, "gradient", f"Moderate gradient anomalies", gradient_score)
    
    # Color correlation analysis
    if color_score >= 0.6:
        rg_corr = color_meta.get('rg_correlation', 0)
        explanations.append(f"Color channel correlations are abnormal (R-G correlation={rg_corr:.2f}), typical of AI processing")
        log_analysis_step(_logger, "color", f"Abnormal color correlations", color_score, rg_correlation=rg_corr)
    elif color_score >= 0.4:
        explanations.append(f"Color analysis shows minor AI-like characteristics")
        log_analysis_step(_logger, "color", f"Minor color anomalies", color_score)
    
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
    
    # Enhanced weighted combination - adjusted for better AI detection
    weights = {
        'pixel': 0.35,      # Increased - pixel patterns are very telling for AI
        'spectral': 0.30,   # Increased - frequency domain is crucial
        'texture': 0.20,    # Maintained
        'gradient': 0.10,   # Decreased
        'color': 0.05       # Decreased - least reliable
    }
    
    overall_probability = (
        weights['pixel'] * pixel_score +
        weights['spectral'] * spectral_score +
        weights['texture'] * texture_score +
        weights['gradient'] * gradient_score +
        weights['color'] * color_score
    )
    
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
        meta=combined_meta
    )
