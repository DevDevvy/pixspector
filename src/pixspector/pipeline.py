from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures

import cv2
import numpy as np

from .config import Config, SandboxConfig
from .core.image_io import load_image, to_float32, ensure_color, LoadedImage
from .core.metadata import read_metadata
from .core import c2pa as c2pa_mod
from .core import evidence as evidence_mod

from .analysis.ela import run_ela
from .analysis.jpeg_ghosts import run_jpeg_ghosts
from .analysis.dct_benford import run_dct_benford
from .analysis.resampling import run_resampling_map
from .analysis.cfa import run_cfa_map
from .analysis.prnu import run_prnu
from .analysis.fft_checks import run_fft_checks
from .analysis.ai_detection import run_ai_detection

from .scoring.rules import score_image
from .utils.visuals import (
    save_image_png,
    save_grayscale_png,
    save_heatmap_png,
    save_overlay_png,
    plot_curve,
)

from .report.report import save_json_report, save_pdf_report
from .utils.logging import get_logger, log_analysis_step

# Pipeline logger  
_logger = get_logger("pipeline")


def _provenance_flags(meta: Dict[str, Any], c2pa_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    exif_consistent = bool(meta.get("format")) and bool(meta.get("size"))
    c2pa_valid = False
    if c2pa_result and c2pa_result.get("found") and c2pa_result.get("valid"):
        c2pa_valid = True
    return {
        "c2pa_valid": c2pa_valid,
        "exif_consistent": exif_consistent,
    }


def _convert_to_serializable(data: Any) -> Any:
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


def analyze_single_image(
    image_path: Path,
    cfg: Config,
    out_dir: Path,
    want_pdf: bool = True,
) -> Dict[str, Any]:
    """
    Run the full pipeline on one image with concurrent analysis and detailed logging.
    """
    log_analysis_step(_logger, "init", f"Starting analysis of {image_path.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    art_dir = out_dir / f"{stem}_artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    intake = evidence_mod.intake_file(image_path, out_dir)

    # --- Load ---------------------------------------------------------------
    sandbox_cfg = SandboxConfig.from_config(cfg)
    loaded: LoadedImage = load_image(
        image_path,
        max_dim=int(cfg.get("app.max_image_dim", 4096)),
        sandbox=sandbox_cfg,
    )
    bgr = loaded.bgr
    rgb = loaded.rgb
    gray = loaded.gray

    # Save the input (downscaled if applied) for reference
    save_image_png(bgr, art_dir / "input_image.png", title="Input (analysis resolution)")
    
    log_analysis_step(_logger, "load", f"Image loaded: {loaded.width}x{loaded.height} (scale factor: {loaded.scale_factor:.2f})")

    # --- Metadata & C2PA ----------------------------------------------------
    md = read_metadata(image_path)
    c2pa_raw = None
    c2pa_found = False
    c2pa_valid = False
    if c2pa_mod.has_c2patool():
        c2pa_res = c2pa_mod.verify(image_path)
        c2pa_found = c2pa_res.found
        c2pa_valid = c2pa_res.valid
        # keep raw json if not too large
        c2pa_raw = c2pa_res.raw_json if c2pa_res.raw_json and len(json.dumps(c2pa_res.raw_json)) < 200_000 else None
    
    # Determine if image is JPEG format
    is_jpeg = md.format and md.format.upper() in ['JPEG', 'JPG']
    log_analysis_step(_logger, "metadata", f"Format: {md.format}, JPEG: {is_jpeg}, C2PA found: {c2pa_found}, valid: {c2pa_valid}")

    # --- Concurrent Forensic Analysis ---
    log_analysis_step(_logger, "forensics", "Starting concurrent forensic analyses")

    # Define analysis tasks that can run concurrently
    def run_ela_task():
        log_analysis_step(_logger, "ela", "Starting ELA analysis")
        ela_cfg = cfg.get("modules.ela", {})
        ela = run_ela(bgr, recompress_quality=int(ela_cfg.get("recompress_quality", 90)),
                      blur_sigma=float(ela_cfg.get("blur_sigma", 0.0)))
        cv2.imwrite(str(art_dir / "ela_diff.png"), ela.diff_u8)
        log_analysis_step(_logger, "ela", f"ELA completed - mean diff: {ela.mean_abs_diff:.3f}, strong regions: {ela.strong_regions_ratio:.3f}")
        return "ela", _convert_to_serializable({
            "mean_abs_diff": ela.mean_abs_diff,
            "p95_abs_diff": ela.p95_abs_diff,
            "strong_regions_ratio": ela.strong_regions_ratio,
            "meta": ela.meta,
        })

    def run_jpeg_ghosts_task():
        if is_jpeg:
            log_analysis_step(_logger, "jpeg_ghosts", "Starting JPEG ghosts analysis")
            jg_cfg = cfg.get("modules.jpeg_ghosts", {})
            jg = run_jpeg_ghosts(
                bgr,
                q_grid=list(jg_cfg.get("q_grid", [60, 70, 80, 85, 90, 95])),
                diff_metric=str(jg_cfg.get("diff_metric", "l1")),
            )
            cv2.imwrite(str(art_dir / "jpeg_ghosts_best_diff.png"), jg.best_diff_u8)
            q_vals = [q for q, _ in jg.quality_curve]
            sc_vals = [s for _, s in jg.quality_curve]
            plot_curve(sc_vals, art_dir / "jpeg_quality_curve.png", title="JPEG Ghosts: mean diff vs Q",
                       xlabel="index of Q-grid", ylabel="mean diff")
            with open(art_dir / "jpeg_quality_grid.txt", "w") as f:
                f.write(", ".join(map(str, q_vals)) + "\n")
            log_analysis_step(_logger, "jpeg_ghosts", f"JPEG ghosts completed - best quality: {jg.best_quality}, misalignment: {jg.misalignment_score:.3f}")
            return "jpeg_ghosts", _convert_to_serializable({
                "best_quality": jg.best_quality,
                "quality_curve": jg.quality_curve,
                "misalignment_score": jg.misalignment_score,
                "meta": jg.meta,
            })
        else:
            log_analysis_step(_logger, "jpeg_ghosts", f"Skipping JPEG ghosts - format is {md.format}, not JPEG")
        return "jpeg_ghosts", None

    def run_dct_benford_task():
        if is_jpeg:
            log_analysis_step(_logger, "dct_benford", "Starting DCT Benford analysis")
            ben_cfg = cfg.get("modules.dct_benford", {})
            ben = run_dct_benford(gray, min_blocks=int(ben_cfg.get("min_blocks", 256)),
                                  strong_z=float(ben_cfg.get("strong_z", 3.0)),
                                  moderate_z=float(ben_cfg.get("moderate_z", 2.0)))
            plot_curve(ben.freq.tolist(), art_dir / "benford_observed.png", title="DCT Benford: observed", xlabel="digit 1..9")
            plot_curve(ben.expected.tolist(), art_dir / "benford_expected.png", title="DCT Benford: expected", xlabel="digit 1..9")
            log_analysis_step(_logger, "dct_benford", f"DCT Benford completed - chi2: {ben.chi2:.3f}, strong: {ben.strong}, moderate: {ben.moderate}")
            return "dct_benford", _convert_to_serializable({
                "n_ac": ben.n_ac,
                "freq": ben.freq.tolist(),
                "expected": ben.expected.tolist(),
                "chi2": ben.chi2,
                "strong": ben.strong,
                "moderate": ben.moderate,
                "meta": ben.meta,
            })
        else:
            log_analysis_step(_logger, "dct_benford", f"Skipping DCT Benford - format is {md.format}, not JPEG")
        return "dct_benford", None

    def run_resampling_task():
        log_analysis_step(_logger, "resampling", "Starting resampling detection")
        rs_cfg = cfg.get("modules.resampling", {})
        rs = run_resampling_map(
            gray,
            patch=int(rs_cfg.get("patch", 128)),
            stride=int(rs_cfg.get("stride", 64)),
            thr_strong=float(rs_cfg.get("periodicity_threshold_strong", 0.35)),
            thr_moderate=float(rs_cfg.get("periodicity_threshold_moderate", 0.25)),
        )
        log_analysis_step(_logger, "resampling", f"Resampling completed - strong ratio: {rs.strong_ratio:.3f}, moderate ratio: {rs.moderate_ratio:.3f}")
        return "resampling", _convert_to_serializable({
            "strong_ratio": rs.strong_ratio,
            "moderate_ratio": rs.moderate_ratio,
            "strong_threshold": rs.strong_threshold,
            "moderate_threshold": rs.moderate_threshold,
            "meta": rs.meta,
        })

    def run_cfa_task():
        log_analysis_step(_logger, "cfa", "Starting CFA analysis")
        cfa_cfg = cfg.get("modules.cfa", {})
        cfa = run_cfa_map(
            rgb,
            patch=int(cfa_cfg.get("patch", 64)),
            stride=int(cfa_cfg.get("stride", 32)),
            thr_strong=float(cfa_cfg.get("inconsistency_threshold_strong", 0.35)),
            thr_moderate=float(cfa_cfg.get("inconsistency_threshold_moderate", 0.25)),
        )
        log_analysis_step(_logger, "cfa", f"CFA completed - strong ratio: {cfa.strong_ratio:.3f}, moderate ratio: {cfa.moderate_ratio:.3f}")
        return "cfa", _convert_to_serializable({
            "strong_ratio": cfa.strong_ratio,
            "moderate_ratio": cfa.moderate_ratio,
            "strong_threshold": cfa.strong_threshold,
            "moderate_threshold": cfa.moderate_threshold,
            "meta": cfa.meta,
        })

    def run_prnu_task():
        log_analysis_step(_logger, "prnu", "Starting PRNU analysis")
        pr_cfg = cfg.get("modules.prnu", {})
        pr = run_prnu(
            gray,
            wavelet_denoise_sigma=float(pr_cfg.get("wavelet_denoise_sigma", 3.0)),
            ref_pattern=None,
            correlate_min_nz=float(pr_cfg.get("correlate_min_nz", 0.05)),
        )
        log_analysis_step(_logger, "prnu", f"PRNU completed - mean abs residual: {pr.mean_abs_residual:.3f}, p95: {pr.p95_abs_residual:.3f}")
        return "prnu", _convert_to_serializable({
            "mean_abs_residual": pr.mean_abs_residual,
            "p95_abs_residual": pr.p95_abs_residual,
            "correlation_with_ref": pr.correlation_with_ref,
            "meta": pr.meta,
        })

    def run_fft_task():
        log_analysis_step(_logger, "fft", "Starting FFT analysis")
        fft_cfg = cfg.get("modules.fft_checks", {})
        fft = run_fft_checks(
            gray,
            radial_bins=int(fft_cfg.get("radial_bins", 64)),
            peak_prominence=float(fft_cfg.get("peak_prominence", 6.0)),
            highfreq_rolloff_warn=float(fft_cfg.get("highfreq_rolloff_warn", 0.75)),
        )
        log_analysis_step(_logger, "fft", f"FFT completed - {len(fft.peak_indices)} peaks detected, rolloff: {fft.highfreq_rolloff:.3f}")
        return "fft_checks", _convert_to_serializable({
            "radial_profile": fft.radial_profile,
            "peak_indices": fft.peak_indices,
            "highfreq_rolloff": fft.highfreq_rolloff,
            "meta": fft.meta,
        })

    def run_ai_detection_task():
        log_analysis_step(_logger, "ai_detection", "Starting AI detection analysis")
        ai_detection = run_ai_detection(rgb)
        
        # Log detailed AI detection results with explanations
        log_analysis_step(_logger, "ai_detection", f"AI detection completed - overall probability: {ai_detection.overall_ai_probability:.3f}")
        log_analysis_step(_logger, "ai_detection", f"  Pixel distribution score: {ai_detection.pixel_distribution_score:.3f}")
        log_analysis_step(_logger, "ai_detection", f"  Spectral anomaly score: {ai_detection.spectral_anomaly_score:.3f}")
        log_analysis_step(_logger, "ai_detection", f"  Texture consistency score: {ai_detection.texture_consistency_score:.3f}")
        log_analysis_step(_logger, "ai_detection", f"  Gradient distribution score: {ai_detection.gradient_distribution_score:.3f}")
        log_analysis_step(_logger, "ai_detection", f"  Color correlation score: {ai_detection.color_correlation_score:.3f}")
        
        # Log human-readable explanations
        for explanation in ai_detection.explanations:
            log_analysis_step(_logger, "ai_explanation", explanation)
            
        return "ai_detection", _convert_to_serializable(ai_detection.to_dict())

    # Execute analyses concurrently - optimize for performance
    analysis_tasks = [
        run_ai_detection_task,  # Prioritize AI detection
        run_ela_task,
        run_resampling_task,
        run_cfa_task,
        run_prnu_task,
        run_fft_task,
    ]
    
    # Add JPEG-specific tasks only for JPEG images
    if is_jpeg:
        analysis_tasks.extend([run_jpeg_ghosts_task, run_dct_benford_task])
        log_analysis_step(_logger, "format", f"Added JPEG-specific analyses for {md.format} image")
    else:
        log_analysis_step(_logger, "format", f"Skipping JPEG-specific analyses for {md.format} image")

    modules = {}
    # Use optimal number of workers - don't exceed CPU count
    max_workers = min(len(analysis_tasks), max(1, (os.cpu_count() or 4) - 1))
    log_analysis_step(_logger, "concurrency", f"Running {len(analysis_tasks)} analyses with {max_workers} workers")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="forensic") as executor:
        # Submit all tasks
        future_to_task = {executor.submit(task): task.__name__ for task in analysis_tasks}

        # Collect results as they complete
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_task, timeout=300):  # 5 minute total timeout
            task_name = future_to_task[future]
            try:
                module_name, result = future.result(timeout=30)  # 30 second timeout per analysis
                if result is not None:
                    modules[module_name] = result
                completed_count += 1
                log_analysis_step(_logger, "progress", f"Completed {module_name} ({completed_count}/{len(analysis_tasks)})")
            except concurrent.futures.TimeoutError:
                log_analysis_step(_logger, "warning", f"Task {task_name} timed out")
            except Exception as e:
                log_analysis_step(_logger, "error", f"Task {task_name} failed: {str(e)}")

    # Provenance evidence flags (for scoring)
    prov = _provenance_flags(
        meta={
            "format": md.format,
            "size": md.size,
        },
        c2pa_result={
            "found": c2pa_found,
            "valid": c2pa_valid,
        } if c2pa_found or c2pa_valid else None,
    )
    modules["provenance"] = prov

    # --- Scoring ------------------------------------------------------------
    log_analysis_step(_logger, "scoring", "Starting scoring analysis")
    log_analysis_step(_logger, "modules_available", f"Available modules for scoring: {list(modules.keys())}")
    
    # Log some key values before scoring
    if "ai_detection" in modules:
        ai_prob = modules["ai_detection"].get("overall_ai_probability", 0)
        log_analysis_step(_logger, "pre_scoring", f"AI probability before scoring: {ai_prob}")
    
    score = score_image(
        modules=modules,
        weights_cfg=cfg.get("rules.weights", {}),
        clamp_min=float(cfg.get("rules.clamp_min", 0)),
        clamp_max=float(cfg.get("rules.clamp_max", 100)),
        buckets_cfg=cfg.get("rules.buckets", {}),
        ai_component_gate=float(cfg.get("rules.ai_component_gate", 0.4)),
    )
    log_analysis_step(_logger, "scoring", f"Scoring completed - final score: {score.suspicion_index}, bucket: {score.bucket_label}, evidence count: {len(score.evidence)}")

    # --- Assemble report dict ----------------------------------------------
    result: Dict[str, Any] = _convert_to_serializable({
        "input": str(image_path),
        "sha256": intake["hashes"].get("sha256", loaded.sha256),
        "intake": intake,
        "width": loaded.width,
        "height": loaded.height,
        "scale_factor": loaded.scale_factor,
        "metadata": {
            "format": md.format,
            "size": md.size,
            "icc_profile_present": md.icc_profile_present,
            "icc_description": md.icc_description,
            "jpeg_subsampling": md.jpeg_subsampling,
            "jpeg_quant_tables_present": bool(md.jpeg_quant_tables),
            "exif": md.exif,  # can be verbose
        },
        "c2pa": {
            "checked": c2pa_mod.has_c2patool(),
            "found": c2pa_found,
            "valid": c2pa_valid,
            "raw": c2pa_raw,
        },
        "modules": modules,
        "suspicion_index": score.suspicion_index,
        "bucket_label": score.bucket_label,
        "evidence": [e.to_dict() for e in score.evidence],
        "notes": score.notes,
        "artifacts_dir": str(art_dir),
        "version": "0.1.0",
    })

    # --- Save JSON + PDF ----------------------------------------------------
    json_path = out_dir / f"{stem}_report.json"
    pdf_path = out_dir / f"{stem}_report.pdf"
    save_json_report(result, json_path)
    if want_pdf:
        save_pdf_report(result, pdf_path, visuals_dir=art_dir)

    log_analysis_step(_logger, "complete", f"Analysis of {image_path.name} completed successfully")

    return result
