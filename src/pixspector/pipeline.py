from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

from .config import Config
from .core.image_io import load_image, to_float32, ensure_color, LoadedImage
from .core.metadata import read_metadata
from .core import c2pa as c2pa_mod

from .analysis.ela import run_ela
from .analysis.jpeg_ghosts import run_jpeg_ghosts
from .analysis.dct_benford import run_dct_benford
from .analysis.resampling import run_resampling_map
from .analysis.cfa import run_cfa_map
from .analysis.prnu import run_prnu
from .analysis.fft_checks import run_fft_checks

from .scoring.rules import score_image
from .utils.visuals import (
    save_image_png,
    save_grayscale_png,
    save_heatmap_png,
    save_overlay_png,
    plot_curve,
)

from .report.report import save_json_report, save_pdf_report


def _provenance_flags(meta: Dict[str, Any], c2pa_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    exif_consistent = bool(meta.get("format")) and bool(meta.get("size"))
    c2pa_valid = False
    if c2pa_result and c2pa_result.get("found") and c2pa_result.get("valid"):
        c2pa_valid = True
    return {
        "c2pa_valid": c2pa_valid,
        "exif_consistent": exif_consistent,
    }


def analyze_single_image(
    image_path: Path,
    cfg: Config,
    out_dir: Path,
    want_pdf: bool = True,
) -> Dict[str, Any]:
    """
    Run the full classical pipeline on one image, save artifacts, return structured result.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    art_dir = out_dir / f"{stem}_artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---------------------------------------------------------------
    loaded: LoadedImage = load_image(image_path, max_dim=int(cfg.get("app.max_image_dim", 4096)))
    bgr = loaded.bgr
    rgb = loaded.rgb
    gray = loaded.gray

    # Save the input (downscaled if applied) for reference
    save_image_png(bgr, art_dir / "input_image.png", title="Input (analysis resolution)")

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

    # --- Modules ------------------------------------------------------------
    modules: Dict[str, Any] = {}

    # ELA
    ela_cfg = cfg.get("modules.ela", {})
    ela = run_ela(bgr, recompress_quality=int(ela_cfg.get("recompress_quality", 90)),
                  blur_sigma=float(ela_cfg.get("blur_sigma", 0.0)))
    modules["ela"] = {
        "mean_abs_diff": ela.mean_abs_diff,
        "p95_abs_diff": ela.p95_abs_diff,
        "strong_regions_ratio": ela.strong_regions_ratio,
        "meta": ela.meta,
    }
    cv2.imwrite(str(art_dir / "ela_diff.png"), ela.diff_u8)

    # JPEG ghosts
    jg_cfg = cfg.get("modules.jpeg_ghosts", {})
    jg = run_jpeg_ghosts(
        bgr,
        q_grid=list(jg_cfg.get("q_grid", [60, 70, 80, 85, 90, 95])),
        diff_metric=str(jg_cfg.get("diff_metric", "l1")),
    )
    modules["jpeg_ghosts"] = {
        "best_quality": jg.best_quality,
        "quality_curve": jg.quality_curve,
        "misalignment_score": jg.misalignment_score,
        "meta": jg.meta,
    }
    cv2.imwrite(str(art_dir / "jpeg_ghosts_best_diff.png"), jg.best_diff_u8)
    # Plot curve (qualities vs score)
    q_vals = [q for q, _ in jg.quality_curve]
    sc_vals = [s for _, s in jg.quality_curve]
    plot_curve(sc_vals, art_dir / "jpeg_quality_curve.png", title="JPEG Ghosts: mean diff vs Q",
               xlabel="index of Q-grid", ylabel="mean diff")
    # Also save the Qs used
    with open(art_dir / "jpeg_quality_grid.txt", "w") as f:
        f.write(", ".join(map(str, q_vals)) + "\n")

    # DCT Benford
    ben_cfg = cfg.get("modules.dct_benford", {})
    ben = run_dct_benford(gray, min_blocks=int(ben_cfg.get("min_blocks", 256)),
                          strong_z=float(ben_cfg.get("strong_z", 3.0)),
                          moderate_z=float(ben_cfg.get("moderate_z", 2.0)))
    modules["dct_benford"] = {
        "n_ac": ben.n_ac,
        "freq": ben.freq.tolist(),
        "expected": ben.expected.tolist(),
        "chi2": ben.chi2,
        "strong": ben.strong,
        "moderate": ben.moderate,
        "meta": ben.meta,
    }
    # Save frequency curve
    plot_curve(ben.freq.tolist(), art_dir / "benford_observed.png", title="DCT Benford: observed", xlabel="digit 1..9")
    plot_curve(ben.expected.tolist(), art_dir / "benford_expected.png", title="DCT Benford: expected", xlabel="digit 1..9")

    # Resampling map
    rs_cfg = cfg.get("modules.resampling", {})
    rs = run_resampling_map(
        gray,
        patch=int(rs_cfg.get("patch", 128)),
        stride=int(rs_cfg.get("stride", 64)),
        thr_strong=float(rs_cfg.get("periodicity_threshold_strong", 0.35)),
        thr_moderate=float(rs_cfg.get("periodicity_threshold_moderate", 0.25)),
    )
    modules["resampling"] = {
        "strong_ratio": rs.strong_ratio,
        "moderate_ratio": rs.moderate_ratio,
        "strong_threshold": rs.strong_threshold,
        "moderate_threshold": rs.moderate_threshold,
        "meta": rs.meta,
    }
    save_heatmap_png(rs.heatmap_u8, art_dir / "resampling_heatmap.png", title="Resampling: periodicity heatmap")
    save_overlay_png(bgr, rs.heatmap_u8, art_dir / "resampling_overlay.png", title="Resampling: overlay")

    # CFA map
    cfa_cfg = cfg.get("modules.cfa", {})
    cfa = run_cfa_map(
        rgb,
        patch=int(cfa_cfg.get("patch", 64)),
        stride=int(cfa_cfg.get("stride", 32)),
        thr_strong=float(cfa_cfg.get("inconsistency_threshold_strong", 0.35)),
        thr_moderate=float(cfa_cfg.get("inconsistency_threshold_moderate", 0.25)),
    )
    modules["cfa"] = {
        "strong_ratio": cfa.strong_ratio,
        "moderate_ratio": cfa.moderate_ratio,
        "strong_threshold": cfa.strong_threshold,
        "moderate_threshold": cfa.moderate_threshold,
        "meta": cfa.meta,
    }
    save_heatmap_png(cfa.inconsistency_map_u8, art_dir / "cfa_inconsistency.png", title="CFA inconsistency heatmap")
    save_overlay_png(bgr, cfa.inconsistency_map_u8, art_dir / "cfa_overlay.png", title="CFA inconsistency overlay")

    # PRNU (blind residual)
    pr_cfg = cfg.get("modules.prnu", {})
    pr = run_prnu(
        gray,
        wavelet_denoise_sigma=float(pr_cfg.get("wavelet_denoise_sigma", 3.0)),
        ref_pattern=None,
        correlate_min_nz=float(pr_cfg.get("correlate_min_nz", 0.05)),
    )
    modules["prnu"] = {
        "mean_abs_residual": pr.mean_abs_residual,
        "p95_abs_residual": pr.p95_abs_residual,
        "correlation_with_ref": pr.correlation_with_ref,
        "meta": pr.meta,
    }
    save_grayscale_png(pr.residual_u8, art_dir / "prnu_residual.png", title="PRNU residual (normalized)")

    # FFT checks
    fft_cfg = cfg.get("modules.fft_checks", {})
    fft = run_fft_checks(
        gray,
        radial_bins=int(fft_cfg.get("radial_bins", 64)),
        peak_prominence=float(fft_cfg.get("peak_prominence", 6.0)),
        highfreq_rolloff_warn=float(fft_cfg.get("highfreq_rolloff_warn", 0.75)),
    )
    modules["fft_checks"] = {
        "radial_profile": fft.radial_profile,
        "peak_indices": fft.peak_indices,
        "highfreq_rolloff": fft.highfreq_rolloff,
        "meta": fft.meta,
    }
    save_grayscale_png(fft.fft_logmag_u8, art_dir / "fft_logmag.png", title="2D FFT log-magnitude")
    plot_curve(fft.radial_profile, art_dir / "fft_radial_profile.png", title="Radial spectrum", xlabel="radial bin")

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
    score = score_image(
        modules=modules,
        weights_cfg=cfg.get("rules.weights", {}),
        clamp_min=float(cfg.get("rules.clamp_min", 0)),
        clamp_max=float(cfg.get("rules.clamp_max", 100)),
        buckets_cfg=cfg.get("rules.buckets", {}),
    )

    # --- Assemble report dict ----------------------------------------------
    result: Dict[str, Any] = {
        "input": str(image_path),
        "sha256": loaded.sha256,
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
    }

    # --- Save JSON + PDF ----------------------------------------------------
    json_path = out_dir / f"{stem}_report.json"
    pdf_path = out_dir / f"{stem}_report.pdf"
    save_json_report(result, json_path)
    if want_pdf:
        save_pdf_report(result, pdf_path, visuals_dir=art_dir)

    return result
