from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger, log_analysis_step, log_scoring_decision


@dataclass
class EvidenceItem:
    key: str              # e.g., "ela_strong", "jpeg_double_misaligned"
    weight: float         # positive increases suspicion, negative reduces
    rationale: str        # human-readable reason
    value: Optional[float] = None  # optional numeric evidence (e.g., ratio/score)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreReport:
    suspicion_index: int
    bucket_label: str
    evidence: List[EvidenceItem]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suspicion_index": self.suspicion_index,
            "bucket_label": self.bucket_label,
            "evidence": [e.to_dict() for e in self.evidence],
            "notes": self.notes,
        }


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bucket_for(score: float, buckets_cfg: Dict[str, Any]) -> str:
    """
    buckets_cfg example:
      low:    {max: 30, label: "Low"}
      medium: {max: 60, label: "Medium"}
      high:   {max: 100, label: "High"}
    """
    for name in ["low", "medium", "high"]:
        cfg = buckets_cfg.get(name, {})
        if score <= float(cfg.get("max", 100)):
            return str(cfg.get("label", name.title()))
    return "High"


def score_image(
    modules: Dict[str, Dict[str, Any]],
    weights_cfg: Dict[str, float],
    clamp_min: float,
    clamp_max: float,
    buckets_cfg: Dict[str, Any],
    ai_component_gate: float = 0.4,
) -> ScoreReport:
    """Score an analysed image by combining AI indicators and classic forensic signals."""

    logger = get_logger("scoring")
    evidence: List[EvidenceItem] = []
    notes: List[str] = []

    def weight(key: str, default: float) -> float:
        return float(weights_cfg.get(key, default))

    def add_evidence(key: str, default_weight: float, rationale: str, value: Optional[float] = None) -> None:
        w = weight(key, default_weight)
        evidence.append(EvidenceItem(key, w, rationale, value=value))
        log_scoring_decision(logger, key, w, rationale, value=value)

    # --- AI detection -------------------------------------------------------
    ai_det = modules.get("ai_detection", {}) or {}
    overall_ai_prob = float(ai_det.get("overall_ai_probability", 0.0))
    pixel_score = float(ai_det.get("pixel_distribution_score", 0.0))
    spectral_score = float(ai_det.get("spectral_anomaly_score", 0.0))
    texture_score = float(ai_det.get("texture_consistency_score", 0.0))
    gradient_score = float(ai_det.get("gradient_distribution_score", 0.0))
    color_score = float(ai_det.get("color_correlation_score", 0.0))
    ai_explanations = ai_det.get("explanations", [])

    component_gate = float(ai_component_gate)
    ai_components_enabled = overall_ai_prob >= component_gate

    log_analysis_step(
        logger,
        "ai_summary",
        "AI detection scores",
        overall_ai_prob,
        details={
            "pixel": pixel_score,
            "spectral": spectral_score,
            "texture": texture_score,
            "gradient": gradient_score,
            "color": color_score,
        },
    )

    if overall_ai_prob >= 0.75:
        add_evidence(
            "ai_detection_high",
            50.0,
            "Multiple AI signatures detected with high confidence.",
            value=overall_ai_prob,
        )
    elif overall_ai_prob >= 0.55:
        add_evidence(
            "ai_detection_medium",
            32.0,
            "Moderate AI indicators detected.",
            value=overall_ai_prob,
        )
    elif overall_ai_prob >= 0.4:
        add_evidence(
            "ai_detection_low",
            18.0,
            "Some AI-like characteristics detected.",
            value=overall_ai_prob,
        )

    if ai_components_enabled:
        if pixel_score >= 0.6:
            add_evidence(
                "ai_pixel_distribution",
                26.0,
                "Pixel distribution patterns indicate artificial generation.",
                value=pixel_score,
            )
        elif pixel_score >= 0.45:
            add_evidence(
                "ai_pixel_distribution_moderate",
                14.0,
                "Moderate pixel distribution anomalies detected.",
                value=pixel_score,
            )

        if spectral_score >= 0.6:
            add_evidence(
                "ai_spectral_anomaly",
                24.0,
                "Frequency domain analysis reveals AI-generated characteristics.",
                value=spectral_score,
            )
        elif spectral_score >= 0.5:
            add_evidence(
                "ai_spectral_anomaly_moderate",
                14.0,
                "Moderate spectral anomalies detected.",
                value=spectral_score,
            )

        if texture_score >= 0.6:
            add_evidence(
                "ai_texture_inconsistency",
                20.0,
                "Texture analysis reveals patterns typical of AI-generated content.",
                value=texture_score,
            )
        elif texture_score >= 0.45:
            add_evidence(
                "ai_texture_consistency_moderate",
                12.0,
                "Moderate texture anomalies detected.",
                value=texture_score,
            )

        if gradient_score >= 0.55:
            add_evidence(
                "ai_gradient_distribution",
                16.0,
                "Gradient distribution patterns suggest AI synthesis.",
                value=gradient_score,
            )
        elif gradient_score >= 0.4:
            add_evidence(
                "ai_gradient_distribution_moderate",
                8.0,
                "Gradient distribution shows atypical coherence patterns.",
                value=gradient_score,
            )

        if color_score >= 0.5:
            add_evidence(
                "ai_color_correlation",
                12.0,
                "Color correlation patterns indicate artificial generation.",
                value=color_score,
            )
        elif color_score >= 0.38:
            add_evidence(
                "ai_color_correlation_moderate",
                6.0,
                "Moderate colour anomalies detected.",
                value=color_score,
            )

    # --- Classic forensic modules -----------------------------------------
    ela = modules.get("ela") or {}
    strong_ratio = float(ela.get("strong_regions_ratio", 0.0) or 0.0)
    if strong_ratio >= 0.1:
        add_evidence(
            "ela_strong",
            12.0,
            "ELA highlights large inconsistent regions indicative of manipulation.",
            value=strong_ratio,
        )
    elif strong_ratio >= 0.05:
        add_evidence(
            "ela_moderate",
            6.0,
            "ELA reveals moderate inconsistencies requiring review.",
            value=strong_ratio,
        )

    jpeg_ghosts = modules.get("jpeg_ghosts") or {}
    misalign = jpeg_ghosts.get("misalignment_score")
    if isinstance(misalign, (int, float)):
        misalign = float(misalign)
        if misalign >= 0.5:
            add_evidence(
                "jpeg_double_misaligned",
                15.0,
                "JPEG ghost analysis detected misaligned double compression.",
                value=misalign,
            )
        elif misalign >= 0.3:
            add_evidence(
                "jpeg_double_aligned",
                8.0,
                "JPEG ghost analysis suggests re-save with aligned grid.",
                value=misalign,
            )

    dct = modules.get("dct_benford") or {}
    if dct.get("strong"):
        add_evidence("dct_benford_strong", 15.0, "Strong deviation from Benford's law in DCT coefficients.")
    elif dct.get("moderate"):
        add_evidence("dct_benford_moderate", 8.0, "Moderate deviation from Benford's law in DCT coefficients.")

    resampling = modules.get("resampling") or {}
    res_strong = float(resampling.get("strong_ratio", 0.0) or 0.0)
    res_mod = float(resampling.get("moderate_ratio", 0.0) or 0.0)
    strong_thr = float(resampling.get("strong_threshold", 0.35) or 0.35)
    moderate_thr = float(resampling.get("moderate_threshold", 0.25) or 0.25)
    if res_strong >= strong_thr:
        add_evidence(
            "resampling_periodicity_strong",
            12.0,
            "Resampling detector found strong periodic interpolation artefacts.",
            value=res_strong,
        )
    elif res_mod >= moderate_thr:
        add_evidence(
            "resampling_periodicity_moderate",
            6.0,
            "Resampling detector found moderate interpolation artefacts.",
            value=res_mod,
        )

    cfa = modules.get("cfa") or {}
    cfa_strong = float(cfa.get("strong_ratio", 0.0) or 0.0)
    cfa_mod = float(cfa.get("moderate_ratio", 0.0) or 0.0)
    cfa_strong_thr = float(cfa.get("strong_threshold", 0.35) or 0.35)
    cfa_moderate_thr = float(cfa.get("moderate_threshold", 0.25) or 0.25)
    if cfa_strong >= cfa_strong_thr:
        add_evidence(
            "cfa_inconsistency_large",
            12.0,
            "CFA interpolation patterns inconsistent with a single capture pipeline.",
            value=cfa_strong,
        )
    elif cfa_mod >= cfa_moderate_thr:
        add_evidence(
            "cfa_inconsistency_small",
            6.0,
            "CFA analysis highlights pockets of inconsistency.",
            value=cfa_mod,
        )

    prnu = modules.get("prnu") or {}
    p95 = prnu.get("p95_abs_residual")
    if isinstance(p95, (int, float)) and float(p95) >= 10.0:
        add_evidence(
            "prnu_mismatch",
            8.0,
            "High sensor noise residual suggests content does not match an authentic capture.",
            value=float(p95),
        )

    fft = modules.get("fft_checks") or {}
    peaks = fft.get("peak_indices") or []
    if isinstance(peaks, list) and len(peaks) >= 2:
        add_evidence(
            "fft_periodic_spikes",
            5.0,
            "FFT analysis detected periodic spikes indicative of synthetic textures.",
            value=float(len(peaks)),
        )

    # Provenance evidence (negative weights)
    provenance = modules.get("provenance") or {}
    if provenance.get("c2pa_valid"):
        add_evidence("provenance_c2pa_valid", -35.0, "C2PA manifest validates provenance.")
    if provenance.get("exif_consistent"):
        add_evidence("provenance_exif_consistent", -12.0, "EXIF structure consistent with camera capture.")

    # --- Final aggregation -------------------------------------------------
    base_score = sum(item.weight for item in evidence)
    ai_boost = 0.0
    if overall_ai_prob >= 0.75:
        ai_boost = 22.0
        log_analysis_step(logger, "boost", "Applying high AI confidence boost", ai_boost)
    elif overall_ai_prob >= 0.55:
        ai_boost = 12.0
        log_analysis_step(logger, "boost", "Applying medium AI confidence boost", ai_boost)
    elif overall_ai_prob >= 0.4:
        ai_boost = 5.0
        log_analysis_step(logger, "boost", "Applying low AI confidence boost", ai_boost)

    final_score = clamp(base_score + ai_boost, clamp_min, clamp_max)
    bucket = bucket_for(final_score, buckets_cfg)

    log_analysis_step(
        logger,
        "final",
        f"Final score computed: {final_score:.2f}",
        final_score,
        details={"base": base_score, "ai_boost": ai_boost, "bucket": bucket},
    )

    # Notes and explanations ------------------------------------------------
    if ai_components_enabled:
        for explanation in ai_explanations:
            notes.append(f"AI Analysis: {explanation}")
            log_analysis_step(logger, "explanation", explanation)
    elif ai_explanations:
        notes.append(
            "AI detection module observed minor anomalies but below the confidence gate,"
            " so they were not factored into scoring."
        )

    if overall_ai_prob >= 0.75:
        notes.append("HIGH CONFIDENCE: Multiple AI generation signatures detected.")
    elif overall_ai_prob >= 0.55:
        notes.append("MODERATE CONFIDENCE: Several AI generation indicators detected.")
    elif overall_ai_prob >= 0.4:
        notes.append("LOW CONFIDENCE: Some technical patterns suggest possible AI generation.")
    elif final_score >= 25:
        notes.append("Technical irregularities detected that may indicate manipulation.")
    else:
        notes.append("Technical analysis consistent with authentic camera content.")

    if not evidence:
        notes.append("No significant technical anomalies detected. Content appears authentic.")

    if ai_components_enabled:
        if pixel_score >= 0.5:
            notes.append("Pixel distribution analysis suggests artificial generation patterns.")
        if spectral_score >= 0.5:
            notes.append("Frequency domain analysis indicates non-natural image characteristics.")
        if texture_score >= 0.5:
            notes.append("Texture analysis reveals patterns typical of AI-generated content.")

    return ScoreReport(
        suspicion_index=int(round(final_score)),
        bucket_label=bucket,
        evidence=evidence,
        notes=notes,
    )
