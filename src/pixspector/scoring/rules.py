from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


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
) -> ScoreReport:
    """
    Enhanced scoring with dedicated AI detection heavily weighted.
    """
    from ..utils.logging import get_logger, log_analysis_step
    _scoring_logger = get_logger("scoring")
    
    ev: List[EvidenceItem] = []
    notes: List[str] = []

    # --- AI Detection (Primary Analysis) ------------------------------------
    ai_det = modules.get("ai_detection", {})
    overall_ai_prob = float(ai_det.get("overall_ai_probability", 0.0))
    pixel_score = float(ai_det.get("pixel_distribution_score", 0.0))
    spectral_score = float(ai_det.get("spectral_anomaly_score", 0.0))
    texture_score = float(ai_det.get("texture_consistency_score", 0.0))
    gradient_score = float(ai_det.get("gradient_distribution_score", 0.0))
    color_score = float(ai_det.get("color_correlation_score", 0.0))
    ai_explanations = ai_det.get("explanations", [])

    log_analysis_step(_scoring_logger, "ai_scoring", f"AI detection scores - Overall: {overall_ai_prob:.3f}, Pixel: {pixel_score:.3f}, Spectral: {spectral_score:.3f}, Texture: {texture_score:.3f}")

    # Enhanced AI detection scoring with higher weights
    if overall_ai_prob >= 0.7:
        ev.append(EvidenceItem("ai_detection_high", 60,  # Increased from 45
                               "Multiple AI signatures detected with high confidence.",
                               value=overall_ai_prob))
        log_analysis_step(_scoring_logger, "evidence", f"HIGH AI confidence detected: {overall_ai_prob:.3f}")
    elif overall_ai_prob >= 0.5:
        ev.append(EvidenceItem("ai_detection_medium", 45,  # Increased from 30
                               "Moderate AI indicators detected.",
                               value=overall_ai_prob))
        log_analysis_step(_scoring_logger, "evidence", f"MEDIUM AI confidence detected: {overall_ai_prob:.3f}")
    elif overall_ai_prob >= 0.3:
        ev.append(EvidenceItem("ai_detection_low", 25,  # Increased from 15
                               "Some AI-like characteristics detected.",
                               value=overall_ai_prob))
        log_analysis_step(_scoring_logger, "evidence", f"LOW AI indicators detected: {overall_ai_prob:.3f}")

    # Specific AI technique results with enhanced scoring
    if pixel_score >= 0.6:
        ev.append(EvidenceItem("ai_pixel_distribution", 25,  # Increased from 20
                               "Pixel distribution patterns indicate artificial generation.",
                               value=pixel_score))
        log_analysis_step(_scoring_logger, "evidence", f"Artificial pixel distribution detected: {pixel_score:.3f}")
    elif pixel_score >= 0.4:
        ev.append(EvidenceItem("ai_pixel_distribution_moderate", 15,
                               "Moderate pixel distribution anomalies detected.",
                               value=pixel_score))

    if spectral_score >= 0.6:
        ev.append(EvidenceItem("ai_spectral_anomaly", 25,  # Increased from 20
                               "Frequency domain analysis reveals AI-generated characteristics.",
                               value=spectral_score))
        log_analysis_step(_scoring_logger, "evidence", f"AI spectral signatures detected: {spectral_score:.3f}")
    elif spectral_score >= 0.4:
        ev.append(EvidenceItem("ai_spectral_anomaly_moderate", 15,
                               "Moderate spectral anomalies detected.",
                               value=spectral_score))

    if texture_score >= 0.6:
        ev.append(EvidenceItem("ai_texture_consistency", 25,  # Increased from 20
                               "Texture analysis reveals artificial generation patterns.",
                               value=texture_score))
        log_analysis_step(_scoring_logger, "evidence", f"AI texture patterns detected: {texture_score:.3f}")
    elif texture_score >= 0.4:
        ev.append(EvidenceItem("ai_texture_consistency_moderate", 15,
                               "Moderate texture anomalies detected.",
                               value=texture_score))

    if gradient_score >= 0.6:
        ev.append(EvidenceItem("ai_gradient_distribution", 20,
                               "Gradient distribution patterns suggest AI synthesis.",
                               value=gradient_score))
        log_analysis_step(_scoring_logger, "evidence", f"AI gradient patterns detected: {gradient_score:.3f}")

    if color_score >= 0.6:
        ev.append(EvidenceItem("ai_color_correlation", 20,
                               "Color correlation patterns indicate artificial generation.",
                               value=color_score))
        log_analysis_step(_scoring_logger, "evidence", f"AI color patterns detected: {color_score:.3f}")

    # --- Traditional Forensic Analysis (Secondary) -------------------------
    prov = modules.get("provenance", {})
    if prov.get("c2pa_valid"):
        w = float(weights_cfg.get("provenance_c2pa_valid", -35))
        ev.append(EvidenceItem("provenance_c2pa_valid", w, "C2PA manifest valid (trusted content credentials)."))
    if prov.get("exif_consistent"):
        w = float(weights_cfg.get("provenance_exif_consistent", -12))
        ev.append(EvidenceItem("provenance_exif_consistent", w, "EXIF appears consistent with format and camera."))

    # --- Final Score Calculation -------------------------------------------
    total_weight = sum(item.weight for item in ev)
    
    # Apply AI detection boost for decisive scoring
    ai_boost = 0
    if overall_ai_prob >= 0.7:
        ai_boost = 20  # Strong boost for high confidence AI
        log_analysis_step(_scoring_logger, "boost", f"Applying high AI boost: +{ai_boost}")
    elif overall_ai_prob >= 0.5:
        ai_boost = 10  # Moderate boost for medium confidence AI
        log_analysis_step(_scoring_logger, "boost", f"Applying medium AI boost: +{ai_boost}")
    
    final = clamp(total_weight + ai_boost, clamp_min, clamp_max)
    bucket = bucket_for(final, buckets_cfg)
    
    log_analysis_step(_scoring_logger, "final", f"Final score: {final} (base: {total_weight}, AI boost: {ai_boost}) -> {bucket}")

    # --- Enhanced Notes with AI Explanations -------------------------------
    # Add AI detection explanations to notes
    for explanation in ai_explanations:
        notes.append(f"AI Analysis: {explanation}")
        log_analysis_step(_scoring_logger, "explanation", explanation)

    if overall_ai_prob >= 0.7:
        notes.append("HIGH CONFIDENCE: Multiple AI generation signatures detected through advanced technical analysis.")
        log_analysis_step(_scoring_logger, "conclusion", "HIGH confidence AI detection")
    elif overall_ai_prob >= 0.5:
        notes.append("MODERATE CONFIDENCE: Several AI generation indicators detected.")
        log_analysis_step(_scoring_logger, "conclusion", "MODERATE confidence AI detection")
    elif overall_ai_prob >= 0.3:
        notes.append("LOW CONFIDENCE: Some technical patterns suggest possible AI generation.")
        log_analysis_step(_scoring_logger, "conclusion", "LOW confidence AI detection")
    elif final >= 25:
        notes.append("Technical irregularities detected that may indicate manipulation.")
        log_analysis_step(_scoring_logger, "conclusion", "Technical irregularities detected")
    else:
        notes.append("Technical analysis consistent with authentic camera content.")
        log_analysis_step(_scoring_logger, "conclusion", "Content appears authentic")

    if not ev:
        notes.append("No significant technical anomalies detected. Content appears authentic.")

    # Specific guidance based on AI detection components
    if pixel_score >= 0.5:
        notes.append("Pixel distribution analysis suggests artificial generation patterns.")
    if spectral_score >= 0.5:
        notes.append("Frequency domain analysis indicates non-natural image characteristics.")
    if texture_score >= 0.5:
        notes.append("Texture analysis reveals patterns typical of AI-generated content.")

    return ScoreReport(suspicion_index=final, bucket_label=bucket, evidence=ev, notes=notes)
