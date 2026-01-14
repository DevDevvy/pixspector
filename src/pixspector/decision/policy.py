from __future__ import annotations

from typing import Any, Dict, List, Optional


ConfidenceLevel = str


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_decision(
    provenance: Dict[str, Any],
    watermark: Dict[str, Any],
    forensics: Dict[str, Any],
    ml_signals: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate overall authenticity confidence based on available signals.

    Rules:
    - HIGH: strong provenance and no high-risk forensic or ML indicators.
    - MEDIUM: clear indicators of manipulation/synthetic origin.
    - INDETERMINATE: limited or conflicting evidence.
    """
    rationale: List[str] = []

    c2pa_valid = bool(provenance.get("c2pa_valid"))
    exif_consistent = bool(provenance.get("exif_consistent"))
    provenance_strong = c2pa_valid or exif_consistent
    if c2pa_valid:
        rationale.append("C2PA signature validated in provenance metadata.")
    elif exif_consistent:
        rationale.append("EXIF metadata appears internally consistent.")

    watermark_conf = _safe_float(watermark.get("max_confidence")) or 0.0
    watermark_hits = watermark.get("hits") or []
    watermark_detected = watermark_conf >= 0.5 or len(watermark_hits) > 0
    if watermark_detected:
        rationale.append(
            f"Watermark signal detected (max confidence {watermark_conf:.2f})."
        )

    suspicion_index = _safe_float(forensics.get("suspicion_index"))
    evidence = forensics.get("evidence") or []
    evidence_count = len(evidence)
    if suspicion_index is not None:
        rationale.append(f"Forensic suspicion index is {suspicion_index:.1f}.")

    high_forensic_anomaly = (suspicion_index is not None and suspicion_index >= 70) or evidence_count >= 4
    moderate_forensic_anomaly = (suspicion_index is not None and suspicion_index >= 40) or evidence_count >= 2

    ai_prob = None
    if ml_signals:
        ai_detection = ml_signals.get("ai_detection") or {}
        ai_prob = _safe_float(ai_detection.get("overall_ai_probability"))
        if ai_prob is not None:
            rationale.append(f"ML synthetic probability is {ai_prob:.2f}.")

    ai_high = ai_prob is not None and ai_prob >= 0.8
    ai_moderate = ai_prob is not None and ai_prob >= 0.6

    if not provenance and not watermark and suspicion_index is None and ai_prob is None:
        return {
            "confidence_level": "INDETERMINATE",
            "rationale": ["No provenance, forensic, or ML signals were available."],
        }

    if provenance_strong and not (high_forensic_anomaly or ai_high or watermark_detected):
        confidence: ConfidenceLevel = "HIGH"
        rationale.append("Strong provenance with limited forensic anomalies.")
    elif high_forensic_anomaly or ai_high or watermark_detected:
        confidence = "MEDIUM"
        rationale.append("Signals indicate possible manipulation or synthetic origin.")
    elif moderate_forensic_anomaly or ai_moderate:
        confidence = "MEDIUM"
        rationale.append("Moderate anomalies reduce authenticity confidence.")
    else:
        confidence = "INDETERMINATE"
        rationale.append("Signals are mixed or insufficient for a confident determination.")

    return {
        "confidence_level": confidence,
        "rationale": rationale,
    }
