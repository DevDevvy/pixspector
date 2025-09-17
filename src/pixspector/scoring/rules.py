from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional


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
    Fuse per-module outputs into a transparent, rule-based Suspicion Index.

    Parameters
    ----------
    modules: dict keyed by module name with structured fields, e.g.
      {
        "ela": {"p95_abs_diff": 120.3, "strong_regions_ratio": 0.08, ...},
        "jpeg_ghosts": {"misalignment_score": 0.41, ...},
        "dct_benford": {"strong": True, "moderate": False, ...},
        "resampling": {"strong_ratio": 0.34, "moderate_ratio": 0.12, ...},
        "cfa": {"strong_ratio": 0.15, ...},
        "prnu": {"mean_abs_residual": 3.2, ...},
        "fft_checks": {"peak_indices": [8, 11], "highfreq_rolloff": 0.82, ...},
        "provenance": {"c2pa_valid": True, "exif_consistent": True}
      }

    weights_cfg: dict of weights keyed by evidence keys in defaults.yaml -> rules.weights
    """
    ev: List[EvidenceItem] = []
    notes: List[str] = []

    # --- Provenance ----------------------------------------------------------
    prov = modules.get("provenance", {})
    if prov.get("c2pa_valid"):
        w = float(weights_cfg.get("provenance_c2pa_valid", -30))
        ev.append(EvidenceItem("provenance_c2pa_valid", w, "C2PA manifest valid (trusted content credentials)."))
    if prov.get("exif_consistent"):
        w = float(weights_cfg.get("provenance_exif_consistent", -5))
        ev.append(EvidenceItem("provenance_exif_consistent", w, "EXIF appears consistent with format and camera."))

    # --- ELA -----------------------------------------------------------------
    ela = modules.get("ela", {})
    p95 = float(ela.get("p95_abs_diff", 0.0))
    strong_ratio = float(ela.get("strong_regions_ratio", 0.0))
    # Heuristic: strong if many very bright ELA pixels OR very high p95
    if strong_ratio >= 0.08 or p95 >= 180.0:
        ev.append(EvidenceItem("ela_strong", float(weights_cfg.get("ela_strong", 12)),
                               "ELA reveals strong localized recompression error.",
                               value=max(strong_ratio, p95)))
    elif strong_ratio >= 0.03 or p95 >= 140.0:
        ev.append(EvidenceItem("ela_moderate", float(weights_cfg.get("ela_moderate", 6)),
                               "ELA shows moderate uneven error.",
                               value=max(strong_ratio, p95)))

    # --- JPEG ghosts / double-JPEG ------------------------------------------
    jpg = modules.get("jpeg_ghosts", {})
    mis = float(jpg.get("misalignment_score", 0.0))
    # Misalignment > ~0.5 suggests double-JPEG with misaligned 8x8 grid
    if mis >= 0.5:
        ev.append(EvidenceItem("jpeg_double_misaligned", float(weights_cfg.get("jpeg_double_misaligned", 15)),
                               "Evidence of double-JPEG with grid misalignment.", value=mis))
    elif mis >= 0.35:
        ev.append(EvidenceItem("jpeg_double_aligned", float(weights_cfg.get("jpeg_double_aligned", 8)),
                               "Possible double-JPEG signatures.", value=mis))

    # --- DCT Benford ---------------------------------------------------------
    ben = modules.get("dct_benford", {})
    if bool(ben.get("strong", False)):
        ev.append(EvidenceItem("dct_benford_strong", float(weights_cfg.get("dct_benford_strong", 10)),
                               "Strong deviation from Benford distribution in DCT ACs."))
    elif bool(ben.get("moderate", False)):
        ev.append(EvidenceItem("dct_benford_moderate", float(weights_cfg.get("dct_benford_moderate", 5)),
                               "Moderate Benford deviation in DCT ACs."))

    # --- Resampling ----------------------------------------------------------
    res = modules.get("resampling", {})
    res_str = float(res.get("strong_ratio", 0.0))
    res_mod = float(res.get("moderate_ratio", 0.0))
    if res_str >= 0.1:
        ev.append(EvidenceItem("resampling_periodicity_strong",
                               float(weights_cfg.get("resampling_periodicity_strong", 20)),
                               "Strong resampling periodicity across patches.", value=res_str))
    elif res_mod >= 0.1:
        ev.append(EvidenceItem("resampling_periodicity_moderate",
                               float(weights_cfg.get("resampling_periodicity_moderate", 10)),
                               "Moderate resampling periodicity.", value=res_mod))

    # --- CFA -----------------------------------------------------------------
    cfa = modules.get("cfa", {})
    cfa_str = float(cfa.get("strong_ratio", 0.0))
    cfa_mod = float(cfa.get("moderate_ratio", 0.0))
    if cfa_str >= 0.08:
        ev.append(EvidenceItem("cfa_inconsistency_large",
                               float(weights_cfg.get("cfa_inconsistency_large", 20)),
                               "Large regions with CFA/demosaicing inconsistencies.", value=cfa_str))
    elif cfa_mod >= 0.08:
        ev.append(EvidenceItem("cfa_inconsistency_small",
                               float(weights_cfg.get("cfa_inconsistency_small", 8)),
                               "Smaller CFA inconsistency regions.", value=cfa_mod))

    # --- PRNU ----------------------------------------------------------------
    pr = modules.get("prnu", {})
    # In this no-gallery mode, use high residual magnitude as a weak cue only
    p95r = float(pr.get("p95_abs_residual", 0.0))
    if p95r >= 15.0:
        ev.append(EvidenceItem("prnu_mismatch", float(weights_cfg.get("prnu_mismatch", 12)),
                               "High PRNU residual magnitude (weak synthesis/overprocessing indicator).",
                               value=p95r))

    # --- FFT checks ----------------------------------------------------------
    fft = modules.get("fft_checks", {})
    peaks = fft.get("peak_indices", []) or []
    roll = float(fft.get("highfreq_rolloff", 0.0))
    if len(peaks) >= 2:
        ev.append(EvidenceItem("fft_periodic_spikes", float(weights_cfg.get("fft_periodic_spikes", 10)),
                               f"Multiple periodic spikes in radial spectrum at bins {peaks[:4]}.",
                               value=float(len(peaks))))

    # Sum weights -> clamp -> bucket
    raw = sum(e.weight for e in ev)
    final = int(round(clamp(raw, clamp_min, clamp_max)))
    bucket = bucket_for(final, buckets_cfg)

    # Notes
    if prov.get("c2pa_valid"):
        notes.append("Content Credentials (C2PA) manifest is valid; treat edits documented therein as expected.")
    if not ev:
        notes.append("No strong forensic cues found. This does not prove authenticity; consider additional context.")

    return ScoreReport(suspicion_index=final, bucket_label=bucket, evidence=ev, notes=notes)
