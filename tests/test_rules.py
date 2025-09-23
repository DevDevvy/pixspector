from __future__ import annotations

from pathlib import Path

from pixspector.analysis.ai_detection import run_ai_detection
from pixspector.config import Config, find_defaults_path
from pixspector.core.image_io import load_image
from pixspector.scoring.rules import score_image


def test_rule_engine_basic():
    modules = {
        "ela": {"p95_abs_diff": 200.0, "strong_regions_ratio": 0.1},
        "jpeg_ghosts": {"misalignment_score": 0.6},
        "dct_benford": {"strong": True, "moderate": False},
        "resampling": {"strong_ratio": 0.2, "moderate_ratio": 0.0},
        "cfa": {"strong_ratio": 0.0, "moderate_ratio": 0.1},
        "prnu": {"p95_abs_residual": 20.0},
        "fft_checks": {"peak_indices": [5, 9], "highfreq_rolloff": 0.8},
        "provenance": {"c2pa_valid": False, "exif_consistent": True},
    }
    weights = {
        "ela_strong": 12,
        "jpeg_double_misaligned": 15,
        "dct_benford_strong": 10,
        "resampling_periodicity_strong": 20,
        "cfa_inconsistency_small": 8,
        "prnu_mismatch": 12,
        "fft_periodic_spikes": 10,
        "provenance_exif_consistent": -5,
    }
    rep = score_image(
        modules,
        weights,
        0,
        100,
        {"low": {"max": 30, "label": "Low"}, "medium": {"max": 60, "label": "Medium"}, "high": {"max": 100, "label": "High"}},
        ai_component_gate=0.4,
    )
    assert 0 <= rep.suspicion_index <= 100
    assert rep.bucket_label in {"Low", "Medium", "High"}
    assert rep.evidence  # should have some items


def test_ai_component_gate_prevents_real_false_positive():
    defaults = find_defaults_path(Path(__file__))
    cfg = Config.load(defaults)

    weights = cfg.get("rules.weights", {})
    clamp_min = float(cfg.get("rules.clamp_min", 0))
    clamp_max = float(cfg.get("rules.clamp_max", 100))
    buckets = cfg.get("rules.buckets", {})
    gate = float(cfg.get("rules.ai_component_gate", 0.4))

    samples_dir = Path("examples/sample_images")

    def score_sample(name: str) -> int:
        img = load_image(samples_dir / name, max_dim=1024)
        ai_res = run_ai_detection(img.rgb)
        report = score_image(
            modules={"ai_detection": ai_res.to_dict()},
            weights_cfg=weights,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            buckets_cfg=buckets,
            ai_component_gate=gate,
        )
        return report.suspicion_index

    ai_score = score_sample("ai_photo.webp")
    real_score = score_sample("real_photo.jpg")

    assert ai_score > real_score
    assert real_score <= 25
    assert ai_score >= 35
