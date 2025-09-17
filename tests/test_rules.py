from __future__ import annotations

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
    rep = score_image(modules, weights, 0, 100, {"low": {"max": 30, "label": "Low"}, "medium": {"max": 60, "label": "Medium"}, "high": {"max": 100, "label": "High"}})
    assert 0 <= rep.suspicion_index <= 100
    assert rep.bucket_label in {"Low", "Medium", "High"}
    assert rep.evidence  # should have some items
