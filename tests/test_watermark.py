import numpy as np

from pixspector.analysis.watermark import run_watermark_detection


def test_watermark_detection_returns_non_stub_detector_statuses():
    rng = np.random.default_rng(1234)
    img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)

    report = run_watermark_detection(img)

    assert report.schemes_checked == ["synthid", "stable_signature", "adobe_cc"]
    assert len(report.detectors) == 3
    assert all(det.status in {"hit", "no_hit"} for det in report.detectors)
    assert all(det.status != "stub" for det in report.detectors)
    assert report.max_confidence >= 0.0


def test_periodic_pattern_boosts_watermark_confidence():
    h, w = 192, 192
    y, x = np.indices((h, w))
    pattern = 127.5 + 70 * np.sin(2 * np.pi * x / 12.0) + 50 * np.sin(2 * np.pi * y / 18.0)
    pattern = np.clip(pattern, 0, 255).astype(np.uint8)
    rgb = np.dstack([pattern, pattern, pattern])

    report = run_watermark_detection(rgb)

    assert report.max_confidence > 0.45
    assert all(det.confidence >= 0.0 for det in report.detectors)
