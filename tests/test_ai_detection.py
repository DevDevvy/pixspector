from __future__ import annotations

from pathlib import Path

import pytest

from pixspector.analysis.ai_detection import run_ai_detection
from pixspector.core.image_io import load_image


SAMPLES_DIR = Path("examples/sample_images")


@pytest.mark.parametrize(
    ("filename", "expected_label"),
    [
        ("ai_photo.webp", "ai"),
        ("ai_photo_2.webp", "ai"),
        ("real_photo.jpg", "real"),
        ("real_photo_2.JPG", "real"),
    ],
)
def test_sample_image_ai_detection(filename: str, expected_label: str) -> None:
    """Ensure bundled sample images are classified as expected."""

    image = load_image(SAMPLES_DIR / filename, max_dim=1024)
    result = run_ai_detection(image.rgb)

    if expected_label == "ai":
        assert (
            result.overall_ai_probability >= 0.42
        ), f"Expected AI image {filename} to score high, got {result.overall_ai_probability:.3f}"
    else:
        assert (
            result.overall_ai_probability <= 0.4
        ), f"Expected real image {filename} to score low, got {result.overall_ai_probability:.3f}"
