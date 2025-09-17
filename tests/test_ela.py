from __future__ import annotations

import numpy as np
import cv2

from pixspector.analysis.ela import run_ela


def test_ela_metrics():
    # Synthetic: left half noise, right half flat -> ELA should show higher p95
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :128, :] = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    img[:, 128:, :] = 128
    res = run_ela(img, recompress_quality=90)
    assert res.p95_abs_diff >= 10.0  # not zero
    assert 0.0 <= res.strong_regions_ratio <= 1.0
