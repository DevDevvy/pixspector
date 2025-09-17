from __future__ import annotations

import numpy as np
import cv2

from pixspector.analysis.resampling import run_resampling_map


def test_resampling_map_ranges():
    base = np.zeros((256, 256), dtype=np.uint8)
    # Diagonal lines + scale to inject resampling traces
    for i in range(0, 256, 4):
        cv2.line(base, (0, i), (255, (i + 64) % 256), 255, 1)
    scaled = cv2.resize(base, (224, 224), interpolation=cv2.INTER_LINEAR)
    scaled = cv2.resize(scaled, (256, 256), interpolation=cv2.INTER_LINEAR)

    res = run_resampling_map(scaled, patch=64, stride=32)
    assert 0.0 <= res.strong_ratio <= 1.0
    assert 0.0 <= res.moderate_ratio <= 1.0
    assert res.heatmap_u8.dtype == np.uint8
