from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any

import cv2
import numpy as np


@dataclass
class CFAResult:
    inconsistency_map_u8: np.ndarray     # visualization (uint8) of CFA inconsistency per patch
    strong_ratio: float
    moderate_ratio: float
    strong_threshold: float
    moderate_threshold: float
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self).copy()
        d["inconsistency_map_u8"] = {"shape": list(self.inconsistency_map_u8.shape), "dtype": "uint8"}
        return d


def _cfa_periodicity_score(rgb_u8: np.ndarray) -> float:
    """
    Heuristic CFA/demosaicing consistency:
    - Work mostly on the green channel (dominant in Bayer)
    - Compare pixel vs. local directional averages at Bayer phases
    - Strong deviations from 2x2 periodicity raise the score
    Returns [0..1].
    """
    rgb = rgb_u8.astype(np.float32)
    g = rgb[..., 1]

    # Expected Bayer periodicity ~2; build 2x2 phase maps
    h, w = g.shape
    phases = np.zeros((2, 2), dtype=np.float32)

    # local prediction (mean of cross neighbors)
    pred = 0.25 * (
        np.pad(g, ((1, 1), (0, 0)), mode="reflect")[0:h, :] +     # up
        np.pad(g, ((0, 0), (0, 0)), mode="reflect")[1:h + 1, :] + # down
        np.pad(g, ((0, 0), (1, 1)), mode="reflect")[:, 0:w] +     # left
        np.pad(g, ((0, 0), (0, 0)), mode="reflect")[:, 1:w + 1]   # right
    )
    resid = np.abs(g - pred)

    # Aggregate residual energy by CFA phase
    counts = np.zeros((2, 2), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            py, px = y & 1, x & 1
            phases[py, px] += resid[y, x]
            counts[py, px] += 1

    phases = phases / (counts + 1e-6)

    # Normalize phase energies; strong imbalance between phases -> inconsistency
    phases = (phases - phases.min()) / (phases.ptp() + 1e-6)
    # score = variance across 4 phases
    score = float(np.var(phases))
    return float(np.clip(score, 0.0, 1.0))


def run_cfa_map(
    rgb_u8: np.ndarray,
    patch: int = 64,
    stride: int = 32,
    thr_strong: float = 0.35,
    thr_moderate: float = 0.25,
) -> CFAResult:
    """
    Slide over RGB image; compute CFA inconsistency score per patch.
    """
    h, w = rgb_u8.shape[:2]
    ys = list(range(0, max(1, h - patch + 1), stride))
    xs = list(range(0, max(1, w - patch + 1), stride))
    heat = np.zeros((len(ys), len(xs)), dtype=np.float32)

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            p = rgb_u8[y:y + patch, x:x + patch]
            if p.shape[0] != patch or p.shape[1] != patch:
                continue
            heat[iy, ix] = _cfa_periodicity_score(p)

    strong_mask = heat >= thr_strong
    mod_mask = (heat >= thr_moderate) & (~strong_mask)

    heat_norm = (255.0 * (heat - heat.min()) / (heat.ptp() + 1e-6)).astype(np.uint8)
    vis = cv2.resize(heat_norm, (w, h), interpolation=cv2.INTER_NEAREST)

    return CFAResult(
        inconsistency_map_u8=vis,
        strong_ratio=float(np.mean(strong_mask)),
        moderate_ratio=float(np.mean(mod_mask)),
        strong_threshold=float(thr_strong),
        moderate_threshold=float(thr_moderate),
        meta={
            "patch": patch,
            "stride": stride,
            "note": "CFA/demosaicing inconsistencies can indicate splices or non-camera synthesis."
        },
    )
