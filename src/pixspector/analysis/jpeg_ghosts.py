from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
from PIL import Image

from ..core.image_io import ensure_color


@dataclass
class GhostsResult:
    best_quality: int                  # JPEG Q with lowest diff (most similar)
    quality_curve: List[Tuple[int, float]]  # [(Q, mean L1 diff)]
    best_diff_u8: np.ndarray           # visualization of abs diff at best Q
    misalignment_score: float          # heuristic estimate of 8x8-grid misalignment (0..1)
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self).copy()
        d["best_diff_u8"] = {"shape": list(self.best_diff_u8.shape), "dtype": "uint8"}
        return d


def _recompress_rgb(rgb: np.ndarray, quality: int) -> np.ndarray:
    from io import BytesIO
    pil_im = Image.fromarray(rgb, mode="RGB")
    buf = BytesIO()
    pil_im.save(buf, format="JPEG", quality=int(quality), subsampling="keep")
    buf.seek(0)
    rec = Image.open(buf).convert("RGB")
    return np.array(rec, dtype=np.uint8)


def _grid_misalignment_score(gray_diff: np.ndarray, block: int = 8) -> float:
    """
    Very lightweight heuristic for double-JPEG grid misalignment.
    We project energy along rows/cols and check periodic peaks offset from multiples of 8.
    Returns a 0..1 score.
    """
    # Emphasize grid edges by Sobel -> abs -> mean across channels
    sobx = cv2.Sobel(gray_diff, cv2.CV_32F, 1, 0, ksize=3)
    soby = cv2.Sobel(gray_diff, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(sobx, soby)
    # Sum over axes to get 1D signals
    row_sig = np.mean(mag, axis=1)
    col_sig = np.mean(mag, axis=0)

    def periodic_offset_strength(sig: np.ndarray, blk: int) -> float:
        # Fold the signal into blk bins; compute variance across phase bins
        n = len(sig)
        bins = [[] for _ in range(blk)]
        for i, v in enumerate(sig):
            bins[i % blk].append(float(v))
        means = np.array([np.mean(b) if b else 0.0 for b in bins], dtype=np.float32)
        # strong if one or two phases dominate (misalignment)
        means = (means - means.min()) / (means.ptp() + 1e-6)
        return float(np.max(means))

    r = periodic_offset_strength(row_sig, block)
    c = periodic_offset_strength(col_sig, block)
    return float(np.clip(0.5 * (r + c), 0.0, 1.0))


def run_jpeg_ghosts(bgr_u8: np.ndarray, q_grid: List[int], diff_metric: str = "l1") -> GhostsResult:
    """
    JPEG ghosts / double-JPEG search:
    - recompress across a grid of qualities
    - compute mean diff per Q (L1 or L2)
    - pick best (lowest diff)
    - compute a grid-misalignment heuristic on the best diff map
    """
    bgr_u8 = ensure_color(bgr_u8)
    rgb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB)

    quality_curve: List[Tuple[int, float]] = []
    best_q = q_grid[0]
    best_diff = None
    best_score = float("inf")

    for q in q_grid:
        rec = _recompress_rgb(rgb, q)
        diff = cv2.absdiff(rgb, rec).astype(np.float32)
        if diff_metric == "l2":
            score = float(np.mean(diff ** 2))
        else:
            score = float(np.mean(np.abs(diff)))
        quality_curve.append((q, score))
        if score < best_score:
            best_score = score
            best_q = q
            best_diff = diff

    # Normalize best diff to 0..255 for visualization
    vis = best_diff.copy()
    for c in range(3):
        ch = vis[..., c]
        m, M = ch.min(), ch.max()
        if M > m:
            vis[..., c] = 255.0 * (ch - m) / (M - m + 1e-6)
    vis_u8 = np.clip(vis, 0, 255).astype(np.uint8)
    vis_u8 = cv2.cvtColor(vis_u8, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(vis_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
    misalign = _grid_misalignment_score(gray, block=8)

    return GhostsResult(
        best_quality=int(best_q),
        quality_curve=quality_curve,
        best_diff_u8=vis_u8,
        misalignment_score=float(misalign),
        meta={
            "metric": diff_metric,
            "note": "Lower diff at some Q can indicate original JPEG quality; misalignment hints at double-compression."
        },
    )
