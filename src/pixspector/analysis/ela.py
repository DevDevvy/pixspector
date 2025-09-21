from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

from ..core.image_io import ensure_color


@dataclass
class ELAResult:
    diff_u8: np.ndarray           # 3-channel visualization image (uint8)
    mean_abs_diff: float          # global mean absolute diff (0..255)
    p95_abs_diff: float           # 95th percentile abs diff (0..255)
    strong_regions_ratio: float   # fraction of pixels above a heuristic threshold
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self).copy()
        # don't inline the full image array in JSON contexts
        d["diff_u8"] = {"shape": list(self.diff_u8.shape), "dtype": "uint8"}
        return d


def _pil_recompress(rgb: np.ndarray, quality: int) -> np.ndarray:
    """
    Recompress an RGB array with Pillow at given JPEG quality, return RGB uint8.
    """
    h, w = rgb.shape[:2]
    pil_im = Image.fromarray(rgb)
    from io import BytesIO
    buf = BytesIO()
    pil_im.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    arr = np.array(recompressed, dtype=np.uint8)
    if arr.shape[0] != h or arr.shape[1] != w:
        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
    return arr


def run_ela(bgr_u8: np.ndarray, recompress_quality: int = 90, blur_sigma: float = 0.0) -> ELAResult:
    """
    Error Level Analysis:
    - recompress at a given JPEG quality
    - take absolute difference
    - (optional) blur the diff slightly for visualization stability
    Returns a visualization and simple global statistics.
    """
    bgr_u8 = ensure_color(bgr_u8)
    rgb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB)
    rec_rgb = _pil_recompress(rgb, quality=recompress_quality)

    diff = cv2.absdiff(rgb, rec_rgb).astype(np.float32)  # [0..255]
    if blur_sigma and blur_sigma > 0:
        k = int(max(3, round(blur_sigma * 4) | 1))
        diff = cv2.GaussianBlur(diff, (k, k), blur_sigma)

    # Normalize per-channel to full 0..255 for visualization
    vis = diff.copy()
    for c in range(3):
        ch = vis[..., c]
        m, M = ch.min(), ch.max()
        if M > m:
            vis[..., c] = (255.0 * (ch - m) / (M - m))
    vis_u8 = np.clip(vis, 0, 255).astype(np.uint8)
    vis_u8 = cv2.cvtColor(vis_u8, cv2.COLOR_RGB2BGR)

    # Global metrics
    gray_diff = cv2.cvtColor(vis_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_abs = float(np.mean(gray_diff))
    p95_abs = float(np.percentile(gray_diff, 95))
    # strong regions: > 200/255 in vis (heuristic)
    strong_ratio = float(np.mean(gray_diff > 200.0))

    return ELAResult(
        diff_u8=vis_u8,
        mean_abs_diff=mean_abs,
        p95_abs_diff=p95_abs,
        strong_regions_ratio=strong_ratio,
        meta={
            "recompress_quality": recompress_quality,
            "blur_sigma": blur_sigma,
            "note": "ELA highlights uneven error introduced by recompression; splices/edits can stand out."
        },
    )
