from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class LoadedImage:
    path: Path
    bgr: np.ndarray            # OpenCV BGR uint8 image
    rgb: np.ndarray            # RGB uint8 image
    gray: np.ndarray           # Grayscale uint8 image
    sha256: str
    width: int
    height: int
    scale_factor: float        # if downscaled on load


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_downscale(img_bgr: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_bgr, scale


def load_image(path: Path, max_dim: int = 4096) -> LoadedImage:
    """
    Robust image loader (JPEG/PNG/TIFF/WEBP/HEIC* if OpenCV supports).
    Returns BGR/RGB/GRAY plus SHA-256 and scale info.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    # Use PIL to robustly decode + handle color profiles, then convert to OpenCV arrays.
    with Image.open(path) as im:
        im = im.convert("RGB")
        rgb = np.array(im)  # uint8 RGB
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr, scale = _maybe_downscale(bgr, max_dim=max_dim)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sha = _compute_sha256(path)
    h, w = gray.shape[:2]

    return LoadedImage(
        path=path,
        bgr=bgr,
        rgb=rgb,
        gray=gray,
        sha256=sha,
        width=w,
        height=h,
        scale_factor=scale,
    )


def to_float32(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0


def ensure_color(img: np.ndarray) -> np.ndarray:
    """Ensure 3-channel color."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def tiles(img: np.ndarray, tile: int, stride: int):
    """
    Generator over sliding-window tiles.
    Yields (y, x, h, w, view).
    """
    h, w = img.shape[:2]
    for y in range(0, max(1, h - tile + 1), stride):
        for x in range(0, max(1, w - tile + 1), stride):
            view = img[y:y + tile, x:x + tile]
            if view.shape[0] == tile and view.shape[1] == tile:
                yield y, x, tile, tile, view
