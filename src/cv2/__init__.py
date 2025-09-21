"""Minimal OpenCV-compatible subset implemented with NumPy/SciPy/Pillow.

This fallback exists so pixspector can run in environments where the native
``opencv-python`` wheels cannot be imported (e.g. due to missing libGL).
It only implements the functions and constants used within the project
and the accompanying test-suite.  The APIs are intentionally lightweight
and not a drop-in replacement for full OpenCV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from matplotlib import cm, colormaps
from scipy import ndimage
from scipy.fftpack import dct as _dct, idct as _idct
from skimage.transform import resize as sk_resize

# --- Constants ---------------------------------------------------------------

COLOR_BGR2RGB = 1
COLOR_RGB2BGR = 2
COLOR_BGR2GRAY = 3
COLOR_RGB2GRAY = 4
COLOR_GRAY2BGR = 5

INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_CUBIC = 3
INTER_AREA = 5

COLORMAP_VIRIDIS = "viridis"

IMWRITE_JPEG_QUALITY = 1

CV_32F = np.float32

# --- Helpers -----------------------------------------------------------------

def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _ensure_float(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.floating):
        return arr
    return arr.astype(np.float32)


def cvtColor(img: np.ndarray, code: int) -> np.ndarray:
    if code == COLOR_BGR2RGB or code == COLOR_RGB2BGR:
        return img[..., ::-1].copy()
    if code == COLOR_BGR2GRAY:
        b, g, r = img[..., 0], img[..., 1], img[..., 2]
        gray = 0.114 * b + 0.587 * g + 0.299 * r
        return np.clip(gray, 0, 255).astype(np.uint8)
    if code == COLOR_RGB2GRAY:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return np.clip(gray, 0, 255).astype(np.uint8)
    if code == COLOR_GRAY2BGR:
        return np.repeat(img[..., None], 3, axis=2).astype(np.uint8)
    raise ValueError(f"Unsupported color conversion code: {code}")


def absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(_ensure_float(a) - _ensure_float(b)).astype(a.dtype)


def GaussianBlur(img: np.ndarray, ksize: Tuple[int, int], sigmaX: float = 0.0, sigmaY: Optional[float] = None) -> np.ndarray:
    if sigmaY is None:
        sigmaY = sigmaX
    sigma = float(max(sigmaX, sigmaY))
    if sigma <= 0:
        return img
    arr = _ensure_float(img)
    if img.ndim == 3:
        channels = [ndimage.gaussian_filter(arr[..., c], sigma=sigma, mode="reflect") for c in range(img.shape[2])]
        out = np.stack(channels, axis=2)
    else:
        out = ndimage.gaussian_filter(arr, sigma=sigma, mode="reflect")
    return out.astype(img.dtype)


def Laplacian(img: np.ndarray, ddepth, ksize: int = 3) -> np.ndarray:
    lap = ndimage.laplace(_ensure_float(img), mode="reflect")
    return lap.astype(ddepth)


def Sobel(img: np.ndarray, ddepth, dx: int, dy: int, ksize: int = 3) -> np.ndarray:
    arr = _ensure_float(img)
    if arr.ndim == 2:
        res = np.zeros_like(arr, dtype=np.float32)
        if dx:
            res += ndimage.sobel(arr, axis=1, mode="reflect")
        if dy:
            res += ndimage.sobel(arr, axis=0, mode="reflect")
        return res.astype(ddepth)
    # color image: process each channel independently
    channels = []
    for c in range(arr.shape[2]):
        chan = np.zeros_like(arr[..., c], dtype=np.float32)
        if dx:
            chan += ndimage.sobel(arr[..., c], axis=1, mode="reflect")
        if dy:
            chan += ndimage.sobel(arr[..., c], axis=0, mode="reflect")
        channels.append(chan)
    stacked = np.stack(channels, axis=2)
    return stacked.astype(ddepth)


def _interp_order(interpolation: int) -> int:
    if interpolation == INTER_NEAREST:
        return 0
    if interpolation == INTER_CUBIC:
        return 3
    return 1


def resize(img: np.ndarray, dsize: Tuple[int, int], interpolation: int = INTER_LINEAR) -> np.ndarray:
    width, height = dsize
    order = _interp_order(interpolation)
    if img.ndim == 2:
        out = sk_resize(img, (height, width), order=order, preserve_range=True, anti_aliasing=(order != 0))
    else:
        out = sk_resize(img, (height, width, img.shape[2]), order=order, preserve_range=True, anti_aliasing=(order != 0))
    return np.clip(out, 0, 255).astype(img.dtype)


def rectangle(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Sequence[int], thickness: int = 1) -> np.ndarray:
    pil = Image.fromarray(_ensure_uint8(img))
    draw = ImageDraw.Draw(pil)
    xy = [pt1[0], pt1[1], pt2[0], pt2[1]]
    if thickness < 0:
        draw.rectangle(xy, fill=tuple(int(c) for c in color))
    else:
        draw.rectangle(xy, outline=tuple(int(c) for c in color), width=int(thickness))
    result = np.array(pil, dtype=img.dtype)
    img[...] = result
    return img


def line(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: int, thickness: int = 1) -> np.ndarray:
    pil = Image.fromarray(_ensure_uint8(img))
    draw = ImageDraw.Draw(pil)
    draw.line([pt1, pt2], fill=int(color), width=int(thickness))
    result = np.array(pil, dtype=img.dtype)
    img[...] = result
    return img


def imwrite(filename: str, img: np.ndarray, params: Optional[Sequence[int]] = None) -> bool:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    quality = None
    if params:
        params = list(params)
        for i in range(0, len(params), 2):
            if params[i] == IMWRITE_JPEG_QUALITY and i + 1 < len(params):
                quality = int(params[i + 1])
    pil_img = Image.fromarray(_ensure_uint8(img))
    save_kwargs = {}
    if quality is not None:
        save_kwargs["quality"] = quality
        save_kwargs["subsampling"] = 0
    pil_img.save(str(path), **save_kwargs)
    return True


def applyColorMap(src: np.ndarray, cmap: str) -> np.ndarray:
    if cmap != COLORMAP_VIRIDIS:
        raise ValueError("Only viridis colormap is supported in the stub")
    arr = _ensure_float(src)
    norm = (arr - arr.min()) / (np.ptp(arr) + 1e-6)
    rgba = colormaps[cmap](norm)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb[..., ::-1]


def addWeighted(src1: np.ndarray, alpha: float, src2: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    out = alpha * _ensure_float(src1) + beta * _ensure_float(src2) + gamma
    return np.clip(out, 0, 255).astype(src1.dtype)


def dct(block: np.ndarray) -> np.ndarray:
    arr = _ensure_float(block)
    return _dct(_dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")


def idct(coeffs: np.ndarray) -> np.ndarray:
    arr = _ensure_float(coeffs)
    return _idct(_idct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")


def imread(filename: str) -> np.ndarray:
    return np.array(Image.open(filename))


__all__ = [name for name in globals() if not name.startswith("_")]
