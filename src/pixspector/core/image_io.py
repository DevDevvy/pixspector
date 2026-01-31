from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import SandboxConfig
from .sandbox import secure_load_image

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
    """Compute SHA-256 hash of file efficiently using chunked reading."""
    h = hashlib.sha256()
    chunk_size = 8 * 1024 * 1024  # 8MB chunks for better performance
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
    except Exception as e:
        raise IOError(f"Failed to compute hash for {path.name}: {e}")
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


def load_image(
    path: Path,
    max_dim: int = 4096,
    sandbox: Optional[SandboxConfig] = None,
) -> LoadedImage:
    """
    Robust image loader (JPEG/PNG/TIFF/WEBP/HEIC* if OpenCV supports).
    Returns BGR/RGB/GRAY plus SHA-256 and scale info.
    
    Args:
        path: Path to image file
        max_dim: Maximum dimension for downscaling (default 4096)
        sandbox: Optional sandbox configuration for security
    
    Raises:
        FileNotFoundError: If path doesn't exist
        ValueError: If file is not a valid image or parameters are invalid
        RuntimeError: If image decoding fails
    """
    # Validate input parameters
    if not isinstance(path, Path):
        path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    if max_dim <= 0:
        raise ValueError(f"max_dim must be positive, got {max_dim}")
    
    # Security check: file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > 500:  # Warn on very large files
        import warnings
        warnings.warn(f"Large image file: {file_size_mb:.1f} MB. Processing may be slow.", UserWarning)

    try:
        decoded = secure_load_image(path, sandbox)
    except Exception as e:
        raise RuntimeError(f"Failed to decode image {path.name}: {str(e)}")
    
    if decoded.width <= 0 or decoded.height <= 0:
        raise ValueError(f"Invalid image dimensions: {decoded.width}x{decoded.height}")
    rgb = np.frombuffer(decoded.data, dtype=np.uint8).reshape(decoded.height, decoded.width, 3)
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
