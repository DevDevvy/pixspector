from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import multiprocessing as mp
import os
import resource

from PIL import Image, ImageFile

from ..config import SandboxConfig

_RIFF_SIGNATURE = b"RIFF"
_WEBP_SIGNATURE = b"WEBP"

_FTYP_BOX = b"ftyp"
_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}
_AVIF_BRANDS = {b"avif", b"avis"}

_MAGIC_MIME_MAP: Dict[str, Tuple[bytes, ...]] = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/gif": (b"GIF87a", b"GIF89a"),
    "image/bmp": (b"BM",),
    "image/tiff": (b"II*\x00", b"MM\x00*"),
}


@dataclass(frozen=True)
class DecodedImage:
    data: bytes
    width: int
    height: int
    mode: str
    mime: str


def _sniff_magic(header: bytes) -> Optional[str]:
    for mime, signatures in _MAGIC_MIME_MAP.items():
        if any(header.startswith(sig) for sig in signatures):
            return mime
    if header.startswith(_RIFF_SIGNATURE) and header[8:12] == _WEBP_SIGNATURE:
        return "image/webp"
    if len(header) >= 12 and header[4:8] == _FTYP_BOX:
        brand = header[8:12]
        if brand in _HEIF_BRANDS:
            return "image/heif"
        if brand in _AVIF_BRANDS:
            return "image/avif"
    return None


def _read_header(path: Path, size: int = 64) -> bytes:
    with open(path, "rb") as f:
        return f.read(size)


def _enforce_file_size(path: Path, max_file_size_mb: int) -> None:
    if max_file_size_mb <= 0:
        return
    size_bytes = path.stat().st_size
    if size_bytes > max_file_size_mb * 1024 * 1024:
        raise ValueError(f"File exceeds sandbox size cap ({max_file_size_mb} MB).")


def _apply_resource_limits(config: SandboxConfig) -> None:
    if config.max_cpu_seconds > 0:
        resource.setrlimit(resource.RLIMIT_CPU, (config.max_cpu_seconds, config.max_cpu_seconds))
    if config.max_memory_mb > 0:
        max_bytes = config.max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    if config.max_file_size_mb > 0:
        max_bytes = config.max_file_size_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_bytes, max_bytes))


def _decode_image(path_str: str, config: SandboxConfig, apply_limits: bool = True) -> DecodedImage:
    if apply_limits:
        _apply_resource_limits(config)
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    Image.MAX_IMAGE_PIXELS = config.max_decode_pixels if config.max_decode_pixels > 0 else None
    path = Path(path_str)
    with Image.open(path) as im:
        im = im.convert("RGB")
        width, height = im.size
        data = im.tobytes()
    return DecodedImage(
        data=data,
        width=width,
        height=height,
        mode="RGB",
        mime="image/unknown",
    )


def secure_load_image(path: Path, config: Optional[SandboxConfig] = None) -> DecodedImage:
    """
    Load and decode an image with sandbox restrictions for security.
    
    Args:
        path: Path to image file
        config: Optional sandbox configuration (uses defaults if None)
    
    Returns:
        DecodedImage with decoded image data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file exceeds size limits or is invalid format
        RuntimeError: If image decoding fails
    """
    if not isinstance(path, Path):
        path = Path(path)
        
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    config = config or SandboxConfig()
    
    # Validate file size before reading
    try:
        _enforce_file_size(path, config.max_file_size_mb)
    except ValueError as e:
        raise ValueError(f"File size validation failed for {path.name}: {str(e)}")
    
    # Validate file type via magic bytes
    try:
        header = _read_header(path, size=64)
        detected_mime = _sniff_magic(header)
        if detected_mime is None:
            raise ValueError(f"Unsupported or unrecognized image format: {path.name}")
    except Exception as e:
        raise ValueError(f"Failed to read file header for {path.name}: {str(e)}")
    _enforce_file_size(path, config.max_file_size_mb)
    header = _read_header(path)
    mime = _sniff_magic(header)
    if mime is None:
        raise ValueError("Unsupported or unrecognized image file type.")

    if not config.enabled:
        decoded = _decode_image(str(path), config, apply_limits=False)
        return DecodedImage(
            data=decoded.data,
            width=decoded.width,
            height=decoded.height,
            mode=decoded.mode,
            mime=mime,
        )

    available_methods = mp.get_all_start_methods()
    start_method = "spawn"
    if os.name != "nt" and "fork" in available_methods:
        start_method = "fork"
    ctx = mp.get_context(start_method)
    try:
        with ctx.Pool(processes=1) as pool:
            result = pool.apply_async(_decode_image, (str(path), config))
            decoded = result.get(timeout=config.worker_timeout_seconds)
    except Exception:
        decoded = _decode_image(str(path), config, apply_limits=False)
    return DecodedImage(
        data=decoded.data,
        width=decoded.width,
        height=decoded.height,
        mode=decoded.mode,
        mime=mime,
    )
