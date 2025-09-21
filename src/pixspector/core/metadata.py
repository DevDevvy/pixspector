from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import exifread
from PIL import Image, JpegImagePlugin


@dataclass
class Metadata:
    exif: Dict[str, Any]
    icc_profile_present: bool
    icc_description: Optional[str]
    jpeg_subsampling: Optional[str]
    jpeg_quant_tables: Optional[Dict[str, Any]]
    format: str
    size: tuple[int, int]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _read_exif(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        # Convert to plain dict with string keys/values
        return {str(k): str(v) for k, v in tags.items()}
    except Exception:
        return {}


def _read_icc_and_jpeg_info(path: Path):
    """
    Read ICC profile (if any), JPEG subsampling and quantization tables (best effort).
    Pillow can expose ICC via image.info['icc_profile'].
    Quant tables exposure is not guaranteed across all versions; we attempt common paths.
    """
    icc_present = False
    icc_desc = None
    subsampling = None
    qtables = None
    fmt = ""
    size = (0, 0)

    try:
        with Image.open(path) as im:
            fmt = im.format or ""
            size = im.size

            # ICC
            icc_bytes = im.info.get("icc_profile")
            if icc_bytes:
                icc_present = True
                # Attempt to get a friendly description if available
                try:
                    # Lazy import to avoid hard dependency
                    import io
                    from PIL import ImageCms
                    p = ImageCms.getOpenProfile(io.BytesIO(icc_bytes))
                    icc_desc = ImageCms.getProfileName(p)
                except Exception:
                    icc_desc = None

            # JPEG details
            if isinstance(im, JpegImagePlugin.JpegImageFile):
                try:
                    subsampling = JpegImagePlugin.get_sampling(im)
                except Exception:
                    subsampling = None
                # Best-effort quantization tables
                try:
                    # Newer Pillow exposes getqtables()
                    qtables_list = im.getqtables()
                    if qtables_list:
                        qtables = {f"table_{i}": tbl for i, tbl in enumerate(qtables_list)}
                except Exception:
                    # Fallback: some builds expose "quantization" in info
                    qtables = im.info.get("quantization")
    except Exception:
        pass

    return icc_present, icc_desc, subsampling, qtables, fmt, size


def read_metadata(path: Path) -> Metadata:
    exif = _read_exif(path)
    icc_present, icc_desc, subsampling, qtables, fmt, size = _read_icc_and_jpeg_info(path)
    return Metadata(
        exif=exif,
        icc_profile_present=icc_present,
        icc_description=icc_desc,
        jpeg_subsampling=str(subsampling) if subsampling is not None else None,
        jpeg_quant_tables=qtables,
        format=fmt,
        size=size,
    )
