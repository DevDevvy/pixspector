from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import cv2


def iter_blocks_8x8(img: np.ndarray) -> Iterator[Tuple[int, int, np.ndarray]]:
    """
    Iterate non-overlapping 8x8 blocks over a 2D array (float32 recommended).
    Yields (by, bx, block).
    """
    h, w = img.shape[:2]
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    for by in range(0, h8, 8):
        for bx in range(0, w8, 8):
            yield by, bx, img[by:by + 8, bx:bx + 8]


def dct2(block: np.ndarray) -> np.ndarray:
    """
    2D DCT-II on an 8x8 block.
    """
    return cv2.dct(block.astype(np.float32))


def idct2(coeffs: np.ndarray) -> np.ndarray:
    """
    2D inverse DCT on an 8x8 block of coefficients.
    """
    return cv2.idct(coeffs.astype(np.float32))
