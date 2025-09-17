from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple

import cv2
import numpy as np

from ..utils.blocks import iter_blocks_8x8, dct2


@dataclass
class BenfordResult:
    n_ac: int                         # number of AC coefficients examined
    freq: np.ndarray                  # observed first-digit frequencies (1..9), shape (9,)
    expected: np.ndarray              # Benford expected distribution, shape (9,)
    chi2: float                       # chi-squared statistic
    strong: bool                      # strong deviation flag
    moderate: bool                    # moderate deviation flag
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self).copy()
        d["freq"] = self.freq.tolist()
        d["expected"] = self.expected.tolist()
        return d


def _first_digit_hist(vals: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Compute first-digit histogram (1..9) for positive values in `vals`.
    Returns (hist[9], n).
    """
    vals = np.abs(vals)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return np.zeros(9, dtype=np.int64), 0
    # first digit in base 10
    exponents = np.floor(np.log10(vals))
    mantissas = vals / (10 ** exponents)
    first_digits = np.floor(mantissas).astype(np.int64)
    first_digits = np.clip(first_digits, 1, 9)
    hist = np.bincount(first_digits, minlength=10)[1:10]  # skip 0
    return hist.astype(np.int64), int(vals.size)


def _benford_expected() -> np.ndarray:
    # Benford law probabilities for digits 1..9: log10(1 + 1/d)
    d = np.arange(1, 10, dtype=np.float64)
    return np.log10(1.0 + 1.0 / d)


def run_dct_benford(gray_u8: np.ndarray, min_blocks: int = 256,
                    strong_z: float = 3.0, moderate_z: float = 2.0) -> BenfordResult:
    """
    Run 8x8 DCT over grayscale image, collect AC coefficients, and test
    first-digit distribution vs. Benford's law using a chi-squared measure.
    Flags 'strong'/'moderate' if normalized deviation is large.
    """
    # Ensure dimensions are multiples of 8 (crop bottom/right as needed)
    h, w = gray_u8.shape[:2]
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    img = gray_u8[:h8, :w8].astype(np.float32) - 128.0

    ac_vals = []
    n_blocks = 0
    for by, bx, block in iter_blocks_8x8(img):
        n_blocks += 1
        c = dct2(block)
        # exclude DC coefficient at [0,0]; collect AC coeffs
        ac = c.copy()
        ac[0, 0] = 0.0
        ac_vals.append(ac.flatten())
    if n_blocks == 0:
        obs = np.zeros(9, dtype=np.float64)
        exp = _benford_expected()
        return BenfordResult(0, obs, exp, chi2=0.0, strong=False, moderate=False, meta={"note": "No 8x8 blocks."})

    if n_blocks < min_blocks:
        # Still compute, but mark as low-sample
        pass

    ac_vals = np.concatenate(ac_vals, axis=0)
    # Remove zeros (DCs already zeroed, but AC can be zero too)
    ac_vals = ac_vals[ac_vals != 0.0]

    hist, n = _first_digit_hist(ac_vals)
    if n == 0:
        obs = np.zeros(9, dtype=np.float64)
        exp = _benford_expected()
        return BenfordResult(0, obs, exp, chi2=0.0, strong=False, moderate=False,
                             meta={"note": "No non-zero AC coefficients."})

    obs = hist.astype(np.float64) / float(n)
    exp = _benford_expected()

    # Chi-squared against expected
    # (Use simple Pearson chi2; small counts are unlikely with many blocks.)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.sum((obs - exp) ** 2 / (exp + 1e-12)))

    # Normalize deviation in a crude "z-like" way for thresholds:
    # max absolute deviation across digits divided by sqrt(exp_var)
    max_dev = float(np.max(np.abs(obs - exp)))
    # approximate variance term for a multinomial per digit:
    var = exp * (1 - exp) / max(1, n)
    z_like = max_dev / (np.sqrt(float(np.max(var)) + 1e-12))

    strong = bool(z_like >= strong_z)
    moderate = bool((not strong) and z_like >= moderate_z)

    return BenfordResult(
        n_ac=int(n),
        freq=obs,
        expected=exp,
        chi2=chi2,
        strong=strong,
        moderate=moderate,
        meta={
            "blocks_considered": n_blocks,
            "min_blocks_recommended": min_blocks,
            "z_like": z_like,
            "note": "Benford on DCT ACs can flag atypical quantization / recompression signatures."
        },
    )
