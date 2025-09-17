"""
Generate sample images for pixspector demos.

- real_1.jpg: noisy natural-like gradient
- edited_1.jpg: same gradient + pasted logo/box
- synthetic_1.jpg: checkerboard + stripes (obvious periodicity)

Run:
    python examples/sample_images/make_samples.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2


def real_like(path: Path):
    h, w = 512, 512
    base = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        base[:, i, :] = i * 255 // (w - 1)
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def edited(path: Path):
    # Start from real-like
    h, w = 512, 512
    base = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        base[:, i, :] = i * 255 // (w - 1)
    cv2.rectangle(base, (140, 140), (372, 372), (255, 255, 255), -1)
    cv2.putText(base, "PASTE", (170, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.imwrite(str(path), base, [int(cv2.IMWRITE_JPEG_QUALITY), 85])


def synthetic(path: Path):
    h, w = 512, 512
    img = np.zeros((h, w), dtype=np.uint8)
    # checkerboard
    for y in range(0, h, 16):
        for x in range(0, w, 16):
            if ((x // 16) + (y // 16)) % 2 == 0:
                img[y:y+16, x:x+16] = 255
    # add vertical stripes
    for x in range(0, w, 32):
        cv2.line(img, (x, 0), (x, h-1), 255, 1)
    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(path), img3, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def main():
    out_dir = Path(__file__).parent
    (out_dir).mkdir(parents=True, exist_ok=True)
    real_like(out_dir / "real_1.jpg")
    edited(out_dir / "edited_1.jpg")
    synthetic(out_dir / "synthetic_1.jpg")
    print(f"Sample images written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
