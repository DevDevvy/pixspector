from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI warnings
import matplotlib.pyplot as plt


def save_heatmap_png(img_u8: np.ndarray, out_path: Path, title: Optional[str] = None) -> None:
    """
    Save a grayscale heatmap (uint8) as a PNG using matplotlib.
    (No style/colors specified to keep deps simple and consistent.)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_u8, cmap="viridis")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_overlay_png(base_bgr: np.ndarray, heat_u8: np.ndarray, out_path: Path,
                     alpha: float = 0.5, title: Optional[str] = None) -> None:
    """
    Overlay a heatmap onto the base image (resize to match).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = base_bgr.shape[:2]
    heat = cv2.resize(heat_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    overlay = cv2.addWeighted(base_bgr, 1.0, heat_color, alpha, 0)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_image_png(img_bgr: np.ndarray, out_path: Path, title: Optional[str] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_grayscale_png(img_u8: np.ndarray, out_path: Path, title: Optional[str] = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_u8, cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_curve(values: Sequence[float], out_path: Path, title: Optional[str] = None,
               xlabel: str = "", ylabel: str = "") -> None:
    """
    Save a simple line plot.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(values))
    plt.figure(figsize=(8, 6))
    plt.plot(x, values)
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
