"""Crop the original frame for Tier2, preserving aspect ratio.

Two things this gets right that a naive `frame[y1:y2, x1:x2]` does not:

1. **Context padding.** The Tier1 box is tight around the subject. An
   ImageNet-pretrained backbone (EfficientNet-B2 and friends) was trained on
   framed photographs, not edge-to-edge subjects, so a tight crop loses
   accuracy. Pad ~20% before cropping.
2. **Letterbox, don't squash.** Resizing a 40x180 crop straight to 260x260
   distorts the aspect ratio. A standing human becomes a squat one. The loss is
   invisible unless you look at the crops, which nobody does until accuracy is
   already bad. Pad to square first, then resize.

Uses OpenCV's resize when available (better interpolation) and falls back to
dependency-free numpy nearest-neighbour so this works in CI and on a bare Pi.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .types import BBox


def crop_for_tier2(
    frame: np.ndarray,
    bbox: BBox,
    target_size: int = 260,
    padding_fraction: float = 0.2,
    pad_value: int = 114,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Return (letterboxed square crop, original crop pixel size).

    The frame array is read-only (capture freezes buffered frames), so the crop
    is always a copy — which is what we want anyway before handing it to a model.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox.padded(padding_fraction).to_pixels(width, height)
    crop = np.array(frame[y1:y2, x1:x2], dtype=np.uint8, copy=True)
    original_size = (crop.shape[1], crop.shape[0])
    return letterbox(crop, target_size, pad_value), original_size


def letterbox(image: np.ndarray, size: int, pad_value: int = 114) -> np.ndarray:
    """Scale to fit inside size x size preserving aspect, pad the remainder."""
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.full((size, size, 3), pad_value, dtype=np.uint8)

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = resize(image, new_w, new_h)

    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2  # type: ignore
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    except ImportError:
        return _resize_nearest(image, width, height)


def _resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Dependency-free fallback. Lower quality than INTER_AREA, but it keeps the
    whole pipeline runnable with numpy alone."""
    src_h, src_w = image.shape[:2]
    rows = (np.arange(height) * src_h // height).clip(0, src_h - 1)
    cols = (np.arange(width) * src_w // width).clip(0, src_w - 1)
    return image[rows[:, None], cols[None, :]]


def normalize_for_model(
    image: np.ndarray,
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
    scale: float = 1.0 / 255.0,
    channels_first: bool = True,
) -> np.ndarray:
    """uint8 HWC RGB -> float32 CHW (or HWC), optionally mean/std normalized."""
    arr = image.astype(np.float32) * scale
    if mean is not None:
        arr = arr - np.asarray(mean, dtype=np.float32)
    if std is not None:
        arr = arr / np.asarray(std, dtype=np.float32)
    if channels_first:
        arr = np.transpose(arr, (2, 0, 1))
    return np.ascontiguousarray(arr)