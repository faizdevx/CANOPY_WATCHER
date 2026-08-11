"""Motion gate — the cheap stage in front of the Tier1 detector.

A detector looks at ONE frame and says "there is a dog here". It has no memory,
so it cannot know anything moved. Motion is frame differencing against a running
background. They answer different questions, and on a solar-powered node
pointed at an empty forest, running a detector on every frame is mostly wasted
power.

    frame -> motion gate (~1 ms, numpy) -> [changed?] -> detector (20-200 ms)
                    |
                    +-> no change: dropped, no inference, no power spent

**The gate has a real cost**: a motionless subject is invisible to it. A poacher
standing still is exactly the thing you most want to catch. So it is OFF by
default on the laptop profile (mains power, and correctness while developing)
and configurable ON for Pi/Jetson where the power budget is real. This is a
policy knob in yaml, not a code branch — and the gap is documented rather than
discovered in the field.

numpy only: no OpenCV dependency, so it runs identically in Codespaces, on a Pi,
and in CI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .types import BBox

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotionResult:
    motion: bool
    changed_fraction: float
    bbox: Optional[BBox] = None
    warming_up: bool = False


class MotionGate:
    """Exponential-moving-average background subtraction on a downsampled grid.

    Downsampling first is what makes this ~1 ms: a 1920x1080 frame at stride 8
    is 240x135, and 32k pixels of absolute difference is nothing. It also acts
    as a crude blur, which suppresses single-pixel sensor noise for free.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        p = dict(params or {})
        self.enabled: bool = bool(p.get("enabled", False))
        self.stride: int = max(1, int(p.get("downsample_stride", 8)))
        #: per-pixel intensity delta (0-255) that counts as "changed"
        self.pixel_threshold: float = float(p.get("pixel_threshold", 18))
        #: fraction of changed pixels needed to declare motion
        self.min_area_fraction: float = float(p.get("min_area_fraction", 0.002))
        #: cap: a whole-frame change is a light switch or auto-exposure, not an animal
        self.max_area_fraction: float = float(p.get("max_area_fraction", 0.6))
        #: background adaptation rate; lower = slower to absorb a parked object
        self.learning_rate: float = float(p.get("learning_rate", 0.05))
        #: frames to build a background before any decision is trusted
        self.warmup_frames: int = int(p.get("warmup_frames", 10))

        self._background: Optional[np.ndarray] = None
        self._seen = 0
        self.triggered = 0
        self.suppressed = 0
        self.rejected_too_large = 0

    # ------------------------------------------------------------------ #

    def evaluate(self, image: np.ndarray) -> MotionResult:
        """Disabled gate always passes — the detector then sees every frame."""
        if not self.enabled:
            return MotionResult(motion=True, changed_fraction=1.0)

        small = self._grayscale_downsample(image)

        if self._background is None:
            self._background = small.copy()
            self._seen = 1
            return MotionResult(True, 0.0, warming_up=True)

        diff = np.abs(small - self._background)
        mask = diff > self.pixel_threshold
        changed = float(mask.mean())

        self._background += self.learning_rate * (small - self._background)
        self._seen += 1

        if self._seen <= self.warmup_frames:
            # Background not settled: pass frames through rather than risk
            # silently dropping real events during the first second.
            return MotionResult(True, changed, warming_up=True)

        if changed > self.max_area_fraction:
            # Whole-frame change: exposure step, headlights, cloud shadow.
            self.rejected_too_large += 1
            self.suppressed += 1
            return MotionResult(False, changed)

        if changed < self.min_area_fraction:
            self.suppressed += 1
            return MotionResult(False, changed)

        self.triggered += 1
        return MotionResult(True, changed, bbox=self._mask_bbox(mask))

    def reset(self) -> None:
        """Called after a camera reconnect — the old background is a different
        scene now (auto-exposure resets, the camera may have been nudged)."""
        self._background = None
        self._seen = 0

    # ------------------------------------------------------------------ #

    def _grayscale_downsample(self, image: np.ndarray) -> np.ndarray:
        return image[:: self.stride, :: self.stride, :].mean(axis=2).astype(np.float32)

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> Optional[BBox]:
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            return None
        h, w = mask.shape
        try:
            return BBox(
                cols[0] / w, rows[0] / h,
                (cols[-1] + 1) / w, (rows[-1] + 1) / h,
            ).clamped()
        except ValueError:
            return None

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "frames_seen": self._seen,
            "triggered": self.triggered,
            "suppressed": self.suppressed,
            "rejected_too_large": self.rejected_too_large,
            "background_ready": self._seen > self.warmup_frames,
        }