"""Value types passed between capture, Tier1, Tier2 and the risk scorer.

Bounding boxes are **normalized 0-1 everywhere**. Pixel coordinates are only
materialised at the moment of cropping, because the same box has to make sense
against a 1920x1080 live frame, a downscaled archive segment, and a dashboard
canvas of arbitrary size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

CATEGORIES = ("human", "vehicle", "animal", "unknown")


@dataclass(frozen=True)
class BBox:
    """Normalized box, origin top-left, x1<x2 and y1<y2."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"degenerate bbox: {self}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Fraction of the frame covered — a crude but useful distance proxy."""
        return self.width * self.height

    def clamped(self) -> "BBox":
        return BBox(
            max(0.0, min(1.0, self.x1)), max(0.0, min(1.0, self.y1)),
            max(0.0, min(1.0, self.x2)), max(0.0, min(1.0, self.y2)),
        )

    def padded(self, fraction: float) -> "BBox":
        """Grow by `fraction` of the box size on every side, then clamp.

        A tight crop is what a fine-tuned model expects; an ImageNet-pretrained
        backbone was trained on framed photographs and does measurably worse on
        an edge-to-edge subject. Context padding splits the difference.
        """
        dx = self.width * fraction
        dy = self.height * fraction
        return BBox(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy).clamped()

    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        x1 = max(0, min(width - 1, int(round(self.x1 * width))))
        y1 = max(0, min(height - 1, int(round(self.y1 * height))))
        x2 = max(x1 + 1, min(width, int(round(self.x2 * width))))
        y2 = max(y1 + 1, min(height, int(round(self.y2 * height))))
        return x1, y1, x2, y2

    @classmethod
    def from_pixels(cls, x1: int, y1: int, x2: int, y2: int,
                    width: int, height: int) -> "BBox":
        return cls(x1 / width, y1 / height, x2 / width, y2 / height).clamped()

    def to_dict(self) -> Dict[str, float]:
        return {"x1": round(self.x1, 6), "y1": round(self.y1, 6),
                "x2": round(self.x2, 6), "y2": round(self.y2, 6)}


@dataclass(frozen=True)
class Candidate:
    """Tier1 output. Coarse triage, not a species identification.

    COCO has no deer, no tiger, no poacher-on-foot-at-night. Tier1's job is to
    answer "is this worth waking Tier2 for", so `category` is deliberately one
    of four buckets and `label` keeps whatever finer string the model produced.
    """

    category: str
    confidence: float
    bbox: BBox
    label: Optional[str] = None
    backend: str = "unknown"
    model_version: Optional[str] = None
    latency_ms: Optional[float] = None
    motion_gated: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "category": self.category,
            "confidence": round(float(self.confidence), 4),
            "backend": self.backend,
            "model_version": self.model_version,
            "latency_ms": (None if self.latency_ms is None
                           else round(float(self.latency_ms), 2)),
            "motion_gated": self.motion_gated,
        }


@dataclass(frozen=True)
class Verification:
    """Tier2 output. A classifier, so no bbox — that passes through from Tier1."""

    category: str
    confidence: float
    label: Optional[str] = None
    backend: str = "unknown"
    model_version: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "category": self.category,
            "confidence": round(float(self.confidence), 4),
            "backend": self.backend,
            "model_version": self.model_version,
            "latency_ms": (None if self.latency_ms is None
                           else round(float(self.latency_ms), 2)),
        }


@dataclass(frozen=True)
class Tier2Task:
    """What actually sits in the Tier2 queue: the CROP, not a frame id.

    Cropping happens microseconds after Tier1 fires, while the frame is
    guaranteed to still be in the capture buffer. Queuing a frame_id instead
    would make Tier2 latency a hidden dependency on buffer retention — a slow
    Tier2 would trigger our own FRAME_EXPIRED error. It also keeps the queued
    object around 200 KB instead of 6 MB.
    """

    frame_id: str
    candidate: Candidate
    crop: Any                       # np.ndarray, RGB uint8, model input size
    original_crop_size: Tuple[int, int]
    frame_width: int
    frame_height: int
    captured_wall_clock: float
    media_timestamp: Optional[float] = None
    enqueued_at: float = 0.0