"""Rolling frame buffer — bounded by BOTH time and memory.

The memory cap is not optional. Run the arithmetic the spec skips:

    1920 x 1080 x 3 bytes        =  6.2 MB per frame
    6.2 MB x 30 fps x 60 seconds = 11.2 GB

Jetson Nano has 4 GB shared between CPU and GPU. A Pi has 2-8 GB. A 60-second
decoded-frame buffer at 1080p30 is impossible on every target device, so
`buffer_seconds` alone is a lie. `buffer_max_memory_mb` is the real constraint
and it always wins; the effective depth is derived at runtime from the actual
frame size and logged, so nobody discovers the limit via an OOM kill on a node
in a forest at 3am.

Retrieval failures are three distinct diagnoses, never collapsed into one:

    FrameExpired    valid id, aged out       -> buffer too small / Tier2 too slow
    FrameNotFound   id never existed         -> malformed id, or wrong source
    FrameNotYet     id is ahead of capture   -> ordering or clock bug

Returning a "close enough" nearby frame instead is how debugging becomes
archaeology; this buffer never does that.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

from .frame import CapturedFrame, freeze, parse_frame_id

logger = logging.getLogger(__name__)


class CaptureError(Exception):
    """Base class for Capture Service failures."""


class FrameRetrievalError(CaptureError):
    """Base for the three retrieval outcomes, so callers can catch one thing."""

    code = "FRAME_ERROR"

    def __init__(self, frame_id: str, detail: str = "") -> None:
        self.frame_id = frame_id
        super().__init__(f"{self.code}: {frame_id}{(' — ' + detail) if detail else ''}")


class FrameExpired(FrameRetrievalError):
    """The id was valid; the frame has already been evicted."""
    code = "FRAME_EXPIRED"


class FrameNotFound(FrameRetrievalError):
    """The id never referred to a frame this buffer produced."""
    code = "FRAME_NOT_FOUND"


class FrameNotYet(FrameRetrievalError):
    """The sequence is ahead of anything captured yet."""
    code = "FRAME_NOT_YET"


class FrameBuffer:
    """Thread-safe, insertion-ordered, evicts oldest first.

    Keyed by frame_id (dict + ordered eviction) rather than positional: O(1)
    exact lookup, and deliberately no "nearest frame to time T" API. The video
    archive tier can build its own index when it needs one.
    """

    def __init__(
        self,
        source_id: str,
        max_seconds: float = 30.0,
        max_memory_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        self.source_id = source_id
        self.max_seconds = float(max_seconds)
        self.max_memory_bytes = int(max_memory_bytes)

        self._lock = threading.RLock()
        self._frames: "OrderedDict[str, CapturedFrame]" = OrderedDict()
        self._bytes = 0
        self._highest_sequence: Optional[int] = None
        self._lowest_sequence: Optional[int] = None
        self._evicted_total = 0
        self._logged_capacity = False

    # ------------------------------------------------------------------ #

    def append(self, frame: CapturedFrame) -> int:
        """Add a frame, evicting as needed. Returns how many were evicted."""
        object.__setattr__(frame, "data", freeze(frame.data))

        with self._lock:
            self._frames[frame.frame_id] = frame
            self._bytes += frame.nbytes
            self._highest_sequence = frame.sequence
            if self._lowest_sequence is None:
                self._lowest_sequence = frame.sequence
            self._log_capacity_once(frame)
            return self._evict(now=frame.received_timestamp)

    def get(self, frame_id: str) -> CapturedFrame:
        """Return the exact original frame, or raise a specific error."""
        with self._lock:
            hit = self._frames.get(frame_id)
            if hit is not None:
                return hit

            parsed = parse_frame_id(frame_id)
            if parsed is None:
                raise FrameNotFound(frame_id, "not a valid frame id")
            source, sequence = parsed
            if source != self.source_id:
                raise FrameNotFound(
                    frame_id, f"belongs to source {source!r}, this node is {self.source_id!r}")

            if self._highest_sequence is None:
                raise FrameNotYet(frame_id, "no frames captured yet")
            if sequence > self._highest_sequence:
                raise FrameNotYet(
                    frame_id, f"latest captured sequence is {self._highest_sequence}")
            if self._lowest_sequence is not None and sequence < self._lowest_sequence:
                raise FrameExpired(
                    frame_id,
                    f"oldest retained sequence is {self._lowest_sequence} "
                    f"({len(self._frames)} frames / {self.seconds_retained:.1f}s retained)")
            raise FrameExpired(frame_id, "evicted")

    def peek_latest(self) -> Optional[CapturedFrame]:
        with self._lock:
            if not self._frames:
                return None
            return next(reversed(self._frames.values()))

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()
            self._bytes = 0
            self._lowest_sequence = None

    # ------------------------------------------------------------------ #

    def _evict(self, now: float) -> int:
        evicted = 0
        while self._frames:
            oldest_id = next(iter(self._frames))
            oldest = self._frames[oldest_id]
            too_old = (now - oldest.received_timestamp) > self.max_seconds
            too_big = self._bytes > self.max_memory_bytes
            if not (too_old or too_big):
                break
            if len(self._frames) == 1 and not too_old:
                break                       # never evict the only frame on size alone
            self._frames.popitem(last=False)
            self._bytes -= oldest.nbytes
            evicted += 1
        if evicted:
            self._evicted_total += evicted
            self._lowest_sequence = next(iter(self._frames.values())).sequence \
                if self._frames else None
        return evicted

    def _log_capacity_once(self, frame: CapturedFrame) -> None:
        """State the real buffer depth in actual numbers, once, at startup."""
        if self._logged_capacity:
            return
        self._logged_capacity = True
        per_frame_mb = frame.nbytes / (1024 * 1024)
        max_frames = max(1, int(self.max_memory_bytes // max(1, frame.nbytes)))
        logger.info(
            "capture buffer: %.1f MB/frame at %dx%d — memory cap %d MB allows "
            "%d frames; requested retention %.0fs",
            per_frame_mb, frame.width, frame.height,
            self.max_memory_bytes // (1024 * 1024), max_frames, self.max_seconds,
        )

    # ------------------------------------------------------------------ #

    @property
    def seconds_retained(self) -> float:
        with self._lock:
            if len(self._frames) < 2:
                return 0.0
            values = list(self._frames.values())
            return values[-1].received_timestamp - values[0].received_timestamp

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "frames": len(self._frames),
                "bytes": self._bytes,
                "megabytes": round(self._bytes / (1024 * 1024), 1),
                "seconds_retained": round(self.seconds_retained, 2),
                "max_seconds": self.max_seconds,
                "max_megabytes": self.max_memory_bytes // (1024 * 1024),
                "lowest_sequence": self._lowest_sequence,
                "highest_sequence": self._highest_sequence,
                "evicted_total": self._evicted_total,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def __contains__(self, frame_id: object) -> bool:
        with self._lock:
            return frame_id in self._frames