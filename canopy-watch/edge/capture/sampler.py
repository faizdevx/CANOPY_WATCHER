"""Frame selection for Tier1, and the queue that carries it.

Two separate concerns, both about staying current rather than staying complete:

**Sampling** is time-based, not "every Nth frame". A modulo counter means the
Tier1 rate silently changes when the camera negotiates a different fps than the
config requested — which the webcam driver warns is common. Asking for a frame
roughly every 200 ms stays correct at 15, 29.97, 30 or 60 fps.

**Backpressure** is drop-oldest. If the camera produces 30 fps and Tier1 manages
10, a FIFO queue accumulates minutes of stale frames and the detector ends up
reporting on a scene that no longer exists. Latest-frame-wins is the right
default for real-time detection — the buffer still holds the originals, so
nothing is lost for Tier2 retrieval, only for immediate inference.

Dropping is counted, never silent. `frames_dropped_backpressure` climbing is how
you find out Tier1 is too heavy for this board's `hardware_tier`, or that the
sample interval is set faster than the device can serve.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Dict, Optional

from .frame import CapturedFrame

logger = logging.getLogger(__name__)


class TimeBasedSampler:
    """Emit at most one frame per `interval_s`, without accumulating drift."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = max(0.0, float(interval_s))
        self._next_due: Optional[float] = None
        self.selected = 0
        self.skipped = 0
        self.resyncs = 0

    def should_select(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now

        if self.interval_s <= 0:            # 0 = send every frame
            self.selected += 1
            return True

        if self._next_due is None:
            self._next_due = now + self.interval_s
            self.selected += 1
            return True

        # Epsilon: repeated float addition of the interval drifts by ~1e-13,
        # which would otherwise skip a frame that is due exactly on time.
        if now < (self._next_due - 1e-9):
            self.skipped += 1
            return False

        # Advance by whole intervals. If we fell more than one interval behind
        # (a stall, a reconnect), resync to now instead of firing a catch-up
        # burst — a burst is the opposite of "stay current".
        behind = now - self._next_due
        if behind > self.interval_s:
            self._next_due = now + self.interval_s
            self.resyncs += 1
        else:
            self._next_due += self.interval_s

        self.selected += 1
        return True

    def reset(self) -> None:
        self._next_due = None

    def stats(self) -> Dict[str, Any]:
        return {
            "interval_ms": round(self.interval_s * 1000),
            "selected": self.selected,
            "skipped": self.skipped,
            "resyncs": self.resyncs,
        }


class LatestFrameQueue:
    """Bounded queue that discards the OLDEST item when full."""

    def __init__(self, depth: int = 1) -> None:
        self.depth = max(1, int(depth))
        self._q: "queue.Queue[CapturedFrame]" = queue.Queue(maxsize=self.depth)
        self._lock = threading.Lock()
        self.dropped = 0
        self.delivered = 0

    def put(self, frame: CapturedFrame) -> bool:
        """Returns False if something older had to be discarded to make room."""
        with self._lock:
            made_room = False
            while True:
                try:
                    self._q.put_nowait(frame)
                    return not made_room
                except queue.Full:
                    try:
                        stale = self._q.get_nowait()
                        self.dropped += 1
                        made_room = True
                        logger.debug("tier1 backpressure: dropped %s", stale.frame_id)
                    except queue.Empty:      # drained by a consumer, retry
                        continue

    def get(self, timeout: Optional[float] = 1.0) -> Optional[CapturedFrame]:
        try:
            frame = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        self.delivered += 1
        return frame

    def drain(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def qsize(self) -> int:
        return self._q.qsize()

    def stats(self) -> Dict[str, Any]:
        return {
            "depth": self.depth,
            "queued": self.qsize(),
            "delivered": self.delivered,
            "frames_dropped_backpressure": self.dropped,
        }