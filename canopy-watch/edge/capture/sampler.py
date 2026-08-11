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
    """Bounded queue with a configurable full-queue policy.

    `drop_oldest` (default) is right for a LIVE camera: there is always another
    frame 33 ms behind, so shed the stale one and stay current.

    `block` is right for a FINITE source. A video file is not "ahead" of the
    consumer in any meaningful sense — it will wait. Dropping there makes the
    run non-deterministic (which sampled frame Tier1 actually sees depends on
    thread scheduling) and silently skips footage, which defeats the point of
    evaluating on a recording: a threshold change must be attributable to the
    threshold, not to how busy the machine was that afternoon.
    """

    def __init__(self, depth: int = 1, policy: str = "drop_oldest") -> None:
        self.depth = max(1, int(depth))
        if policy not in ("drop_oldest", "block"):
            raise ValueError(
                f"tier1 queue policy must be drop_oldest|block, got {policy!r}")
        self.policy = policy
        self._q: "queue.Queue[CapturedFrame]" = queue.Queue(maxsize=self.depth)
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self.dropped = 0
        self.delivered = 0
        self.blocked_waits = 0

    def close(self) -> None:
        """Release any producer blocked in put() so shutdown cannot deadlock."""
        self._closed.set()

    def put(self, frame: CapturedFrame) -> bool:
        """False if something was dropped (or the queue closed while blocking)."""
        if self.policy == "block":
            while not self._closed.is_set():
                try:
                    self._q.put(frame, timeout=0.1)
                    return True
                except queue.Full:
                    self.blocked_waits += 1
            return False

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
            "policy": self.policy,
            "queued": self.qsize(),
            "delivered": self.delivered,
            "blocked_waits": self.blocked_waits,
            "frames_dropped_backpressure": self.dropped,
        }