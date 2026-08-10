"""Capture Service.

    1. Give frame an ID          capture owns the sequence, not the driver
    2. Add timestamps            media (nullable) / received (monotonic) / wall clock
    3. Keep recent frames        memory-capped ring buffer
    4. Select frames             time-based sampling
    5. Send to Tier 1            drop-oldest queue, counted
    6. Retrieve exact frame      by frame_id, three explicit failure modes
    7. Handle camera loss        close -> backoff -> reopen, forever

What it deliberately does NOT do:

  * decode        — the driver boundary already emits canonical RGB uint8
  * resize        — Tier1 knows its own model geometry; capture must not
  * detect        — no classification, no bboxes, no events
  * crop for Tier2 — capture returns the ORIGINAL frame; the caller crops

Threading: one capture thread runs the loop. Tier1 consumes the queue from
another thread; Tier2 / the event manager calls `get_frame()` from a third. The
buffer is locked, and frames are handed out as read-only references rather than
copies — copying 6 MB per retrieval defeats the point of having a buffer.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..drivers import CameraDriver, CameraError, create_camera_driver
from .buffer import (
    CaptureError,
    FrameBuffer,
    FrameExpired,
    FrameNotFound,
    FrameNotYet,
    FrameRetrievalError,
)
from .frame import CapturedFrame, make_frame_id
from .sampler import LatestFrameQueue, TimeBasedSampler

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "buffer_seconds": 30.0,
    "buffer_max_memory_mb": 512,
    "tier1_sample_interval_ms": 200,
    "tier1_queue_depth": 1,
    "reopen_backoff_s": [1, 2, 5, 10, 30],
    "max_reopen_attempts": 0,          # 0 = never give up (field default)
    "idle_sleep_s": 0.001,
}


class CaptureService:
    """Owns exactly one camera driver and the frames it produces."""

    def __init__(
        self,
        driver: CameraDriver,
        source_id: str,
        capture_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        cfg = dict(DEFAULTS)
        cfg.update(dict(capture_config or {}))
        self.config = cfg

        self.driver = driver
        self.source_id = source_id

        self.buffer = FrameBuffer(
            source_id=source_id,
            max_seconds=float(cfg["buffer_seconds"]),
            max_memory_bytes=int(cfg["buffer_max_memory_mb"]) * 1024 * 1024,
        )
        self.sampler = TimeBasedSampler(float(cfg["tier1_sample_interval_ms"]) / 1000.0)
        self.tier1_queue = LatestFrameQueue(int(cfg["tier1_queue_depth"]))

        self._backoff: List[float] = [float(x) for x in cfg["reopen_backoff_s"]] or [1.0]
        self._max_reopen = int(cfg["max_reopen_attempts"])
        self._idle_sleep = float(cfg["idle_sleep_s"])

        # --- state that must survive a camera reconnect --------------------- #
        self._sequence = 0                  # NEVER reset on reopen
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = False
        self._started_at: Optional[float] = None

        # --- counters ------------------------------------------------------- #
        self._frames_captured = 0
        self._read_errors = 0
        self._reconnects = 0
        self._last_reconnect: Optional[float] = None
        self._last_error: Optional[str] = None
        self._last_frame_at: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Construction from resolved config
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        driver: Optional[CameraDriver] = None,
    ) -> "CaptureService":
        """Build from the merged config the loader produces.

            base.yaml -> <profile>.yaml -> hardware.generated.yaml -> env

        `hardware.driver.camera` picks the driver; `capture.*` is pure policy
        and lives in the profile layer only — nothing here is auto-detected.
        `identity.node_id` becomes the source id, which the loader already
        fails fast on if it is unset.
        """
        identity = config.get("identity") or {}
        source_id = identity.get("node_id")
        if not source_id:
            raise CaptureError(
                "capture: identity.node_id is not set — frame ids would not be "
                "attributable to a node. Set CW_NODE_ID."
            )
        cam = driver if driver is not None else create_camera_driver(config)
        return cls(cam, str(source_id), config.get("capture") or {})

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._running:
            return
        self._open_with_retry(initial=True)
        self._stop.clear()
        self._running = True
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()
        logger.info("capture: started for source %s", self.source_id)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._running = False
        try:
            self.driver.close()
        except Exception:                   # pragma: no cover
            logger.debug("capture: driver close failed", exc_info=True)
        logger.info("capture: stopped (%d frames captured)", self._frames_captured)

    def __enter__(self) -> "CaptureService":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.driver.read_frame()
            except CameraError as exc:
                self._read_errors += 1
                self._last_error = str(exc)
                if not self.driver.is_healthy():
                    logger.warning("capture: camera unhealthy (%s) — reconnecting", exc)
                    self._reconnect()
                else:
                    self._stop.wait(self._idle_sleep)
                continue
            except Exception as exc:        # pragma: no cover - defensive
                self._read_errors += 1
                self._last_error = repr(exc)
                logger.exception("capture: unexpected read failure")
                self._stop.wait(self._idle_sleep)
                continue

            self._ingest(frame)

    def _ingest(self, frame) -> CapturedFrame:
        """(1) id, (2) timestamps, (3) buffer, (4) select, (5) enqueue."""
        self._sequence += 1
        captured = CapturedFrame(
            frame_id=make_frame_id(self.source_id, self._sequence),
            source_id=self.source_id,
            sequence=self._sequence,
            data=frame.data,
            received_timestamp=frame.timestamp,
            wall_clock=frame.metadata.get("wall_clock", time.time()),
            # Only the CSI drivers actually have a sensor timestamp. Absent
            # stays absent — never backfilled from the receive time.
            media_timestamp=frame.metadata.get("media_timestamp"),
            driver=frame.source,
            metadata=dict(frame.metadata),
        )

        self.buffer.append(captured)
        self._frames_captured += 1
        self._last_frame_at = captured.received_timestamp

        if self.sampler.should_select(captured.received_timestamp):
            self.tier1_queue.put(captured)

        return captured

    # ------------------------------------------------------------------ #
    # (6) Exact retrieval
    # ------------------------------------------------------------------ #

    def get_frame(self, frame_id: str) -> CapturedFrame:
        """Return the exact original frame for `frame_id`.

        Raises FrameExpired / FrameNotFound / FrameNotYet — never a nearby
        frame. The array is read-only; crop or copy before drawing on it.
        """
        return self.buffer.get(frame_id)

    def try_get_frame(self, frame_id: str) -> Optional[CapturedFrame]:
        try:
            return self.buffer.get(frame_id)
        except FrameRetrievalError:
            return None

    def latest_frame(self) -> Optional[CapturedFrame]:
        return self.buffer.peek_latest()

    # ------------------------------------------------------------------ #
    # (5) Tier1 side
    # ------------------------------------------------------------------ #

    def next_for_tier1(self, timeout: Optional[float] = 1.0) -> Optional[CapturedFrame]:
        """Blocking pull of the next selected frame. None on timeout."""
        return self.tier1_queue.get(timeout=timeout)

    def run_tier1_loop(
        self,
        handler: Callable[[CapturedFrame], Any],
        stop_event: Optional[threading.Event] = None,
        timeout: float = 1.0,
    ) -> None:
        """Convenience driver-loop for a Tier1 consumer thread."""
        stop_event = stop_event or self._stop
        while not stop_event.is_set():
            frame = self.next_for_tier1(timeout=timeout)
            if frame is None:
                continue
            try:
                handler(frame)
            except Exception:
                logger.exception("capture: tier1 handler raised on %s", frame.frame_id)

    # ------------------------------------------------------------------ #
    # (7) Camera loss
    # ------------------------------------------------------------------ #

    def _reconnect(self) -> None:
        """Close, back off, reopen — indefinitely by default.

        Across a reconnect the sequence counter does NOT reset (that is why
        capture owns it rather than the driver) and the buffer is NOT cleared —
        Tier2 may still be retrieving a frame captured before the drop. A node
        that stops capturing because a cable was nudged, and needs someone to
        walk to it, is a failed design.
        """
        try:
            self.driver.close()
        except Exception:                   # pragma: no cover
            logger.debug("capture: close during reconnect failed", exc_info=True)

        self._reconnects += 1
        self._last_reconnect = time.monotonic()
        self.sampler.reset()
        self.tier1_queue.drain()            # queued frames are stale by now
        self._open_with_retry(initial=False)

    def _open_with_retry(self, initial: bool) -> None:
        attempt = 0
        while not self._stop.is_set():
            try:
                self.driver.open()
                if not initial:
                    logger.info("capture: camera reconnected after %d attempt(s)", attempt + 1)
                return
            except CameraError as exc:
                attempt += 1
                self._last_error = str(exc)
                if initial and attempt == 1:
                    # First open on startup is a configuration problem, not a
                    # field dropout — surface it instead of silently retrying.
                    raise
                if self._max_reopen and attempt >= self._max_reopen:
                    raise CaptureError(
                        f"capture: camera did not come back after {attempt} attempts: {exc}"
                    ) from exc
                delay = self._backoff[min(attempt - 1, len(self._backoff) - 1)]
                logger.warning("capture: reopen attempt %d failed (%s) — retrying in %.0fs",
                               attempt, exc, delay)
                self._stop.wait(delay)

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    def health(self) -> Dict[str, Any]:
        now = time.monotonic()
        return {
            "source_id": self.source_id,
            "running": self._running,
            "uptime_s": None if self._started_at is None else round(now - self._started_at, 1),
            "frames_captured": self._frames_captured,
            "sequence": self._sequence,
            "capture_fps": self._measured_fps(now),
            "last_frame_age_s": (None if self._last_frame_at is None
                                 else round(now - self._last_frame_at, 3)),
            "read_errors": self._read_errors,
            "reconnects": self._reconnects,
            "last_reconnect_age_s": (None if self._last_reconnect is None
                                     else round(now - self._last_reconnect, 1)),
            "last_error": self._last_error,
            "camera": self.driver.health_detail(),
            "buffer": self.buffer.stats(),
            "sampler": self.sampler.stats(),
            "tier1": self.tier1_queue.stats(),
        }

    def _measured_fps(self, now: float) -> Optional[float]:
        """Achieved fps, measured. Not the number the config asked for."""
        if self._started_at is None or self._frames_captured == 0:
            return None
        elapsed = now - self._started_at
        return round(self._frames_captured / elapsed, 2) if elapsed > 0 else None


__all__ = [
    "CaptureService",
    "CapturedFrame",
    "CaptureError",
    "FrameBuffer",
    "FrameExpired",
    "FrameNotFound",
    "FrameNotYet",
    "FrameRetrievalError",
]