"""Edge pipeline — the wiring.

    capture thread ─► tier1 queue (depth 1, drop-OLDEST)      stay current
                          │
                     Tier1 thread
                       motion gate ──► detector
                          │ fires
                       crop NOW from the capture buffer
                          │
                     tier2 queue (depth 32, drop-NEWEST)      don't waste work
                          │
                     Tier2 thread
                       verify ──► RiskScorer ──► DetectionEvent
                          │
                     bus.publish("detection.scored")
                          │
                ┌─────────┴──────────┐
                ▼                    ▼
          event store (later)   alerting / debug API

Two queues with opposite shedding policies, for the reason argued in
`edge/inference/queues.py`: raw frames are cheap and replaceable, Tier2 tasks
are expensive and already paid for.

**Cropping happens on the Tier1 thread, immediately.** Queuing a `frame_id` and
resolving it on the Tier2 thread would make Tier2 latency a hidden dependency on
buffer retention — a slow Tier2 would trigger our own FRAME_EXPIRED. Cropping at
enqueue time removes that coupling entirely, and the queued object is ~200 KB
instead of 6 MB.

It does NOT remove the coupling to Tier1's own latency: capture -> tier1 queue
-> thread wake -> inference is still real time, and a very small buffer can roll
past in it. That case is survivable rather than fatal, because Tier1 is holding
a strong reference to the identical CapturedFrame — the crop comes from that and
`buffer_misses` counts it. Sizing rule:

    minimum buffer frames  >  capture_fps x (tier1 queue wait + tier1 latency)

When `ai.tier2.mode` is `defer_to_cloud` or `disabled`, the candidate skips the
Tier2 queue and goes straight to scoring with the matching `tier2_status`. The
pipeline shape does not change per profile — only which stage is active.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Mapping, Optional

from .bus import EventBus, InProcessBus
from .capture import CaptureService, FrameRetrievalError
from .events import DetectionEvent, SnapshotWriter
from .inference import (
    DropNewestQueue,
    MotionGate,
    Tier2Task,
    create_tier1,
    create_tier2,
    crop_for_tier2,
    tier2_mode,
)
from .inference.tier2.base import Tier2Error
from .scoring import PersistenceWindow, RiskScorer, ScoringContext

logger = logging.getLogger(__name__)

TOPIC_CANDIDATE = "detection.candidate"
TOPIC_SCORED = "detection.scored"

PIPELINE_DEFAULTS: Dict[str, Any] = {
    "tier2_queue_depth": 32,
    "crop_padding_fraction": 0.2,
    "publish_candidates": False,
    "validate_events": True,
    "utc_offset_hours": 0.0,
}


class EdgePipeline:
    def __init__(
        self,
        capture: CaptureService,
        config: Optional[Mapping[str, Any]] = None,
        bus: Optional[EventBus] = None,
    ) -> None:
        cfg = dict(config or {})
        self.config = cfg
        pipe_cfg = {**PIPELINE_DEFAULTS, **(cfg.get("pipeline") or {})}
        self.pipeline_config = pipe_cfg

        identity = cfg.get("identity") or {}
        self.node_id: str = capture.source_id
        self.site_id: Optional[str] = identity.get("site_id")
        self.zone: Optional[str] = identity.get("zone")

        self.capture = capture
        self.bus = bus or InProcessBus()

        ai = cfg.get("ai") or {}
        self.motion_gate = MotionGate((ai.get("tier1") or {}).get("motion_gate"))
        self.tier1 = create_tier1(cfg)
        self.tier2_mode = tier2_mode(cfg)
        self.tier2 = create_tier2(cfg) if self.tier2_mode == "local" else None

        self.tier2_queue: "DropNewestQueue[Tier2Task]" = DropNewestQueue(
            int(pipe_cfg["tier2_queue_depth"]))

        self.snapshots = SnapshotWriter(cfg.get("snapshots"))

        scoring_cfg = cfg.get("scoring") or {}
        self.scorer = RiskScorer(scoring_cfg)
        self.persistence = PersistenceWindow(scoring_cfg.get("persistence"))

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._running = False
        self._started_at: Optional[float] = None

        # counters
        self.frames_examined = 0
        self.motion_suppressed = 0
        self.candidates = 0
        self.crops_failed = 0
        self.buffer_misses = 0
        self.events_emitted = 0
        self.tier2_failures = 0

    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config: Mapping[str, Any],
                    bus: Optional[EventBus] = None) -> "EdgePipeline":
        capture = CaptureService.from_config(config)
        return cls(capture, config, bus=bus)

    def start(self) -> None:
        if self._running:
            return
        self.tier1.load()
        if self.tier2 is not None:
            self.tier2.load()

        self.bus.start()
        self.capture.start()
        self._stop.clear()
        self._running = True
        self._started_at = time.monotonic()

        self._spawn(self._tier1_loop, "tier1")
        if self.tier2 is not None:
            self._spawn(self._tier2_loop, "tier2")

        logger.info("pipeline: running (tier1=%s tier2=%s/%s motion_gate=%s)",
                    self.tier1.name,
                    self.tier2.name if self.tier2 else "-",
                    self.tier2_mode,
                    "on" if self.motion_gate.enabled else "off")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()
        self.capture.stop(timeout=timeout)
        self.bus.stop(timeout=timeout)
        self._running = False
        logger.info("pipeline: stopped (%d events emitted)", self.events_emitted)

    def __enter__(self) -> "EdgePipeline":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def _spawn(self, target, name: str) -> None:
        thread = threading.Thread(target=target, name=f"pipeline-{name}", daemon=True)
        thread.start()
        self._threads.append(thread)

    # ------------------------------------------------------------------ #
    # Tier1
    # ------------------------------------------------------------------ #

    def _tier1_loop(self) -> None:
        last_reconnects = self.capture.health()["reconnects"]
        while not self._stop.is_set():
            frame = self.capture.next_for_tier1(timeout=0.25)
            if frame is None:
                continue

            # A reconnect means a different scene — auto-exposure resets, the
            # camera may have been nudged. The old background is worthless.
            reconnects = self.capture.health()["reconnects"]
            if reconnects != last_reconnects:
                self.motion_gate.reset()
                last_reconnects = reconnects

            try:
                self.process_frame(frame)
            except Exception:
                logger.exception("pipeline: tier1 stage failed on %s", frame.frame_id)

    def process_frame(self, frame) -> int:
        """One frame through gate + Tier1. Returns candidates produced.

        Public and synchronous so tests (and a future single-shot CLI) can drive
        the stage without threads.
        """
        self.frames_examined += 1

        motion = self.motion_gate.evaluate(frame.data)
        if not motion.motion:
            self.motion_suppressed += 1
            return 0

        candidates = self.tier1.detect(frame.data)
        if not candidates:
            return 0

        for candidate in candidates:
            self.candidates += 1
            candidate = _with_motion_flag(candidate, self.motion_gate.enabled)
            if self.pipeline_config["publish_candidates"]:
                self.bus.publish(TOPIC_CANDIDATE,
                                 {"frame_id": frame.frame_id, **candidate.to_dict()})
            self._handoff(frame, candidate)
        return len(candidates)

    def _handoff(self, frame, candidate) -> None:
        """Tier2 dispatch, or straight to scoring when Tier2 isn't running here."""
        if self.tier2_mode != "local":
            status = "deferred" if self.tier2_mode == "defer_to_cloud" else "disabled"
            self._emit(frame_id=frame.frame_id,
                       snapshot_source=(frame.data, candidate.bbox),
                       captured_at=frame.wall_clock,
                       media_timestamp=frame.media_timestamp,
                       candidate=candidate,
                       verification=None,
                       tier2_status=status,
                       frame_width=frame.width,
                       frame_height=frame.height)
            return

        # Canonical path: pull the original back out of the buffer by id. This
        # is the retrieval path Tier2 is architecturally supposed to use, and
        # exercising it here keeps it honest.
        #
        # It CAN miss. Cropping at enqueue time removes the dependency on Tier2
        # latency, but not on Tier1's own: capture -> tier1 queue -> thread
        # wake -> inference is still time in which a very small buffer can roll
        # past. When that happens the frame is not lost — Tier1 is holding a
        # strong reference to the identical CapturedFrame, so we crop from that
        # and count the miss. A rising `buffer_misses` means the buffer is
        # undersized for this node's capture rate, which is worth surfacing
        # rather than silently absorbing.
        try:
            original = self.capture.get_frame(frame.frame_id)
        except FrameRetrievalError as exc:
            self.buffer_misses += 1
            logger.debug("pipeline: buffer rolled past %s (%s) — cropping from the "
                         "frame Tier1 still holds", frame.frame_id, exc)
            original = frame

        crop, original_size = crop_for_tier2(
            original.data,
            candidate.bbox,
            target_size=self.tier2.input_size,
            padding_fraction=float(self.pipeline_config["crop_padding_fraction"]),
        )

        task = Tier2Task(
            frame_id=frame.frame_id,
            candidate=candidate,
            crop=crop,
            original_crop_size=original_size,
            frame_width=original.width,
            frame_height=original.height,
            captured_wall_clock=original.wall_clock,
            media_timestamp=original.media_timestamp,
            enqueued_at=time.monotonic(),
        )
        self.tier2_queue.put(task)

    # ------------------------------------------------------------------ #
    # Tier2
    # ------------------------------------------------------------------ #

    def _tier2_loop(self) -> None:
        while not self._stop.is_set():
            task = self.tier2_queue.get(timeout=0.25)
            if task is None:
                continue
            try:
                self.process_task(task)
            except Exception:
                logger.exception("pipeline: tier2 stage failed on %s", task.frame_id)

    def process_task(self, task: Tier2Task) -> DetectionEvent:
        """Verify a queued crop, score it, emit the event."""
        verification = None
        status = "local"
        try:
            verification = self.tier2.verify(task.crop, task.candidate)
        except Tier2Error as exc:
            # A Tier2 failure must not swallow the detection — Tier1 already
            # saw something. Ship it with the status that says so.
            self.tier2_failures += 1
            status = "failed"
            logger.warning("pipeline: tier2 failed on %s: %s", task.frame_id, exc)

        return self._emit(
            frame_id=task.frame_id,
            captured_at=task.captured_wall_clock,
            media_timestamp=task.media_timestamp,
            candidate=task.candidate,
            verification=verification,
            tier2_status=status,
            frame_width=task.frame_width,
            frame_height=task.frame_height,
            queue_latency_ms=round((time.monotonic() - task.enqueued_at) * 1000, 2),
            snapshot_crop=task.crop,
        )

    # ------------------------------------------------------------------ #
    # Scoring + emission
    # ------------------------------------------------------------------ #

    def _emit(
        self,
        *,
        frame_id: str,
        captured_at: float,
        media_timestamp: Optional[float],
        candidate,
        verification,
        tier2_status: str,
        frame_width: int,
        frame_height: int,
        queue_latency_ms: Optional[float] = None,
        snapshot_crop=None,
        snapshot_source=None,
    ) -> DetectionEvent:
        category = verification.category if verification else candidate.category

        # count() BEFORE record() — persistence measures prior sightings, not
        # this one, or every first detection would score as if it recurred.
        recent = self.persistence.count(category, self.zone, now=captured_at)
        self.persistence.record(category, self.zone, now=captured_at)

        context = ScoringContext(
            zone=self.zone,
            wall_clock=captured_at,
            utc_offset_hours=float(self.pipeline_config["utc_offset_hours"]),
            bbox_area=candidate.bbox.area,
            recent_similar=recent,
            node_id=self.node_id,
        )
        risk = self.scorer.score(candidate, verification, context)

        # Pixels stay on disk; the event carries only a reference. Written after
        # scoring so `min_band` can suppress the write for low-risk detections.
        snapshot_ref = None
        if self.snapshots.should_write(risk.band):
            if snapshot_crop is None and snapshot_source is not None:
                from .inference import crop_for_tier2
                snapshot_crop, _ = crop_for_tier2(
                    snapshot_source[0], snapshot_source[1],
                    target_size=260,
                    padding_fraction=float(self.pipeline_config["crop_padding_fraction"]))
            snapshot_ref = self.snapshots.write(frame_id, crop=snapshot_crop)

        event = DetectionEvent.build(
            node_id=self.node_id,
            frame_id=frame_id,
            captured_at=captured_at,
            candidate=candidate,
            verification=verification,
            tier2_status=tier2_status,
            risk=risk,
            frame_width=frame_width,
            frame_height=frame_height,
            site_id=self.site_id,
            zone=self.zone,
            media_timestamp=media_timestamp,
            snapshot_ref=snapshot_ref,
            pipeline={
                "tier2_mode": self.tier2_mode,
                "queue_latency_ms": queue_latency_ms,
                "motion_gate": self.motion_gate.enabled,
            },
        )

        if self.pipeline_config["validate_events"]:
            # Fail on the node, not after a six-hour offline queue.
            event.validate()

        self.events_emitted += 1
        self.bus.publish(TOPIC_SCORED, event)
        return event

    # ------------------------------------------------------------------ #

    def wait_until_complete(self, timeout: Optional[float] = None,
                            drain_grace: float = 2.0) -> bool:
        """Block until a finite source is exhausted AND the Tier2 queue drains.

        Only meaningful for the video driver — a live camera never ends. Waiting
        on capture alone is not enough: Tier2 work is still queued behind it, and
        returning early would silently truncate the last detections of a clip.
        """
        if not self.capture.wait_for_end(timeout):
            return False
        deadline = time.monotonic() + drain_grace
        while time.monotonic() < deadline:
            if self.tier2_queue.qsize() == 0:
                time.sleep(0.05)          # let the in-flight task finish emitting
                if self.tier2_queue.qsize() == 0:
                    return True
            time.sleep(0.02)
        return True

    def stats(self) -> Dict[str, Any]:
        uptime = None if self._started_at is None else round(
            time.monotonic() - self._started_at, 1)
        return {
            "node_id": self.node_id,
            "running": self._running,
            "uptime_s": uptime,
            "frames_examined": self.frames_examined,
            "motion_suppressed": self.motion_suppressed,
            "candidates": self.candidates,
            "crops_failed": self.crops_failed,
            "buffer_misses": self.buffer_misses,
            "events_emitted": self.events_emitted,
            "tier2_mode": self.tier2_mode,
            "tier2_failures": self.tier2_failures,
            "motion_gate": self.motion_gate.stats(),
            "snapshots": self.snapshots.stats(),
            "tier1": self.tier1.stats(),
            "tier2": self.tier2.stats() if self.tier2 else None,
            "tier2_queue": self.tier2_queue.stats(),
            "capture": self.capture.health(),
            "bus": self.bus.stats() if hasattr(self.bus, "stats") else None,
        }


def _with_motion_flag(candidate, gated: bool):
    from dataclasses import replace
    return replace(candidate, motion_gated=gated)