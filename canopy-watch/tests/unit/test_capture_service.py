"""Capture Service contract tests — all seven responsibilities, no hardware.

    python -m pytest tests/unit/test_capture_service.py -v
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from edge.capture import (
    CaptureError,
    CaptureService,
    CapturedFrame,
    FrameBuffer,
    FrameExpired,
    FrameNotFound,
    FrameNotYet,
    LatestFrameQueue,
    TimeBasedSampler,
    make_frame_id,
    parse_frame_id,
)
from edge.drivers.mock.camera import MockCameraDriver


def build(driver_params=None, capture_config=None, source_id="cam01"):
    driver = MockCameraDriver({"width": 64, "height": 48, **(driver_params or {})})
    return CaptureService(driver, source_id, capture_config or {})


def make_captured(seq, source="cam01", nbytes_shape=(4, 4, 3), received=None):
    return CapturedFrame(
        frame_id=make_frame_id(source, seq),
        source_id=source,
        sequence=seq,
        data=np.zeros(nbytes_shape, dtype=np.uint8),
        received_timestamp=time.monotonic() if received is None else received,
        wall_clock=time.time(),
    )


# --------------------------------------------------------------------------- #
# 1. Frame IDs
# --------------------------------------------------------------------------- #

def test_frame_id_format_and_roundtrip():
    fid = make_frame_id("cam01", 184392)
    assert fid == "cam01-000184392"
    assert parse_frame_id(fid) == ("cam01", 184392)


def test_parse_rejects_non_ids():
    assert parse_frame_id("not-an-id") is None
    assert parse_frame_id("cam01-42") is None      # unpadded
    assert parse_frame_id(None) is None


def test_ids_are_sequential_and_unique():
    svc = build()
    svc.start()
    time.sleep(0.15)
    svc.stop()
    seqs = [f.sequence for f in svc.buffer._frames.values()]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


def test_sequence_survives_camera_reconnect():
    """The driver's own frame_index resets on reopen; capture's must not, or
    two different frames in one deployment could share an id."""
    svc = build()
    svc.driver.open()
    for _ in range(3):
        svc._ingest(svc.driver.read_frame())
    before = svc._sequence

    svc._reconnect()
    svc._ingest(svc.driver.read_frame())

    # Capture's sequence is continuous across the dropout — the driver's own
    # frame_index is session/debug metadata and is not what ids are built from.
    assert svc._sequence == before + 1
    assert svc.buffer.stats()["highest_sequence"] == before + 1
    svc.driver.close()


def test_buffer_is_not_cleared_on_reconnect():
    svc = build()
    svc.driver.open()
    first = svc._ingest(svc.driver.read_frame())
    svc._reconnect()
    assert svc.get_frame(first.frame_id).frame_id == first.frame_id
    svc.driver.close()


# --------------------------------------------------------------------------- #
# 2. Timestamps
# --------------------------------------------------------------------------- #

def test_all_three_timestamps_present_or_explicitly_absent():
    svc = build()
    svc.driver.open()
    frame = svc._ingest(svc.driver.read_frame())
    svc.driver.close()

    assert isinstance(frame.received_timestamp, float)   # monotonic
    assert frame.wall_clock > 1_600_000_000              # real epoch seconds
    assert frame.media_timestamp is None                 # mock has no sensor


def test_media_timestamp_is_never_backfilled():
    """Absent is debuggable; fabricated looks correct until it isn't."""
    svc = build()
    svc.driver.open()
    frame = svc._ingest(svc.driver.read_frame())
    svc.driver.close()
    assert frame.media_timestamp is None
    assert frame.media_timestamp != frame.received_timestamp


def test_media_timestamp_is_passed_through_when_the_driver_has_one():
    class SensorCam(MockCameraDriver):
        def _frame_metadata(self):
            meta = super()._frame_metadata()
            meta["media_timestamp"] = 172345.533
            return meta

    svc = CaptureService(SensorCam({"width": 32, "height": 24}), "cam01")
    svc.driver.open()
    frame = svc._ingest(svc.driver.read_frame())
    svc.driver.close()
    assert frame.media_timestamp == pytest.approx(172345.533)


def test_received_timestamps_are_monotonic():
    svc = build()
    svc.start()
    time.sleep(0.1)
    svc.stop()
    stamps = [f.received_timestamp for f in svc.buffer._frames.values()]
    assert stamps == sorted(stamps)


# --------------------------------------------------------------------------- #
# 3. Rolling buffer
# --------------------------------------------------------------------------- #

def test_memory_cap_wins_over_requested_seconds():
    """buffer_seconds alone is a lie on a 4 GB Jetson — bytes decide."""
    per_frame = 4 * 4 * 3
    buf = FrameBuffer("cam01", max_seconds=3600, max_memory_bytes=per_frame * 5)
    for seq in range(1, 21):
        buf.append(make_captured(seq))
    assert len(buf) <= 5
    assert buf.stats()["bytes"] <= per_frame * 5
    assert buf.stats()["evicted_total"] >= 15


def test_age_cap_evicts_even_when_memory_is_free():
    buf = FrameBuffer("cam01", max_seconds=0.05, max_memory_bytes=10 ** 9)
    now = time.monotonic()
    buf.append(make_captured(1, received=now))
    buf.append(make_captured(2, received=now + 1.0))
    assert len(buf) == 1
    assert buf.stats()["lowest_sequence"] == 2


def test_buffer_never_evicts_the_only_frame_on_size_alone():
    buf = FrameBuffer("cam01", max_seconds=3600, max_memory_bytes=1)
    buf.append(make_captured(1))
    assert len(buf) == 1


def test_buffered_frames_are_read_only():
    """Capture hands out references, not 6 MB copies — so mutation must fail
    loudly rather than corrupt shared history."""
    svc = build()
    svc.driver.open()
    frame = svc._ingest(svc.driver.read_frame())
    svc.driver.close()
    with pytest.raises(ValueError):
        svc.get_frame(frame.frame_id).data[0, 0] = 255


# --------------------------------------------------------------------------- #
# 4. Selection
# --------------------------------------------------------------------------- #

def test_sampling_is_time_based_not_every_nth_frame():
    s = TimeBasedSampler(0.2)
    t = 1000.0
    assert s.should_select(t) is True          # first frame always
    assert s.should_select(t + 0.05) is False
    assert s.should_select(t + 0.19) is False
    assert s.should_select(t + 0.20) is True
    assert s.should_select(t + 0.39) is False
    assert s.should_select(t + 0.40) is True


def test_sampler_resyncs_instead_of_bursting_after_a_stall():
    s = TimeBasedSampler(0.2)
    t = 1000.0
    s.should_select(t)
    assert s.should_select(t + 10.0) is True   # long stall
    assert s.should_select(t + 10.05) is False # no catch-up burst
    assert s.resyncs == 1


def test_zero_interval_selects_every_frame():
    s = TimeBasedSampler(0)
    assert all(s.should_select(1000.0 + i) for i in range(5))


def test_sampling_rate_holds_when_camera_fps_changes():
    """A modulo counter would silently change rate with negotiated fps."""
    s = TimeBasedSampler(0.1)
    slow = sum(s.should_select(1000.0 + i / 15.0) for i in range(15))   # 15 fps, 1s
    s2 = TimeBasedSampler(0.1)
    fast = sum(s2.should_select(1000.0 + i / 60.0) for i in range(60))  # 60 fps, 1s
    assert slow == fast


# --------------------------------------------------------------------------- #
# 5. Tier1 delivery + backpressure
# --------------------------------------------------------------------------- #

def test_queue_drops_oldest_and_counts_it():
    q = LatestFrameQueue(depth=1)
    q.put(make_captured(1))
    q.put(make_captured(2))
    q.put(make_captured(3))
    assert q.stats()["frames_dropped_backpressure"] == 2
    assert q.get(timeout=0).sequence == 3      # latest wins


def test_dropped_frames_remain_retrievable_from_the_buffer():
    """Backpressure loses frames for INFERENCE, never for Tier2 retrieval."""
    svc = build(capture_config={"tier1_sample_interval_ms": 0, "tier1_queue_depth": 1})
    svc.driver.open()
    frames = [svc._ingest(svc.driver.read_frame()) for _ in range(5)]
    svc.driver.close()

    assert svc.tier1_queue.stats()["frames_dropped_backpressure"] == 4
    for f in frames:
        assert svc.get_frame(f.frame_id).sequence == f.sequence


def test_tier1_consumer_loop():
    svc = build(capture_config={"tier1_sample_interval_ms": 10})
    seen = []
    stop = threading.Event()
    svc.start()
    t = threading.Thread(
        target=svc.run_tier1_loop, args=(seen.append, stop, 0.05), daemon=True)
    t.start()
    time.sleep(0.3)
    stop.set()
    t.join(timeout=1)
    svc.stop()
    assert len(seen) >= 2
    assert all(isinstance(f, CapturedFrame) for f in seen)


# --------------------------------------------------------------------------- #
# 6. Exact retrieval — three distinct diagnoses
# --------------------------------------------------------------------------- #

def test_expired_frame_raises_expired_not_a_nearby_frame():
    buf = FrameBuffer("cam01", max_seconds=3600, max_memory_bytes=4 * 4 * 3 * 2)
    for seq in range(1, 11):
        buf.append(make_captured(seq))
    with pytest.raises(FrameExpired):
        buf.get(make_frame_id("cam01", 1))


def test_unknown_source_raises_not_found():
    buf = FrameBuffer("cam01", max_memory_bytes=10 ** 9)
    buf.append(make_captured(1))
    with pytest.raises(FrameNotFound):
        buf.get(make_frame_id("cam99", 1))


def test_malformed_id_raises_not_found():
    buf = FrameBuffer("cam01", max_memory_bytes=10 ** 9)
    buf.append(make_captured(1))
    with pytest.raises(FrameNotFound):
        buf.get("garbage")


def test_future_sequence_raises_not_yet():
    buf = FrameBuffer("cam01", max_memory_bytes=10 ** 9)
    buf.append(make_captured(5))
    with pytest.raises(FrameNotYet):
        buf.get(make_frame_id("cam01", 900))


def test_error_codes_are_explicit_strings():
    buf = FrameBuffer("cam01", max_memory_bytes=10 ** 9)
    with pytest.raises(FrameNotYet) as err:
        buf.get(make_frame_id("cam01", 1))
    assert "FRAME_NOT_YET" in str(err.value)


def test_try_get_frame_returns_none_instead_of_raising():
    svc = build()
    assert svc.try_get_frame(make_frame_id("cam01", 999)) is None


def test_tier1_to_tier2_handoff():
    """The sequence that matters: Tier1 gets a sampled frame, reports a
    frame_id, Tier2 pulls the ORIGINAL back out by that id."""
    svc = build(capture_config={"tier1_sample_interval_ms": 0})
    svc.start()
    time.sleep(0.1)

    tier1_frame = svc.next_for_tier1(timeout=1.0)
    assert tier1_frame is not None

    detection = {"frame_id": tier1_frame.frame_id, "class": "animal",
                 "confidence": 0.94, "bbox": [10, 5, 40, 30]}

    original = svc.get_frame(detection["frame_id"])
    x1, y1, x2, y2 = detection["bbox"]
    crop = original.data[y1:y2, x1:x2]

    assert original.frame_id == tier1_frame.frame_id
    assert crop.shape == (25, 30, 3)
    svc.stop()


# --------------------------------------------------------------------------- #
# 7. Camera loss
# --------------------------------------------------------------------------- #

def test_service_recovers_from_camera_dropout():
    svc = build(driver_params={"fail_after_frames": 3, "max_consecutive_failures": 1,
                               "read_retries": 1},
                capture_config={"reopen_backoff_s": [0.01]})
    svc.start()
    time.sleep(0.4)
    health = svc.health()
    svc.stop()

    assert health["reconnects"] >= 1
    assert health["running"] is True
    assert svc._sequence > 0


def test_first_open_failure_is_raised_not_retried_forever():
    """A camera that was never there on boot is a config problem, not a field
    dropout — surface it instead of silently looping."""
    class DeadCam(MockCameraDriver):
        def _open(self):
            from edge.drivers import CameraOpenError
            raise CameraOpenError("no such device")

    svc = CaptureService(DeadCam(), "cam01", {"reopen_backoff_s": [0.01]})
    with pytest.raises(Exception):
        svc.start()


def test_stale_queue_is_drained_on_reconnect():
    svc = build(capture_config={"tier1_sample_interval_ms": 0, "tier1_queue_depth": 4})
    svc.driver.open()
    for _ in range(3):
        svc._ingest(svc.driver.read_frame())
    assert svc.tier1_queue.qsize() > 0
    svc._reconnect()
    assert svc.tier1_queue.qsize() == 0
    svc.driver.close()


# --------------------------------------------------------------------------- #
# Config wiring + health
# --------------------------------------------------------------------------- #

def test_from_config_uses_node_id_as_source_id():
    cfg = {
        "identity": {"node_id": "rpi-northgate-01"},
        "hardware": {"driver": {"camera": "mock"}, "camera_params": {"width": 32, "height": 24}},
        "capture": {"buffer_seconds": 5, "tier1_sample_interval_ms": 100},
    }
    svc = CaptureService.from_config(cfg)
    assert svc.source_id == "rpi-northgate-01"
    assert svc.buffer.max_seconds == 5
    assert svc.sampler.interval_s == pytest.approx(0.1)


def test_from_config_fails_fast_without_node_id():
    with pytest.raises(CaptureError) as err:
        CaptureService.from_config({"hardware": {"driver": {"camera": "mock"}}})
    assert "node_id" in str(err.value)


def test_capture_defaults_apply_when_capture_section_missing():
    cfg = {"identity": {"node_id": "dev-01"}, "hardware": {"driver": {"camera": "mock"}}}
    svc = CaptureService.from_config(cfg)
    assert svc.buffer.max_seconds == 30.0
    assert svc.tier1_queue.depth == 1


def test_health_reports_measured_fps_not_configured_fps():
    svc = build(driver_params={"fps": 999})
    svc.start()
    time.sleep(0.2)
    health = svc.health()
    svc.stop()

    assert health["capture_fps"] is not None
    assert health["capture_fps"] != 999
    for key in ("buffer", "sampler", "tier1", "camera", "reconnects", "read_errors"):
        assert key in health
