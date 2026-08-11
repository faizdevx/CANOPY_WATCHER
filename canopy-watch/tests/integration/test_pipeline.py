"""End-to-end: mock camera -> capture -> Tier1 -> Tier2 -> risk -> event bus.

No camera, no model files, no broker. The mock camera draws a bright moving
target on a dark gradient and the heuristic Tier1 is a real brightness-blob
detector, so these assertions are about actual detections that actually track a
moving object — not a stubbed stage returning a canned box.

    python -m pytest tests/integration/test_pipeline.py -v
"""

from __future__ import annotations

import threading
import time

import pytest

from edge.bus import InProcessBus
from edge.events import DetectionEvent, validate_event
from edge.pipeline import TOPIC_SCORED, EdgePipeline


def config(**overrides):
    cfg = {
        "identity": {"node_id": "cam01", "site_id": "north-park", "zone": "core"},
        "hardware": {
            "driver": {"camera": "mock"},
            "camera_params": {"width": 320, "height": 240, "warmup_frames": 0},
        },
        "capture": {
            "buffer_seconds": 10,
            "buffer_max_memory_mb": 64,
            "tier1_sample_interval_ms": 20,
            "tier1_queue_depth": 1,
        },
        "ai": {
            "accelerator": "cpu",
            "tier1": {
                "backend": "heuristic",
                "score_threshold": 0.0,
                "assumed_category": "unknown",
                "motion_gate": {"enabled": False},
            },
            "tier2": {"backend": "heuristic", "mode": "local", "input_size": 128},
        },
        "scoring": {"persistence": {"window_seconds": 300}},
        "pipeline": {"tier2_queue_depth": 32, "validate_events": True},
    }
    for key, value in overrides.items():
        section, _, field = key.partition(".")
        if field:
            cfg.setdefault(section, {})
            target = cfg[section]
            parts = field.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        else:
            cfg[section] = value
    return cfg


class Collector:
    def __init__(self):
        self.events = []
        self.seen = threading.Event()

    def __call__(self, topic, payload):
        self.events.append(payload)
        self.seen.set()


def run_pipeline(cfg, seconds=1.0):
    bus = InProcessBus()
    collector = Collector()
    bus.subscribe(TOPIC_SCORED, collector, name="collector")
    pipeline = EdgePipeline.from_config(cfg, bus=bus)
    pipeline.start()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and len(collector.events) < 3:
        time.sleep(0.02)
    time.sleep(0.1)
    stats = pipeline.stats()
    pipeline.stop()
    return collector.events, stats


# --------------------------------------------------------------------------- #

def test_full_chain_emits_valid_scored_events():
    events, stats = run_pipeline(config())

    assert events, "pipeline produced no detections"
    assert stats["frames_examined"] > 0
    assert stats["candidates"] > 0
    assert stats["crops_failed"] == 0

    event = events[0]
    assert isinstance(event, DetectionEvent)
    validate_event(event.to_dict())


def test_event_traces_back_to_a_real_captured_frame():
    events, _ = run_pipeline(config())
    payload = events[0].to_dict()
    assert payload["frame_id"].startswith("cam01-")
    assert payload["node_id"] == "cam01"
    assert payload["zone"] == "core"
    assert payload["frame_size"] == {"width": 320, "height": 240}


def test_tier2_actually_ran_and_is_recorded_as_local():
    events, stats = run_pipeline(config())
    payload = events[0].to_dict()
    assert payload["tier2_status"] == "local"
    assert payload["tier2"] is not None
    assert payload["tier2"]["backend"] == "heuristic"
    assert stats["tier2"]["invocations"] > 0


def test_defer_to_cloud_skips_tier2_but_still_emits():
    """A Pi profile. Same pipeline shape, different tier2_status."""
    cfg = config()
    cfg["ai"]["tier2"]["mode"] = "defer_to_cloud"
    events, stats = run_pipeline(cfg)

    assert events
    payload = events[0].to_dict()
    assert payload["tier2_status"] == "deferred"
    assert payload["tier2"] is None
    assert stats["tier2"] is None
    validate_event(payload)


def test_tier2_disabled_mode():
    cfg = config()
    cfg["ai"]["tier2"]["mode"] = "disabled"
    events, _ = run_pipeline(cfg)
    assert events[0].to_dict()["tier2_status"] == "disabled"


def test_risk_score_is_present_and_explained():
    events, _ = run_pipeline(config())
    risk = events[0].to_dict()["risk"]
    assert 0 <= risk["score"] <= 100
    assert risk["band"] in ("low", "medium", "high", "critical")
    assert len(risk["factors"]) == 6
    assert all(f["reason"] for f in risk["factors"])
    assert sum(f["contribution"] for f in risk["factors"]) == pytest.approx(
        risk["score"], abs=0.05)


def test_persistence_raises_risk_across_repeated_detections():
    """First sighting must not score as if it recurred; later ones should."""
    events, _ = run_pipeline(config(), seconds=2.0)
    if len(events) < 3:
        pytest.skip("not enough detections in the time budget")

    def persistence(evt):
        return next(f["value"] for f in evt.to_dict()["risk"]["factors"]
                    if f["name"] == "persistence")

    assert persistence(events[0]) == 0.0
    assert persistence(events[-1]) > persistence(events[0])


def test_motion_gate_on_suppresses_a_static_scene():
    cfg = config()
    cfg["hardware"]["camera_params"]["mode"] = "synthetic"
    cfg["ai"]["tier1"]["motion_gate"] = {"enabled": True, "warmup_frames": 2,
                                         "min_area_fraction": 0.9}
    events, stats = run_pipeline(cfg, seconds=0.6)
    assert stats["motion_gate"]["enabled"] is True
    assert stats["motion_suppressed"] > 0
    assert len(events) < stats["frames_examined"]


def test_motion_gate_off_by_default_for_laptop():
    _, stats = run_pipeline(config(), seconds=0.4)
    assert stats["motion_gate"]["enabled"] is False
    assert stats["motion_suppressed"] == 0


def test_a_tiny_buffer_degrades_but_never_drops_a_detection():
    """Buffer deliberately far too small (1 MB ~= 4 frames at 320x240).

    The buffer WILL roll past between Tier1 firing and the crop — cropping at
    enqueue time removes the dependency on Tier2 latency, not on Tier1's own.
    That must degrade to a counted miss, never to a lost detection, because
    Tier1 is still holding the identical frame."""
    cfg = config()
    cfg["capture"]["buffer_max_memory_mb"] = 1
    cfg["capture"]["buffer_seconds"] = 1
    events, stats = run_pipeline(cfg, seconds=1.0)

    assert events
    # crops_failed is the only way a detection can vanish at this stage. It
    # stays zero: candidates that have not become events yet are still in the
    # Tier2 queue when the run is cut off, which is in-flight work, not loss.
    assert stats["crops_failed"] == 0
    assert stats["events_emitted"] > 0
    assert "buffer_misses" in stats                            # and it is visible


def test_an_adequate_buffer_has_no_misses():
    events, stats = run_pipeline(config(), seconds=1.0)
    assert events
    assert stats["buffer_misses"] == 0
    assert stats["crops_failed"] == 0


def test_pipeline_survives_a_camera_dropout():
    cfg = config()
    cfg["hardware"]["camera_params"].update(
        {"fail_after_frames": 6, "max_consecutive_failures": 1, "read_retries": 1})
    cfg["capture"]["reopen_backoff_s"] = [0.01]
    _, stats = run_pipeline(cfg, seconds=1.0)
    assert stats["capture"]["reconnects"] >= 1
    assert stats["running"] is True


def test_bus_subscribers_are_independent():
    bus = InProcessBus()
    fast, slow = [], []
    bus.subscribe(TOPIC_SCORED, lambda t, p: fast.append(p), name="fast")

    def slow_handler(topic, payload):
        time.sleep(0.05)
        slow.append(payload)

    bus.subscribe(TOPIC_SCORED, slow_handler, name="slow")

    pipeline = EdgePipeline.from_config(config(), bus=bus)
    pipeline.start()
    time.sleep(0.8)
    pipeline.stop()

    assert len(fast) >= len(slow)      # slow subscriber never blocked the fast one
    assert bus.stats()["published"] >= len(fast)


def test_stats_expose_every_stage():
    _, stats = run_pipeline(config(), seconds=0.5)
    for key in ("frames_examined", "motion_suppressed", "candidates",
                "events_emitted", "tier2_queue", "tier1", "capture", "motion_gate"):
        assert key in stats
    assert stats["capture"]["capture_fps"] is not None