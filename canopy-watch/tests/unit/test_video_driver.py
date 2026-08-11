"""Video file driver + full pipeline over a clip.

Clips are generated on the fly by the same code as `scripts/make_test_video.py`,
so these run in CI with no fixture checked in and no real footage needed.

    python -m pytest tests/unit/test_video_driver.py -v
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from edge.bus import InProcessBus                      # noqa: E402
from edge.capture import CaptureService                # noqa: E402
from edge.drivers import (                             # noqa: E402
    CameraEndOfStream,
    CameraOpenError,
    available_drivers,
    create_camera_driver,
)
from edge.events import validate_event                 # noqa: E402
from edge.pipeline import TOPIC_SCORED, EdgePipeline   # noqa: E402


# --------------------------------------------------------------------------- #
# Clip generation
# --------------------------------------------------------------------------- #

def make_clip(path: Path, seconds=4.0, fps=15, width=320, height=240,
              empty=False, seed=7) -> Path:
    total = int(seconds * fps)
    enter, leave = int(total * 0.2), int(total * 0.8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    assert writer.isOpened(), "this OpenCV build cannot write mp4v"
    rng = np.random.default_rng(seed)
    size = max(20, min(width, height) // 6)

    for tick in range(total):
        ramp = np.linspace(18, 70, height, dtype=np.float32)[:, None]
        frame = np.zeros((height, width, 3), dtype=np.float32)
        frame[:, :, 1] = ramp
        frame[:, :, 0] = ramp / 3.0
        frame[:, :, 2] = ramp / 4.0
        frame += rng.integers(0, 5, size=(height, width, 3))
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        if not empty and enter <= tick < leave:
            progress = (tick - enter) / max(1, leave - enter)
            x = int(progress * (width - size))
            y = max(0, min(height - size, int(height * 0.45)))
            frame[y:y + size, x:x + size] = (245, 200, 90)

        writer.write(frame[:, :, ::-1])
    writer.release()
    return path


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    return make_clip(tmp_path_factory.mktemp("video") / "clip.mp4")


@pytest.fixture(scope="module")
def empty_clip(tmp_path_factory) -> Path:
    return make_clip(tmp_path_factory.mktemp("video") / "empty.mp4", empty=True)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def test_video_driver_is_registered():
    assert "video" in available_drivers()


def test_missing_file_fails_with_a_clear_message():
    driver = create_camera_driver(driver="video", params={"path": "/nope/x.mp4"})
    with pytest.raises(CameraOpenError) as err:
        driver.open()
    assert "not found" in str(err.value)


def test_missing_path_param_is_actionable():
    with pytest.raises(CameraOpenError) as err:
        create_camera_driver(driver="video", params={}).open()
    assert "path" in str(err.value)


def test_frames_are_canonical_rgb(clip):
    with create_camera_driver(driver="video", params={"path": str(clip)}) as driver:
        frame = driver.read_frame()
    assert frame.data.dtype == np.uint8
    assert frame.data.shape == (240, 320, 3)
    assert frame.source == "video"


def test_media_timestamp_is_real_for_a_file(clip):
    """POS_MSEC is meaningless for live capture but genuine for a file — this is
    the third media-timestamp source outside the CSI drivers."""
    with create_camera_driver(driver="video", params={"path": str(clip)}) as driver:
        stamps = [driver.read_frame().metadata["media_timestamp"] for _ in range(5)]
    assert all(s is not None for s in stamps)
    assert stamps == sorted(stamps)
    assert stamps[-1] > stamps[0]


def test_end_of_stream_is_raised_not_a_read_error(clip):
    """MUST be its own type. Wrapping it in CameraReadError makes the Capture
    Service treat a finished file as a dropout and reopen it forever."""
    driver = create_camera_driver(driver="video", params={"path": str(clip)})
    driver.open()
    count = 0
    with pytest.raises(CameraEndOfStream):
        while count < 1000:
            driver.read_frame()
            count += 1
    assert 50 < count < 70          # ~60 frames in a 4s @ 15fps clip
    driver.close()


def test_loop_mode_never_ends(clip):
    driver = create_camera_driver(driver="video",
                                  params={"path": str(clip), "loop": True})
    driver.open()
    for _ in range(150):            # more than the clip contains
        driver.read_frame()
    assert driver.loops_completed >= 1
    driver.close()


def test_frame_stride_subsamples(clip):
    def count(stride):
        driver = create_camera_driver(driver="video",
                                      params={"path": str(clip), "frame_stride": stride})
        driver.open()
        n = 0
        try:
            while n < 500:
                driver.read_frame()
                n += 1
        except CameraEndOfStream:
            pass
        driver.close()
        return n

    assert count(3) < count(1) / 2


def test_start_and_end_frame_bound_the_range(clip):
    driver = create_camera_driver(
        driver="video",
        params={"path": str(clip), "start_frame": 10, "end_frame": 25})
    driver.open()
    n = 0
    try:
        while n < 500:
            driver.read_frame()
            n += 1
    except CameraEndOfStream:
        pass
    driver.close()
    assert 10 <= n <= 16


def test_resize_on_read(clip):
    with create_camera_driver(driver="video",
                              params={"path": str(clip), "width": 160,
                                      "height": 120}) as driver:
        assert driver.read_frame().data.shape == (120, 160, 3)


def test_capabilities_report_the_source(clip):
    with create_camera_driver(driver="video", params={"path": str(clip)}) as driver:
        caps = driver.get_capabilities()
    assert caps["total_frames"] > 0
    assert caps["fps"] == pytest.approx(15, abs=1)
    assert caps["media_timestamp_available"] is True
    assert caps["finite"] is True


def test_realtime_pacing_takes_roughly_the_clip_duration(clip):
    driver = create_camera_driver(
        driver="video",
        params={"path": str(clip), "realtime": True, "speed": 4.0})
    driver.open()
    started = time.monotonic()
    try:
        for _ in range(500):
            driver.read_frame()
    except CameraEndOfStream:
        pass
    elapsed = time.monotonic() - started
    driver.close()
    assert 0.4 < elapsed < 2.5      # 4s clip at 4x, generously bounded for CI


# --------------------------------------------------------------------------- #
# Capture service over a finite source
# --------------------------------------------------------------------------- #

def capture_config(clip: Path, **capture):
    return {
        "identity": {"node_id": "vid01", "zone": "core"},
        "hardware": {"driver": {"camera": "video"},
                     "camera_params": {"path": str(clip)}},
        "capture": {"buffer_seconds": 10, "buffer_max_memory_mb": 64,
                    "tier1_sample_interval_ms": 200, **capture},
    }


def test_capture_ends_cleanly_instead_of_reconnecting(clip):
    """The bug this guards: EOS wrapped as a read error makes capture decide the
    camera is unhealthy, reconnect, reopen the same file, and never terminate."""
    service = CaptureService.from_config(capture_config(clip))
    service.start()
    assert service.wait_for_end(timeout=20), "capture never reached end of stream"
    health = service.health()
    service.stop()

    assert health["ended"] is True
    assert health["reconnects"] == 0
    assert "end of" in (health["end_reason"] or "")
    assert health["frames_captured"] > 40


def test_media_clock_sampling_covers_the_whole_clip(clip):
    """Wall-clock sampling under-samples a fast replay: an 8s clip finishing in
    1.4s of wall time yields 7 samples instead of 40."""
    media = CaptureService.from_config(
        capture_config(clip, tier1_sample_clock="media"))
    media.start()
    media.wait_for_end(timeout=20)
    media_selected = media.sampler.stats()["selected"]
    media.stop()

    wall = CaptureService.from_config(capture_config(clip, tier1_sample_clock="wall"))
    wall.start()
    wall.wait_for_end(timeout=20)
    wall_selected = wall.sampler.stats()["selected"]
    wall.stop()

    assert media_selected > wall_selected
    assert media_selected >= 15          # 4s of clip at 200ms intervals


def test_auto_clock_falls_back_for_a_live_driver():
    cfg = {
        "identity": {"node_id": "cam01"},
        "hardware": {"driver": {"camera": "mock"},
                     "camera_params": {"width": 64, "height": 48}},
        "capture": {"tier1_sample_clock": "auto"},
    }
    service = CaptureService.from_config(cfg)
    service.start()
    time.sleep(0.2)
    stats = service.health()["sampler"]
    service.stop()
    assert stats["using_media_time"] is False


# --------------------------------------------------------------------------- #
# Full pipeline over a clip
# --------------------------------------------------------------------------- #

def pipeline_config(clip: Path, snapshots=None, **overrides):
    cfg = {
        "identity": {"node_id": "vid01", "site_id": "test", "zone": "core"},
        "hardware": {"driver": {"camera": "video"},
                     "camera_params": {"path": str(clip)}},
        "capture": {"buffer_seconds": 10, "buffer_max_memory_mb": 64,
                    "tier1_sample_interval_ms": 200,
                    "tier1_sample_clock": "media",
                    "tier1_backpressure": "block"},
        "ai": {"accelerator": "cpu",
               "tier1": {"backend": "heuristic", "score_threshold": 0.3,
                         "motion_gate": {"enabled": False}},
               "tier2": {"backend": "heuristic", "mode": "local", "input_size": 128}},
        "scoring": {},
        "snapshots": {"enabled": bool(snapshots), "directory": str(snapshots or "."),
                      "min_band": "low"},
        "pipeline": {"tier2_queue_depth": 32, "validate_events": True},
    }
    cfg.update(overrides)
    return cfg


def run_clip(cfg):
    bus = InProcessBus()
    events = []
    bus.subscribe(TOPIC_SCORED, lambda t, e: events.append(e), name="collect")
    pipeline = EdgePipeline.from_config(cfg, bus=bus)
    pipeline.start()
    completed = pipeline.wait_until_complete(timeout=30)
    stats = pipeline.stats()
    pipeline.stop()
    assert completed, "pipeline did not finish the clip"
    return events, stats


def test_pipeline_processes_a_clip_to_completion(clip):
    events, stats = run_clip(pipeline_config(clip))
    assert events
    assert stats["capture"]["ended"] is True
    assert stats["capture"]["reconnects"] == 0
    for event in events:
        validate_event(event.to_dict())


def test_detections_land_where_the_subject_actually_is(clip):
    """The subject enters at 20% and leaves at 80% of a 4s clip, so detections
    should cluster in roughly 0.8s-3.2s of MEDIA time — not wall time."""
    events, _ = run_clip(pipeline_config(clip))
    stamps = [e.to_dict()["media_timestamp"] for e in events]
    assert all(s is not None for s in stamps)
    assert min(stamps) >= 0.6
    assert max(stamps) <= 3.4


def test_empty_clip_produces_no_false_positives(empty_clip):
    """The negative control. A pipeline reporting detections on subject-free
    background is telling you its false-positive rate."""
    events, stats = run_clip(pipeline_config(empty_clip))
    assert events == []
    assert stats["candidates"] == 0
    assert stats["frames_examined"] > 10        # it did look at the footage


def test_drop_oldest_is_lossy_on_a_file_which_is_why_block_exists(clip):
    cfg = pipeline_config(clip)
    cfg["capture"]["tier1_backpressure"] = "drop_oldest"
    _, dropping = run_clip(cfg)
    _, blocking = run_clip(pipeline_config(clip))

    assert blocking["capture"]["tier1"]["policy"] == "block"
    assert blocking["capture"]["tier1"]["frames_dropped_backpressure"] == 0
    assert (blocking["frames_examined"]
            >= dropping["frames_examined"])


def test_bbox_tracks_the_subject_across_the_clip(clip):
    events, _ = run_clip(pipeline_config(clip))
    xs = [e.to_dict()["bbox"]["x1"] for e in events]
    assert len(xs) >= 3
    assert xs[-1] > xs[0]                        # subject moves left to right


def test_snapshots_are_written_and_referenced(clip, tmp_path):
    out = tmp_path / "snaps"
    events, stats = run_clip(pipeline_config(clip, snapshots=out))
    assert events
    refs = [e.to_dict()["snapshot_ref"] for e in events]
    assert all(r for r in refs)
    assert all(Path(r).exists() for r in refs)
    assert stats["snapshots"]["written"] == len(events)


def test_snapshot_min_band_suppresses_low_risk_writes(clip, tmp_path):
    cfg = pipeline_config(clip, snapshots=tmp_path / "s")
    cfg["snapshots"]["min_band"] = "critical"
    events, stats = run_clip(cfg)
    assert events
    assert stats["snapshots"]["written"] < len(events)


def test_snapshots_are_bounded_on_disk(clip, tmp_path):
    cfg = pipeline_config(clip, snapshots=tmp_path / "s")
    cfg["snapshots"]["max_files"] = 3
    events, _ = run_clip(cfg)
    assert len(events) > 3
    assert len(list((tmp_path / "s").glob("*.jpg"))) <= 3


def test_deferred_mode_over_a_clip(empty_clip, clip):
    cfg = pipeline_config(clip)
    cfg["ai"]["tier2"]["mode"] = "defer_to_cloud"
    events, _ = run_clip(cfg)
    assert events
    payload = events[0].to_dict()
    assert payload["tier2_status"] == "deferred"
    assert payload["tier2"] is None


def test_same_clip_gives_identical_results_twice(clip):
    """Repeatability is the whole reason to evaluate on a file rather than a
    camera pointed at a garden — a threshold change must be attributable to the
    threshold, not to how busy the machine was.

    Requires `tier1_backpressure: block`. With the live-camera default
    (drop_oldest) this test fails, because WHICH sampled frame Tier1 sees
    depends on thread scheduling."""
    def signature(events):
        return [(e.to_dict()["media_timestamp"],
                 round(e.to_dict()["confidence"], 4),
                 e.to_dict()["bbox"]["x1"]) for e in events]

    first, _ = run_clip(pipeline_config(clip))
    second, _ = run_clip(pipeline_config(clip))
    assert signature(first) == signature(second)