"""Contract tests for the camera driver layer.

These run anywhere — no camera, no OpenCV, no Pi, no Jetson. Everything that
needs real hardware is asserted through the mock driver plus a fake backend, so
CI can prove the contract holds without a single physical device.

    python -m pytest tests/unit/test_camera_drivers.py -v
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from edge.drivers import (
    CameraDriver,
    CameraNotOpenError,
    CameraOpenError,
    CameraReadError,
    Frame,
    UnknownDriverError,
    available_drivers,
    create_camera_driver,
    load_camera_driver_class,
)
from edge.drivers.interfaces.camera import validate_frame_array
from edge.drivers.mock.camera import MockCameraDriver


# --------------------------------------------------------------------------- #
# Canonical output contract — the one non-negotiable rule
# --------------------------------------------------------------------------- #

def test_mock_emits_canonical_rgb_frame():
    with create_camera_driver(driver="mock", params={"width": 320, "height": 240}) as cam:
        frame = cam.read_frame()

    assert isinstance(frame, Frame)
    assert validate_frame_array(frame.data)
    assert frame.data.shape == (240, 320, 3)
    assert frame.data.dtype == np.uint8
    assert frame.source == "mock"
    assert isinstance(frame.timestamp, float)
    assert frame.width == 320 and frame.height == 240


def test_frame_timestamps_are_monotonic():
    with create_camera_driver(driver="mock") as cam:
        stamps = [cam.read_frame().timestamp for _ in range(5)]
    assert stamps == sorted(stamps)


def test_frame_index_increments_and_is_reported():
    with create_camera_driver(driver="mock") as cam:
        indices = [cam.read_frame().metadata["frame_index"] for _ in range(3)]
    assert indices == [0, 1, 2]


def test_synthetic_frames_actually_change():
    """Tier1 is a motion filter — a mock that emits an identical frame forever
    would make every downstream test vacuously pass."""
    with create_camera_driver(driver="mock", params={"width": 160, "height": 120}) as cam:
        a = cam.read_frame().data.copy()
        for _ in range(20):
            b = cam.read_frame().data
        assert not np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #

def test_read_before_open_raises():
    cam = MockCameraDriver()
    with pytest.raises(CameraNotOpenError):
        cam.read_frame()


def test_close_is_idempotent_and_open_is_reentrant():
    cam = MockCameraDriver()
    assert cam.open() is True
    assert cam.open() is True
    cam.close()
    cam.close()
    with pytest.raises(CameraNotOpenError):
        cam.read_frame()


def test_context_manager_closes():
    with MockCameraDriver() as cam:
        cam.read_frame()
        assert cam.is_healthy()
    assert cam.is_healthy() is False


# --------------------------------------------------------------------------- #
# Health is an active check, not a cached boolean
# --------------------------------------------------------------------------- #

def test_unhealthy_before_open():
    assert MockCameraDriver().is_healthy() is False


def test_unhealthy_when_frames_go_stale():
    cam = MockCameraDriver({"stale_frame_seconds": 0.05})
    cam.open()
    cam.read_frame()
    assert cam.is_healthy()
    time.sleep(0.08)
    assert cam.is_healthy() is False
    cam.close()


def test_unhealthy_after_repeated_read_failures():
    cam = MockCameraDriver({
        "fail_after_frames": 1,     # one good frame, then the camera "dies"
        "read_retries": 1,
        "max_consecutive_failures": 2,
    })
    cam.open()
    cam.read_frame()
    assert cam.is_healthy()
    for _ in range(3):
        with pytest.raises(CameraReadError):
            cam.read_frame()
    assert cam.is_healthy() is False
    detail = cam.health_detail()
    assert detail["consecutive_failures"] >= 2
    assert detail["last_error"]
    cam.close()


# --------------------------------------------------------------------------- #
# Warm-up and the "opened but never yields a frame" path (macOS permissions)
# --------------------------------------------------------------------------- #

class _DeadCamera(CameraDriver):
    """Opens successfully, reads nothing — exactly what macOS does when camera
    permission is denied, and what some V4L2 devices do when held elsewhere."""
    name = "dead"
    source_type = "usb"

    def _open(self) -> None:
        return None

    def _read(self):
        raise CameraReadError("no frame")

    def _close(self) -> None:
        return None


class _WarmupCamera(CameraDriver):
    """First 3 frames are black, then real ones."""
    name = "warmup"
    source_type = "usb"

    def __init__(self, params=None):
        super().__init__(params)
        self.reads = 0

    def _open(self) -> None:
        return None

    def _read(self):
        self.reads += 1
        if self.reads <= 3:
            return np.zeros((48, 64, 3), dtype=np.uint8)
        return np.full((48, 64, 3), 128, dtype=np.uint8)

    def _close(self) -> None:
        return None


def test_open_fails_loudly_when_no_frame_ever_arrives():
    with pytest.raises(CameraOpenError) as err:
        _DeadCamera({"open_read_attempts": 5}).open()
    assert "no usable frames" in str(err.value)


def test_open_discards_warmup_frames_before_declaring_ready():
    cam = _WarmupCamera({"warmup_frames": 3, "required_valid_frames": 1})
    assert cam.open() is True
    assert cam.reads == 4          # 3 discarded + 1 accepted
    cam.close()


def test_blank_frames_can_be_rejected_during_open():
    cam = _WarmupCamera({
        "warmup_frames": 0,
        "required_valid_frames": 1,
        "reject_blank_on_open": True,
        "open_read_attempts": 2,   # only the 2 black frames get read
    })
    with pytest.raises(CameraOpenError) as err:
        cam.open()
    assert "blank" in str(err.value)


# --------------------------------------------------------------------------- #
# Non-canonical output is caught at the driver boundary
# --------------------------------------------------------------------------- #

class _BgrFloatCamera(_WarmupCamera):
    name = "bad"

    def _read(self):
        return np.zeros((48, 64, 3), dtype=np.float32)   # wrong dtype


def test_non_canonical_array_is_rejected():
    cam = _BgrFloatCamera({"warmup_frames": 0, "open_read_attempts": 3})
    with pytest.raises(CameraOpenError):
        cam.open()


# --------------------------------------------------------------------------- #
# Factory / registry
# --------------------------------------------------------------------------- #

def test_registry_contains_the_camera_drivers():
    assert available_drivers() == (
        "csi_gstreamer",
        "mock",
        "picamera",
        "video",
        "webcam",
    )


def test_unknown_driver_names_the_valid_options():
    with pytest.raises(UnknownDriverError) as err:
        create_camera_driver(driver="usb3_vision")
    assert "csi_gstreamer" in str(err.value)


def test_factory_reads_nested_config():
    cfg = {
        "hardware": {
            "driver": {"camera": "mock"},
            "camera_params": {"width": 128, "height": 96},
        }
    }
    cam = create_camera_driver(cfg)
    assert isinstance(cam, MockCameraDriver)
    with cam:
        assert cam.read_frame().data.shape == (96, 128, 3)


def test_explicit_params_override_config():
    cfg = {"hardware": {"driver": {"camera": "mock"}, "camera_params": {"width": 128}}}
    cam = create_camera_driver(cfg, params={"width": 64, "height": 64})
    with cam:
        assert cam.read_frame().data.shape == (64, 64, 3)


def test_missing_driver_key_fails_fast():
    with pytest.raises(UnknownDriverError):
        create_camera_driver({"hardware": {"driver": {}}})


@pytest.mark.parametrize("driver_name", ["mock", "webcam", "picamera", "csi_gstreamer"])
def test_every_registered_driver_class_imports_and_subclasses_the_interface(driver_name):
    """Import must succeed on ANY machine — the hardware SDKs (picamera2,
    GStreamer) are only imported inside open(), never at module scope."""
    cls = load_camera_driver_class(driver_name)
    assert issubclass(cls, CameraDriver)
    assert cls.name == driver_name
    assert cls.source_type in {"usb", "csi", "synthetic"}


@pytest.mark.parametrize("driver_name", ["mock", "webcam", "picamera", "csi_gstreamer"])
def test_capabilities_shape_is_uniform(driver_name):
    caps = load_camera_driver_class(driver_name)({}).get_capabilities()
    for key in ("driver", "source_type", "width", "height", "fps", "backend"):
        assert key in caps


# --------------------------------------------------------------------------- #
# Pipeline construction is testable without a Jetson
# --------------------------------------------------------------------------- #

def test_jetson_pipeline_string_uses_argus_and_converts_to_bgr():
    from edge.drivers.jetson.csi_gstreamer import JetsonCSIDriver

    pipeline = JetsonCSIDriver({"sensor_id": 1, "width": 640, "height": 360, "fps": 21}) \
        .build_pipeline()
    assert "nvarguscamerasrc sensor-id=1" in pipeline
    assert "memory:NVMM" in pipeline
    assert "nvvidconv" in pipeline
    assert "format=(string)BGR " in pipeline + " "
    assert "framerate=(fraction)21/1" in pipeline
    assert "appsink" in pipeline


# --------------------------------------------------------------------------- #
# Fixture replay
# --------------------------------------------------------------------------- #

def test_fixture_replay_loops_in_order(tmp_path):
    cv2 = pytest.importorskip("cv2")
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:03d}.png"),
                    np.full((32, 32, 3), i * 40 + 20, dtype=np.uint8))

    cam = create_camera_driver(driver="mock",
                               params={"mode": "fixtures", "fixture_dir": str(tmp_path)})
    with cam:
        names = [cam.read_frame().metadata["fixture"] for _ in range(4)]
    assert names == ["000.png", "001.png", "002.png", "000.png"]


def test_missing_fixture_dir_fails_fast():
    cam = create_camera_driver(driver="mock",
                               params={"mode": "fixtures", "fixture_dir": "/nope/nowhere"})
    with pytest.raises(CameraOpenError):
        cam.open()