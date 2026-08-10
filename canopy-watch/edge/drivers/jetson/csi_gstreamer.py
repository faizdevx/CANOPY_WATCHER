"""Jetson CSI camera driver — `hardware.driver.camera: csi_gstreamer`.

Backend: NVIDIA Argus, reached through the `nvarguscamerasrc` GStreamer
element, consumed by OpenCV with `cv2.CAP_GSTREAMER`.

Why not plain V4L2: the V4L2 subdevices exist underneath, but going around
Argus loses the ISP — auto-exposure, auto white balance, sensor mode selection.
For a camera pointed at a forest across a full day/night cycle, that matters.

Two things that bite people, both checked explicitly at open():

1. The default pip `opencv-python` wheel is built WITHOUT GStreamer support.
   `VideoCapture(pipeline, CAP_GSTREAMER)` then fails with no useful message.
   JetPack's bundled OpenCV normally has `WITH_GSTREAMER=ON`; this driver reads
   `cv2.getBuildInformation()` and says so plainly rather than failing opaquely.
2. `nvarguscamerasrc` frames live in GPU memory (NVMM). Converting them to a
   CPU numpy array is a deliberate cost paid for portability — every driver in
   this project returns the same canonical CPU RGB array. If profiling later
   shows the round-trip is the bottleneck, the fix is an optional
   `get_gpu_frame()` that only Tier2-on-Jetson opts into. The base interface
   stays untouched. Nano and Orin use the identical mechanism.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from ..interfaces.camera import CameraDriver, CameraOpenError, CameraReadError

logger = logging.getLogger(__name__)


class JetsonCSIDriver(CameraDriver):
    name = "csi_gstreamer"
    source_type = "csi"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        self.sensor_id: int = int(self.params.get("sensor_id", 0))
        self.sensor_mode: Optional[int] = self.params.get("sensor_mode")
        # capture_* = what the sensor emits; width/height = what we hand upstream
        self.capture_width: int = int(self.params.get("capture_width", 1920))
        self.capture_height: int = int(self.params.get("capture_height", 1080))
        self.width: int = int(self.params.get("width", 1280))
        self.height: int = int(self.params.get("height", 720))
        self.fps: int = int(self.params.get("fps", 30))
        self.flip_method: int = int(self.params.get("flip_method", 0))
        self.pipeline_override: Optional[str] = self.params.get("pipeline")

        self._cv2 = None
        self._cap = None
        self._pipeline: str = ""

    # ------------------------------------------------------------------ #

    def _open(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise CameraOpenError("csi_gstreamer: OpenCV is not installed") from exc
        self._cv2 = cv2

        if not _has_gstreamer(cv2):
            raise CameraOpenError(
                "csi_gstreamer: this OpenCV build has no GStreamer support "
                "(getBuildInformation() reports GStreamer: NO). The pip "
                "`opencv-python` wheel is built without it. Use JetPack's bundled "
                "OpenCV (python3-opencv), or rebuild with -D WITH_GSTREAMER=ON. "
                "A USB camera on this board does not need any of this — set "
                "driver.camera: webcam instead."
            )

        self._pipeline = self.pipeline_override or self.build_pipeline()
        logger.info("csi_gstreamer: pipeline = %s", self._pipeline)

        cap = cv2.VideoCapture(self._pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap.release()
            raise CameraOpenError(
                f"csi_gstreamer: GStreamer pipeline failed to open for "
                f"sensor-id={self.sensor_id}. Check: the CSI ribbon cable "
                f"orientation (reversed = no image and no error), that the board "
                f"is flashed with JetPack, and that `gst-launch-1.0 "
                f"nvarguscamerasrc num-buffers=1 ! fakesink` succeeds."
            )
        self._cap = cap

    def build_pipeline(self) -> str:
        """nvarguscamerasrc -> NVMM caps -> nvvidconv (GPU->CPU) -> BGRx ->
        videoconvert -> BGR -> appsink. OpenCV hands us BGR; `_read` flips it to
        the canonical RGB."""
        src = f"nvarguscamerasrc sensor-id={self.sensor_id}"
        if self.sensor_mode is not None:
            src += f" sensor-mode={int(self.sensor_mode)}"
        return (
            f"{src} ! "
            f"video/x-raw(memory:NVMM), width=(int){self.capture_width}, "
            f"height=(int){self.capture_height}, framerate=(fraction){self.fps}/1 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            f"video/x-raw, width=(int){self.width}, height=(int){self.height}, "
            f"format=(string)BGRx ! "
            f"videoconvert ! video/x-raw, format=(string)BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )

    def _read(self) -> np.ndarray:
        if self._cap is None:
            raise CameraReadError("csi_gstreamer: capture handle is gone")
        ok, bgr = self._cap.read()
        # Argus buffers carry a PTS, but OpenCV's appsink wrapper does not
        # expose it (CAP_PROP_POS_MSEC is meaningless for live capture).
        # Reporting None is correct; a raw appsink consumer could surface the
        # real PTS later without changing this interface.
        if not ok or bgr is None:
            raise CameraReadError("csi_gstreamer: appsink produced no frame")
        return np.ascontiguousarray(bgr[:, :, ::-1])  # BGR -> RGB

    def _close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    # ------------------------------------------------------------------ #

    def _capabilities(self) -> Dict[str, Any]:
        return {
            "backend": "nvarguscamerasrc/GStreamer",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "capture_width": self.capture_width,
            "capture_height": self.capture_height,
            "sensor_id": self.sensor_id,
            "sensor_mode": self.sensor_mode,
            "zero_copy": False,  # deliberate: NVMM -> CPU numpy for portability
            "media_timestamp_available": False,
            "pipeline": self._pipeline or None,
        }

    def _frame_metadata(self) -> Dict[str, Any]:
        return {"media_timestamp": None}


def _has_gstreamer(cv2) -> bool:
    try:
        info = cv2.getBuildInformation()
    except Exception:  # pragma: no cover
        return False
    for line in info.splitlines():
        if "GStreamer" in line:
            return "YES" in line.upper()
    return False