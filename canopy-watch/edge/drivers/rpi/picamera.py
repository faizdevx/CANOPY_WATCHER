"""Raspberry Pi CSI camera driver — `hardware.driver.camera: picamera`.

Backend: libcamera, via the officially maintained **picamera2** binding.

This is NOT the USB webcam path. libcamera does not reliably expose a CSI
sensor as a plain `/dev/video0` for OpenCV, and the legacy raspicam/MMAL stack
is deprecated — so a Pi with a CSI module and a Pi with a USB webcam use two
genuinely different implementations, selected by `driver.camera`.

`picamera2` is imported lazily inside `_open()`. Contributor laptops and Jetson
boards never install it, and importing it at module scope would break the
factory's registry import on every non-Pi machine.

Colour ordering note: picamera2's "RGB888" format label describes the packing,
and the numpy array it hands back is in BGR channel order. That is corrected
here, at the driver boundary. `swap_rb: false` disables the correction if a
future picamera2 changes this behaviour.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from ..interfaces.camera import CameraDriver, CameraOpenError, CameraReadError

logger = logging.getLogger(__name__)


class PiCameraDriver(CameraDriver):
    name = "picamera"
    source_type = "csi"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        self.width: int = int(self.params.get("width", 1280))
        self.height: int = int(self.params.get("height", 720))
        self.fps: Optional[float] = self.params.get("fps")
        self.camera_num: int = int(self.params.get("camera_num", 0))
        self.pixel_format: str = str(self.params.get("format", "RGB888"))
        self.swap_rb: bool = bool(self.params.get("swap_rb", True))
        self.hflip: bool = bool(self.params.get("hflip", False))
        self.vflip: bool = bool(self.params.get("vflip", False))
        self.controls: Dict[str, Any] = dict(self.params.get("controls", {}))
        # SensorTimestamp is a real hardware capture time — one of only two
        # places in this project where a media timestamp genuinely exists.
        self.report_sensor_timestamp: bool = bool(
            self.params.get("report_sensor_timestamp", True))

        self._last_sensor_ts: Optional[float] = None
        self._picam2 = None
        self._sensor_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #

    def _open(self) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise CameraOpenError(
                "picamera: picamera2 is not installed. On Raspberry Pi OS: "
                "sudo apt install -y python3-picamera2 "
                "(it is not reliably pip-installable). "
                "If this device has a USB webcam rather than a CSI module, set "
                "driver.camera: webcam instead."
            ) from exc

        try:
            available = Picamera2.global_camera_info()
        except Exception as exc:
            raise CameraOpenError(f"picamera: libcamera could not enumerate cameras: {exc}") from exc

        if not available:
            raise CameraOpenError(
                "picamera: libcamera reports no cameras attached. Check the CSI "
                "ribbon cable orientation, and `rpicam-hello --list-cameras` "
                "(or `libcamera-hello --list-cameras` on older OS images)."
            )
        if self.camera_num >= len(available):
            raise CameraOpenError(
                f"picamera: camera_num={self.camera_num} but libcamera sees "
                f"{len(available)} camera(s)"
            )
        self._sensor_info = dict(available[self.camera_num])

        try:
            picam2 = Picamera2(self.camera_num)
            cfg_kwargs: Dict[str, Any] = {
                "main": {"format": self.pixel_format, "size": (self.width, self.height)},
            }
            transform = _transform(self.hflip, self.vflip)
            if transform is not None:
                cfg_kwargs["transform"] = transform
            config = picam2.create_video_configuration(**cfg_kwargs)
            picam2.configure(config)
            controls = dict(self.controls)
            if self.fps:
                frame_us = int(1_000_000 / float(self.fps))
                controls.setdefault("FrameDurationLimits", (frame_us, frame_us))
            if controls:
                picam2.set_controls(controls)
            picam2.start()
        except Exception as exc:
            raise CameraOpenError(f"picamera: failed to start camera: {exc}") from exc

        self._picam2 = picam2
        logger.info(
            "picamera: started %s at %dx%d (format=%s)",
            self._sensor_info.get("Model", "unknown sensor"),
            self.width, self.height, self.pixel_format,
        )

    def _read(self) -> np.ndarray:
        if self._picam2 is None:
            raise CameraReadError("picamera: camera handle is gone")
        self._last_sensor_ts = None
        if self.report_sensor_timestamp:
            # capture_request() gets pixels and metadata from the SAME request,
            # so the timestamp provably belongs to this frame. Two separate
            # calls could straddle a frame boundary.
            try:
                request = self._picam2.capture_request()
            except Exception as exc:
                raise CameraReadError(f"picamera: capture_request failed: {exc}") from exc
            try:
                arr = request.make_array("main")
                sensor_ns = (request.get_metadata() or {}).get("SensorTimestamp")
                if sensor_ns:
                    self._last_sensor_ts = float(sensor_ns) / 1e9   # ns -> s
            finally:
                request.release()
        else:
            try:
                arr = self._picam2.capture_array("main")
            except Exception as exc:
                raise CameraReadError(f"picamera: capture_array failed: {exc}") from exc
        if arr is None:
            raise CameraReadError("picamera: capture returned None")

        if arr.ndim == 3 and arr.shape[2] == 4:      # XBGR8888 / XRGB8888
            arr = arr[:, :, :3]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise CameraReadError(f"picamera: unexpected array shape {arr.shape}")
        if self.swap_rb:
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr.astype(np.uint8, copy=False))

    def _close(self) -> None:
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:  # pragma: no cover
                logger.debug("picamera: error during close", exc_info=True)
            finally:
                self._picam2 = None

    # ------------------------------------------------------------------ #

    def _capabilities(self) -> Dict[str, Any]:
        return {
            "backend": "picamera2/libcamera",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "format": self.pixel_format,
            "sensor": self._sensor_info.get("Model"),
            "camera_num": self.camera_num,
            "media_timestamp_available": self.report_sensor_timestamp,
        }

    def _frame_metadata(self) -> Dict[str, Any]:
        # None when unavailable. Capture must never backfill this from the
        # receive time — absent is debuggable, fabricated is not.
        return {"media_timestamp": self._last_sensor_ts}


def _transform(hflip: bool, vflip: bool):
    """libcamera Transform, or None when no flip was asked for (keeps this
    module importable for introspection without libcamera present)."""
    if not (hflip or vflip):
        return None
    from libcamera import Transform  # type: ignore
    return Transform(hflip=int(hflip), vflip=int(vflip))