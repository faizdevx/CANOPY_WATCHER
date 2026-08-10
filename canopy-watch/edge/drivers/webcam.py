"""Webcam driver — `hardware.driver.camera: webcam`.

ONE implementation shared by every USB/V4L2-family camera:

    laptop (Windows MSMF / macOS AVFoundation / Linux V4L2)
    Raspberry Pi with a USB webcam
    Jetson with a USB webcam

It lives at `edge/drivers/webcam.py` rather than being copied into
`laptop/`, `rpi/` and `jetson/` because the capture path is genuinely
identical in all three cases — OpenCV's `VideoCapture` over the platform
backend. Only the two CSI stacks (Pi libcamera, Jetson Argus) are different
enough to need their own modules, and those get their own folders.

Handled here, once, so nothing upstream ever sees it:
  * OS backend hint          (CAP_MSMF / CAP_AVFOUNDATION / CAP_V4L2)
  * name-first, index-fallback device resolution (indices reshuffle on replug)
  * warm-up (first frames are black/garbage on MSMF and others) — in the base class
  * "opened but never yields a frame" (macOS camera permission denied) — base class
  * BGR -> RGB conversion
"""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .interfaces.camera import CameraDriver, CameraOpenError, CameraReadError

logger = logging.getLogger(__name__)


class WebcamDriver(CameraDriver):
    name = "webcam"
    source_type = "usb"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        self.device_name: Optional[str] = self.params.get("device_name")
        self.device_index: int = int(self.params.get("device_index", 0))
        self.width: Optional[int] = self.params.get("width")
        self.height: Optional[int] = self.params.get("height")
        self.fps: Optional[float] = self.params.get("fps")
        self.fourcc: Optional[str] = self.params.get("fourcc")  # e.g. "MJPG"
        self.backend_pref: str = str(self.params.get("backend", "auto"))

        self._cv2 = None
        self._cap = None
        self._resolved_index: Optional[int] = None
        self._resolved_via: str = "unresolved"
        self._backend_name: str = "unknown"

    # ------------------------------------------------------------------ #

    def _open(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise CameraOpenError(
                "webcam: opencv-python is not installed (pip install opencv-python)"
            ) from exc
        self._cv2 = cv2

        index, via = self._resolve_device(cv2)
        self._resolved_index, self._resolved_via = index, via

        api, api_name = self._backend(cv2)
        self._backend_name = api_name

        cap = cv2.VideoCapture(index, api) if api is not None else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            raise CameraOpenError(
                f"webcam: could not open device index {index} "
                f"(resolved via {via}, backend {api_name}). "
                f"Check the camera is connected and not in use by another application."
            )

        # Format first: on USB, MJPG vs YUYV changes throughput more than
        # resolution does, and some devices only offer high fps under MJPG.
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        self._cap = cap
        logger.info(
            "webcam: opened index=%s (via %s) backend=%s requested=%sx%s@%s",
            index, via, api_name, self.width, self.height, self.fps,
        )
        # NOTE: isOpened() being True proves nothing on macOS with camera
        # permission denied. The base class warm-up is what actually decides
        # whether this driver reaches READY.

    def _read(self) -> np.ndarray:
        if self._cap is None:
            raise CameraReadError("webcam: capture handle is gone")
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise CameraReadError("webcam: VideoCapture.read() returned no frame")
        if bgr.ndim == 2:  # some V4L2 devices hand back single-channel
            bgr = self._cv2.cvtColor(bgr, self._cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(bgr[:, :, ::-1])  # BGR -> RGB, once, here

    def _close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    # ------------------------- device resolution ----------------------- #

    def _resolve_device(self, cv2) -> Tuple[int, str]:
        """Name first, index fallback.

        `device_index: 0` does not durably mean "my laptop webcam" — plug in a
        second USB camera and the numbering reshuffles. If `device_name` is
        configured and we can enumerate names on this OS, we match on it.
        """
        if self.device_name:
            found = _find_index_by_name(self.device_name)
            if found is not None:
                return found, f"name match {self.device_name!r}"
            logger.warning(
                "webcam: device_name %r not found on this system — "
                "falling back to device_index=%s",
                self.device_name, self.device_index,
            )
        return self.device_index, "configured index"

    def _backend(self, cv2) -> Tuple[Optional[int], str]:
        """Pass an explicit backend hint. Without one, OpenCV probes backends
        sequentially on open, which is slow and occasionally picks the legacy
        one (DirectShow instead of MSMF)."""
        pref = self.backend_pref.lower()
        explicit = {
            "v4l2": ("CAP_V4L2", "CAP_V4L2"),
            "msmf": ("CAP_MSMF", "CAP_MSMF"),
            "dshow": ("CAP_DSHOW", "CAP_DSHOW"),
            "avfoundation": ("CAP_AVFOUNDATION", "CAP_AVFOUNDATION"),
            "any": (None, "CAP_ANY"),
        }
        if pref in explicit:
            attr, label = explicit[pref]
            return (getattr(cv2, attr, None) if attr else None), label

        system = platform.system()
        attr = {
            "Linux": "CAP_V4L2",
            "Darwin": "CAP_AVFOUNDATION",
            "Windows": "CAP_MSMF",
        }.get(system)
        if attr is None:
            return None, "CAP_ANY"
        return getattr(cv2, attr, None), attr

    # ---------------------------- reporting ---------------------------- #

    def _capabilities(self) -> Dict[str, Any]:
        caps: Dict[str, Any] = {
            "backend": self._backend_name,
            "resolved_index": self._resolved_index,
            "resolved_via": self._resolved_via,
            "device_name": self.device_name,
            "os": platform.system(),
        }
        if self._cap is not None and self._cv2 is not None:
            cv2 = self._cv2
            caps["width"] = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            caps["height"] = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            caps["fps"] = self._cap.get(cv2.CAP_PROP_FPS) or None
        else:
            caps.update({"width": self.width, "height": self.height, "fps": self.fps})
        return caps


# --------------------------------------------------------------------------- #
# Device-name enumeration (best effort, per OS)
# --------------------------------------------------------------------------- #

def list_devices() -> List[Dict[str, Any]]:
    """Enumerate cameras as [{index, name}]. Best effort — OpenCV itself has no
    portable device-name API, so each OS is probed differently and an empty list
    is a normal, non-fatal result."""
    system = platform.system()
    if system == "Linux":
        return _list_linux()
    if system == "Darwin":
        return _list_macos()
    if system == "Windows":
        return _list_windows()
    return []


def _find_index_by_name(wanted: str) -> Optional[int]:
    needle = wanted.strip().lower()
    for dev in list_devices():
        if needle in str(dev.get("name", "")).lower():
            return int(dev["index"])
    return None


def _list_linux() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    base = Path("/sys/class/video4linux")
    if not base.is_dir():
        return out
    for node in sorted(base.iterdir()):
        name_file = node / "name"
        if not name_file.exists():
            continue
        try:
            idx = int("".join(c for c in node.name if c.isdigit()))
            out.append({"index": idx, "name": name_file.read_text().strip()})
        except (ValueError, OSError):
            continue
    return out


def _list_macos() -> List[Dict[str, Any]]:
    # AVFoundation has no /dev/video* equivalent; system_profiler is the only
    # dependency-free way to see device names. Order here is not guaranteed to
    # match AVFoundation's index order, so this is a hint, not gospel.
    try:
        raw = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    names: List[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.endswith(":") and line.startswith("    ") and not line.startswith("      "):
            label = stripped[:-1].strip()
            if label and label != "Camera":
                names.append(label)
    return [{"index": i, "name": n} for i, n in enumerate(names)]


def _list_windows() -> List[Dict[str, Any]]:
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore
    except ImportError:
        logger.debug("webcam: pygrabber not installed — cannot enumerate names on Windows")
        return []
    try:
        return [{"index": i, "name": n}
                for i, n in enumerate(FilterGraph().get_input_devices())]
    except Exception:  # pragma: no cover
        return []