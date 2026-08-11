"""Camera driver factory.

The ONLY place in the codebase that maps a config string to an implementation.
Application code calls `create_camera_driver(cfg)` and gets back something that
satisfies `CameraDriver` — it never imports a concrete driver, and never asks
what OS or board it is running on.

    hardware:
      driver:
        camera: webcam          # mock | webcam | picamera | csi_gstreamer
      camera_params:
        device_name: "Logitech C920"
        device_index: 0

Registry entries are (module path, class name) pairs imported lazily, so a
laptop never needs picamera2 installed and a Pi never needs the Jetson path to
resolve.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Mapping, Optional, Tuple, Type

from .interfaces.camera import CameraDriver, CameraError

logger = logging.getLogger(__name__)

#: driver name -> (module, class). Add a new sensor here, nowhere else.
CAMERA_REGISTRY: Dict[str, Tuple[str, str]] = {
    "mock":          ("edge.drivers.mock.camera",           "MockCameraDriver"),
    "webcam":        ("edge.drivers.webcam",                "WebcamDriver"),
    "picamera":      ("edge.drivers.rpi.picamera",          "PiCameraDriver"),
    "csi_gstreamer": ("edge.drivers.jetson.csi_gstreamer",  "JetsonCSIDriver"),
    "video": ("edge.drivers.video", "VideoFileDriver"),
}


class UnknownDriverError(CameraError):
    """Config named a camera driver that isn't registered."""


def available_drivers() -> Tuple[str, ...]:
    return tuple(sorted(CAMERA_REGISTRY))


def load_camera_driver_class(driver_name: str) -> Type[CameraDriver]:
    """Import and return the driver class for `driver_name` (no instantiation)."""
    try:
        module_path, class_name = CAMERA_REGISTRY[driver_name]
    except KeyError as exc:
        raise UnknownDriverError(
            f"unknown camera driver {driver_name!r}; "
            f"valid values are {', '.join(available_drivers())}"
        ) from exc

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise CameraError(
            f"camera driver {driver_name!r} could not be imported from "
            f"{module_path}: {exc}"
        ) from exc

    cls = getattr(module, class_name)
    if not issubclass(cls, CameraDriver):
        raise CameraError(f"{module_path}.{class_name} does not implement CameraDriver")
    return cls


def create_camera_driver(
    config: Optional[Mapping[str, Any]] = None,
    *,
    driver: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> CameraDriver:
    """Build (but do not open) the camera driver selected by config.

    Accepts either a resolved Canopy Watch config mapping::

        create_camera_driver(cfg)          # reads hardware.driver.camera

    or explicit arguments, which is what tests and the CLI use::

        create_camera_driver(driver="mock", params={"width": 320})

    Explicit arguments win over the config mapping.
    """
    cfg_driver, cfg_params = _extract(config)
    name = driver or cfg_driver
    if not name:
        raise UnknownDriverError(
            "no camera driver configured — set hardware.driver.camera to one of "
            + ", ".join(available_drivers())
        )

    merged: Dict[str, Any] = dict(cfg_params)
    if params:
        merged.update(params)

    cls = load_camera_driver_class(name)
    logger.debug("camera factory: %s -> %s", name, cls.__name__)
    return cls(merged)


def _extract(config: Optional[Mapping[str, Any]]) -> Tuple[Optional[str], Dict[str, Any]]:
    """Pull driver name + params out of a resolved config, tolerating both the
    nested `hardware.*` form and an already-narrowed `hardware` sub-mapping."""
    if not config:
        return None, {}
    hardware = config.get("hardware", config)
    if not isinstance(hardware, Mapping):
        return None, {}
    drivers = hardware.get("driver") or hardware.get("drivers") or {}
    name = drivers.get("camera") if isinstance(drivers, Mapping) else None
    raw_params = (
        (drivers.get("camera_params") if isinstance(drivers, Mapping) else None)
        or hardware.get("camera_params")
        or {}
    )
    params = dict(raw_params) if isinstance(raw_params, Mapping) else {}
    return name, params