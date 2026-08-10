"""Hardware abstraction layer.

Business logic imports from here and nothing deeper:

    from edge.drivers import create_camera_driver, CameraDriver, Frame
"""

from .factory import (
    CAMERA_REGISTRY,
    UnknownDriverError,
    available_drivers,
    create_camera_driver,
    load_camera_driver_class,
)
from .interfaces.camera import (
    CameraDriver,
    CameraError,
    CameraNotOpenError,
    CameraOpenError,
    CameraReadError,
    Frame,
)

__all__ = [
    "CAMERA_REGISTRY",
    "CameraDriver",
    "CameraError",
    "CameraNotOpenError",
    "CameraOpenError",
    "CameraReadError",
    "Frame",
    "UnknownDriverError",
    "available_drivers",
    "create_camera_driver",
    "load_camera_driver_class",
]
