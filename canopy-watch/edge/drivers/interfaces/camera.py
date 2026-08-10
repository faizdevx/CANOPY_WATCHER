"""Camera driver contract for Canopy Watch.

The rest of the application (Capture Service, Tier1, Tier2, scoring) talks ONLY
to :class:`CameraDriver` and only ever receives a :class:`Frame`.

No caller upstream of this module is allowed to know whether the pixels came
from OpenCV, picamera2, GStreamer/Argus, or a synthetic generator.

Canonical output (non-negotiable):

    Frame.data       numpy.ndarray, RGB, H x W x 3, uint8
    Frame.timestamp  monotonic float (seconds)
    Frame.source     driver name, for logging/debugging

All colour-space conversion (BGR->RGB, NV12->RGB, YUYV->RGB) happens inside the
driver, at this boundary, exactly once — never downstream.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class CameraError(Exception):
    """Base class for every camera driver failure."""


class CameraOpenError(CameraError):
    """Driver could not reach a READY state (device missing, permission denied,
    backend unavailable, or no valid frame produced during warm-up)."""


class CameraReadError(CameraError):
    """Driver was open but could not produce a frame."""


class CameraNotOpenError(CameraError):
    """read_frame() was called before a successful open()."""


# --------------------------------------------------------------------------- #
# Canonical frame
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Frame:
    """One image, normalised. This is the only thing that crosses the boundary."""

    data: np.ndarray                       # RGB, HxWx3, uint8
    timestamp: float                       # time.monotonic() at capture
    source: str                            # driver name, e.g. "webcam"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    def __repr__(self) -> str:  # keep logs readable, never dump pixels
        return (f"Frame(source={self.source!r}, {self.width}x{self.height}, "
                f"ts={self.timestamp:.3f}, meta_keys={sorted(self.metadata)})")


def validate_frame_array(arr: Any) -> bool:
    """True if `arr` satisfies the canonical contract (RGB HxWx3 uint8)."""
    return (
        isinstance(arr, np.ndarray)
        and arr.ndim == 3
        and arr.shape[2] == 3
        and arr.dtype == np.uint8
        and arr.shape[0] > 0
        and arr.shape[1] > 0
    )


def assert_canonical(arr: Any, source: str) -> np.ndarray:
    """Raise if a driver tried to emit a non-canonical array. Drivers call this
    on their way out so a contract violation is caught at the boundary, not
    three services downstream."""
    if not validate_frame_array(arr):
        shape = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", None)
        raise CameraReadError(
            f"driver {source!r} emitted a non-canonical frame "
            f"(shape={shape}, dtype={dtype}); expected RGB HxWx3 uint8"
        )
    return arr


def is_blank(arr: np.ndarray, std_threshold: float = 1.0) -> bool:
    """Cheap 'this frame is black/garbage' heuristic used during warm-up only.

    Deliberately NOT used as a runtime health signal — a legitimately dark night
    scene has low variance too, and this platform points cameras at forests at
    night on purpose.
    """
    return float(arr.std()) < std_threshold


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #

class CameraDriver(abc.ABC):
    """One abstract contract. Four implementations. Zero platform logic upstream.

        open()              -> bool
        read_frame()        -> Frame
        is_healthy()        -> bool
        close()             -> None
        get_capabilities()  -> dict

    Subclasses implement the three underscore hooks (`_open`, `_read`, `_close`)
    and inherit lifecycle guards, health bookkeeping and warm-up for free.
    """

    #: value of `hardware.driver.camera` that selects this implementation
    name: str = "unset"
    #: "usb" | "csi" | "synthetic" — what the physical source actually is
    source_type: str = "unknown"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params: Dict[str, Any] = dict(params or {})

        # --- health / warm-up policy (overridable per profile) -------------- #
        self.warmup_frames: int = int(self.params.get("warmup_frames", 5))
        self.open_read_attempts: int = int(self.params.get("open_read_attempts", 30))
        self.required_valid_frames: int = int(self.params.get("required_valid_frames", 2))
        self.reject_blank_on_open: bool = bool(self.params.get("reject_blank_on_open", False))
        self.read_retries: int = int(self.params.get("read_retries", 3))
        self.max_consecutive_failures: int = int(self.params.get("max_consecutive_failures", 10))
        self.stale_frame_seconds: float = float(self.params.get("stale_frame_seconds", 5.0))

        # --- state ---------------------------------------------------------- #
        self._opened: bool = False
        self._frame_index: int = 0
        self._last_frame_ts: Optional[float] = None
        self._consecutive_failures: int = 0
        self._last_error: Optional[str] = None

    # ----------------------------- lifecycle ------------------------------- #

    def open(self) -> bool:
        """Acquire the device, warm it up, and prove it produces real frames.

            open device -> discard N frames -> read until K valid frames -> READY

        `isOpened()`-style booleans are not trusted: macOS with camera
        permission denied, and some V4L2 devices, report "open" while every read
        returns nothing usable. Raises CameraOpenError instead of silently
        handing back a driver that will never produce a frame.
        """
        if self._opened:
            return True

        self._open()          # backend-specific acquisition
        self._opened = True

        try:
            self._warmup()
        except Exception:
            self._opened = False
            try:
                self._close()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("cleanup after failed open() also failed", exc_info=True)
            raise

        logger.info("camera driver %r ready (%s)", self.name, self.get_capabilities())
        return True

    def _warmup(self) -> None:
        """Discard the first N frames, then require K genuinely valid reads."""
        discarded = 0
        valid = 0
        attempts = 0
        last_reason = "no read attempted"

        while attempts < self.open_read_attempts and valid < self.required_valid_frames:
            attempts += 1
            try:
                arr = self._read()
            except Exception as exc:
                last_reason = f"read raised: {exc}"
                continue

            if not validate_frame_array(arr):
                last_reason = "read returned a non-canonical array"
                continue

            if discarded < self.warmup_frames:
                discarded += 1
                continue

            if self.reject_blank_on_open and is_blank(arr):
                last_reason = "frames were blank (black/garbage) after warm-up"
                continue

            valid += 1

        if valid < self.required_valid_frames:
            raise CameraOpenError(
                f"{self.name}: opened but produced no usable frames "
                f"({valid}/{self.required_valid_frames} valid in {attempts} reads) — "
                f"last reason: {last_reason}. "
                f"On macOS check Privacy & Security > Camera; on Linux check that the "
                f"device is not held by another process."
            )

        self._last_frame_ts = time.monotonic()
        self._consecutive_failures = 0
        self._after_warmup()

    def close(self) -> None:
        if not self._opened:
            return
        try:
            self._close()
        finally:
            self._opened = False
            self._last_frame_ts = None

    # ------------------------------ capture -------------------------------- #

    def read_frame(self) -> Frame:
        """Return the next canonical Frame, retrying transient read failures."""
        if not self._opened:
            raise CameraNotOpenError(f"{self.name}: read_frame() before open()")

        last_exc: Optional[Exception] = None
        for _ in range(max(1, self.read_retries)):
            try:
                arr = self._read()
            except Exception as exc:
                last_exc = exc
                continue

            if not validate_frame_array(arr):
                last_exc = CameraReadError("non-canonical array from backend")
                continue

            assert_canonical(arr, self.name)
            ts = time.monotonic()
            self._last_frame_ts = ts
            self._consecutive_failures = 0
            self._last_error = None
            meta = {
                "frame_index": self._frame_index,
                "wall_clock": time.time(),
                "source_type": self.source_type,
            }
            meta.update(self._frame_metadata())
            self._frame_index += 1
            return Frame(data=arr, timestamp=ts, source=self.name, metadata=meta)

        self._consecutive_failures += 1
        self._last_error = str(last_exc) if last_exc else "unknown read failure"
        raise CameraReadError(
            f"{self.name}: read failed after {self.read_retries} attempts "
            f"({self._consecutive_failures} consecutive) — {self._last_error}"
        )

    # ------------------------------- health -------------------------------- #

    def is_healthy(self) -> bool:
        """Active check, not a cached boolean from open().

        Unhealthy when: not open, too many consecutive read failures, or the
        last good frame is older than `stale_frame_seconds`.
        """
        if not self._opened:
            return False
        if self._consecutive_failures >= self.max_consecutive_failures:
            return False
        if self._last_frame_ts is None:
            return False
        if self.stale_frame_seconds > 0:
            if (time.monotonic() - self._last_frame_ts) > self.stale_frame_seconds:
                return False
        return True

    def health_detail(self) -> Dict[str, Any]:
        """Structured health for the Health Monitor / local debug API."""
        age = None if self._last_frame_ts is None else (time.monotonic() - self._last_frame_ts)
        return {
            "driver": self.name,
            "opened": self._opened,
            "healthy": self.is_healthy(),
            "frames_emitted": self._frame_index,
            "last_frame_age_s": age,
            "consecutive_failures": self._consecutive_failures,
            "last_error": self._last_error,
        }

    # --------------------------- capabilities ------------------------------ #

    def get_capabilities(self) -> Dict[str, Any]:
        caps = {
            "driver": self.name,
            "source_type": self.source_type,
            "width": self.params.get("width"),
            "height": self.params.get("height"),
            "fps": self.params.get("fps"),
        }
        caps.update(self._capabilities())
        return caps

    # ----------------------- context manager sugar ------------------------- #

    def __enter__(self) -> "CameraDriver":
        self.open()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -------------------------- backend hooks ------------------------------ #

    @abc.abstractmethod
    def _open(self) -> None:
        """Acquire the underlying device. Raise CameraOpenError on failure."""

    @abc.abstractmethod
    def _read(self) -> np.ndarray:
        """Return ONE frame already normalised to RGB HxWx3 uint8."""

    @abc.abstractmethod
    def _close(self) -> None:
        """Release the underlying device. Must be safe to call twice."""

    def _after_warmup(self) -> None:
        """Called once, after warm-up succeeds and before the first real
        read_frame(). Override when warm-up consumed something the driver needs
        back — e.g. the mock driver rewinds its fixture reel so replay always
        starts at the first file."""
        return None

    def _capabilities(self) -> Dict[str, Any]:
        """Backend-specific capability extras. Override if useful."""
        return {}

    def _frame_metadata(self) -> Dict[str, Any]:
        """Backend-specific per-frame metadata. Override if useful."""
        return {}