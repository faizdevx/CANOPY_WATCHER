"""Video file driver — `hardware.driver.camera: video`.

Replays an mp4/mov/avi/mkv through the exact same `CameraDriver` contract as a
live webcam. Nothing downstream — capture, Tier1, Tier2, scoring — knows or
cares that the frames came off disk. That is the point of the driver boundary,
and a recorded clip is the only honest way to evaluate detection quality
repeatably: the same file produces the same detections every run.

Three things this driver does that a live camera cannot:

**Real media timestamps.** `CAP_PROP_POS_MSEC` is meaningless for live capture,
but for a file it is exactly what it claims — the presentation time of this
frame within the recording. So `media_timestamp` is genuinely populated here,
and the capture service's three-timestamp design finally has a third source
outside the CSI paths.

**End of stream.** A finite source finishing is success, not a fault. The driver
raises `CameraEndOfStream`, which the Capture Service treats as a clean shutdown
rather than a dropout to reconnect from. With `loop: true` it rewinds instead,
which is what you want when using a clip as a stand-in for a live feed.

**Two clocks.** `realtime: false` (default) replays as fast as the machine can
process it — right for batch analysis, and it makes CI fast. `realtime: true`
paces to the file's own fps, which is what you want when testing whether a
device can actually keep up with 30 fps of real footage.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .interfaces.camera import (
    CameraDriver,
    CameraEndOfStream,
    CameraOpenError,
    CameraReadError,
)

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}


class VideoFileDriver(CameraDriver):
    name = "video"
    source_type = "file"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        self.path: Optional[str] = self.params.get("path") or self.params.get("video_path")
        self.loop: bool = bool(self.params.get("loop", False))
        self.realtime: bool = bool(self.params.get("realtime", False))
        self.speed: float = float(self.params.get("speed", 1.0))
        self.start_frame: int = int(self.params.get("start_frame", 0))
        self.end_frame: Optional[int] = self.params.get("end_frame")
        self.stride: int = max(1, int(self.params.get("frame_stride", 1)))
        #: resize on read — useful for feeding 4K footage to a Pi-sized pipeline
        self.width: Optional[int] = self.params.get("width")
        self.height: Optional[int] = self.params.get("height")

        # A file has no warm-up problem: frame 0 is already a real frame.
        self.warmup_frames = int(self.params.get("warmup_frames", 0))
        self.required_valid_frames = int(self.params.get("required_valid_frames", 1))
        # Nor a staleness problem — a paused decode is not an unhealthy camera.
        self.stale_frame_seconds = float(self.params.get("stale_frame_seconds", 0))

        self._cv2 = None
        self._cap = None
        self._source_fps: float = 0.0
        self._total_frames: int = 0
        self._position: int = 0          # index of the NEXT frame to read
        self._last_pts_ms: Optional[float] = None
        self._playback_started: Optional[float] = None
        self._first_pts_ms: Optional[float] = None
        self.loops_completed = 0
        self.frames_read = 0

    # ------------------------------------------------------------------ #

    def _open(self) -> None:
        if not self.path:
            raise CameraOpenError(
                "video: camera_params.path is not set — point it at an mp4")
        path = Path(self.path).expanduser()
        if not path.exists():
            raise CameraOpenError(f"video: file not found: {path}")
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            logger.warning("video: %s has an unusual suffix — trying anyway", path.name)

        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise CameraOpenError(
                "video: opencv-python is not installed (pip install opencv-python)"
            ) from exc
        self._cv2 = cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise CameraOpenError(
                f"video: OpenCV could not open {path}. The container may be fine "
                f"but the codec unsupported by this build — try re-encoding with "
                f"`ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4`."
            )

        self._cap = cap
        self._source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._position = 0
        self._playback_started = None
        self._first_pts_ms = None

        if self.start_frame:
            self._seek(self.start_frame)

        logger.info(
            "video: %s — %d frames @ %.2f fps (%.1fs), replay=%s stride=%d loop=%s",
            path.name, self._total_frames, self._source_fps,
            self._duration_seconds(), "realtime" if self.realtime else "fast",
            self.stride, self.loop,
        )

    def _read(self) -> np.ndarray:
        cap, cv2 = self._cap, self._cv2
        if cap is None:
            raise CameraReadError("video: capture handle is gone")

        if self.end_frame is not None and self._position >= self.end_frame:
            return self._finish()

        pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        ok, bgr = cap.read()
        self._position += 1

        if not ok or bgr is None:
            return self._finish()

        # Skip ahead when subsampling. Reading and discarding is faster and far
        # more reliable than seeking, which is only frame-accurate on keyframes.
        for _ in range(self.stride - 1):
            if not cap.grab():
                break
            self._position += 1

        self._last_pts_ms = float(pts_ms) if pts_ms is not None else None
        self.frames_read += 1
        self._pace()

        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        if self.width and self.height:
            bgr = cv2.resize(bgr, (int(self.width), int(self.height)),
                             interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(bgr[:, :, ::-1])       # BGR -> RGB

    def _finish(self) -> np.ndarray:
        """End of the clip: rewind, or declare the stream over."""
        if self.loop:
            self.loops_completed += 1
            self._seek(self.start_frame)
            self._playback_started = None
            self._first_pts_ms = None
            logger.debug("video: looped (%d)", self.loops_completed)
            return self._read()
        raise CameraEndOfStream(
            f"video: reached the end of {Path(self.path).name} after "
            f"{self.frames_read} frame(s)")

    def _close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    # ------------------------------------------------------------------ #

    def _seek(self, frame_index: int) -> None:
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        self._position = int(frame_index)

    def _pace(self) -> None:
        """Sleep so playback tracks the file's own timeline.

        Paced against the media PTS rather than a per-frame sleep, so decode
        time doesn't accumulate into drift over a long clip.
        """
        if not self.realtime or self._last_pts_ms is None:
            return
        now = time.monotonic()
        if self._playback_started is None:
            self._playback_started = now
            self._first_pts_ms = self._last_pts_ms
            return
        media_elapsed = (self._last_pts_ms - (self._first_pts_ms or 0.0)) / 1000.0
        target = self._playback_started + media_elapsed / max(0.01, self.speed)
        gap = target - now
        if gap > 0:
            time.sleep(min(gap, 1.0))

    def _duration_seconds(self) -> float:
        if self._source_fps > 0 and self._total_frames > 0:
            return self._total_frames / self._source_fps
        return 0.0

    # ------------------------------------------------------------------ #

    def _frame_metadata(self) -> Dict[str, Any]:
        # Unlike live capture, POS_MSEC on a file is a real presentation time.
        return {
            "media_timestamp": (None if self._last_pts_ms is None
                                else self._last_pts_ms / 1000.0),
            "video_frame_index": self._position - 1,
            "loop": self.loops_completed,
        }

    def _capabilities(self) -> Dict[str, Any]:
        return {
            "backend": "opencv/file",
            "path": self.path,
            "width": self.width or (int(self._cap.get(self._cv2.CAP_PROP_FRAME_WIDTH))
                                    if self._cap else None),
            "height": self.height or (int(self._cap.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
                                      if self._cap else None),
            "fps": self._source_fps or None,
            "total_frames": self._total_frames,
            "duration_s": round(self._duration_seconds(), 2),
            "frame_stride": self.stride,
            "loop": self.loop,
            "realtime": self.realtime,
            "media_timestamp_available": True,
            "finite": not self.loop,
        }