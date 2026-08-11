"""Snapshot writer — populates `DetectionEvent.snapshot_ref`.

Events carry no image bytes: they travel over LoRa/4G and must stay small, and
they must move independently of media. But an event pointing at nothing is
useless for review, so the node writes the pixels locally and puts a *reference*
in the event. The Sync Agent uploads media separately, on its own schedule, and
the reference resolves to an object key once it has.

Bounded on purpose. An edge node runs for months on an SD card, and a snapshot
per detection with no ceiling is a filesystem that fills silently at 3am. Oldest
files are evicted once the count or byte budget is exceeded — the same rule as
the frame buffer, applied to disk.

Off by default. Turn it on for video analysis and field review; leave it off on
a Pi where SD-card write cycles are the constraint.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "directory": "snapshots",
    "mode": "crop",            # crop | frame | both
    "format": "jpg",           # jpg | png
    "jpeg_quality": 85,
    "max_files": 500,
    "max_megabytes": 200,
    "min_band": "low",         # only persist at or above this risk band
}

BAND_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


class SnapshotWriter:
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        cfg = {**DEFAULTS, **dict(params or {})}
        self.enabled: bool = bool(cfg["enabled"])
        self.directory = Path(cfg["directory"]).expanduser()
        self.mode: str = str(cfg["mode"])
        self.format: str = str(cfg["format"]).lstrip(".")
        self.jpeg_quality: int = int(cfg["jpeg_quality"])
        self.max_files: int = int(cfg["max_files"])
        self.max_bytes: int = int(cfg["max_megabytes"]) * 1024 * 1024
        self.min_band: str = str(cfg["min_band"])

        self._lock = threading.Lock()
        self._written: Deque[Tuple[Path, int]] = deque()
        self._bytes = 0
        self.count = 0
        self.errors = 0
        self.skipped = 0

    # ------------------------------------------------------------------ #

    def should_write(self, band: str) -> bool:
        if not self.enabled:
            return False
        return BAND_ORDER.get(band, 0) >= BAND_ORDER.get(self.min_band, 0)

    def write(
        self,
        frame_id: str,
        *,
        crop: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None,
        suffix: str = "",
    ) -> Optional[str]:
        """Write the snapshot and return the reference to embed in the event."""
        if not self.enabled:
            return None

        image = crop if (self.mode in ("crop", "both") and crop is not None) else frame
        if image is None:
            return None

        safe = frame_id.replace("/", "_")
        name = f"{safe}{('-' + suffix) if suffix else ''}.{self.format}"
        target = self.directory / name

        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            written = _encode(image, target, self.jpeg_quality)
        except Exception as exc:
            self.errors += 1
            logger.warning("snapshot: failed to write %s: %s", target, exc)
            return None

        with self._lock:
            self._written.append((target, written))
            self._bytes += written
            self.count += 1
            self._evict()

        if self.mode == "both" and crop is not None and frame is not None:
            try:
                full = self.directory / f"{safe}-full.{self.format}"
                extra = _encode(frame, full, self.jpeg_quality)
                with self._lock:
                    self._written.append((full, extra))
                    self._bytes += extra
                    self._evict()
            except Exception:
                logger.debug("snapshot: full-frame write failed", exc_info=True)

        return str(target)

    def _evict(self) -> None:
        while self._written and (len(self._written) > self.max_files
                                 or self._bytes > self.max_bytes):
            path, size = self._written.popleft()
            self._bytes -= size
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.debug("snapshot: could not unlink %s", path)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "directory": str(self.directory),
                "written": self.count,
                "retained": len(self._written),
                "megabytes": round(self._bytes / (1024 * 1024), 2),
                "errors": self.errors,
            }


def _encode(image: np.ndarray, target: Path, quality: int) -> int:
    """RGB in, file out. Returns bytes written."""
    try:
        import cv2  # type: ignore
        params = ([int(cv2.IMWRITE_JPEG_QUALITY), quality]
                  if target.suffix.lower() in (".jpg", ".jpeg") else [])
        if not cv2.imwrite(str(target), image[:, :, ::-1], params):
            raise OSError(f"cv2.imwrite returned False for {target}")
    except ImportError:
        from PIL import Image  # type: ignore
        Image.fromarray(image).save(target, quality=quality)
    return target.stat().st_size