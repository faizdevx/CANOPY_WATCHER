"""Mock camera driver — `hardware.driver.camera: mock`.

Used for: contributor laptops with no camera, CI, and the simulation harness.
Depends on numpy only, so it runs anywhere the project runs.

Two modes:

    mode: synthetic   generated frames (moving target on a gradient background)
    mode: fixtures    replays image files from `fixture_dir`, in sorted order

Fixture replay needs an image decoder (OpenCV or Pillow); synthetic mode needs
neither, which is why synthetic is the default.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..interfaces.camera import CameraDriver, CameraOpenError, CameraReadError

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class MockCameraDriver(CameraDriver):
    name = "mock"
    source_type = "synthetic"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        self.mode: str = str(self.params.get("mode", "synthetic"))
        self.width: int = int(self.params.get("width", 640))
        self.height: int = int(self.params.get("height", 480))
        self.fps: float = float(self.params.get("fps", 30))
        self.fixture_dir: Optional[str] = self.params.get("fixture_dir")
        self.loop_fixtures: bool = bool(self.params.get("loop_fixtures", True))
        # Test hook: after N successful frames, every read fails. Used to
        # exercise the health / degradation path without unplugging anything.
        # 0 disables it. Warm-up is unaffected — the counter is rewound after
        # open(), so open() always succeeds and the failures start in steady state.
        self.fail_after_frames: int = int(self.params.get("fail_after_frames", 0))
        self.pace: bool = bool(self.params.get("pace_to_fps", False))

        self._fixtures: List[Path] = []
        self._cursor = 0
        self._tick = 0
        self._steady_state = False
        self._last_emit: Optional[float] = None

        # Mock frames are synthetic and always well-formed; no warm-up needed.
        self.warmup_frames = int(self.params.get("warmup_frames", 0))

    # ------------------------------------------------------------------ #

    def _open(self) -> None:
        if self.mode == "fixtures":
            if not self.fixture_dir:
                raise CameraOpenError("mock: mode=fixtures requires camera_params.fixture_dir")
            d = Path(self.fixture_dir)
            if not d.is_dir():
                raise CameraOpenError(f"mock: fixture_dir not found: {d}")
            self._fixtures = sorted(
                p for p in d.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES
            )
            if not self._fixtures:
                raise CameraOpenError(f"mock: no image fixtures in {d}")
            logger.info("mock: replaying %d fixtures from %s", len(self._fixtures), d)
        elif self.mode != "synthetic":
            raise CameraOpenError(f"mock: unknown mode {self.mode!r} (synthetic|fixtures)")
        self._cursor = 0
        self._tick = 0
        self._steady_state = False

    def _read(self) -> np.ndarray:
        if self._steady_state and self.fail_after_frames and self._tick >= self.fail_after_frames:
            raise CameraReadError("mock: injected read failure (fail_after_frames)")

        if self.pace and self._last_emit is not None and self.fps > 0:
            gap = (1.0 / self.fps) - (time.monotonic() - self._last_emit)
            if gap > 0:
                time.sleep(gap)

        arr = self._next_fixture() if self.mode == "fixtures" else self._synthesise()
        self._tick += 1
        self._last_emit = time.monotonic()
        return arr

    def _after_warmup(self) -> None:
        # Warm-up reads consumed frames off the front of the reel. Rewind so a
        # replay always starts at the first fixture — CI tests assert on exact
        # frame order and shouldn't have to know the warm-up policy.
        self._cursor = 0
        self._tick = 0
        self._steady_state = True

    def _close(self) -> None:
        self._fixtures = []
        self._cursor = 0

    # ------------------------------------------------------------------ #

    def _synthesise(self) -> np.ndarray:
        """Vertical gradient background + a target that moves every frame, so a
        motion-based Tier1 filter has something real to fire on."""
        h, w = self.height, self.width
        img = np.zeros((h, w, 3), dtype=np.uint8)

        ramp = np.linspace(20, 90, h, dtype=np.uint8)[:, None]
        img[:, :, 1] = ramp                       # green-ish canopy backdrop
        img[:, :, 0] = (ramp // 3).astype(np.uint8)
        img[:, :, 2] = (ramp // 4).astype(np.uint8)

        box = max(8, min(h, w) // 8)
        travel = max(1, w - box)
        x = int((self._tick * max(1, travel // 40)) % travel)
        y = int(h // 2 - box // 2 + (h // 6) * np.sin(self._tick / 9.0))
        y = max(0, min(h - box, y))
        img[y:y + box, x:x + box] = (235, 180, 60)

        # low-amplitude noise so std() is never exactly zero (blank-detection)
        rng = np.random.default_rng(self._tick)
        noise = rng.integers(0, 6, size=(h, w, 3), dtype=np.uint8)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    def _next_fixture(self) -> np.ndarray:
        if self._cursor >= len(self._fixtures):
            if not self.loop_fixtures:
                raise CameraReadError("mock: fixture list exhausted (loop_fixtures=false)")
            self._cursor = 0
        path = self._fixtures[self._cursor]
        self._cursor += 1
        return _decode_rgb(path)

    def _capabilities(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "mode": self.mode,
            "backend": "numpy",
            "fixtures": len(self._fixtures) or None,
        }

    def _frame_metadata(self) -> Dict[str, Any]:
        if self.mode == "fixtures" and self._fixtures:
            idx = (self._cursor - 1) % len(self._fixtures)
            return {"fixture": self._fixtures[idx].name}
        return {"synthetic_tick": self._tick}


def _decode_rgb(path: Path) -> np.ndarray:
    """Decode an image file to canonical RGB uint8, via OpenCV or Pillow."""
    try:
        import cv2  # type: ignore
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise CameraReadError(f"mock: could not decode fixture {path}")
        return np.ascontiguousarray(bgr[:, :, ::-1])
    except ImportError:
        pass
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except ImportError as exc:  # pragma: no cover
        raise CameraOpenError(
            "mock: fixture replay needs opencv-python or Pillow installed"
        ) from exc