#!/usr/bin/env python3
"""Generate a synthetic test clip, so the video path can be exercised without
hunting for real footage.

    python scripts/make_test_video.py                          # tests/fixtures/test_clip.mp4
    python scripts/make_test_video.py --seconds 20 --fps 30 --out /tmp/clip.mp4
    python scripts/make_test_video.py --empty                  # no subject at all

The clip is a bright subject crossing a dark textured background, entering at
~20% and leaving at ~80% of the duration. That gives three distinguishable
phases to assert against: quiet, active, quiet. `--empty` produces background
only, which is the negative control — a pipeline that reports detections on it
is producing false positives, and you want to know that number.

Real footage is better. This exists so the plumbing is testable today and in CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("needs opencv-python (pip install opencv-python)", file=sys.stderr)
    raise SystemExit(2)


def background(width: int, height: int, tick: int, rng: np.random.Generator) -> np.ndarray:
    """Dark canopy-ish gradient with mild noise and slow swaying texture.

    The sway matters: a perfectly static background makes the motion gate look
    better than it is on real footage, where leaves move constantly.
    """
    ramp = np.linspace(18, 70, height, dtype=np.float32)[:, None]
    img = np.zeros((height, width, 3), dtype=np.float32)
    img[:, :, 1] = ramp
    img[:, :, 0] = ramp / 3.0
    img[:, :, 2] = ramp / 4.0

    xs = np.arange(width, dtype=np.float32)
    sway = 6.0 * np.sin(xs / 23.0 + tick / 11.0)
    img += sway[None, :, None]
    img += rng.integers(0, 7, size=(height, width, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description="Make a synthetic Canopy Watch test clip")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "tests" / "fixtures" / "test_clip.mp4")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--empty", action="store_true",
                    help="background only — the false-positive control")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    total = int(args.seconds * args.fps)
    enter, leave = int(total * 0.2), int(total * 0.8)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (args.width, args.height))
    if not writer.isOpened():
        print(f"could not open {args.out} for writing", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    size = max(24, min(args.width, args.height) // 7)

    for tick in range(total):
        frame = background(args.width, args.height, tick, rng)

        if not args.empty and enter <= tick < leave:
            progress = (tick - enter) / max(1, leave - enter)
            x = int(progress * (args.width - size))
            y = int(args.height * 0.45 + args.height * 0.10 * np.sin(progress * 6.0))
            y = max(0, min(args.height - size, y))
            frame[y:y + size, x:x + size] = (245, 200, 90)
            # softer core so the crop has internal structure, not a flat block
            inner = size // 4
            frame[y + inner:y + size - inner, x + inner:x + size - inner] = (255, 240, 170)

        writer.write(frame[:, :, ::-1])          # RGB -> BGR for OpenCV

    writer.release()
    kind = "background only" if args.empty else f"subject visible frames {enter}-{leave}"
    print(f"wrote {args.out} — {total} frames @ {args.fps} fps "
          f"({args.seconds:.1f}s), {kind}")
    return 0


if __name__ == "__main__":
    sys.exit(main())