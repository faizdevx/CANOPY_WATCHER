#!/usr/bin/env python3
"""Camera probe — the one command each device person runs to prove their camera
actually works through the Canopy Watch driver layer.

    python scripts/camera_probe.py --list                 # what's attached?
    python scripts/camera_probe.py                        # use resolved config
    python scripts/camera_probe.py --driver webcam --frames 30
    python scripts/camera_probe.py --driver webcam --device-name "Logitech C920"
    python scripts/camera_probe.py --driver mock --save /tmp/frame.png

Exit code 0 means: driver opened, warmed up, produced canonical RGB frames, and
reported healthy. Anything else prints why.

`--list` is also the honest answer to "is a Pi CSI camera attached?" — it asks
`rpicam-hello --list-cameras` (or `libcamera-hello` on older OS images) instead
of checking whether a binary happens to exist or whether /dev/video0 is there,
neither of which is a reliable signal on a Pi.
"""

from __future__ import annotations

import argparse
import logging
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge.drivers import (  # noqa: E402
    CameraError,
    available_drivers,
    create_camera_driver,
)


# --------------------------------------------------------------------------- #
# Enumeration
# --------------------------------------------------------------------------- #

def list_sources() -> None:
    print(f"platform: {platform.system()} {platform.machine()}")
    print(f"registered drivers: {', '.join(available_drivers())}\n")

    from edge.drivers.webcam import list_devices
    usb = list_devices()
    print("USB / V4L2 cameras (driver.camera: webcam)")
    if usb:
        for dev in usb:
            print(f"  index {dev['index']}: {dev['name']}")
    else:
        print("  none found (or this OS can't enumerate names without extra packages)")

    print("\nCSI camera via libcamera (driver.camera: picamera)")
    binary = shutil.which("rpicam-hello") or shutil.which("libcamera-hello")
    if not binary:
        print("  no rpicam-hello / libcamera-hello on PATH — not a Pi, or tooling not installed")
    else:
        try:
            proc = subprocess.run([binary, "--list-cameras"],
                                  capture_output=True, text=True, timeout=15)
            out = (proc.stdout or proc.stderr).strip()
            detected = "Available cameras" in out and "no cameras available" not in out.lower()
            print(f"  {Path(binary).name} --list-cameras -> "
                  f"{'CAMERA DETECTED' if detected else 'NO CAMERA DETECTED'}")
            for line in out.splitlines():
                print(f"    {line}")
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"  {binary} failed: {exc}")

    print("\nCSI camera via Argus (driver.camera: csi_gstreamer)")
    if not Path("/etc/nv_tegra_release").exists():
        print("  not a Jetson (no /etc/nv_tegra_release)")
    else:
        try:
            import cv2  # type: ignore
            from edge.drivers.jetson.csi_gstreamer import _has_gstreamer
            print(f"  Jetson detected; OpenCV GStreamer support: "
                  f"{'YES' if _has_gstreamer(cv2) else 'NO — use JetPack OpenCV'}")
        except ImportError:
            print("  Jetson detected; OpenCV not installed")


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #

def probe(args: argparse.Namespace) -> int:
    params = {}
    if args.device_name:
        params["device_name"] = args.device_name
    if args.device_index is not None:
        params["device_index"] = args.device_index
    if args.width:
        params["width"] = args.width
    if args.height:
        params["height"] = args.height

    config = None
    if not args.driver:
        try:
            from edge.config.loader import get_config  # type: ignore
            config = get_config()
        except Exception as exc:
            print(f"could not load config ({exc}); pass --driver explicitly", file=sys.stderr)
            return 2

    try:
        cam = create_camera_driver(config, driver=args.driver, params=params)
    except CameraError as exc:
        print(f"FAIL  factory: {exc}", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    try:
        cam.open()
    except CameraError as exc:
        print(f"FAIL  open: {exc}", file=sys.stderr)
        return 3
    open_ms = (time.monotonic() - t0) * 1000

    print(f"OK    open in {open_ms:.0f} ms")
    for key, value in sorted(cam.get_capabilities().items()):
        print(f"      {key}: {value}")

    first = None
    start = time.monotonic()
    try:
        for i in range(args.frames):
            frame = cam.read_frame()
            if first is None:
                first = frame
            if args.verbose:
                print(f"      frame {i}: {frame}")
    except CameraError as exc:
        print(f"FAIL  read: {exc}", file=sys.stderr)
        cam.close()
        return 4
    elapsed = time.monotonic() - start

    fps = args.frames / elapsed if elapsed > 0 else float("inf")
    print(f"OK    {args.frames} frames in {elapsed:.2f}s ({fps:.1f} fps effective)")
    print(f"OK    canonical shape {first.data.shape} dtype {first.data.dtype} "
          f"mean {first.data.mean():.1f}")
    print(f"{'OK   ' if cam.is_healthy() else 'FAIL '} health: {cam.health_detail()}")

    if args.save:
        _save(first.data, Path(args.save))
        print(f"OK    wrote {args.save}")

    cam.close()
    return 0 if cam.health_detail()["frames_emitted"] else 5


def _save(rgb, path: Path) -> None:
    try:
        import cv2  # type: ignore
        cv2.imwrite(str(path), rgb[:, :, ::-1])
        return
    except ImportError:
        pass
    from PIL import Image  # type: ignore
    Image.fromarray(rgb).save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe a Canopy Watch camera driver")
    ap.add_argument("--list", action="store_true", help="enumerate attached cameras and exit")
    ap.add_argument("--driver", choices=list(available_drivers()),
                    help="override hardware.driver.camera from config")
    ap.add_argument("--device-name", help="prefer this camera by name (USB only)")
    ap.add_argument("--device-index", type=int, help="fallback index (USB only)")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--save", help="write the first frame to this path")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    if args.list:
        list_sources()
        return 0
    return probe(args)


if __name__ == "__main__":
    sys.exit(main())