#!/usr/bin/env python3
"""Run the full Canopy Watch pipeline over a video file.

    python scripts/analyze_video.py clip.mp4
    python scripts/analyze_video.py clip.mp4 --snapshots out/ --events out/events.jsonl
    python scripts/analyze_video.py clip.mp4 --realtime --motion-gate
    python scripts/analyze_video.py clip.mp4 --tier1 tflite_ssd \
        --tier1-model models/ssd_mobilenet_v2_coco_int8.tflite
    python scripts/analyze_video.py clip.mp4 --stride 3 --zone core --tz 5.5

The video driver satisfies the same `CameraDriver` contract as a webcam, so
capture, Tier1, Tier2 and scoring are byte-for-byte the code that runs in the
field. Nothing here is a special "offline mode".

Why this is the right way to evaluate detection quality: the same file produces
the same detections every run, so a threshold change is attributable. A live
camera pointed at a garden is not a repeatable test.

Exits 0 when the clip is fully processed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging  # noqa: E402

from edge.bus import InProcessBus            # noqa: E402
from edge.pipeline import TOPIC_SCORED, EdgePipeline   # noqa: E402


def build_config(args: argparse.Namespace) -> dict:
    tier1 = {
        "backend": args.tier1,
        "score_threshold": args.threshold,
        "motion_gate": {
            "enabled": args.motion_gate,
            "warmup_frames": 10,
            "min_area_fraction": args.motion_area,
        },
    }
    if args.tier1_model:
        tier1["model_path"] = args.tier1_model
    if args.tier1_labels:
        tier1["labels_path"] = args.tier1_labels

    tier2 = {"backend": args.tier2, "mode": args.tier2_mode, "input_size": 260}
    if args.tier2_model:
        tier2["model_path"] = args.tier2_model

    return {
        "identity": {"node_id": args.node_id, "site_id": args.site, "zone": args.zone},
        "hardware": {
            "driver": {"camera": "video"},
            "camera_params": {
                "path": str(args.video),
                "realtime": args.realtime,
                "speed": args.speed,
                "frame_stride": args.stride,
                "start_frame": args.start_frame,
                "end_frame": args.end_frame,
                "loop": False,           # finite on purpose: we want it to END
                "width": args.width,
                "height": args.height,
            },
        },
        "capture": {
            "buffer_seconds": 15,
            "buffer_max_memory_mb": args.buffer_mb,
            "tier1_sample_interval_ms": args.sample_ms,
            "tier1_queue_depth": 1,
            # Sample on the CLIP's timeline, not the wall clock — otherwise an
            # 8s clip finishing in 1.4s of real time gets sampled 7 times.
            "tier1_sample_clock": args.sample_clock,
            # Block rather than drop: a file will wait, so losing frames here
            # would only make the run non-reproducible for no benefit.
            "tier1_backpressure": args.backpressure,
        },
        "ai": {"accelerator": args.accelerator, "tier1": tier1, "tier2": tier2},
        "scoring": {"persistence": {"window_seconds": 600}},
        "snapshots": {
            "enabled": bool(args.snapshots),
            "directory": args.snapshots or "snapshots",
            "mode": "crop",
            "min_band": args.snapshot_min_band,
            "max_files": args.snapshot_max_files,
        },
        "pipeline": {
            "tier2_queue_depth": 32,
            "validate_events": True,
            "utc_offset_hours": args.tz,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze a video file with Canopy Watch")
    ap.add_argument("video", type=Path)

    ap.add_argument("--events", type=Path, help="write events as JSONL to this path")
    ap.add_argument("--snapshots", help="directory for detection crops (enables them)")
    ap.add_argument("--snapshot-min-band", default="low",
                    choices=["low", "medium", "high", "critical"])
    ap.add_argument("--snapshot-max-files", type=int, default=500)

    ap.add_argument("--tier1", default="heuristic", choices=["heuristic", "tflite_ssd"])
    ap.add_argument("--tier1-model")
    ap.add_argument("--tier1-labels")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--tier2", default="heuristic", choices=["heuristic", "onnx"])
    ap.add_argument("--tier2-model")
    ap.add_argument("--tier2-mode", default="local",
                    choices=["local", "defer_to_cloud", "disabled"])
    ap.add_argument("--accelerator", default="cpu",
                    choices=["cpu", "cuda", "tensorrt", "edgetpu"])

    ap.add_argument("--motion-gate", action="store_true",
                    help="enable the motion pre-filter (off by default, as on laptop)")
    ap.add_argument("--motion-area", type=float, default=0.002)

    ap.add_argument("--realtime", action="store_true",
                    help="pace to the file's fps instead of running flat out")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int)
    ap.add_argument("--width", type=int, help="downscale on read")
    ap.add_argument("--height", type=int)
    ap.add_argument("--sample-ms", type=int, default=200,
                    help="Tier1 sampling interval in ms of wall clock")
    ap.add_argument("--buffer-mb", type=int, default=256)
    ap.add_argument("--sample-clock", default="media", choices=["media", "wall", "auto"],
                    help="gate sampling on clip time (default) or wall clock")
    ap.add_argument("--backpressure", default="block", choices=["block", "drop_oldest"],
                    help="block = lose nothing, reproducible; drop_oldest = live behaviour")

    ap.add_argument("--node-id", default="video-analysis-01")
    ap.add_argument("--site", default="offline")
    ap.add_argument("--zone", default="core", choices=["core", "buffer", "periphery"])
    ap.add_argument("--tz", type=float, default=0.0, help="node UTC offset in hours")

    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"no such file: {args.video}", file=sys.stderr)
        return 2

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else (
            logging.WARNING if args.quiet else logging.INFO),
        format="%(levelname)-7s %(name)s: %(message)s")

    events_file = None
    if args.events:
        args.events.parent.mkdir(parents=True, exist_ok=True)
        events_file = args.events.open("w")

    collected = []

    def on_event(topic, event):
        payload = event.to_dict()
        collected.append(payload)
        if events_file:
            events_file.write(json.dumps(payload) + "\n")
        if args.quiet:
            return
        risk = payload["risk"]
        top = max(risk["factors"], key=lambda f: f["contribution"])
        media = payload.get("media_timestamp")
        stamp = f"{media:7.2f}s" if media is not None else "   --  "
        print(f"{stamp}  [{risk['band'].upper():8}] {risk['score']:5.1f}  "
              f"{payload['category']:8} conf={payload['confidence']:.2f}  "
              f"{payload['frame_id']}  top={top['name']}")

    bus = InProcessBus()
    bus.subscribe(TOPIC_SCORED, on_event, name="console")

    pipeline = EdgePipeline.from_config(build_config(args), bus=bus)
    started = time.monotonic()
    pipeline.start()

    caps = pipeline.capture.driver.get_capabilities()
    if not args.quiet:
        print(f"\n{args.video.name}: {caps.get('total_frames')} frames @ "
              f"{caps.get('fps')} fps ({caps.get('duration_s')}s), "
              f"tier1={args.tier1} tier2={args.tier2}/{args.tier2_mode} "
              f"motion_gate={'on' if args.motion_gate else 'off'}\n")

    completed = pipeline.wait_until_complete(timeout=args.timeout)
    elapsed = time.monotonic() - started
    stats = pipeline.stats()
    pipeline.stop()
    if events_file:
        events_file.close()

    if not completed:
        print(f"\ntimed out after {args.timeout}s — clip not fully processed",
              file=sys.stderr)

    print(f"\n--- summary ---")
    print(f"wall clock         {elapsed:.1f}s")
    print(f"frames captured    {stats['capture']['frames_captured']}")
    print(f"frames to tier1    {stats['frames_examined']}")
    dropped = stats["capture"]["tier1"]["frames_dropped_backpressure"]
    if dropped:
        print(f"frames dropped     {dropped} (use --backpressure block to keep all)")
    print(f"motion suppressed  {stats['motion_suppressed']}")
    print(f"candidates         {stats['candidates']}")
    print(f"events             {stats['events_emitted']}")
    if collected:
        bands = {}
        for payload in collected:
            bands[payload["risk"]["band"]] = bands.get(payload["risk"]["band"], 0) + 1
        print(f"risk bands         {bands}")
        print(f"peak score         {max(p['risk']['score'] for p in collected):.1f}")
    if stats["snapshots"]["enabled"]:
        print(f"snapshots          {stats['snapshots']['written']} "
              f"-> {stats['snapshots']['directory']}")
    if args.events:
        print(f"events written     {args.events}")
    if stats["buffer_misses"]:
        print(f"buffer misses      {stats['buffer_misses']} "
              f"(buffer undersized for the capture rate)")
    if stats["tier2_queue"]["tasks_dropped_overload"]:
        print(f"tier2 shed         {stats['tier2_queue']['tasks_dropped_overload']} "
              f"(tier2 undersized for the detection rate)")

    return 0 if completed else 1


if __name__ == "__main__":
    sys.exit(main())