#!/usr/bin/env python3
"""Run the full edge pipeline and print scored detections as they arrive.

    python scripts/run_edge.py --profile laptop
    python scripts/run_edge.py --demo              # no config file needed
    python scripts/run_edge.py --demo --json       # full event payloads

Ctrl-C prints a final stats block for every stage.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge.bus import InProcessBus          # noqa: E402
from edge.pipeline import TOPIC_SCORED, EdgePipeline   # noqa: E402

DEMO_CONFIG = {
    "identity": {"node_id": "dev-demo-01", "site_id": "demo", "zone": "core"},
    "hardware": {"driver": {"camera": "mock"},
                 "camera_params": {"width": 640, "height": 480}},
    "capture": {"buffer_seconds": 10, "buffer_max_memory_mb": 256,
                "tier1_sample_interval_ms": 200},
    "ai": {"accelerator": "cpu",
           "tier1": {"backend": "heuristic", "score_threshold": 0.3,
                     "motion_gate": {"enabled": False}},
           "tier2": {"backend": "heuristic", "mode": "local", "input_size": 260}},
    "scoring": {},
    "pipeline": {"tier2_queue_depth": 32},
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Canopy Watch edge pipeline")
    ap.add_argument("--profile", help="CW_PROFILE to load via edge.config.loader")
    ap.add_argument("--demo", action="store_true",
                    help="use a built-in mock-camera config, no yaml needed")
    ap.add_argument("--json", action="store_true", help="print full event payloads")
    ap.add_argument("--seconds", type=float, default=0, help="0 = run until Ctrl-C")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)-7s %(name)s: %(message)s")

    if args.demo:
        config = DEMO_CONFIG
    else:
        try:
            from edge.config.loader import get_config
            config = get_config()
        except Exception as exc:
            print(f"could not load config ({exc}); try --demo", file=sys.stderr)
            return 2

    bus = InProcessBus()

    def on_event(topic, event):
        payload = event.to_dict()
        if args.json:
            print(json.dumps(payload, indent=2))
            return
        risk = payload["risk"]
        top = max(risk["factors"], key=lambda f: f["contribution"])
        print(f"[{risk['band'].upper():8}] {risk['score']:5.1f}  "
              f"{payload['category']:8} conf={payload['confidence']:.2f}  "
              f"{payload['frame_id']}  tier2={payload['tier2_status']}  "
              f"top factor: {top['name']} ({top['contribution']:.1f})")

    bus.subscribe(TOPIC_SCORED, on_event, name="console")

    pipeline = EdgePipeline.from_config(config, bus=bus)
    stopping = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stopping.update(flag=True))

    pipeline.start()
    print("running — Ctrl-C to stop\n")
    deadline = time.monotonic() + args.seconds if args.seconds else None
    try:
        while not stopping["flag"]:
            if deadline and time.monotonic() > deadline:
                break
            time.sleep(0.1)
    finally:
        stats = pipeline.stats()
        pipeline.stop()
        print("\n--- stats ---")
        print(json.dumps(stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())