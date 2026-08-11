#!/usr/bin/env python3
"""Download the Tier1/Tier2 model artifacts.

The pipeline runs end-to-end with `backend: heuristic` and no model files at
all, which is how CI and Codespaces work. This script is what you run on a real
machine to swap in the actual models — the code path does not change.

    python scripts/fetch_models.py                 # all
    python scripts/fetch_models.py --only tier1
    python scripts/fetch_models.py --list

Then point config at them:

    ai:
      tier1:
        backend: tflite_ssd
        model_path: models/ssd_mobilenet_v2_coco_int8.tflite
        labels_path: models/coco_labels.txt
      tier2:
        backend: onnx
        model_path: models/efficientnet_b2.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models"

ARTIFACTS = {
    "tier1": {
        "filename": "ssd_mobilenet_v2_coco_int8.tflite",
        "url": ("https://storage.googleapis.com/download.tensorflow.org/models/"
                "tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"),
        "note": ("SSD-MobileNet COCO int8. Ships with TFLite_Detection_PostProcess "
                 "baked in, so no manual anchor decoding or NMS. Also the standard "
                 "Coral model, so the EdgeTPU path stays open."),
        "unzip_member": "detect.tflite",
    },
    "tier1_labels": {
        "filename": "coco_labels.txt",
        "url": None,
        "note": "Optional — a built-in COCO list is used when this is absent.",
    },
    "tier2": {
        "filename": "efficientnet_b2.onnx",
        "url": None,
        "note": ("EfficientNet-B2, 260x260. Export from torchvision:\n"
                 "  import torch, torchvision\n"
                 "  m = torchvision.models.efficientnet_b2(weights='DEFAULT').eval()\n"
                 "  torch.onnx.export(m, torch.randn(1,3,260,260), "
                 "'models/efficientnet_b2.onnx', opset_version=17)\n"
                 "Replace with the fine-tuned artifact once the retraining "
                 "pipeline exists (cloud/retraining/finetune.py)."),
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(key: str, spec: dict) -> int:
    target = MODEL_DIR / spec["filename"]
    if target.exists():
        print(f"OK    {key}: {target} already present ({sha256(target)[:12]})")
        return 0
    if not spec.get("url"):
        print(f"SKIP  {key}: no automatic download available")
        for line in spec["note"].splitlines():
            print(f"      {line}")
        return 0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"...   {key}: downloading {spec['url']}")
    try:
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(spec["url"], tmp)
        member = spec.get("unzip_member")
        if member:
            import zipfile
            with zipfile.ZipFile(tmp) as zf:
                target.write_bytes(zf.read(member))
            tmp.unlink()
        else:
            tmp.rename(target)
    except Exception as exc:
        print(f"FAIL  {key}: {exc}", file=sys.stderr)
        print("      The pipeline still runs with backend: heuristic.", file=sys.stderr)
        return 1
    print(f"OK    {key}: {target} ({sha256(target)[:12]})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Canopy Watch model artifacts")
    ap.add_argument("--only", choices=sorted(ARTIFACTS))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for key, spec in ARTIFACTS.items():
            state = "present" if (MODEL_DIR / spec["filename"]).exists() else "missing"
            print(f"{key:14} {spec['filename']:38} {state}")
        return 0

    keys = [args.only] if args.only else list(ARTIFACTS)
    return max(fetch(k, ARTIFACTS[k]) for k in keys)


if __name__ == "__main__":
    sys.exit(main())