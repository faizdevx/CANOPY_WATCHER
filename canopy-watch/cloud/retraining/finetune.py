"""Tier2 fine-tuning pipeline — NOT IMPLEMENTED, deliberately.

Documented here so the shape is agreed before anyone writes it, and so the edge
side can be built against the interface it will eventually produce.

The loop:

    edge detections
        -> Data Lake bronze  (raw events + snapshot refs, nothing dropped)
        -> silver            (deduplicated, joined to snapshots, bad crops removed)
        -> gold              (human-labelled, class-balanced, train/val/test split
                              by CAMERA and by DATE — random splits leak, because
                              consecutive frames of one animal are near-duplicates
                              and would appear in both train and val)
        -> train             EfficientNet-B2, ImageNet init, last-block unfreeze
        -> evaluate          against the FROZEN test split of the previous model,
                             never a freshly drawn one
        -> gate              promote only if it beats the incumbent on the metrics
                             that matter here: recall on `human` at fixed FPR,
                             plus calibration (ECE) — an overconfident model
                             corrupts every downstream risk score
        -> export            ONNX, opset pinned, then TensorRT engine per device
                             class (an engine is not portable across JetPack
                             versions or GPU architectures — build per target)
        -> sign              artifact + SHA + label map, signed
        -> Model Registry    versioned; edge pulls on next connectivity window

What already exists on the edge side, so none of the above touches edge code:

  * `tier2.model_path` / `model_version` / `labels_path` come from config
  * `Tier2Verifier.class_map` projects model labels onto the four categories
  * `model_version` travels inside every DetectionEvent, so a regression can be
    attributed to a specific artifact after the fact

Open questions to settle before implementing:

  * Where do labels come from? Ranger review UI, or an external annotation
    vendor? This decides the whole silver->gold stage.
  * Federated variant: nodes contribute gradients rather than raw crops. Lower
    bandwidth and better privacy, considerably harder to debug. Not for v1.
  * Rollout strategy: canary per node group before fleet-wide, which needs the
    fleet-management service that also does not exist yet.
"""

from __future__ import annotations

from typing import Any, Dict


def build_gold_dataset(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("retraining pipeline is deferred — see module docstring")


def train_tier2(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("retraining pipeline is deferred — see module docstring")


def evaluate_and_gate(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    raise NotImplementedError("retraining pipeline is deferred — see module docstring")


def export_and_publish(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError("retraining pipeline is deferred — see module docstring")