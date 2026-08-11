"""SSD-MobileNetV2 COCO int8 TFLite — `tier1.backend: tflite_ssd`.

The default *real* Tier1. Chosen over a YOLO export for one concrete reason:
this graph ships with `TFLite_Detection_PostProcess` baked in, so `invoke()`
returns four ready tensors — boxes (already normalized), classes, scores, count.
A YOLOv8n TFLite export typically hands back a raw `[1, 84, 8400]` tensor and
leaves anchor decoding and NMS to you, which is exactly the code that silently
produces subtly wrong boxes when the stride maths is off.

It is also the standard Coral model, so the EdgeTPU path stays open: point
`delegate: libedgetpu.so.1` at it and the same class runs on a Coral stick.

Runtime is imported lazily, `tflite_runtime` first (small, what you install on a
Pi) then TensorFlow's bundled interpreter (what a laptop usually already has).
Neither is imported at module scope, so the registry stays importable on a
machine with no ML stack at all.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..types import BBox, Candidate
from .base import Tier1Detector, Tier1Error

logger = logging.getLogger(__name__)


class TFLiteSSDTier1(Tier1Detector):
    name = "tflite_ssd"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        self.model_path: Optional[str] = self.params.get("model_path")
        self.labels_path: Optional[str] = self.params.get("labels_path")
        self.num_threads: int = int(self.params.get("num_threads", 2))
        self.delegate: Optional[str] = self.params.get("delegate")   # e.g. EdgeTPU
        self.input_size: Optional[int] = self.params.get("input_size")

        self._interpreter = None
        self._input_detail: Dict[str, Any] = {}
        self._output_details: List[Dict[str, Any]] = []
        self._labels: List[str] = []
        self._quantized_input = False

    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if not self.model_path:
            raise Tier1Error("tflite_ssd: tier1.model_path is not set")
        path = Path(self.model_path)
        if not path.exists():
            raise Tier1Error(
                f"tflite_ssd: model not found at {path}. "
                f"Run `python scripts/fetch_models.py` to download it, or switch "
                f"to tier1.backend: heuristic for a model-free pipeline."
            )

        interpreter_cls, delegate_loader = _load_runtime()
        kwargs: Dict[str, Any] = {"model_path": str(path), "num_threads": self.num_threads}
        if self.delegate:
            if delegate_loader is None:
                raise Tier1Error("tflite_ssd: this runtime cannot load delegates")
            kwargs["experimental_delegates"] = [delegate_loader(self.delegate, {})]
            kwargs.pop("num_threads", None)

        try:
            interpreter = interpreter_cls(**kwargs)
            interpreter.allocate_tensors()
        except Exception as exc:
            raise Tier1Error(f"tflite_ssd: failed to load {path}: {exc}") from exc

        self._interpreter = interpreter
        self._input_detail = interpreter.get_input_details()[0]
        self._output_details = interpreter.get_output_details()
        self._quantized_input = self._input_detail["dtype"] == np.uint8

        shape = self._input_detail["shape"]
        self.input_size = int(shape[1])
        if len(self._output_details) < 4:
            raise Tier1Error(
                "tflite_ssd: this graph has fewer than 4 outputs — it does not "
                "contain TFLite_Detection_PostProcess. Use a detection-postprocess "
                "SSD export, or the tflite_yolo backend for raw-tensor models."
            )

        self._labels = _load_labels(self.labels_path)
        self.model_version = self.model_version or path.stem

    def _detect(self, image: np.ndarray) -> List[Candidate]:
        interpreter = self._interpreter
        if interpreter is None:
            raise Tier1Error("tflite_ssd: interpreter is not loaded")

        tensor = self._preprocess(image)
        interpreter.set_tensor(self._input_detail["index"], tensor)
        interpreter.invoke()

        boxes, class_ids, scores, count = self._read_outputs(interpreter)

        candidates: List[Candidate] = []
        for i in range(count):
            score = float(scores[i])
            # ymin, xmin, ymax, xmax — SSD's order, not xyxy.
            ymin, xmin, ymax, xmax = (float(v) for v in boxes[i])
            try:
                bbox = BBox(xmin, ymin, xmax, ymax).clamped()
            except ValueError:
                continue
            label = self._label_for(int(class_ids[i]))
            candidates.append(Candidate(
                category=self.default_category,
                confidence=score,
                bbox=bbox,
                label=label,
                metadata={"class_id": int(class_ids[i])},
            ))
        return candidates

    # ------------------------------------------------------------------ #

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        from ..crop import letterbox
        size = int(self.input_size or 300)
        resized = letterbox(image, size)
        if self._quantized_input:
            return np.expand_dims(resized.astype(np.uint8), axis=0)
        scale, zero = self._input_detail.get("quantization", (0.0, 0))
        arr = resized.astype(np.float32)
        if scale:
            arr = arr / 255.0 / scale + zero
        else:
            arr = (arr - 127.5) / 127.5
        return np.expand_dims(arr.astype(self._input_detail["dtype"]), axis=0)

    def _read_outputs(self, interpreter):
        """Output ORDER varies between exports. Identify tensors by shape rather
        than trusting index 0..3 — getting this wrong yields boxes that look
        plausible and are entirely wrong."""
        tensors = [interpreter.get_tensor(d["index"]) for d in self._output_details]
        boxes = class_ids = scores = None
        count = 0
        for arr in tensors:
            squeezed = np.squeeze(arr)
            if squeezed.ndim == 2 and squeezed.shape[-1] == 4:
                boxes = squeezed
            elif squeezed.ndim == 0:
                count = int(squeezed)
            elif squeezed.ndim == 1:
                if scores is None and squeezed.dtype.kind == "f" and squeezed.max(initial=0) <= 1.0:
                    scores = squeezed
                elif class_ids is None:
                    class_ids = squeezed
                else:
                    scores = squeezed
        if boxes is None or scores is None:
            raise Tier1Error("tflite_ssd: could not identify box/score tensors")
        if class_ids is None:
            class_ids = np.zeros(len(scores))
        if not count:
            count = len(scores)
        return boxes, class_ids, scores, min(int(count), len(scores), len(boxes))

    def _label_for(self, class_id: int) -> Optional[str]:
        if 0 <= class_id < len(self._labels):
            return self._labels[class_id]
        return None

    def _describe(self) -> Dict[str, Any]:
        return {
            "model_path": self.model_path,
            "input_size": self.input_size,
            "quantized": self._quantized_input,
            "labels": len(self._labels),
            "delegate": self.delegate,
            "requires_model_file": True,
        }


def _load_runtime():
    """tflite_runtime first (what you install on a Pi), then TensorFlow."""
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate  # type: ignore
        return Interpreter, load_delegate
    except ImportError:
        pass
    try:
        import tensorflow as tf  # type: ignore
        return tf.lite.Interpreter, getattr(tf.lite.experimental, "load_delegate", None)
    except ImportError as exc:
        raise Tier1Error(
            "tflite_ssd: no TFLite runtime. Install `tflite-runtime` (Pi/Jetson) "
            "or `tensorflow` (laptop), or use tier1.backend: heuristic."
        ) from exc


def _load_labels(path: Optional[str]) -> List[str]:
    if not path:
        return list(COCO_LABELS)
    p = Path(path)
    if not p.exists():
        logger.warning("tflite_ssd: labels file %s not found — using built-in COCO list", p)
        return list(COCO_LABELS)
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    # Some label files are "0  person" per line.
    return [ln.split(maxsplit=1)[-1] if ln[0].isdigit() else ln for ln in lines]


#: COCO in the 90-slot SSD indexing (blanks are the retired class ids).
COCO_LABELS: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "???", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "???", "backpack", "umbrella", "???", "???", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "???", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "???",
    "dining table", "???", "???", "toilet", "???", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "???", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]