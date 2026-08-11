"""Tier2 verifier contract.

Tier2 is a **classifier**, not a detector. It receives a crop that Tier1 already
localised, so it emits class + confidence and the bbox passes straight through.
Target model is EfficientNet-B2 (260x260 input), fine-tuned on park data later.

Whether Tier2 runs at all is a per-profile config decision, never autodetected:

    ai.tier2.mode: local          run here (laptop, Jetson)
                   defer_to_cloud  Tier1 result ships, cloud verifies (Pi)
                   disabled        Tier1 result ships as final

The pipeline shape does not change between profiles — only which node is active
and what `tier2_status` the event carries.
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..types import CATEGORIES, Candidate, Verification

logger = logging.getLogger(__name__)


class Tier2Error(Exception):
    """Tier2 backend could not load or run."""


class Tier2Verifier(abc.ABC):
    name: str = "unset"
    input_size: int = 260                 # EfficientNet-B2

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params: Dict[str, Any] = dict(params or {})
        self.input_size = int(self.params.get("input_size", self.input_size))
        self.model_version: Optional[str] = self.params.get("model_version")
        self.min_confidence: float = float(self.params.get("min_confidence", 0.0))
        #: label -> category, same projection trick as Tier1's class_map
        self.class_map: Dict[str, str] = dict(self.params.get("class_map") or {})
        self.default_category: str = str(self.params.get("default_category", "unknown"))

        self._loaded = False
        self.invocations = 0
        self.total_latency_ms = 0.0
        self.failures = 0

    # ------------------------------------------------------------------ #

    def load(self) -> None:
        if self._loaded:
            return
        self._load()
        self._loaded = True
        logger.info("tier2: %s ready (%s)", self.name, self.describe())

    def verify(self, crop: np.ndarray, candidate: Candidate) -> Verification:
        if not self._loaded:
            self.load()
        started = time.monotonic()
        result = self._verify(crop, candidate)
        latency_ms = (time.monotonic() - started) * 1000
        self.invocations += 1
        self.total_latency_ms += latency_ms

        category = self.map_category(result.label, result.category)
        return Verification(
            category=category,
            confidence=float(result.confidence),
            label=result.label,
            backend=self.name,
            model_version=self.model_version,
            latency_ms=latency_ms,
            metadata=result.metadata,
        )

    def map_category(self, label: Optional[str], fallback: Optional[str]) -> str:
        if label and label in self.class_map:
            return self.class_map[label]
        if fallback in CATEGORIES:
            return fallback
        return self.default_category

    def close(self) -> None:
        self._loaded = False

    # ------------------------------------------------------------------ #

    def describe(self) -> Dict[str, Any]:
        info = {
            "backend": self.name,
            "model_version": self.model_version,
            "input_size": self.input_size,
            "loaded": self._loaded,
        }
        info.update(self._describe())
        return info

    def stats(self) -> Dict[str, Any]:
        avg = (self.total_latency_ms / self.invocations) if self.invocations else None
        return {
            "backend": self.name,
            "invocations": self.invocations,
            "failures": self.failures,
            "avg_latency_ms": None if avg is None else round(avg, 2),
        }

    @abc.abstractmethod
    def _load(self) -> None:
        ...

    @abc.abstractmethod
    def _verify(self, crop: np.ndarray, candidate: Candidate) -> Verification:
        ...

    def _describe(self) -> Dict[str, Any]:
        return {}


# --------------------------------------------------------------------------- #
# Heuristic — no model file, deterministic
# --------------------------------------------------------------------------- #

class HeuristicTier2(Tier2Verifier):
    """`tier2.backend: heuristic`.

    Runs the whole Tier2 code path — crop in, verification out, latency
    measured, category mapped — with numpy statistics standing in for a network.
    It is **deterministic**: the same crop always produces the same output, so
    pipeline tests assert on real values rather than mocking the stage away.

    It is not a classifier and does not pretend to be. `label` is always None
    and the category is inherited from Tier1; what it contributes is a
    confidence adjustment based on how structured the crop is (a crop that is
    flat, or almost entirely padding, is a weak detection). Honest-weak beats
    fake-precise.
    """

    name = "heuristic"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        self.model_version = self.model_version or "heuristic-1"
        self.pad_value: int = int(self.params.get("pad_value", 114))

    def _load(self) -> None:
        return None

    def _verify(self, crop: np.ndarray, candidate: Candidate) -> Verification:
        gray = crop.mean(axis=2)
        contrast = float(gray.std()) / 64.0
        # Letterbox padding carries no information — a crop that is mostly pad
        # means Tier1 boxed something at the very edge of the frame.
        padding = float(np.mean(np.abs(gray - self.pad_value) < 2.0))
        structure = float(np.clip(contrast * (1.0 - padding), 0.0, 1.0))

        # Agreement model: Tier2 pulls Tier1's confidence toward what the crop
        # actually supports, rather than inventing a number.
        confidence = float(np.clip(0.5 * candidate.confidence + 0.5 * structure, 0.01, 0.99))

        return Verification(
            category=candidate.category,
            confidence=confidence,
            label=None,
            metadata={"structure": round(structure, 4), "padding_fraction": round(padding, 4)},
        )

    def _describe(self) -> Dict[str, Any]:
        return {"requires_model_file": False, "classifies": False}


# --------------------------------------------------------------------------- #
# ONNX Runtime — laptop CPU/CUDA, and the Jetson path until TensorRT lands
# --------------------------------------------------------------------------- #

class OnnxTier2(Tier2Verifier):
    """`tier2.backend: onnx`. EfficientNet-B2 or any image classifier.

    Execution providers come from `ai.accelerator`, so the same class covers
    laptop CPU, laptop CUDA and Jetson — the config already carries that field
    and no code branches on device name.

    TensorRT: onnxruntime on JetPack exposes `TensorrtExecutionProvider`, so
    `accelerator: tensorrt` is handled here too. A native .engine loader can be
    added later as a separate backend behind this same interface.
    """

    name = "onnx"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        self.model_path: Optional[str] = self.params.get("model_path")
        self.labels_path: Optional[str] = self.params.get("labels_path")
        self.accelerator: str = str(self.params.get("accelerator", "cpu"))
        self.mean = tuple(self.params.get("mean", (0.485, 0.456, 0.406)))
        self.std = tuple(self.params.get("std", (0.229, 0.224, 0.225)))
        self.channels_first: bool = bool(self.params.get("channels_first", True))

        self._session = None
        self._input_name: Optional[str] = None
        self._labels: List[str] = []

    def _load(self) -> None:
        if not self.model_path:
            raise Tier2Error("onnx: tier2.model_path is not set")
        from pathlib import Path
        path = Path(self.model_path)
        if not path.exists():
            raise Tier2Error(
                f"onnx: model not found at {path}. Run "
                f"`python scripts/fetch_models.py`, or set tier2.backend: heuristic."
            )
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise Tier2Error(
                "onnx: onnxruntime is not installed "
                "(pip install onnxruntime, or onnxruntime-gpu on CUDA hosts)"
            ) from exc

        providers = _providers_for(self.accelerator, ort.get_available_providers())
        try:
            self._session = ort.InferenceSession(str(path), providers=providers)
        except Exception as exc:
            raise Tier2Error(f"onnx: failed to load {path}: {exc}") from exc

        self._input_name = self._session.get_inputs()[0].name
        shape = self._session.get_inputs()[0].shape
        for dim in shape:
            if isinstance(dim, int) and dim > 8:
                self.input_size = dim
        self._labels = _read_labels(self.labels_path)
        self.model_version = self.model_version or path.stem
        logger.info("onnx: providers=%s", self._session.get_providers())

    def _verify(self, crop: np.ndarray, candidate: Candidate) -> Verification:
        from ..crop import normalize_for_model

        tensor = normalize_for_model(
            crop, mean=self.mean, std=self.std, channels_first=self.channels_first)
        batch = np.expand_dims(tensor, axis=0).astype(np.float32)

        try:
            outputs = self._session.run(None, {self._input_name: batch})
        except Exception as exc:
            self.failures += 1
            raise Tier2Error(f"onnx: inference failed: {exc}") from exc

        logits = np.squeeze(outputs[0])
        probs = _softmax(logits)
        index = int(np.argmax(probs))
        label = self._labels[index] if index < len(self._labels) else str(index)

        return Verification(
            category=candidate.category,       # remapped via class_map in verify()
            confidence=float(probs[index]),
            label=label,
            metadata={"class_index": index, "top1": label},
        )

    def _describe(self) -> Dict[str, Any]:
        providers = self._session.get_providers() if self._session else None
        return {
            "model_path": self.model_path,
            "accelerator": self.accelerator,
            "providers": providers,
            "labels": len(self._labels),
            "requires_model_file": True,
            "classifies": True,
        }


def _providers_for(accelerator: str, available: List[str]) -> List[str]:
    preferred = {
        "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
        "cuda": ["CUDAExecutionProvider"],
        "cpu": [],
    }.get(accelerator.lower(), [])
    chosen = [p for p in preferred if p in available]
    missing = [p for p in preferred if p not in available]
    if missing:
        logger.warning("onnx: accelerator %r requested but %s unavailable — "
                       "falling back to CPU", accelerator, ", ".join(missing))
    return chosen + ["CPUExecutionProvider"]


def _read_labels(path: Optional[str]) -> List[str]:
    if not path:
        return []
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        logger.warning("onnx: labels file %s not found", p)
        return []
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)