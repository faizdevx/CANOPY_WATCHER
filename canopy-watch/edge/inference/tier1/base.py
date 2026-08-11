"""Tier1 detector contract.

Same shape as the camera driver layer: one abstract interface, several backends,
selected by config, and no branching in application code. The pipeline only ever
sees `detect(image) -> List[Candidate]`.

Tier1 is a **triage filter**, not a species identifier. It answers "is this
worth waking Tier2 for". COCO's 80 classes contain no deer, no tiger, no
poacher — so the class map projects whatever the model says onto four coarse
categories, and that mapping lives in yaml so retraining on park data changes
config rather than code.
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..types import CATEGORIES, Candidate

logger = logging.getLogger(__name__)


class Tier1Error(Exception):
    """Tier1 backend could not load or run."""


#: COCO-80 -> our four buckets. Anything unlisted becomes `default_category`.
DEFAULT_CLASS_MAP: Dict[str, str] = {
    "person": "human",
    "bicycle": "vehicle",
    "car": "vehicle",
    "motorcycle": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "boat": "vehicle",
    "bird": "animal",
    "cat": "animal",
    "dog": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    "elephant": "animal",
    "bear": "animal",
    "zebra": "animal",
    "giraffe": "animal",
}


class Tier1Detector(abc.ABC):
    name: str = "unset"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params: Dict[str, Any] = dict(params or {})
        self.score_threshold: float = float(self.params.get("score_threshold", 0.4))
        self.max_detections: int = int(self.params.get("max_detections", 10))
        self.default_category: str = str(self.params.get("default_category", "unknown"))
        self.class_map: Dict[str, str] = dict(
            self.params.get("class_map") or DEFAULT_CLASS_MAP)
        self.model_version: Optional[str] = self.params.get("model_version")
        #: categories that are simply not interesting for this deployment
        self.ignore_categories = set(self.params.get("ignore_categories") or [])

        self._loaded = False
        self.invocations = 0
        self.total_latency_ms = 0.0

    # ------------------------------------------------------------------ #

    def load(self) -> None:
        if self._loaded:
            return
        self._load()
        self._loaded = True
        logger.info("tier1: %s ready (%s)", self.name, self.describe())

    def detect(self, image: np.ndarray) -> List[Candidate]:
        """Run inference and return filtered, category-mapped candidates."""
        if not self._loaded:
            self.load()
        started = time.monotonic()
        raw = self._detect(image)
        latency_ms = (time.monotonic() - started) * 1000
        self.invocations += 1
        self.total_latency_ms += latency_ms

        out: List[Candidate] = []
        for cand in raw:
            if cand.confidence < self.score_threshold:
                continue
            category = self.map_category(cand.label, cand.category)
            if category in self.ignore_categories:
                continue
            out.append(Candidate(
                category=category,
                confidence=float(cand.confidence),
                bbox=cand.bbox,
                label=cand.label,
                backend=self.name,
                model_version=self.model_version,
                latency_ms=latency_ms,
                metadata=cand.metadata,
            ))
        out.sort(key=lambda c: c.confidence, reverse=True)
        return out[: self.max_detections]

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
            "score_threshold": self.score_threshold,
            "loaded": self._loaded,
        }
        info.update(self._describe())
        return info

    def stats(self) -> Dict[str, Any]:
        avg = (self.total_latency_ms / self.invocations) if self.invocations else None
        return {
            "backend": self.name,
            "invocations": self.invocations,
            "avg_latency_ms": None if avg is None else round(avg, 2),
        }

    # ------------------------------- hooks ------------------------------ #

    @abc.abstractmethod
    def _load(self) -> None:
        ...

    @abc.abstractmethod
    def _detect(self, image: np.ndarray) -> List[Candidate]:
        """Raw candidates. Thresholding and category mapping happen in detect()."""

    def _describe(self) -> Dict[str, Any]:
        return {}