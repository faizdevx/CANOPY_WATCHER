"""Heuristic Tier1 — `tier1.backend: heuristic`.

No model file, no TFLite, numpy only. This is the Tier1 equivalent of the mock
camera driver: it makes the entire pipeline runnable and testable in CI, in
Codespaces, and on any laptop before a single weight has been downloaded.

It is not a toy stub that returns a hardcoded box. It runs a real (if crude)
detector: coarse-grid brightness blob detection. Frames are reduced to a grid of
cell means, cells standing out from the frame's own statistics are marked, and
connected marked cells become a box.

That is deliberately the thing the mock camera produces — a bright moving target
on a dark gradient — so `mock camera -> heuristic Tier1` yields genuine
detections that track a genuinely moving object. An end-to-end test asserting
"a detection appeared and its box moved" is therefore meaningful rather than
tautological.

Also usable in the field as a crude thermal/IR fallback, where a warm subject
really is the bright blob. Not the plan, but not nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..types import BBox, Candidate
from .base import Tier1Detector


class HeuristicTier1(Tier1Detector):
    name = "heuristic"

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(params)
        self.grid: int = max(4, int(self.params.get("grid", 24)))
        #: how many standard deviations above the frame mean a cell must sit
        self.sigma: float = float(self.params.get("sigma", 2.5))
        self.min_cells: int = max(1, int(self.params.get("min_cells", 2)))
        self.max_area_fraction: float = float(self.params.get("max_area_fraction", 0.5))
        self.assumed_label: Optional[str] = self.params.get("assumed_label")
        self.assumed_category: str = str(self.params.get("assumed_category", "unknown"))
        self.model_version = self.model_version or "heuristic-1"

    def _load(self) -> None:
        return None

    def _detect(self, image: np.ndarray) -> List[Candidate]:
        cells = self._cell_means(image)
        mean = float(cells.mean())
        std = float(cells.std())
        if std < 1e-6:                       # perfectly flat frame, nothing to find
            return []

        mask = cells > (mean + self.sigma * std)
        if int(mask.sum()) < self.min_cells:
            return []

        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        gh, gw = mask.shape
        try:
            bbox = BBox(cols[0] / gw, rows[0] / gh,
                        (cols[-1] + 1) / gw, (rows[-1] + 1) / gh).clamped()
        except ValueError:
            return []

        if bbox.area > self.max_area_fraction:
            # Covers most of the frame — that's a lighting change, not a subject.
            return []

        # Confidence from how far the blob stands out, squashed into 0-1.
        peak = float(cells.max())
        z = (peak - mean) / std
        confidence = float(np.clip((z - self.sigma) / 6.0 + 0.5, 0.0, 0.99))

        return [Candidate(
            category=self.assumed_category,
            confidence=confidence,
            bbox=bbox,
            label=self.assumed_label,
            metadata={"z_score": round(z, 3), "cells": int(mask.sum())},
        )]

    def _cell_means(self, image: np.ndarray) -> np.ndarray:
        """Mean intensity per grid cell, trimming any ragged remainder."""
        gray = image.mean(axis=2).astype(np.float32)
        h, w = gray.shape
        gh = min(self.grid, h)
        gw = min(self.grid, w)
        ch, cw = h // gh, w // gw
        trimmed = gray[: ch * gh, : cw * gw]
        return trimmed.reshape(gh, ch, gw, cw).mean(axis=(1, 3))

    def _describe(self) -> Dict[str, Any]:
        return {"grid": self.grid, "sigma": self.sigma, "requires_model_file": False}