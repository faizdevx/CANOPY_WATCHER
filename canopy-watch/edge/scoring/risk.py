"""Risk Scorer — deterministic arithmetic, not a model.

Same determinism boundary the rest of this system uses: **models produce
observations, scoring produces numbers.** Nothing in this path may hallucinate,
guess, or vary between runs on identical input. Given the same detection and the
same context, the score is byte-identical, forever.

    score = 100 * sum(weight_i * factor_i) / sum(weight_i)

Every factor is normalised to 0-1 and every weight comes from yaml, so a park
operator retunes without a redeploy and two parks can weight differently. The
full breakdown ships inside the event:

    {"name": "nocturnal", "value": 1.0, "weight": 0.20, "contribution": 20.0,
     "reason": "local hour 02 is within night window 18:00-06:00"}

That array is what the Explainability Service will later feed to SHAP/LIME. Until
that exists, it is already human-readable, which is worth more than a bare
number a ranger has no reason to trust.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..inference.types import Candidate, Verification

logger = logging.getLogger(__name__)

SCORER_VERSION = "risk-1.0.0"

DEFAULT_WEIGHTS: Dict[str, float] = {
    "class_base": 0.30,
    "confidence": 0.20,
    "nocturnal": 0.20,
    "zone_sensitivity": 0.15,
    "persistence": 0.10,
    "proximity": 0.05,
}

#: A person at 2am in a core zone is the actual threat this system exists for.
#: A cow is not. Known fauna scores low; unknown sits between.
DEFAULT_CLASS_BASE: Dict[str, float] = {
    "human": 1.0,
    "vehicle": 0.85,
    "unknown": 0.45,
    "animal": 0.20,
}

DEFAULT_ZONE_SENSITIVITY: Dict[str, float] = {
    "core": 1.0,
    "buffer": 0.6,
    "periphery": 0.3,
}

DEFAULT_BANDS = {"critical": 80.0, "high": 60.0, "medium": 35.0}


@dataclass(frozen=True)
class RiskFactor:
    name: str
    value: float
    weight: float
    contribution: float
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(float(self.value), 4),
            "weight": round(float(self.weight), 4),
            "contribution": round(float(self.contribution), 3),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RiskScore:
    score: float
    band: str
    factors: List[RiskFactor] = field(default_factory=list)
    scorer_version: str = SCORER_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(float(self.score), 2),
            "band": self.band,
            "scorer_version": self.scorer_version,
            "factors": [f.to_dict() for f in self.factors],
        }


@dataclass
class ScoringContext:
    """Everything outside the detection itself that changes what it means."""

    zone: Optional[str] = None
    wall_clock: Optional[float] = None          # unix epoch, UTC
    utc_offset_hours: float = 0.0               # node's local offset
    bbox_area: float = 0.0
    recent_similar: int = 0                     # from PersistenceWindow
    node_id: Optional[str] = None


class RiskScorer:
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        p = dict(params or {})
        self.weights = {**DEFAULT_WEIGHTS, **(p.get("weights") or {})}
        self.class_base = {**DEFAULT_CLASS_BASE, **(p.get("class_base") or {})}
        self.zone_sensitivity = {
            **DEFAULT_ZONE_SENSITIVITY, **(p.get("zone_sensitivity") or {})}
        self.bands = {**DEFAULT_BANDS, **(p.get("bands") or {})}
        self.night_start_hour: int = int(p.get("night_start_hour", 18))
        self.night_end_hour: int = int(p.get("night_end_hour", 6))
        #: how many recent similar detections saturate the persistence factor
        self.persistence_saturation: int = max(1, int(p.get("persistence_saturation", 4)))
        #: bbox area fraction at which "close" saturates
        self.proximity_saturation: float = float(p.get("proximity_saturation", 0.25))
        self.unknown_zone_value: float = float(p.get("unknown_zone_value", 0.5))

        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("risk scorer: weights must sum to more than zero")
        self._weight_total = total

    # ------------------------------------------------------------------ #

    def score(
        self,
        candidate: Candidate,
        verification: Optional[Verification],
        context: ScoringContext,
    ) -> RiskScore:
        """Pure function. Same inputs -> same output, always."""
        category = (verification.category if verification else candidate.category)
        confidence = (verification.confidence if verification else candidate.confidence)

        factors = [
            self._class_base(category),
            self._confidence(confidence, used_tier2=verification is not None),
            self._nocturnal(context),
            self._zone(context),
            self._persistence(context),
            self._proximity(context, candidate),
        ]

        weighted = sum(f.contribution for f in factors)
        total = round(max(0.0, min(100.0, weighted)), 4)
        return RiskScore(score=total, band=self.band_for(total), factors=factors)

    def band_for(self, score: float) -> str:
        if score >= self.bands["critical"]:
            return "critical"
        if score >= self.bands["high"]:
            return "high"
        if score >= self.bands["medium"]:
            return "medium"
        return "low"

    # ----------------------------- factors ----------------------------- #

    def _factor(self, name: str, value: float, reason: str) -> RiskFactor:
        value = float(max(0.0, min(1.0, value)))
        weight = float(self.weights.get(name, 0.0))
        contribution = 100.0 * weight * value / self._weight_total
        return RiskFactor(name, value, weight, contribution, reason)

    def _class_base(self, category: str) -> RiskFactor:
        value = self.class_base.get(category, self.class_base.get("unknown", 0.45))
        return self._factor("class_base", value, f"category={category}")

    def _confidence(self, confidence: float, used_tier2: bool) -> RiskFactor:
        source = "tier2" if used_tier2 else "tier1"
        return self._factor("confidence", confidence,
                            f"{source} confidence {confidence:.2f}")

    def _nocturnal(self, ctx: ScoringContext) -> RiskFactor:
        if ctx.wall_clock is None:
            return self._factor("nocturnal", 0.5, "no timestamp — neutral")
        hour = self._local_hour(ctx)
        night = (hour >= self.night_start_hour or hour < self.night_end_hour)
        return self._factor(
            "nocturnal", 1.0 if night else 0.0,
            f"local hour {hour:02d} is {'within' if night else 'outside'} night window "
            f"{self.night_start_hour:02d}:00-{self.night_end_hour:02d}:00")

    def _zone(self, ctx: ScoringContext) -> RiskFactor:
        if not ctx.zone:
            return self._factor("zone_sensitivity", self.unknown_zone_value,
                                "zone not configured — neutral")
        value = self.zone_sensitivity.get(ctx.zone, self.unknown_zone_value)
        return self._factor("zone_sensitivity", value, f"zone={ctx.zone}")

    def _persistence(self, ctx: ScoringContext) -> RiskFactor:
        # One pass through is not the same as someone loitering.
        value = min(1.0, ctx.recent_similar / self.persistence_saturation)
        return self._factor("persistence", value,
                            f"{ctx.recent_similar} similar detection(s) in window")

    def _proximity(self, ctx: ScoringContext, candidate: Candidate) -> RiskFactor:
        area = ctx.bbox_area or candidate.bbox.area
        value = min(1.0, area / self.proximity_saturation)
        return self._factor("proximity", value,
                            f"bbox covers {area * 100:.1f}% of frame")

    def _local_hour(self, ctx: ScoringContext) -> int:
        dt = datetime.fromtimestamp(ctx.wall_clock, tz=timezone.utc)
        return int((dt.hour + ctx.utc_offset_hours) % 24)

    # ------------------------------------------------------------------ #

    def describe(self) -> Dict[str, Any]:
        return {
            "scorer_version": SCORER_VERSION,
            "weights": self.weights,
            "class_base": self.class_base,
            "zone_sensitivity": self.zone_sensitivity,
            "bands": self.bands,
            "night_window": [self.night_start_hour, self.night_end_hour],
        }