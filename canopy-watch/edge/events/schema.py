"""DetectionEvent — the Python side of `shared/schemas/detection_event.json`.

The schema file is the contract; this module is a typed constructor for it plus
a validator, so an event that would be rejected by the Ingest Gateway is caught
on the node instead of after a six-hour offline queue.

Design points that matter for v1 and cannot be retrofitted:

  * `schema_version` present from the start
  * wall clock only — monotonic time is node-local and meaningless in the cloud
  * normalized bbox + frame_size — pixel coords break when archive resolution
    differs from capture resolution
  * `tier2_status` explicit — "ran and found nothing" must be distinguishable
    from "never ran because this is a Pi"
  * no image bytes — only `snapshot_ref`. Events travel over LoRa/4G.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..inference.types import CATEGORIES, BBox, Candidate, Verification
from ..scoring.risk import RiskScore

SCHEMA_VERSION = "1.0.0"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "shared" / "schemas" / "detection_event.json"

TIER2_STATUSES = ("local", "deferred", "disabled", "failed", "skipped")
RISK_BANDS = ("low", "medium", "high", "critical")


class SchemaError(ValueError):
    """Event does not satisfy the DetectionEvent contract."""


def utc_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class DetectionEvent:
    node_id: str
    frame_id: str
    captured_at: float                    # unix epoch UTC
    detected_at: float                    # unix epoch UTC
    category: str
    confidence: float
    bbox: BBox
    frame_width: int
    frame_height: int
    tier1: Candidate
    risk: RiskScore
    tier2_status: str = "skipped"
    tier2: Optional[Verification] = None
    label: Optional[str] = None
    site_id: Optional[str] = None
    zone: Optional[str] = None
    media_timestamp: Optional[float] = None
    snapshot_ref: Optional[str] = None
    pipeline: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        *,
        node_id: str,
        frame_id: str,
        captured_at: float,
        candidate: Candidate,
        risk: RiskScore,
        frame_width: int,
        frame_height: int,
        verification: Optional[Verification] = None,
        tier2_status: str = "skipped",
        detected_at: Optional[float] = None,
        **extra: Any,
    ) -> "DetectionEvent":
        """Final category/confidence come from Tier2 when it ran, else Tier1."""
        import time

        winner = verification if (verification and tier2_status == "local") else candidate
        return cls(
            node_id=node_id,
            frame_id=frame_id,
            captured_at=captured_at,
            detected_at=time.time() if detected_at is None else detected_at,
            category=winner.category,
            confidence=float(winner.confidence),
            label=winner.label,
            bbox=candidate.bbox,
            frame_width=frame_width,
            frame_height=frame_height,
            tier1=candidate,
            tier2=verification,
            tier2_status=tier2_status,
            risk=risk,
            **extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "node_id": self.node_id,
            "site_id": self.site_id,
            "zone": self.zone,
            "frame_id": self.frame_id,
            "captured_at": utc_iso(self.captured_at),
            "detected_at": utc_iso(self.detected_at),
            "media_timestamp": self.media_timestamp,
            "category": self.category,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox": self.bbox.to_dict(),
            "frame_size": {"width": int(self.frame_width), "height": int(self.frame_height)},
            "tier1": self.tier1.to_dict(),
            "tier2": self.tier2.to_dict() if self.tier2 else None,
            "tier2_status": self.tier2_status,
            "risk": self.risk.to_dict(),
            "snapshot_ref": self.snapshot_ref,
            "pipeline": self.pipeline or None,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def validate(self) -> "DetectionEvent":
        validate_event(self.to_dict())
        return self


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def load_schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text())


def validate_event(payload: Dict[str, Any]) -> None:
    """Validate against the JSON Schema when `jsonschema` is installed, and
    always run the structural checks below.

    The built-in checks exist because a field node cannot be assumed to have
    `jsonschema` available, and shipping a malformed event into a six-hour
    offline outbox is a failure you only discover after reconnect.
    """
    try:
        import jsonschema  # type: ignore
        jsonschema.validate(payload, load_schema())
    except ImportError:
        pass
    except Exception as exc:                    # jsonschema.ValidationError
        raise SchemaError(str(exc)) from exc

    _structural_checks(payload)


def _structural_checks(payload: Dict[str, Any]) -> None:
    required = ("schema_version", "event_id", "node_id", "frame_id", "captured_at",
                "detected_at", "category", "confidence", "bbox", "frame_size",
                "tier1", "tier2_status", "risk")
    missing = [k for k in required if payload.get(k) is None]
    if missing:
        raise SchemaError(f"missing required field(s): {', '.join(missing)}")

    if payload["category"] not in CATEGORIES:
        raise SchemaError(f"category {payload['category']!r} not in {CATEGORIES}")

    if payload["tier2_status"] not in TIER2_STATUSES:
        raise SchemaError(f"tier2_status {payload['tier2_status']!r} not in {TIER2_STATUSES}")

    if payload["tier2_status"] != "local" and payload.get("tier2") is not None:
        raise SchemaError("tier2 payload present but tier2_status is not 'local'")
    if payload["tier2_status"] == "local" and payload.get("tier2") is None:
        raise SchemaError("tier2_status is 'local' but no tier2 payload")

    if not 0.0 <= float(payload["confidence"]) <= 1.0:
        raise SchemaError("confidence must be within 0..1")

    box = payload["bbox"]
    for key in ("x1", "y1", "x2", "y2"):
        if not 0.0 <= float(box[key]) <= 1.0:
            raise SchemaError(f"bbox.{key} must be normalized 0..1, got {box[key]}")
    if box["x2"] <= box["x1"] or box["y2"] <= box["y1"]:
        raise SchemaError("bbox is degenerate")

    risk = payload["risk"]
    if not 0.0 <= float(risk["score"]) <= 100.0:
        raise SchemaError("risk.score must be within 0..100")
    if risk["band"] not in RISK_BANDS:
        raise SchemaError(f"risk.band {risk['band']!r} not in {RISK_BANDS}")
    if not risk.get("factors"):
        raise SchemaError("risk.factors must not be empty — a bare score is not explainable")

    for key in ("captured_at", "detected_at"):
        if not isinstance(payload[key], str) or "T" not in payload[key]:
            raise SchemaError(f"{key} must be an ISO-8601 UTC string, not a raw timestamp")


def event_bytes(payload: Dict[str, Any]) -> int:
    """Wire size. Worth watching — these go over LoRa."""
    return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))