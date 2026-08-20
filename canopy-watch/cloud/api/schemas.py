"""cloud/ingest_api/schemas.py — validate what the edge actually sends.

This is the schema contract from the README's Detection Events section
made literal: event ID, timestamp, detection info, normalized bbox,
risk score, schema version. If the edge's Event/DetectionEvent shape
drifts from this, ingestion should reject it loudly (422) rather than
silently store garbage — same fail-fast philosophy as the config loader.
"""
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class DetectionPayload(BaseModel):
    """The `payload` field inside an edge Event — i.e. the DetectionEvent."""
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: Optional[list[float]] = None          # normalized [x, y, w, h]
    frame_id: Optional[str] = None
    risk_score: Optional[float] = None
    risk_band: Optional[str] = None
    snapshot_ref: Optional[str] = None
    occurred_at: Optional[str] = None            # falls back to Event.created_at if absent

    @field_validator("bbox")
    @classmethod
    def bbox_is_normalized(cls, v):
        if v is None:
            return v
        if len(v) != 4 or any((c < 0 or c > 1) for c in v):
            raise ValueError("bbox must be 4 normalized coordinates in [0, 1]")
        return v

    @field_validator("risk_band")
    @classmethod
    def known_band(cls, v):
        if v is not None and v not in ("low", "medium", "high"):
            raise ValueError("risk_band must be one of: low, medium, high")
        return v


class IngestEvent(BaseModel):
    """Mirrors edge/store/event.py::Event exactly — the wire format the
    Sync Agent already produces via Event.to_json()."""
    event_id: UUID
    schema_version: int = 1
    payload: DetectionPayload
    created_at: str