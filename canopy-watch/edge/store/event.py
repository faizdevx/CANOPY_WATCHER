"""edge/store/event.py — What is this event about?"""
from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Event:
    """Immutable outbox event. Wraps a DetectionEvent (or any JSON-serializable payload)."""
    event_id: str
    payload: dict
    schema_version: int = 1
    created_at: str = field(default_factory=_now_iso)

    @staticmethod
    def new(payload: dict, schema_version: int = 1) -> "Event":
        return Event(
            event_id=str(uuid.uuid4()),
            payload=payload,
            schema_version=schema_version,
            created_at=_now_iso(),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_row(event_id: str, payload_json: str, schema_version: int, created_at: str) -> "Event":
        return Event(
            event_id=event_id,
            payload=json.loads(payload_json),
            schema_version=schema_version,
            created_at=created_at,
        )