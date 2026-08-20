"""edge/store/outbox.py — append-only local truth + sync bookkeeping.

Every detection lands here first. Nothing downstream (sync, cloud,
dashboards) is trusted more than this table. Sync is just something
that eventually happens to rows that are already durable.
"""
from datetime import datetime, timezone
from typing import List

from .db import Database
from .event import Event


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Outbox:
    def __init__(self, db: Database):
        self.db = db

    def append(self, event: Event) -> None:
        self.db.execute(
            """INSERT OR IGNORE INTO events
               (event_id, schema_version, payload, created_at, synced)
               VALUES (?, ?, ?, ?, 0)""",
            (event.event_id, event.schema_version, event.to_json(), event.created_at),
        )

    def pending_events(self, limit: int = 50) -> List[Event]:
        """Events not yet synced and not currently in backoff."""
        now = _now_iso()
        rows = self.db.query(
            """SELECT event_id, payload, schema_version, created_at
               FROM events
               WHERE synced = 0
                 AND (next_retry_at IS NULL OR next_retry_at <= ?)
               ORDER BY created_at ASC
               LIMIT ?""",
            (now, limit),
        )
        return [Event.from_row(*row) for row in rows]

    def mark_synced(self, event_id: str) -> None:
        self.db.execute(
            "UPDATE events SET synced = 1, synced_at = ? WHERE event_id = ?",
            (_now_iso(), event_id),
        )

    def mark_failed(self, event_id: str, next_retry_at: str) -> None:
        self.db.execute(
            """UPDATE events
               SET sync_attempts = sync_attempts + 1,
                   last_attempt_at = ?,
                   next_retry_at = ?
               WHERE event_id = ?""",
            (_now_iso(), next_retry_at, event_id),
        )

    def pending_count(self) -> int:
        return self.db.query("SELECT COUNT(*) FROM events WHERE synced = 0")[0][0]

    def attempts(self, event_id: str) -> int:
        row = self.db.query("SELECT sync_attempts FROM events WHERE event_id = ?", (event_id,))
        return row[0][0] if row else 0