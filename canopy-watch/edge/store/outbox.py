"""edge/store/outbox.py — append-only local truth + sync bookkeeping.

Every detection lands here first. Nothing downstream (sync, cloud,
dashboards) is trusted more than this table. Sync is just something
that eventually happens to rows that are already durable.
"""
"""edge/store/outbox.py — append-only local truth + sync bookkeeping.

Every detection lands here first. Nothing downstream (sync, cloud,
dashboards) is trusted more than this table. Sync is just something
that eventually happens to rows that are already durable.
"""
import json
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from .db import Database
from .event import Event


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Outbox:
    def __init__(self, db: Database, cipher: Optional["object"] = None):
        """
        cipher: optional EventCipher (edge/store/crypto.py). When set,
        the payload is encrypted before it touches disk and decrypted
        on read — everything else about the outbox is unaware this is
        happening. Off by default; opt in per profile.
        """
        self.db = db
        self.cipher = cipher

    def append(self, event: Event) -> None:
        # Store only the inner detection payload here — NOT
        # event.to_json(), which would nest the whole Event (event_id,
        # schema_version, created_at, and payload) inside this column.
        # from_row() below expects exactly what Event.new() put in
        # .payload; storing the full envelope silently double-nests it
        # on the next read.
        payload_json = json.dumps(event.payload)
        stored = self.cipher.encrypt(payload_json) if self.cipher else payload_json
        self.db.execute(
            """INSERT OR IGNORE INTO events
               (event_id, schema_version, payload, created_at, synced)
               VALUES (?, ?, ?, ?, 0)""",
            (event.event_id, event.schema_version, stored, event.created_at),
        )

    def pending_events(self, limit: int = 50) -> List[Event]:
        """Events not yet synced, not dead-lettered, not currently in
        backoff. Dead-lettered events are excluded on purpose — they've
        already been judged permanently unsendable, so they shouldn't
        keep occupying the retry queue forever."""
        now = _now_iso()
        rows = self.db.query(
            """SELECT event_id, payload, schema_version, created_at
               FROM events
               WHERE synced = 0
                 AND dead_letter = 0
                 AND (next_retry_at IS NULL OR next_retry_at <= ?)
               ORDER BY created_at ASC
               LIMIT ?""",
            (now, limit),
        )
        events = []
        for event_id, stored, schema_version, created_at in rows:
            payload_json = self.cipher.decrypt(stored) if self.cipher else stored
            events.append(Event.from_row(event_id, payload_json, schema_version, created_at))
        return events

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

    def mark_dead_letter(self, event_id: str, reason: str) -> None:
        """Backend has permanently rejected this event (bad schema,
        revoked token). Stop retrying it — but keep the row, same as a
        synced one, so it's still visible for debugging/audit."""
        self.db.execute(
            "UPDATE events SET dead_letter = 1, dead_letter_reason = ? WHERE event_id = ?",
            (reason, event_id),
        )

    def pending_count(self) -> int:
        """Only events still actively being retried — excludes
        dead-lettered ones, since those are done being tried, not
        merely delayed. This is the number the health/heartbeat report
        should show."""
        return self.db.query(
            "SELECT COUNT(*) FROM events WHERE synced = 0 AND dead_letter = 0"
        )[0][0]

    def dead_letter_count(self) -> int:
        return self.db.query("SELECT COUNT(*) FROM events WHERE dead_letter = 1")[0][0]

    def attempts(self, event_id: str) -> int:
        row = self.db.query("SELECT sync_attempts FROM events WHERE event_id = ?", (event_id,))
        return row[0][0] if row else 0

    def prune_synced(self, older_than_days: int = 30) -> int:
        """Retention for the local outbox, same spirit as the video
        archive retention policy: synced rows older than N days are
        gone-for-good locally (the cloud copy in PostgreSQL is now the
        record of truth for them). Dead-lettered rows are NOT pruned by
        this — they need a human to look at them first."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
        cur = self.db.execute(
            "DELETE FROM events WHERE synced = 1 AND synced_at < ?", (cutoff,)
        )
        return cur.rowcount