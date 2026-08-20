"""edge/store/db.py — where events live on disk."""
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    schema_version  INTEGER NOT NULL DEFAULT 1,
    payload         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    synced          INTEGER NOT NULL DEFAULT 0,
    sync_attempts   INTEGER NOT NULL DEFAULT 0,
    last_attempt_at TEXT,
    next_retry_at   TEXT,
    synced_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_pending
    ON events (synced, next_retry_at);
"""


class Database:
    """One writer-safe connection per process. WAL mode so the capture
    pipeline (writer) and sync agent (reader/writer) don't block each
    other under normal load."""

    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def execute(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def query(self, sql: str, params: tuple = ()):
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self):
        with self._lock:
            self._conn.close()