"""edge/sync/agent.py — Run the Sync Agent.

Internet OFF -> detection -> SQLite -> Internet ON -> automatic sync.

The agent never blocks Capture/Tier1/Tier2. It's a separate loop that
only reads from the Outbox and pushes forward. Detection keeps
succeeding regardless of what this loop is doing.
"""
import logging
import threading
from typing import Optional

from ..store.outbox import Outbox
from .backoff import next_retry_at
from .client import SyncClient, SyncError

log = logging.getLogger("edge.sync.agent")


class SyncAgent:
    def __init__(self, outbox: Outbox, client: SyncClient,
                 poll_interval: float = 5.0, batch_size: int = 50):
        self.outbox = outbox
        self.client = client
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def run_once(self) -> dict:
        """One sync pass. Returns counters. Called directly in tests
        without spinning up a thread."""
        synced, failed = 0, 0
        for event in self.outbox.pending_events(limit=self.batch_size):
            try:
                self.client.send(event)
                self.outbox.mark_synced(event.event_id)
                synced += 1
            except SyncError as e:
                attempt = self.outbox.attempts(event.event_id)
                retry_at = next_retry_at(attempt)
                self.outbox.mark_failed(event.event_id, retry_at)
                failed += 1
                log.debug("sync failed for %s: %s (retry at %s)",
                          event.event_id, e, retry_at)
                # network down for one means it's down for the rest of
                # this batch too — stop burning through it.
                break
        return {"synced": synced, "failed": failed,
                "pending": self.outbox.pending_count()}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.poll_interval + 1)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                result = self.run_once()
                if result["synced"]:
                    log.info("synced %d event(s), %d pending",
                             result["synced"], result["pending"])
            except Exception:
                log.exception("sync loop crashed on iteration, continuing")
            self._stop.wait(self.poll_interval)