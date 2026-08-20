"""scripts/test_offline_sync.py

Internet OFF -> detection -> SQLite -> Internet ON -> automatic sync.

Run: python scripts/test_offline_sync.py
"""
import logging
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.store.db import Database
from edge.store.event import Event
from edge.store.outbox import Outbox
from edge.sync.agent import SyncAgent
from edge.sync.client import MockNetworkClient

logging.basicConfig(level=logging.INFO, format="%(message)s")


def fake_detection(i: int) -> dict:
    return {"category": "unknown", "confidence": 0.9, "bbox": [0.1, 0.1, 0.3, 0.3],
            "frame_id": f"cam01-{i:09d}"}


def main():
    db_path = tempfile.mktemp(suffix=".sqlite")
    print(f"db: {db_path}\n")

    print("--- phase 1: network OFF, detections arriving ---")
    db = Database(db_path)
    outbox = Outbox(db)
    client = MockNetworkClient()
    client.online = False
    agent = SyncAgent(outbox, client, poll_interval=0.2)

    for i in range(5):
        outbox.append(Event.new(fake_detection(i)))
    assert outbox.pending_count() == 5
    print(f"events stored locally: {outbox.pending_count()}")

    result = agent.run_once()
    print(f"sync attempt while offline: {result}")
    assert result["synced"] == 0
    assert outbox.pending_count() == 5

    print("\n--- phase 2: restart edge service (new process, same db) ---")
    db.close()
    db2 = Database(db_path)
    outbox2 = Outbox(db2)
    assert outbox2.pending_count() == 5
    print(f"events survived restart: {outbox2.pending_count()}")

    print("\n--- phase 3: network ON, agent syncs automatically ---")
    client2 = MockNetworkClient()
    client2.online = True
    agent2 = SyncAgent(outbox2, client2, poll_interval=0.2)

    deadline = time.time() + 6
    while outbox2.pending_count() > 0 and time.time() < deadline:
        agent2.run_once()
        time.sleep(0.3)

    print(f"pending after sync: {outbox2.pending_count()}")
    print(f"sent via client: {len(client2.sent)}")
    assert outbox2.pending_count() == 0
    assert len(client2.sent) == 5

    print("\nPASS: internet OFF -> detection -> SQLite -> internet ON -> automatic sync")
    db2.close()
    Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()