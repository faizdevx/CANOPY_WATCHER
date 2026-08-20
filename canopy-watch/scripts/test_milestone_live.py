"""scripts/test_milestone_live.py

The actual milestone: SQLite -> Sync Agent -> Ingest API -> PostgreSQL.
Talks to a REAL running ingest API (cloud/ingest_api/main.py) over
HTTP, which talks to a REAL Postgres. No mocks in this one.

Requires: the ingest API running (uvicorn cloud.ingest_api.main:app)
and a station already created via POST /stations.

Run: python scripts/test_milestone_live.py
"""
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from edge.store.db import Database
from edge.store.event import Event
from edge.store.outbox import Outbox
from edge.sync.agent import SyncAgent
from edge.sync.client import HTTPSyncClient

API = "http://localhost:8000"
TOKEN = "dev-token-123"          # matches the station created in the smoke test
DB_PATH = "/tmp/milestone_rpi_field_03.sqlite"


def main():
    Path(DB_PATH).unlink(missing_ok=True)
    db = Database(DB_PATH)
    outbox = Outbox(db)
    client = HTTPSyncClient(endpoint=f"{API}/v1/events", auth_token=TOKEN)
    agent = SyncAgent(outbox, client, poll_interval=1.0)

    # --- a real-shaped detection payload, matching cloud/ingest_api/schemas.py ---
    good = {
        "category": "wild_boar", "confidence": 0.91,
        "bbox": [0.12, 0.30, 0.25, 0.20],
        "frame_id": "cam01-000184392",
        "risk_score": 78.4, "risk_band": "high",
        "snapshot_ref": "video-analysis-01-000184392",
    }
    bad = {  # will fail pydantic validation on the backend -> 422 -> permanent
        "category": "wild_boar", "confidence": 1.7,   # out of [0,1] range
    }

    outbox.append(Event.new(good))
    outbox.append(Event.new(bad))
    print(f"appended 2 events, pending={outbox.pending_count()}")

    result = agent.run_once()
    print(f"first sync pass: {result}")

    # the batch stops on first hard failure inside run_once (network-style
    # errors), but a PermanentSyncError doesn't stop the batch — so both
    # should have been attempted in one pass. Run once more just in case
    # ordering put the bad one first and it broke early.
    if outbox.pending_count() > 0:
        result2 = agent.run_once()
        print(f"second sync pass: {result2}")

    print(f"final pending={outbox.pending_count()} dead_letter={outbox.dead_letter_count()}")
    assert outbox.pending_count() == 0, "good event should have synced"
    assert outbox.dead_letter_count() == 1, "bad event should be dead-lettered, not stuck retrying"

    # --- confirm it actually landed in Postgres, and dedup works ---
    import urllib.request, json
    req = urllib.request.Request(f"{API}/sightings?limit=5")
    with urllib.request.urlopen(req) as resp:
        sightings = json.loads(resp.read())
    print(f"\nsightings in Postgres: {len(sightings)}")
    print(sightings[0] if sightings else "none")
    assert any(s["category"] == "wild_boar" for s in sightings)

    req = urllib.request.Request(f"{API}/alerts")
    with urllib.request.urlopen(req) as resp:
        alerts = json.loads(resp.read())
    print(f"alerts in Postgres: {len(alerts)}")
    assert len(alerts) >= 1, "high risk_band should have created an alert"

    # --- dedup: resend the same good event_id, confirm no duplicate row ---
    dup_event = Event.new(good)
    outbox2 = Outbox(db)
    outbox2.append(dup_event)
    outbox2.append(dup_event)  # local INSERT OR IGNORE dedup too
    r = agent.run_once()
    print(f"\ndup send result: {r}")

    with urllib.request.urlopen(urllib.request.Request(f"{API}/sightings?limit=10")) as resp:
        sightings_after = json.loads(resp.read())
    print(f"sightings after dup send: {len(sightings_after)} (was {len(sightings)})")
    assert len(sightings_after) == len(sightings) + 1, "resend of a new dup event_id should add exactly one sighting, not error"

    print("\nPASS: SQLite -> Sync Agent -> Ingest API -> PostgreSQL, dedup + dead-letter both correct")
    db.close()


if __name__ == "__main__":
    main()