"""cloud/ingest_api/main.py — the smallest ingest API that proves the
data flow: SQLite -> Sync Agent -> Ingest API -> PostgreSQL.

No Kafka, no Flink, no Kubernetes. One process, one Postgres. The
full architecture's Ingest Gateway / broker / stream processor can
replace this later without changing what the edge device sends or
how /stations, /sightings, /alerts read.
"""
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from .auth import authenticate_station
from .db import get_cursor
from .schemas import IngestEvent

app = FastAPI(title="Canopy Watch Ingest API")


# ---------------------------------------------------------------- events --
@app.post("/v1/events", status_code=201)
def ingest_event(event: IngestEvent, station: dict = Depends(authenticate_station)):
    """
    Validate → authenticate → dedup → store → derive sighting → maybe alert.

    Status codes matter here because the edge's HTTPSyncClient treats
    them differently: 2xx = synced, 4xx = permanent (dead-letter, don't
    retry), 5xx = retryable (backoff, try again later).
    """
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO events (event_id, station_id, schema_version, payload, device_created_at)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (event_id) DO NOTHING
               RETURNING event_id""",
            (str(event.event_id), station["id"], event.schema_version,
             _payload_json(event), event.created_at),
        )
        inserted = cur.fetchone()

        if inserted is None:
            # Same event_id already stored — the Sync Agent resent
            # after a dropped ack. Idempotent: 200, not 201, nothing
            # re-derived.
            return {"event_id": str(event.event_id), "duplicate": True}

        occurred_at = event.payload.occurred_at or event.created_at
        cur.execute(
            """INSERT INTO sightings
               (event_id, station_id, category, confidence, bbox,
                risk_score, risk_band, occurred_at, snapshot_ref)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (event_id) DO NOTHING
               RETURNING id, risk_band""",
            (str(event.event_id), station["id"], event.payload.category,
             event.payload.confidence, _json(event.payload.bbox),
             event.payload.risk_score, event.payload.risk_band,
             occurred_at, event.payload.snapshot_ref),
        )
        sighting = cur.fetchone()

        alert_id = None
        if sighting and sighting["risk_band"] == "high":
            cur.execute(
                """INSERT INTO alerts (sighting_id, station_id, risk_band, message)
                   VALUES (%s, %s, %s, %s) RETURNING id""",
                (sighting["id"], station["id"], "high",
                 f"High-risk {event.payload.category} detected "
                 f"(confidence {event.payload.confidence:.2f})"),
            )
            alert_id = cur.fetchone()["id"]

    return {"event_id": str(event.event_id), "duplicate": False, "alert_id": alert_id}


def _payload_json(event: IngestEvent) -> str:
    import json
    return json.dumps(event.payload.model_dump())


def _json(value):
    import json
    return json.dumps(value) if value is not None else None


# -------------------------------------------------------------- stations --
class StationCreate(BaseModel):
    node_id: str
    org: str
    site: str
    zone: str | None = None
    token: str          # raw token, provided once at provisioning time; we hash it and forget it


@app.post("/stations", status_code=201)
def create_station(body: StationCreate):
    from .auth import _hash
    with get_cursor() as cur:
        cur.execute(
            """INSERT INTO stations (node_id, org, site, zone, token_hash)
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (body.node_id, body.org, body.site, body.zone, _hash(body.token)),
        )
        return {"id": cur.fetchone()["id"], "node_id": body.node_id}


@app.get("/stations")
def list_stations():
    with get_cursor() as cur:
        cur.execute(
            """SELECT id, node_id, org, site, zone, last_seen_at,
                      last_pending_sync, last_battery_pct, last_thermal_c
               FROM stations ORDER BY node_id"""
        )
        return cur.fetchall()


class Heartbeat(BaseModel):
    pending_sync_count: int | None = None
    battery_pct: float | None = None
    thermal_c: float | None = None


@app.post("/stations/heartbeat")
def heartbeat(body: Heartbeat, station: dict = Depends(authenticate_station)):
    """
    Surfaces edge/store/outbox.py::pending_count() alongside battery and
    thermal — the health-monitoring axis from the main config doc,
    closing the "pending_count exists but nothing exposes it" follow-up.
    """
    with get_cursor() as cur:
        cur.execute(
            """UPDATE stations
               SET last_seen_at = %s, last_pending_sync = %s,
                   last_battery_pct = %s, last_thermal_c = %s
               WHERE id = %s""",
            (datetime.now(timezone.utc), body.pending_sync_count,
             body.battery_pct, body.thermal_c, station["id"]),
        )
    return {"ok": True}


# ------------------------------------------------------------- sightings --
@app.get("/sightings")
def list_sightings(station_id: int | None = None, limit: int = 50):
    with get_cursor() as cur:
        if station_id:
            cur.execute(
                """SELECT * FROM sightings WHERE station_id = %s
                   ORDER BY occurred_at DESC LIMIT %s""",
                (station_id, limit),
            )
        else:
            cur.execute("SELECT * FROM sightings ORDER BY occurred_at DESC LIMIT %s", (limit,))
        return cur.fetchall()


# ---------------------------------------------------------------- alerts --
@app.get("/alerts")
def list_alerts(unacknowledged_only: bool = True):
    with get_cursor() as cur:
        if unacknowledged_only:
            cur.execute("SELECT * FROM alerts WHERE NOT acknowledged ORDER BY created_at DESC")
        else:
            cur.execute("SELECT * FROM alerts ORDER BY created_at DESC")
        return cur.fetchall()


@app.post("/alerts/{alert_id}/ack")
def ack_alert(alert_id: int):
    with get_cursor() as cur:
        cur.execute(
            "UPDATE alerts SET acknowledged = true WHERE id = %s RETURNING id",
            (alert_id,),
        )
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="alert not found")
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok"}