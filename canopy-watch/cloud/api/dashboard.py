"""
Dashboard read API.

Read-only aggregation + filtering over the tables the Ingest API (Phase 3)
already writes to. This does not receive data from the edge — it only
answers questions for the dashboard/app.js frontend.

Mount into your existing FastAPI app:

    from dashboard import router as dashboard_router
    app.include_router(dashboard_router, prefix="/api/dashboard", tags=["dashboard"])

Assumes an asyncpg pool is already sitting on app.state.db_pool, same as
the Ingest API routes (Phase 3's "db pool, auth, schemas, and routes" file).
If you used psycopg2 / SQLAlchemy instead, swap `_fetch` for your equivalent
— the SQL and response shapes are what matters here, not the driver.

SCHEMA ASSUMPTIONS — adjust column names below to match what Phase 3
actually created:

    sightings
        event_id       text primary key
        station_id     text references stations(station_id)
        ts             timestamptz         -- detection timestamp
        class_name     text
        confidence     real                -- 0.0 - 1.0
        risk_score     real                -- 0 - 100
        model_version  text
        bbox           jsonb null
        image_url      text null

    stations
        station_id     text primary key
        sector         text
        is_connected   boolean
        last_heartbeat timestamptz
        lat            double precision null
        lng            double precision null

    alerts
        event_id       text references sightings(event_id)
        status         text        -- LOGGED | SMS_DISPATCHED | ACKNOWLEDGED
        assigned_patrol text null
        dispatched_at  timestamptz null

If alerts live as a column on `sightings` instead of a separate table,
drop the join below and read straight off `sightings.alert_status`.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

router = APIRouter()

RISK_BANDS = {"high": (70, 100000), "medium": (40, 70), "low": (-1, 40)}
WINDOWS = {"1h": timedelta(hours=1), "24h": timedelta(hours=24), "7d": timedelta(days=7)}


# --------------------------------------------------------------------------
# Response models — keep this shape stable, dashboard/app.js depends on it
# --------------------------------------------------------------------------
class StationOut(BaseModel):
    id: str
    sector: Optional[str] = None
    is_connected: bool
    last_heartbeat: Optional[datetime] = None


class SummaryOut(BaseModel):
    total_detections: int
    high_risk_detections: int
    high_risk_pct: int
    alerts_dispatched: int
    window_label: Optional[str] = None
    stations: list[StationOut]
    generated_at: datetime


class SightingOut(BaseModel):
    sighting_id: str
    timestamp: datetime
    station: dict
    inference: dict
    alert: dict


class SightingsOut(BaseModel):
    sightings: list[SightingOut]
    total: int


# --------------------------------------------------------------------------
# Shared filter parsing
# --------------------------------------------------------------------------
def _since_cutoff(since: Optional[str]) -> Optional[datetime]:
    if not since or since == "all":
        return None
    delta = WINDOWS.get(since)
    if not delta:
        return None
    return datetime.now(timezone.utc) - delta


def _risk_range(risk: Optional[str]):
    if not risk:
        return None
    return RISK_BANDS.get(risk)


async def _pool(request: Request):
    """Pull the asyncpg pool your Phase 3 app already created."""
    return request.app.state.db_pool


# --------------------------------------------------------------------------
# GET /summary
# --------------------------------------------------------------------------
@router.get("/summary", response_model=SummaryOut)
async def get_summary(
    request: Request,
    station: Optional[str] = None,
    risk: Optional[str] = Query(None, pattern="^(high|medium|low)$"),
    since: str = "24h",
    pool=Depends(_pool),
):
    cutoff = _since_cutoff(since)
    risk_range = _risk_range(risk)

    where = ["1=1"]
    args: list = []

    if station:
        args.append(station)
        where.append(f"station_id = ${len(args)}")
    if cutoff:
        args.append(cutoff)
        where.append(f"ts >= ${len(args)}")
    if risk_range:
        args.append(risk_range[0])
        args.append(risk_range[1])
        where.append(f"risk_score > ${len(args) - 1} AND risk_score <= ${len(args)}")

    where_sql = " AND ".join(where)

    async with pool.acquire() as conn:
        total = await conn.fetchval(f"SELECT count(*) FROM sightings WHERE {where_sql}", *args)
        high = await conn.fetchval(
            f"SELECT count(*) FROM sightings WHERE {where_sql} AND risk_score >= 70", *args
        )
        dispatched = await conn.fetchval(
            f"""
            SELECT count(*) FROM sightings s
            JOIN alerts a ON a.event_id = s.event_id
            WHERE {where_sql.replace('station_id', 's.station_id').replace('ts', 's.ts').replace('risk_score', 's.risk_score')}
              AND a.status = 'SMS_DISPATCHED'
            """,
            *args,
        )
        station_rows = await conn.fetch(
            "SELECT station_id, sector, is_connected, last_heartbeat FROM stations ORDER BY station_id"
        )

    total = total or 0
    high = high or 0
    return SummaryOut(
        total_detections=total,
        high_risk_detections=high,
        high_risk_pct=round((high / total) * 100) if total else 0,
        alerts_dispatched=dispatched or 0,
        window_label={"1h": "last 1h", "24h": "last 24h", "7d": "last 7d"}.get(since, "all time"),
        stations=[
            StationOut(
                id=r["station_id"],
                sector=r["sector"],
                is_connected=r["is_connected"],
                last_heartbeat=r["last_heartbeat"],
            )
            for r in station_rows
        ],
        generated_at=datetime.now(timezone.utc),
    )


# --------------------------------------------------------------------------
# GET /sightings
# --------------------------------------------------------------------------
@router.get("/sightings", response_model=SightingsOut)
async def get_sightings(
    request: Request,
    station: Optional[str] = None,
    risk: Optional[str] = Query(None, pattern="^(high|medium|low)$"),
    since: str = "24h",
    q: Optional[str] = None,
    limit: int = Query(200, le=1000),
    offset: int = 0,
    pool=Depends(_pool),
):
    cutoff = _since_cutoff(since)
    risk_range = _risk_range(risk)

    where = ["1=1"]
    args: list = []

    if station:
        args.append(station)
        where.append(f"s.station_id = ${len(args)}")
    if cutoff:
        args.append(cutoff)
        where.append(f"s.ts >= ${len(args)}")
    if risk_range:
        args.append(risk_range[0])
        args.append(risk_range[1])
        where.append(f"s.risk_score > ${len(args) - 1} AND s.risk_score <= ${len(args)}")
    if q:
        args.append(f"%{q}%")
        where.append(f"s.class_name ILIKE ${len(args)}")

    where_sql = " AND ".join(where)
    args.append(limit)
    limit_idx = len(args)
    args.append(offset)
    offset_idx = len(args)

    sql = f"""
        SELECT
            s.event_id, s.ts, s.station_id, s.class_name, s.confidence,
            s.risk_score, s.model_version, s.bbox, s.image_url,
            st.sector, st.is_connected,
            COALESCE(a.status, 'LOGGED') AS alert_status,
            a.assigned_patrol
        FROM sightings s
        JOIN stations st ON st.station_id = s.station_id
        LEFT JOIN alerts a ON a.event_id = s.event_id
        WHERE {where_sql}
        ORDER BY s.ts DESC
        LIMIT ${limit_idx} OFFSET ${offset_idx}
    """
    count_sql = f"SELECT count(*) FROM sightings s WHERE {where_sql.replace('s.', '')}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)
        total = await conn.fetchval(count_sql, *args[:-2])  # drop limit/offset for the count query

    sightings = [
        SightingOut(
            sighting_id=r["event_id"],
            timestamp=r["ts"],
            station={
                "id": r["station_id"],
                "sector": r["sector"],
                "is_connected": r["is_connected"],
            },
            inference={
                "class_name": r["class_name"],
                "confidence": r["confidence"],
                "risk_score": r["risk_score"],
                "model_version": r["model_version"],
                "bounding_box": r["bbox"],
                "image_url": r["image_url"],
            },
            alert={
                "status": r["alert_status"],
                "assigned_patrol": r["assigned_patrol"],
            },
        )
        for r in rows
    ]
    return SightingsOut(sightings=sightings, total=total or 0)