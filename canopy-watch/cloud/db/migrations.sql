-- Smallest schema that proves the data flow. No partitioning, no
-- time-series extension, no read replicas. Add those when there's an
-- actual reason to, not in advance of one.

CREATE TABLE IF NOT EXISTS stations (
    id                  SERIAL PRIMARY KEY,
    node_id             TEXT UNIQUE NOT NULL,     -- matches edge CW_NODE_ID
    org                 TEXT NOT NULL,
    site                TEXT NOT NULL,
    zone                TEXT,
    token_hash          TEXT NOT NULL,            -- sha256(CW_SYNC_TOKEN), never store raw
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at        TIMESTAMPTZ,
    last_pending_sync   INTEGER,                  -- outbox.pending_count() at last heartbeat
    last_battery_pct    REAL,
    last_thermal_c      REAL
);

-- Raw ingest log. This is the durable copy of exactly what the edge
-- device sent, dedup'd on event_id. Nothing here is ever mutated by
-- business logic — sightings/alerts are derived, this is the source.
CREATE TABLE IF NOT EXISTS events (
    event_id        UUID PRIMARY KEY,             -- generated on-device by Event.new()
    station_id      INTEGER NOT NULL REFERENCES stations(id),
    schema_version  INTEGER NOT NULL,
    payload         JSONB NOT NULL,
    device_created_at TIMESTAMPTZ NOT NULL,        -- Event.created_at (on-device clock)
    received_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_events_station ON events (station_id, received_at);

-- Business-level record derived from a validated event. One row per
-- event (unique on event_id) — this is what dashboards/queries use,
-- not the raw events table.
CREATE TABLE IF NOT EXISTS sightings (
    id              SERIAL PRIMARY KEY,
    event_id        UUID UNIQUE NOT NULL REFERENCES events(event_id),
    station_id      INTEGER NOT NULL REFERENCES stations(id),
    category        TEXT NOT NULL,
    confidence      REAL NOT NULL,
    bbox            JSONB,
    risk_score      REAL,
    risk_band       TEXT,                          -- low / medium / high
    occurred_at     TIMESTAMPTZ NOT NULL,
    snapshot_ref    TEXT
);
CREATE INDEX IF NOT EXISTS idx_sightings_station_time ON sightings (station_id, occurred_at);

CREATE TABLE IF NOT EXISTS alerts (
    id              SERIAL PRIMARY KEY,
    sighting_id     INTEGER NOT NULL REFERENCES sightings(id),
    station_id      INTEGER NOT NULL REFERENCES stations(id),
    risk_band       TEXT NOT NULL,
    message         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    acknowledged    BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS idx_alerts_unacked ON alerts (station_id) WHERE NOT acknowledged;