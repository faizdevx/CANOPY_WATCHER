#!/usr/bin/env bash
# cloud/scripts/run_local.sh — smallest possible local stack.
set -euo pipefail

service postgresql start 2>/dev/null || true

psql -h localhost -U postgres -tc "SELECT 1 FROM pg_database WHERE datname='canopy_watch'" \
  | grep -q 1 || createdb -h localhost -U postgres canopy_watch

psql -h localhost -U canopy -d canopy_watch -f "$(dirname "$0")/../ingest_api/migrations.sql"

uvicorn cloud.ingest_api.main:app --host 0.0.0.0 --port 8000 --reload