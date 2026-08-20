"""cloud/ingest_api/db.py — connection pool. Raw SQL, no ORM.

MVP-sized backend, MVP-sized data layer. An ORM buys abstraction we
don't need yet at the cost of hiding the four queries this service
actually runs. Add SQLAlchemy when there's a second developer who
wants it, not before.
"""
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2.pool import SimpleConnectionPool

DSN = os.environ.get(
    "CANOPY_DB_DSN",
    "host=localhost dbname=canopy_watch user=canopy password=canopy_dev",
)

_pool = SimpleConnectionPool(1, 10, DSN)


@contextmanager
def get_conn():
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


@contextmanager
def get_cursor():
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur