"""cloud/ingest_api/auth.py — identify the edge device.

One token per station, provisioned the same way CW_SYNC_TOKEN is
provisioned on the Pi/Jetson (main README: "get this from whoever
provisions the fleet"). We only ever store the hash.
"""
import hashlib

from fastapi import Header, HTTPException

from .db import get_cursor


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def authenticate_station(authorization: str = Header(default=None)) -> dict:
    """FastAPI dependency. Raises 401 (permanent — do not retry) on
    anything wrong with the token itself."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()

    with get_cursor() as cur:
        cur.execute(
            "SELECT id, node_id, org, site, zone FROM stations WHERE token_hash = %s",
            (_hash(token),),
        )
        station = cur.fetchone()

    if not station:
        raise HTTPException(status_code=401, detail="unknown or revoked token")
    return station