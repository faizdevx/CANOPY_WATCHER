"""edge/sync/backoff.py — the curve, not just a number."""
import random
from datetime import datetime, timedelta, timezone


def next_retry_at(attempt: int, base_seconds: float = 2.0,
                   max_seconds: float = 900.0, jitter: float = 0.3) -> str:
    """
    Exponential backoff with full jitter, capped.
    attempt=0 -> ~base, attempt=1 -> ~2*base, ... capped at max_seconds.
    max_seconds should come from profile.yaml: high on rpi/jetson
    (tolerate long offline stretches), low on laptop (fast dev loop).
    """
    delay = min(base_seconds * (2 ** attempt), max_seconds)
    jittered = max(0.5, delay * (1 + random.uniform(-jitter, jitter)))
    when = datetime.now(timezone.utc) + timedelta(seconds=jittered)
    return when.isoformat()