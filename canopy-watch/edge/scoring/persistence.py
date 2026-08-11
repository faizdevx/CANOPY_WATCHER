"""Bounded recent-detection window feeding the `persistence` risk factor.

One animal walking through is not the same as something circling back for the
third time in ten minutes. This tracks how often a category has been seen
recently per zone.

Bounded and self-evicting, exactly like the frame buffer — an edge node runs for
months without a restart, so anything that only grows is a slow-motion outage.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional, Tuple


class PersistenceWindow:
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        p = dict(params or {})
        self.window_seconds: float = float(p.get("window_seconds", 600))
        self.max_entries_per_key: int = int(p.get("max_entries_per_key", 200))
        self.max_keys: int = int(p.get("max_keys", 64))
        self._events: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def record(self, category: str, zone: Optional[str] = None,
               now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        key = (zone or "_", category)
        with self._lock:
            if key not in self._events and len(self._events) >= self.max_keys:
                oldest = min(self._events, key=lambda k: self._events[k][-1]
                             if self._events[k] else 0)
                self._events.pop(oldest, None)
            bucket = self._events[key]
            bucket.append(now)
            self._prune(bucket, now)
            while len(bucket) > self.max_entries_per_key:
                bucket.popleft()

    def count(self, category: str, zone: Optional[str] = None,
              now: Optional[float] = None) -> int:
        """Detections in the window, EXCLUDING the current one if just recorded."""
        now = time.time() if now is None else now
        key = (zone or "_", category)
        with self._lock:
            bucket = self._events.get(key)
            if not bucket:
                return 0
            self._prune(bucket, now)
            return len(bucket)

    def _prune(self, bucket: Deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "window_seconds": self.window_seconds,
                "tracked_keys": len(self._events),
                "total_entries": sum(len(v) for v in self._events.values()),
            }