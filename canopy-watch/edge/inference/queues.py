"""The Tier2 queue — bounded, drop-NEWEST.

Deliberately the opposite policy to the Tier1 queue, for a concrete reason.

A raw frame is cheap and there is another one 33 ms behind it, so when Tier1
falls behind, the right move is to throw away the stale frame and work on the
current one: drop-oldest, latest-frame-wins, stay current.

A Tier2 task is not cheap and not frequent. Tier1 already filtered, the frame
was already retrieved from the buffer, and the crop was already computed. Under
overload, drop-oldest would discard the work already paid for and start again on
something equally expensive — thrashing. Drop-newest instead: finish what has
been started, and shed the arrivals you have no capacity for.

Both policies count what they shed. `dropped` climbing here means Tier2 is
undersized for this node's detection rate — either the model is too heavy for
this `hardware_tier`, or Tier1's threshold is too permissive.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Deque, Dict, Generic, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DropNewestQueue(Generic[T]):
    def __init__(self, maxsize: int = 32) -> None:
        self.maxsize = max(1, int(maxsize))
        self._items: Deque[T] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self.dropped = 0
        self.delivered = 0
        self.high_water = 0

    def put(self, item: T) -> bool:
        """False if the queue was full and `item` was rejected."""
        with self._not_empty:
            if len(self._items) >= self.maxsize:
                self.dropped += 1
                logger.debug("tier2 queue full (%d) — rejecting new task", self.maxsize)
                return False
            self._items.append(item)
            self.high_water = max(self.high_water, len(self._items))
            self._not_empty.notify()
            return True

    def get(self, timeout: Optional[float] = 1.0) -> Optional[T]:
        with self._not_empty:
            if not self._items:
                self._not_empty.wait(timeout)
            if not self._items:
                return None
            self.delivered += 1
            return self._items.popleft()

    def drain(self) -> None:
        with self._lock:
            self._items.clear()

    def qsize(self) -> int:
        with self._lock:
            return len(self._items)

    def stats(self) -> Dict[str, Any]:
        return {
            "maxsize": self.maxsize,
            "queued": self.qsize(),
            "delivered": self.delivered,
            "high_water": self.high_water,
            "tasks_dropped_overload": self.dropped,
        }