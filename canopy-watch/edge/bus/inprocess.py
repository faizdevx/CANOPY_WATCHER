"""In-process event bus.

The architecture calls for "local bus in dev, Kafka/MQTT in prod, toggled by
config — no code change, only wiring". This is the local bus, and `EventBus` is
the seam a Redis/MQTT/Kafka implementation later slots into.

Every subscriber gets its OWN bounded queue and its OWN thread. That matters:
`publish()` is called from the Tier2 thread, and a slow subscriber (an SQLite
write, an HTTP alert) must never block inference. A subscriber that cannot keep
up sheds its own messages and counts them, without affecting anyone else.

Topics used by the edge pipeline:

    detection.scored     a fully scored DetectionEvent
    detection.candidate  Tier1 fired (pre-Tier2) — useful for debug UIs
    node.health          periodic health snapshot
"""

from __future__ import annotations

import abc
import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

Handler = Callable[[str, Any], None]


class EventBus(abc.ABC):
    """The seam. An MQTT/Kafka bus implements this and nothing upstream changes."""

    @abc.abstractmethod
    def publish(self, topic: str, payload: Any) -> None: ...

    @abc.abstractmethod
    def subscribe(self, topic: str, handler: Handler, name: Optional[str] = None) -> Any: ...

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self, timeout: float = 2.0) -> None: ...


@dataclass
class Subscription:
    topic: str
    handler: Handler
    name: str
    maxsize: int = 256
    queue: "queue.Queue" = field(default_factory=queue.Queue)
    thread: Optional[threading.Thread] = None
    delivered: int = 0
    dropped: int = 0
    errors: int = 0

    def stats(self) -> Dict[str, Any]:
        return {"name": self.name, "topic": self.topic, "queued": self.queue.qsize(),
                "delivered": self.delivered, "dropped": self.dropped,
                "errors": self.errors}


class InProcessBus(EventBus):
    def __init__(self, queue_size: int = 256) -> None:
        self.queue_size = int(queue_size)
        self._subs: List[Subscription] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._running = False
        self.published = 0

    # ------------------------------------------------------------------ #

    def subscribe(self, topic: str, handler: Handler,
                  name: Optional[str] = None) -> Subscription:
        sub = Subscription(
            topic=topic, handler=handler,
            name=name or getattr(handler, "__name__", "subscriber"),
            maxsize=self.queue_size,
            queue=queue.Queue(maxsize=self.queue_size),
        )
        with self._lock:
            self._subs.append(sub)
        if self._running:
            self._spawn(sub)
        return sub

    def publish(self, topic: str, payload: Any) -> None:
        """Never blocks. A saturated subscriber drops its own messages."""
        self.published += 1
        with self._lock:
            targets = [s for s in self._subs if _matches(s.topic, topic)]
        for sub in targets:
            try:
                sub.queue.put_nowait((topic, payload))
            except queue.Full:
                sub.dropped += 1
                logger.warning("bus: subscriber %s is saturated — dropped %s",
                               sub.name, topic)

    def start(self) -> None:
        if self._running:
            return
        self._stop.clear()
        self._running = True
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            self._spawn(sub)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        with self._lock:
            subs = list(self._subs)
        for sub in subs:
            if sub.thread is not None:
                sub.thread.join(timeout=timeout)
                sub.thread = None
        self._running = False

    # ------------------------------------------------------------------ #

    def _spawn(self, sub: Subscription) -> None:
        if sub.thread is not None:
            return
        sub.thread = threading.Thread(
            target=self._pump, args=(sub,), name=f"bus-{sub.name}", daemon=True)
        sub.thread.start()

    def _pump(self, sub: Subscription) -> None:
        while not self._stop.is_set():
            try:
                topic, payload = sub.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                sub.handler(topic, payload)
                sub.delivered += 1
            except Exception:
                sub.errors += 1
                logger.exception("bus: subscriber %s raised on %s", sub.name, topic)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"published": self.published, "running": self._running,
                    "subscribers": [s.stats() for s in self._subs]}


def _matches(pattern: str, topic: str) -> bool:
    """`*` matches everything; `prefix.*` matches by prefix."""
    if pattern in ("*", topic):
        return True
    return pattern.endswith(".*") and topic.startswith(pattern[:-1])