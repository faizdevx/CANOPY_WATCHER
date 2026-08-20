"""edge/sync/heartbeat.py — surface outbox.pending_count() somewhere.

Closes the metrics follow-up: pending_count() existed but nothing
exposed it. This posts it to the Ingest API's /stations/heartbeat
alongside whatever health.py collects (battery/thermal, per the
health-monitoring axis in the main config doc) so it shows up next to
the same signals a field engineer already checks.

Runs on its own slow interval, separate from the sync agent's poll
loop — a heartbeat every 60s doesn't need the sync agent's 5s cadence.
"""
import json
import logging
import threading
import urllib.error
import urllib.request

from ..store.outbox import Outbox

log = logging.getLogger("edge.sync.heartbeat")


class HeartbeatSender:
    def __init__(self, outbox: Outbox, endpoint: str, auth_token: str,
                 interval: float = 60.0, health_probe=None):
        """
        health_probe: optional zero-arg callable returning
        {"battery_pct": float|None, "thermal_c": float|None}. Left
        pluggable because battery/thermal availability is itself
        hardware-tier dependent (a Pi without a UPS has no battery
        sensor at all — see the main README's Health monitoring axis).
        """
        self.outbox = outbox
        self.endpoint = endpoint
        self.auth_token = auth_token
        self.interval = interval
        self.health_probe = health_probe or (lambda: {"battery_pct": None, "thermal_c": None})
        self._stop = threading.Event()
        self._thread = None

    def send_once(self) -> bool:
        health = self.health_probe()
        body = json.dumps({
            "pending_sync_count": self.outbox.pending_count(),
            "battery_pct": health.get("battery_pct"),
            "thermal_c": health.get("thermal_c"),
        }).encode()
        req = urllib.request.Request(
            self.endpoint, data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.auth_token}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status < 300
        except (urllib.error.URLError, OSError) as e:
            log.debug("heartbeat failed (expected while offline): %s", e)
            return False

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 1)

    def _loop(self):
        while not self._stop.is_set():
            self.send_once()
            self._stop.wait(self.interval)