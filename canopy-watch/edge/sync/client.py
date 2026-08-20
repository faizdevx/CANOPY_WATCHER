"""edge/sync/client.py — how an event leaves the device.

Swappable, same idea as camera drivers: the agent doesn't know or care
if this is HTTPS, MQTT, or a mock used in tests.
"""
from abc import ABC, abstractmethod


class SyncError(Exception):
    """Any transport failure: network down, 5xx, timeout."""


class SyncClient(ABC):
    @abstractmethod
    def send(self, event) -> None:
        """Send one event. Raise SyncError on failure. No retry logic here —
        that belongs to the agent."""


class MockNetworkClient(SyncClient):
    """Used in tests/offline dev. `online` flag simulates the internet
    being stupid on command."""

    def __init__(self):
        self.online = True
        self.sent: list = []

    def send(self, event) -> None:
        if not self.online:
            raise SyncError("simulated network down")
        self.sent.append(event.event_id)


class HTTPSyncClient(SyncClient):
    def __init__(self, endpoint: str, auth_token: str = None, timeout: float = 5.0):
        self.endpoint = endpoint
        self.auth_token = auth_token
        self.timeout = timeout

    def send(self, event) -> None:
        import urllib.request
        import urllib.error

        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        req = urllib.request.Request(
            self.endpoint, data=event.to_json().encode(), headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status >= 300:
                    raise SyncError(f"server returned {resp.status}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise SyncError(str(e)) from e