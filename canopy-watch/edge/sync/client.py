"""edge/sync/client.py — how an event leaves the device.

Swappable, same idea as camera drivers: the agent doesn't know or care
if this is HTTPS, MQTT, or a mock used in tests.
"""
from abc import ABC, abstractmethod


class SyncError(Exception):
    """Retryable transport failure: network down, timeout, 5xx. The
    agent will back off and try this event again later."""


class PermanentSyncError(SyncError):
    """The backend has definitively rejected this event — bad schema
    (422), unknown/revoked token (401/403), or similar 4xx. Retrying
    with the same payload will fail the same way forever, so the agent
    dead-letters it instead of backing off and trying again."""


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
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if 400 <= e.code < 500:
                # Schema rejected, token rejected, etc. — the backend
                # has spoken, and sending the same bytes again won't
                # change its mind.
                raise PermanentSyncError(f"{e.code}: {body}") from e
            raise SyncError(f"{e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # No route, connection refused, DNS failure, read timeout —
            # all retryable, this is exactly "the internet being stupid".
            raise SyncError(str(e)) from e