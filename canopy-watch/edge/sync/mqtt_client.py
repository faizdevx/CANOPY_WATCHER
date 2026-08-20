"""edge/sync/mqtt_client.py — MQTT transport, same SyncClient contract.

For the fleet-management case noted in the main README as a future
scalability item, and because Pi field devices already speak MQTT for
their sync backoff curve (per the main README's Networking axis) — an
MQTT event transport reuses that same connection story instead of
opening a second one just for HTTPS.

Requires paho-mqtt (`pip install paho-mqtt`). Not exercised against a
live broker in this environment — the interface is what's proven here
(SyncAgent doesn't know or care that this isn't HTTPSyncClient), the
broker round-trip is a deploy-environment test.
"""
from .client import PermanentSyncError, SyncClient, SyncError


class MQTTSyncClient(SyncClient):
    def __init__(self, broker_host: str, topic: str, port: int = 1883,
                 client_id: str = None, qos: int = 1, timeout: float = 5.0):
        self.broker_host = broker_host
        self.topic = topic
        self.port = port
        self.client_id = client_id
        self.qos = qos
        self.timeout = timeout

    def send(self, event) -> None:
        try:
            import paho.mqtt.publish as publish
            import paho.mqtt.client as mqtt_client
        except ImportError as e:
            raise PermanentSyncError(
                "paho-mqtt not installed — this is a config error, not a "
                "network one, so retrying won't help"
            ) from e

        try:
            publish.single(
                self.topic,
                payload=event.to_json(),
                qos=self.qos,
                hostname=self.broker_host,
                port=self.port,
                client_id=self.client_id or "",
            )
        except (mqtt_client.MQTTException, OSError, TimeoutError) as e:
            # Broker unreachable, connection refused, publish timeout —
            # same "internet being stupid" bucket as HTTPSyncClient's
            # URLError/OSError. QoS 1 means at-least-once delivery once
            # the connection *does* succeed; event_id dedup on the
            # backend covers the "at-least" part.
            raise SyncError(str(e)) from e