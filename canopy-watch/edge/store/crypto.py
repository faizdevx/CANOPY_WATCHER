"""edge/store/crypto.py — optional encryption at rest for the outbox.

Follow-up: DetectionEvent payloads may eventually carry something
sensitive (exact GPS of a station, for instance) sitting on an SD card
for hours/days before it syncs. This is opt-in — most deployments
don't need it, and it adds a key-management problem (CW_EVENT_KEY_env,
same *_env indirection pattern as CW_SYNC_TOKEN) that isn't worth
forcing on every profile.

Uses Fernet (AES-128-CBC + HMAC, from `cryptography`) — authenticated,
so a corrupted or tampered row fails loudly on decrypt rather than
silently returning garbage.
"""
from cryptography.fernet import Fernet, InvalidToken


class EventCipher:
    def __init__(self, key: bytes):
        self._fernet = Fernet(key)

    @staticmethod
    def generate_key() -> bytes:
        return Fernet.generate_key()

    def encrypt(self, plaintext_json: str) -> str:
        return self._fernet.encrypt(plaintext_json.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        try:
            return self._fernet.decrypt(ciphertext.encode()).decode()
        except InvalidToken as e:
            raise ValueError("event payload failed to decrypt — wrong key or corrupted row") from e