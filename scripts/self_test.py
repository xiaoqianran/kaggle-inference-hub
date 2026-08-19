import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from hub.config import canonical_model
from hub.crypto import decrypt_blob
from hub.state import HubState


def main():
    assert canonical_model("sana") == "sana-sprint-1.6b"
    assert canonical_model("zimage") == "z-image-turbo-gguf"

    state = HubState()
    state.enqueue({"id": 1, "model": "sana-sprint-1.6b", "prompt": "a", "attempt": 0})
    state.enqueue({"id": 2, "model": "z-image-turbo-gguf", "prompt": "b", "attempt": 0})
    assert state.claim("z-image-turbo-gguf", "z-test", 0.01)["id"] == 2
    assert state.claim("sana-sprint-1.6b", "s-test", 0.01)["id"] == 1

    password = "test-password"
    key = hashlib.sha256(password.encode()).digest()
    nonce = os.urandom(12)
    plain = b"webp-bytes"
    encrypted = nonce + AESGCM(key).encrypt(nonce, plain, None)
    assert decrypt_blob(encrypted, password) == plain

    print("OK: model routing + queue isolation + AES-GCM protocol")


if __name__ == "__main__":
    main()
