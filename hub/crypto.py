import hashlib

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def derive_key(password: str) -> bytes:
    return hashlib.sha256(password.encode()).digest()


def decrypt_blob(data: bytes, password: str) -> bytes:
    if len(data) < 13:
        raise ValueError("encrypted payload is too short")
    nonce, ciphertext = data[:12], data[12:]
    return AESGCM(derive_key(password)).decrypt(nonce, ciphertext, None)
