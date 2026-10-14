"""Compiler-version helpers shared by memory compilers."""

from __future__ import annotations

import hashlib


def stable_version(*parts: str) -> str:
    """Return a deterministic fingerprint for compiler-version components.

    Each part is length-prefixed before hashing so different component
    boundaries cannot produce the same payload.
    """
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"
