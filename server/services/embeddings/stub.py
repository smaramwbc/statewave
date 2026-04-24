"""Stub embedding provider — deterministic vectors from content hashing.

Produces consistent vectors for the same input text, making it useful for:
- Local development without external API keys
- Testing that the embedding pipeline works end-to-end
- Deterministic integration tests

The vectors are NOT semantically meaningful — similar texts don't produce
similar vectors. For real semantic search, use the OpenAI provider.
"""

from __future__ import annotations

import hashlib
import struct


class StubEmbeddingProvider:
    """Deterministic hash-based embedding provider."""

    def __init__(self, dimensions: int = 1536) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._hash_to_vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._hash_to_vector(text)

    def _hash_to_vector(self, text: str) -> list[float]:
        """Generate a deterministic unit-length vector from text content."""
        # Use SHA-512 repeatedly to fill the required dimensions
        vector: list[float] = []
        seed = text.encode("utf-8")
        counter = 0
        while len(vector) < self._dimensions:
            h = hashlib.sha512(seed + struct.pack(">I", counter)).digest()
            # Each SHA-512 produces 64 bytes = 16 floats (4 bytes each)
            for i in range(0, 64, 4):
                if len(vector) >= self._dimensions:
                    break
                # Convert 4 bytes to a float in [-1, 1]
                raw = struct.unpack(">I", h[i : i + 4])[0]
                vector.append((raw / 2147483647.5) - 1.0)
            counter += 1

        # Normalize to unit length for cosine similarity
        magnitude = sum(v * v for v in vector) ** 0.5
        if magnitude > 0:
            vector = [v / magnitude for v in vector]
        return vector
