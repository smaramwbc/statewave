"""LiteLLM embedding provider — supports OpenAI, Azure, Cohere, Bedrock, Ollama, etc.

Uses litellm.embedding() for unified multi-provider embedding generation.
Backward compatible: STATEWAVE_EMBEDDING_PROVIDER=openai still works.

Requires:
- pip install 'statewave[llm]'
- Appropriate API key for the chosen model (e.g. OPENAI_API_KEY)
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import Generic, TypeVar

import structlog

logger = structlog.stdlib.get_logger()


# ─── Query embedding cache ───────────────────────────────────────────
#
# /v1/context calls embed_query(task) on every request. In production
# the task text is highly repetitive (the widget wraps user messages in
# a deterministic template, so identical demo questions produce identical
# task strings). Each call costs a network round-trip to OpenAI — we
# observed 16–40s p95 latency per /v1/context, which broke the dev proxy
# at 30s and stretched the production turn time to ~20s. Caching the
# task→vector pair eliminates the repeat cost entirely.
#
# Properties:
#   * In-process only. Each Fly machine has its own cache; that's fine,
#     repeats per machine are still common.
#   * Bounded size + TTL — caps memory, drops stale entries even if the
#     embedding model is rotated.
#   * No locking. asyncio is single-threaded per event loop; concurrent
#     misses on the same key may both call the provider once (wasteful
#     but correct). A futures-by-key dedup is deliberately deferred.
#   * Failures bypass the cache: a provider exception propagates as
#     today, no swallowing, no fallback.

_QUERY_CACHE_MAX_SIZE = 256
_QUERY_CACHE_TTL_SECONDS = 60 * 60  # 1 hour


K = TypeVar("K")
V = TypeVar("V")


class _TTLCache(Generic[K, V]):
    """Bounded LRU + TTL cache. Single-process, single-event-loop.

    Designed for query-embedding caching. Public surface intentionally
    minimal so the contract is easy to read and test:
    `get`, `set`, `clear`, `__len__`, plus `hits` / `misses` counters.
    """

    __slots__ = ("_max_size", "_ttl", "_store", "hits", "misses")

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: "OrderedDict[K, tuple[float, V]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: K) -> V | None:
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            # Expired — drop and treat as miss.
            del self._store[key]
            self.misses += 1
            return None
        # Refresh LRU position on hit.
        self._store.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: K, value: V) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (time.monotonic() + self._ttl, value)
        # Evict oldest until under capacity. Loop handles the rare case
        # where someone shrinks max_size mid-flight.
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)


class OpenAIEmbeddingProvider:
    """LiteLLM-based embedding provider. Name kept for backward compat."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        *,
        query_cache_max_size: int = _QUERY_CACHE_MAX_SIZE,
        query_cache_ttl_seconds: float = _QUERY_CACHE_TTL_SECONDS,
    ) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise ImportError(
                "litellm package is required for embeddings. "
                "Install with: pip install 'statewave[llm]'"
            )
        # Set API key for backward compat (STATEWAVE_OPENAI_API_KEY → OPENAI_API_KEY)
        if api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
        self._model = model
        self._dimensions = dimensions
        self._query_cache: _TTLCache[str, list[float]] = _TTLCache(
            max_size=query_cache_max_size,
            ttl_seconds=query_cache_ttl_seconds,
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provides_semantic_similarity(self) -> bool:
        # Real embedding API — vectors carry semantic meaning, callers may
        # safely use cosine distance as a relevance signal.
        return True

    @property
    def query_cache_stats(self) -> dict[str, int]:
        """Hit/miss/size visibility for ops + tests. Read-only snapshot."""
        return {
            "hits": self._query_cache.hits,
            "misses": self._query_cache.misses,
            "size": len(self._query_cache),
        }

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import litellm

        response = await litellm.aembedding(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        logger.debug(
            "litellm_embeddings_generated",
            count=len(texts),
            model=self._model,
            usage=response.usage.total_tokens if response.usage else None,
        )
        return [item["embedding"] for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        # Cache by exact text — different casing/whitespace produces
        # different embeddings on the OpenAI side, so exact match is the
        # safe key. The widget already sends a deterministic template so
        # repeat queries hit the same key naturally.
        cached = self._query_cache.get(text)
        if cached is not None:
            return cached
        results = await self.embed_texts([text])
        vector = results[0]
        self._query_cache.set(text, vector)
        return vector
