"""Cross-machine query embedding cache (L2).

Wraps `provider.embed_query(text)` with a Postgres-backed lookup so that
identical task text across multiple Fly machines pays the OpenAI round-trip
exactly once. The provider's own in-process LRU+TTL (the L1 layer in
`OpenAIEmbeddingProvider._query_cache`) is unchanged — this layer sits in
front of it.

Layering, in order of precedence:

    1. L1 — provider's in-process LRU (~0.1ms hit, machine-local)
    2. L2 — Postgres `query_embedding_cache` table (~1-5ms hit,
            cross-machine via the shared DB)
    3. Provider call — OpenAI API (~500ms-30s, occasional spikes)

The implementation here is L2 + the provider's own L1+API path; we don't
re-implement L1 inside the helper because it already lives correctly
inside the provider. The order at runtime is:

    L2 hit                → return immediately (saves API + L1)
    L2 miss + L1 hit      → provider returns from L1 (saves API)
    L2 miss + L1 miss     → provider calls API, populates L1; we then
                            populate L2 for cross-machine reuse

Failure modes — explicit and bounded:

  * Stub provider (no real semantic similarity): bypassed entirely.
    Caching hash garbage cross-machine has no value.
  * L2 read fails (e.g. transient DB glitch): fall through to provider.
    Don't bubble a DB error back to /v1/context.
  * Provider call fails: don't write to L2 (avoids poisoning the cache
    with a bad value). The exception propagates to the caller as today.
  * L2 write fails: log a warning but return the embedding. We have a
    valid result; the cross-machine miss next time is acceptable.
"""

from __future__ import annotations

from typing import Any

import structlog

from server.db import repositories as repo

logger = structlog.stdlib.get_logger()


# 24h TTL — query embeddings are stable across long windows (the OpenAI
# model doesn't change without a deliberate rotation, which we'd handle
# by rotating the `model` cache key). Picked higher than the in-process
# 1h because there's no memory pressure on Postgres and longer reuse is
# pure win for demo/repeat traffic.
DEFAULT_TTL_SECONDS = 24 * 60 * 60


async def cached_embed_query(
    session_factory: Any,
    provider: Any,
    text: str,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> list[float]:
    """Return an embedding for `text`, hitting the cross-machine L2 cache
    when possible.

    `session_factory` is the async sessionmaker from `db.engine`. We open
    short-lived sessions per cache operation to avoid coupling the
    embedding helper to the caller's session lifetime.
    """
    if not _provider_provides_semantic_similarity(provider):
        # Stub providers produce hash garbage; skip the L2 cache entirely.
        return await provider.embed_query(text)

    model = getattr(provider, "model", None) or "unknown"

    # L2 read — best-effort. Any DB error falls through to the provider.
    try:
        async with session_factory() as session:
            cached = await repo.query_cache_get(session, text_key=text, model=model)
        if cached is not None:
            return cached
    except Exception:
        logger.warning("query_cache_l2_read_failed", exc_info=True)

    # L2 miss — fall through to the provider (which checks L1 + API).
    embedding = await provider.embed_query(text)

    # L2 write — best-effort. A failure here just means the next
    # cross-machine request pays the API cost; not worth failing the
    # current request over.
    try:
        async with session_factory() as session:
            await repo.query_cache_set(
                session,
                text_key=text,
                model=model,
                embedding=embedding,
                ttl_seconds=ttl_seconds,
            )
            await session.commit()
    except Exception:
        logger.warning("query_cache_l2_write_failed", exc_info=True)

    return embedding


def _provider_provides_semantic_similarity(provider: Any) -> bool:
    """Mirror of the same check in context.py — only real embedding
    providers should populate the cross-machine cache."""
    return bool(provider and getattr(provider, "provides_semantic_similarity", True))
