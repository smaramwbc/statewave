"""Integration tests for the cross-machine query embedding cache.

Hits a real Postgres + pgvector via the test fixtures and verifies that
the L2 cache path (Postgres `query_embedding_cache` table) actually
persists, returns matches, and respects TTL. The pure-unit tests live
in tests/test_query_cache.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import text

from server.db import repositories as repo
from server.services.embeddings.query_cache import cached_embed_query


def _real_provider(model: str = "text-embedding-3-small", vector_value: float = 0.42):
    p = MagicMock()
    p.provides_semantic_similarity = True
    p.model = model
    p.embed_query = AsyncMock(return_value=[vector_value] * 1536)
    return p


@pytest.mark.anyio
async def test_l2_round_trip_writes_then_reads_via_real_postgres(session_factory):
    """End-to-end: provider miss writes to Postgres, second call hits."""
    provider = _real_provider(vector_value=0.123)

    # First call — should miss, hit provider, write to DB.
    e1 = await cached_embed_query(session_factory, provider, "what database?")
    assert provider.embed_query.await_count == 1
    assert e1[0] == pytest.approx(0.123, abs=1e-5)

    # Second call (same text) — should hit L2, NOT call provider.
    e2 = await cached_embed_query(session_factory, provider, "what database?")
    assert provider.embed_query.await_count == 1, "second call must hit L2 cache"
    assert e2[0] == pytest.approx(0.123, abs=1e-5)

    # Direct DB inspection — entry exists with the expected model.
    async with session_factory() as session:
        result = await session.execute(
            text(
                "SELECT model, expires_at FROM query_embedding_cache "
                "WHERE text_key = :tk"
            ),
            {"tk": "what database?"},
        )
        row = result.first()
        assert row is not None
        assert row[0] == "text-embedding-3-small"
        # expires_at should be ~24h in the future (default TTL)
        assert row[1] > datetime.now(timezone.utc) + timedelta(hours=23)
        await session.execute(text("DELETE FROM query_embedding_cache"))
        await session.commit()


@pytest.mark.anyio
async def test_l2_expired_entry_is_treated_as_miss(session_factory):
    """An expired entry must be ignored — provider should be called again."""
    provider = _real_provider(vector_value=0.5)

    # Manually plant an expired entry.
    async with session_factory() as session:
        await session.execute(
            text(
                "INSERT INTO query_embedding_cache "
                "(text_key, model, embedding, expires_at, created_at) "
                "VALUES ('expired', 'text-embedding-3-small', "
                "CAST(:emb AS vector), :exp, :now)"
            ),
            {
                "emb": "[" + ",".join(["0.99"] * 1536) + "]",
                "exp": datetime.now(timezone.utc) - timedelta(hours=1),
                "now": datetime.now(timezone.utc) - timedelta(days=1),
            },
        )
        await session.commit()

    # Cached call should treat as miss → hit provider → overwrite.
    result = await cached_embed_query(session_factory, provider, "expired")
    assert provider.embed_query.await_count == 1
    assert result[0] == pytest.approx(0.5, abs=1e-5)

    # Cleanup
    async with session_factory() as session:
        await session.execute(text("DELETE FROM query_embedding_cache"))
        await session.commit()


@pytest.mark.anyio
async def test_l2_keys_by_model_so_rotations_dont_alias(session_factory):
    """Same text under different models = two distinct cache entries."""
    p_small = _real_provider(model="text-embedding-3-small", vector_value=0.1)
    p_large = _real_provider(model="text-embedding-3-large", vector_value=0.2)

    e_small = await cached_embed_query(session_factory, p_small, "shared question")
    e_large = await cached_embed_query(session_factory, p_large, "shared question")

    assert e_small[0] == pytest.approx(0.1, abs=1e-5)
    assert e_large[0] == pytest.approx(0.2, abs=1e-5)
    # Both providers were called once (no cross-model cache pollution)
    assert p_small.embed_query.await_count == 1
    assert p_large.embed_query.await_count == 1

    # Two distinct rows in the table
    async with session_factory() as session:
        result = await session.execute(
            text(
                "SELECT count(*) FROM query_embedding_cache "
                "WHERE text_key = 'shared question'"
            )
        )
        assert result.scalar() == 2
        await session.execute(text("DELETE FROM query_embedding_cache"))
        await session.commit()


@pytest.mark.anyio
async def test_repo_query_cache_get_returns_none_for_missing(session_factory):
    """Direct repo-level test — get on absent key returns None cleanly."""
    async with session_factory() as session:
        result = await repo.query_cache_get(
            session, text_key="never-cached", model="text-embedding-3-small"
        )
    assert result is None


@pytest.mark.anyio
async def test_repo_query_cache_prunes_old_expired_on_write(session_factory):
    """The opportunistic-cleanup-on-write should remove rows expired > 7 days."""
    # Plant a very-old expired row
    async with session_factory() as session:
        await session.execute(
            text(
                "INSERT INTO query_embedding_cache "
                "(text_key, model, embedding, expires_at, created_at) "
                "VALUES ('ancient', 'text-embedding-3-small', "
                "CAST(:emb AS vector), :exp, :now)"
            ),
            {
                "emb": "[" + ",".join(["0.0"] * 1536) + "]",
                "exp": datetime.now(timezone.utc) - timedelta(days=10),
                "now": datetime.now(timezone.utc) - timedelta(days=11),
            },
        )
        await session.commit()
        # Sanity — the row exists
        result = await session.execute(
            text("SELECT count(*) FROM query_embedding_cache WHERE text_key = 'ancient'")
        )
        assert result.scalar() == 1

    # Any write should trigger the cleanup
    async with session_factory() as session:
        await repo.query_cache_set(
            session,
            text_key="fresh",
            model="text-embedding-3-small",
            embedding=[0.5] * 1536,
            ttl_seconds=86400,
        )
        await session.commit()

    # Ancient should be gone, fresh should remain
    async with session_factory() as session:
        result = await session.execute(
            text("SELECT text_key FROM query_embedding_cache ORDER BY text_key")
        )
        keys = [r[0] for r in result.all()]
        assert "ancient" not in keys
        assert "fresh" in keys
        await session.execute(text("DELETE FROM query_embedding_cache"))
        await session.commit()
