"""Unit tests for the cross-machine query embedding cache helper.

The helper sits in front of `provider.embed_query` and adds a
Postgres-backed L2 layer. These tests pin the contract:

  * Stub provider → bypassed (no cache lookup, no cache write)
  * L2 hit → return cached vector, provider NOT called
  * L2 miss → provider called, result written to L2
  * L2 read failure → falls through to provider, request still succeeds
  * Provider failure → exception propagates, L2 not poisoned
  * L2 write failure → embedding still returned (best-effort write)

Integration with a real Postgres + pgvector lives in
tests/integration/test_query_cache.py.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.services.embeddings.query_cache import (
    DEFAULT_TTL_SECONDS,
    cached_embed_query,
)


def _fake_session_factory():
    """Returns a callable that yields a MagicMock async-context-manager
    session — enough for the helper to attempt L2 ops; we patch
    repositories to control hit/miss/error behavior."""
    session = MagicMock()

    @asynccontextmanager
    async def factory():
        yield session

    return factory, session


def _stub_provider():
    """A provider that flags itself as NOT semantically meaningful (the
    same shape as StubEmbeddingProvider) — the helper must bypass L2."""
    p = MagicMock()
    p.provides_semantic_similarity = False
    p.embed_query = AsyncMock(return_value=[0.1] * 8)
    return p


def _real_provider(model: str = "text-embedding-3-small"):
    """A provider with the real-semantic flag and a `model` property —
    the helper should use L2 for these."""
    p = MagicMock()
    p.provides_semantic_similarity = True
    p.model = model
    p.embed_query = AsyncMock(return_value=[0.42] * 8)
    return p


# ---------------------------------------------------------------------------
# Stub-provider bypass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stub_provider_bypasses_l2_cache():
    factory, _ = _fake_session_factory()
    provider = _stub_provider()
    with patch("server.services.embeddings.query_cache.repo") as mock_repo:
        result = await cached_embed_query(factory, provider, "hello")
    assert result == [0.1] * 8
    provider.embed_query.assert_awaited_once_with("hello")
    # Critical: no DB ops attempted for stub provider
    mock_repo.query_cache_get.assert_not_called()
    mock_repo.query_cache_set.assert_not_called()


# ---------------------------------------------------------------------------
# Hit / miss flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_hit_returns_cached_and_skips_provider():
    factory, _ = _fake_session_factory()
    provider = _real_provider()
    with patch("server.services.embeddings.query_cache.repo") as mock_repo:
        mock_repo.query_cache_get = AsyncMock(return_value=[0.99] * 8)
        result = await cached_embed_query(factory, provider, "what database?")
    assert result == [0.99] * 8
    # Provider must NOT be called on L2 hit — that's the whole point
    provider.embed_query.assert_not_called()
    mock_repo.query_cache_get.assert_awaited_once()
    mock_repo.query_cache_set.assert_not_called()


@pytest.mark.asyncio
async def test_l2_miss_calls_provider_and_writes_l2():
    factory, _ = _fake_session_factory()
    provider = _real_provider()
    with patch("server.services.embeddings.query_cache.repo") as mock_repo:
        mock_repo.query_cache_get = AsyncMock(return_value=None)
        mock_repo.query_cache_set = AsyncMock(return_value=None)
        result = await cached_embed_query(factory, provider, "what database?")
    assert result == [0.42] * 8
    provider.embed_query.assert_awaited_once_with("what database?")
    mock_repo.query_cache_get.assert_awaited_once()
    mock_repo.query_cache_set.assert_awaited_once()
    # The call should pass the provider's model and the default TTL
    _, kwargs = mock_repo.query_cache_set.call_args
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["ttl_seconds"] == DEFAULT_TTL_SECONDS


@pytest.mark.asyncio
async def test_l2_miss_keys_by_model_so_model_rotation_doesnt_alias():
    """If the same text is embedded under a different model, the cache
    key includes the model and the two embeddings are separate entries."""
    factory, _ = _fake_session_factory()
    p1 = _real_provider(model="text-embedding-3-small")
    p2 = _real_provider(model="text-embedding-3-large")
    with patch("server.services.embeddings.query_cache.repo") as mock_repo:
        mock_repo.query_cache_get = AsyncMock(return_value=None)
        mock_repo.query_cache_set = AsyncMock(return_value=None)
        await cached_embed_query(factory, p1, "Q")
        await cached_embed_query(factory, p2, "Q")
    assert mock_repo.query_cache_set.await_count == 2
    set_models = {c.kwargs["model"] for c in mock_repo.query_cache_set.call_args_list}
    assert set_models == {"text-embedding-3-small", "text-embedding-3-large"}


# ---------------------------------------------------------------------------
# Failure modes — none should break correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2_read_failure_falls_through_to_provider():
    factory, _ = _fake_session_factory()
    provider = _real_provider()
    with patch("server.services.embeddings.query_cache.repo") as mock_repo:
        mock_repo.query_cache_get = AsyncMock(side_effect=RuntimeError("DB glitch"))
        mock_repo.query_cache_set = AsyncMock(return_value=None)
        # Must not raise — the helper logs the failure and falls through.
        result = await cached_embed_query(factory, provider, "q")
    assert result == [0.42] * 8
    provider.embed_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_failure_does_not_poison_l2():
    """When the provider raises, the L2 cache must NOT be written. A
    poisoned cache would replay the failure forever."""
    factory, _ = _fake_session_factory()
    provider = _real_provider()
    provider.embed_query = AsyncMock(side_effect=RuntimeError("provider 5xx"))
    with patch("server.services.embeddings.query_cache.repo") as mock_repo:
        mock_repo.query_cache_get = AsyncMock(return_value=None)
        mock_repo.query_cache_set = AsyncMock(return_value=None)
        with pytest.raises(RuntimeError, match="provider 5xx"):
            await cached_embed_query(factory, provider, "q")
    mock_repo.query_cache_set.assert_not_called()


@pytest.mark.asyncio
async def test_l2_write_failure_does_not_break_request():
    """If the cache write fails (e.g. transient DB issue), we still have
    a valid embedding from the provider — return it, don't raise."""
    factory, _ = _fake_session_factory()
    provider = _real_provider()
    with patch("server.services.embeddings.query_cache.repo") as mock_repo:
        mock_repo.query_cache_get = AsyncMock(return_value=None)
        mock_repo.query_cache_set = AsyncMock(side_effect=RuntimeError("DB write failed"))
        result = await cached_embed_query(factory, provider, "q")
    assert result == [0.42] * 8


@pytest.mark.asyncio
async def test_provider_without_model_attribute_uses_unknown():
    """Defensive: a provider implementation without a `model` property
    shouldn't break the cache helper. Stored as 'unknown'."""
    factory, _ = _fake_session_factory()
    provider = MagicMock()
    provider.provides_semantic_similarity = True
    # Explicitly delete the auto-mocked `model` attr so getattr() returns None
    del provider.model
    provider.embed_query = AsyncMock(return_value=[0.5] * 8)
    with patch("server.services.embeddings.query_cache.repo") as mock_repo:
        mock_repo.query_cache_get = AsyncMock(return_value=None)
        mock_repo.query_cache_set = AsyncMock(return_value=None)
        await cached_embed_query(factory, provider, "q")
    _, kwargs = mock_repo.query_cache_set.call_args
    assert kwargs["model"] == "unknown"
