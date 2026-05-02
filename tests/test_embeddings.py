"""Unit tests for the embedding provider abstraction and stub provider."""

from __future__ import annotations

import pytest

from server.services.embeddings.stub import StubEmbeddingProvider


# ---------------------------------------------------------------------------
# StubEmbeddingProvider
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_stub_embed_texts_returns_correct_count():
    provider = StubEmbeddingProvider(dimensions=64)
    results = await provider.embed_texts(["hello", "world", "test"])
    assert len(results) == 3


@pytest.mark.anyio
async def test_stub_embed_texts_correct_dimensions():
    provider = StubEmbeddingProvider(dimensions=128)
    results = await provider.embed_texts(["hello world"])
    assert len(results[0]) == 128


@pytest.mark.anyio
async def test_stub_embed_texts_empty_input():
    provider = StubEmbeddingProvider()
    results = await provider.embed_texts([])
    assert results == []


@pytest.mark.anyio
async def test_stub_embed_query_correct_dimensions():
    provider = StubEmbeddingProvider(dimensions=64)
    result = await provider.embed_query("test query")
    assert len(result) == 64


@pytest.mark.anyio
async def test_stub_deterministic():
    """Same input text always produces the same vector."""
    provider = StubEmbeddingProvider(dimensions=64)
    v1 = await provider.embed_query("hello world")
    v2 = await provider.embed_query("hello world")
    assert v1 == v2


@pytest.mark.anyio
async def test_stub_different_texts_different_vectors():
    """Different texts produce different vectors."""
    provider = StubEmbeddingProvider(dimensions=64)
    v1 = await provider.embed_query("hello world")
    v2 = await provider.embed_query("goodbye world")
    assert v1 != v2


@pytest.mark.anyio
async def test_stub_vectors_are_unit_length():
    """Stub vectors should be normalized to approximately unit length."""
    provider = StubEmbeddingProvider(dimensions=256)
    v = await provider.embed_query("normalize me")
    magnitude = sum(x * x for x in v) ** 0.5
    assert abs(magnitude - 1.0) < 1e-6


@pytest.mark.anyio
async def test_stub_dimensions_property():
    provider = StubEmbeddingProvider(dimensions=384)
    assert provider.dimensions == 384


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def test_get_provider_returns_none_when_disabled(monkeypatch):
    from server.services.embeddings import get_provider, reset_provider
    import server.core.config

    original_settings = server.core.config.settings
    try:
        monkeypatch.setenv("STATEWAVE_EMBEDDING_PROVIDER", "none")
        reset_provider()
        from server.core.config import Settings

        server.core.config.settings = Settings()
        assert get_provider() is None
    finally:
        server.core.config.settings = original_settings
        reset_provider()


def test_get_provider_returns_stub_by_default():
    from server.services.embeddings import get_provider, reset_provider
    import server.core.config

    original_settings = server.core.config.settings
    try:
        from server.core.config import Settings

        # Force stub provider
        import os

        os.environ.pop("STATEWAVE_EMBEDDING_PROVIDER", None)
        reset_provider()
        server.core.config.settings = Settings()
        provider = get_provider()
        assert provider is not None
        assert type(provider).__name__ == "StubEmbeddingProvider"
    finally:
        server.core.config.settings = original_settings
        reset_provider()


# ---------------------------------------------------------------------------
# _TTLCache (used by OpenAIEmbeddingProvider for query embedding cache)
# ---------------------------------------------------------------------------
#
# These tests pin the cache contract independently of the LiteLLM-bound
# provider so they don't need network/API access. The OpenAI provider's
# import of litellm is lazy (inside __init__) and we'd otherwise have to
# guard the whole file with a litellm-availability skip.

from server.services.embeddings.openai import _TTLCache  # noqa: E402


def test_ttl_cache_first_get_misses():
    cache: _TTLCache[str, list[float]] = _TTLCache(max_size=8, ttl_seconds=60)
    assert cache.get("hello") is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_ttl_cache_set_then_get_hits():
    cache: _TTLCache[str, list[float]] = _TTLCache(max_size=8, ttl_seconds=60)
    cache.set("hello", [1.0, 2.0, 3.0])
    assert cache.get("hello") == [1.0, 2.0, 3.0]
    assert cache.hits == 1
    assert cache.misses == 0


def test_ttl_cache_repeated_hits_increment_counter():
    cache: _TTLCache[str, list[float]] = _TTLCache(max_size=8, ttl_seconds=60)
    cache.set("k", [0.5])
    for _ in range(5):
        assert cache.get("k") == [0.5]
    assert cache.hits == 5
    assert cache.misses == 0


def test_ttl_cache_expired_entry_treated_as_miss(monkeypatch):
    """When TTL has elapsed, the entry should be evicted and a miss recorded."""
    import time as time_module
    cache: _TTLCache[str, list[float]] = _TTLCache(max_size=8, ttl_seconds=10)
    fake_now = [1000.0]
    monkeypatch.setattr(time_module, "monotonic", lambda: fake_now[0])
    cache.set("k", [1.0])
    # Within TTL
    fake_now[0] = 1005.0
    assert cache.get("k") == [1.0]
    # Past TTL
    fake_now[0] = 1100.0
    assert cache.get("k") is None
    # Expired entry was dropped
    assert len(cache) == 0
    # Miss counter incremented for the expired access
    assert cache.misses == 1


def test_ttl_cache_size_bound_evicts_oldest():
    cache: _TTLCache[str, list[float]] = _TTLCache(max_size=3, ttl_seconds=60)
    cache.set("a", [1.0])
    cache.set("b", [2.0])
    cache.set("c", [3.0])
    cache.set("d", [4.0])  # Should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == [2.0]
    assert cache.get("c") == [3.0]
    assert cache.get("d") == [4.0]
    assert len(cache) == 3


def test_ttl_cache_lru_promotion_on_hit_keeps_recent_entries():
    """Hitting an entry should refresh its LRU position so it survives eviction."""
    cache: _TTLCache[str, list[float]] = _TTLCache(max_size=3, ttl_seconds=60)
    cache.set("a", [1.0])
    cache.set("b", [2.0])
    cache.set("c", [3.0])
    # Touch "a" — moves it to most-recent
    cache.get("a")
    cache.set("d", [4.0])  # Should evict "b" (now least recent), not "a"
    assert cache.get("a") == [1.0]
    assert cache.get("b") is None
    assert cache.get("c") == [3.0]
    assert cache.get("d") == [4.0]


def test_ttl_cache_set_overwrites_and_refreshes_position():
    cache: _TTLCache[str, list[float]] = _TTLCache(max_size=3, ttl_seconds=60)
    cache.set("a", [1.0])
    cache.set("b", [2.0])
    cache.set("c", [3.0])
    cache.set("a", [9.0])  # Overwrite + promote "a" to most-recent
    cache.set("d", [4.0])  # Should evict "b" (least recent), not "a"
    assert cache.get("a") == [9.0]
    assert cache.get("b") is None


def test_ttl_cache_clear_resets_state():
    cache: _TTLCache[str, list[float]] = _TTLCache(max_size=8, ttl_seconds=60)
    cache.set("k", [1.0])
    cache.get("k")  # +1 hit
    cache.get("missing")  # +1 miss
    cache.clear()
    assert len(cache) == 0
    assert cache.hits == 0
    assert cache.misses == 0


# ---------------------------------------------------------------------------
# OpenAIEmbeddingProvider — query cache integration (mocked litellm)
# ---------------------------------------------------------------------------
#
# These tests verify the cache wires correctly into embed_query without
# hitting the real OpenAI API. We mock the provider's own embed_texts so
# we can count calls and assert that cached repeats skip the network path.


@pytest.mark.anyio
async def test_openai_provider_embed_query_caches_repeat(monkeypatch):
    pytest.importorskip("litellm")
    from server.services.embeddings.openai import OpenAIEmbeddingProvider

    provider = OpenAIEmbeddingProvider(api_key="test-key", dimensions=4)

    call_count = {"n": 0}

    async def fake_embed_texts(self, texts):  # noqa: ANN001
        call_count["n"] += 1
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    monkeypatch.setattr(OpenAIEmbeddingProvider, "embed_texts", fake_embed_texts)

    v1 = await provider.embed_query("How do I deploy on Fly.io?")
    v2 = await provider.embed_query("How do I deploy on Fly.io?")
    v3 = await provider.embed_query("How do I deploy on Fly.io?")

    assert v1 == v2 == v3
    assert call_count["n"] == 1, "second/third identical query must hit cache"
    stats = provider.query_cache_stats
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["size"] == 1


@pytest.mark.anyio
async def test_openai_provider_embed_query_distinct_texts_each_miss(monkeypatch):
    pytest.importorskip("litellm")
    from server.services.embeddings.openai import OpenAIEmbeddingProvider

    provider = OpenAIEmbeddingProvider(api_key="test-key", dimensions=4)

    call_count = {"n": 0}

    async def fake_embed_texts(self, texts):  # noqa: ANN001
        call_count["n"] += 1
        # Different vector per text so we can spot mix-ups
        return [[float(call_count["n"])] * 4 for _ in texts]

    monkeypatch.setattr(OpenAIEmbeddingProvider, "embed_texts", fake_embed_texts)

    v_a = await provider.embed_query("question A")
    v_b = await provider.embed_query("question B")
    assert v_a != v_b
    assert call_count["n"] == 2
    assert provider.query_cache_stats["misses"] == 2
    assert provider.query_cache_stats["hits"] == 0


@pytest.mark.anyio
async def test_openai_provider_embed_query_does_not_cache_on_provider_error(monkeypatch):
    """A failing OpenAI call must not poison the cache with a stale or
    fallback value. The error propagates and the cache stays empty so the
    next call retries fresh."""
    pytest.importorskip("litellm")
    from server.services.embeddings.openai import OpenAIEmbeddingProvider

    provider = OpenAIEmbeddingProvider(api_key="test-key", dimensions=4)

    async def fake_embed_texts(self, texts):  # noqa: ANN001
        raise RuntimeError("simulated upstream failure")

    monkeypatch.setattr(OpenAIEmbeddingProvider, "embed_texts", fake_embed_texts)

    with pytest.raises(RuntimeError, match="simulated upstream failure"):
        await provider.embed_query("never cached")
    assert provider.query_cache_stats["size"] == 0
    assert provider.query_cache_stats["hits"] == 0


@pytest.mark.anyio
async def test_openai_provider_embed_query_eviction_under_size_bound(monkeypatch):
    pytest.importorskip("litellm")
    from server.services.embeddings.openai import OpenAIEmbeddingProvider

    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        dimensions=4,
        query_cache_max_size=2,
    )
    call_count = {"n": 0}

    async def fake_embed_texts(self, texts):  # noqa: ANN001
        call_count["n"] += 1
        return [[float(call_count["n"])] * 4 for _ in texts]

    monkeypatch.setattr(OpenAIEmbeddingProvider, "embed_texts", fake_embed_texts)

    await provider.embed_query("a")
    await provider.embed_query("b")
    await provider.embed_query("c")  # evicts "a"
    # "a" should be a miss again — not in cache
    await provider.embed_query("a")
    # 4 misses total: a, b, c, a-again
    assert provider.query_cache_stats["misses"] == 4
    assert provider.query_cache_stats["size"] == 2
