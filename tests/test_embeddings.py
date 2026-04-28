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
