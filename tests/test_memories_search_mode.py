"""GET /v1/memories/search reports which path actually ran via `search_mode`
(issue #281), so callers can tell semantic search from a silent text fallback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from httpx import AsyncClient


async def test_search_mode_text_when_semantic_not_requested(client: AsyncClient):
    with patch(
        "server.api.memories.repo.search_memories", new=AsyncMock(return_value=[])
    ):
        resp = await client.get(
            "/v1/memories/search", params={"subject_id": "u", "q": "hello"}
        )
    assert resp.status_code == 200
    assert resp.json()["search_mode"] == "text"


async def test_search_mode_text_when_semantic_without_query(client: AsyncClient):
    # semantic=true but no q → the `if semantic and query` guard is false → text.
    with patch(
        "server.api.memories.repo.search_memories", new=AsyncMock(return_value=[])
    ):
        resp = await client.get(
            "/v1/memories/search",
            params={"subject_id": "u", "semantic": "true"},
        )
    assert resp.status_code == 200
    assert resp.json()["search_mode"] == "text"


async def test_search_mode_text_fallback_when_no_provider(client: AsyncClient):
    # semantic + q requested, but no embedding provider configured → text_fallback.
    with (
        patch("server.api.memories.get_embedding_provider", return_value=None),
        patch(
            "server.api.memories.repo.search_memories", new=AsyncMock(return_value=[])
        ),
    ):
        resp = await client.get(
            "/v1/memories/search",
            params={"subject_id": "u", "q": "hello", "semantic": "true"},
        )
    assert resp.status_code == 200
    assert resp.json()["search_mode"] == "text_fallback"


async def test_search_mode_text_fallback_when_provider_errors(client: AsyncClient):
    # provider present, but the embedding lookup raises → caught → text_fallback.
    with (
        patch("server.api.memories.get_embedding_provider", return_value=object()),
        patch(
            "server.services.embeddings.query_cache.cached_embed_query",
            new=AsyncMock(side_effect=RuntimeError("provider down")),
        ),
        patch(
            "server.api.memories.repo.search_memories", new=AsyncMock(return_value=[])
        ),
    ):
        resp = await client.get(
            "/v1/memories/search",
            params={"subject_id": "u", "q": "hello", "semantic": "true"},
        )
    assert resp.status_code == 200
    assert resp.json()["search_mode"] == "text_fallback"


async def test_search_mode_semantic_when_embedding_search_runs(client: AsyncClient):
    # provider + embedding + (hybrid) search all succeed → semantic.
    with (
        patch("server.api.memories.get_embedding_provider", return_value=object()),
        patch(
            "server.services.embeddings.query_cache.cached_embed_query",
            new=AsyncMock(return_value=[0.0] * 8),
        ),
        patch(
            "server.api.memories.repo.search_memories_hybrid",
            new=AsyncMock(return_value=[]),
        ),
    ):
        resp = await client.get(
            "/v1/memories/search",
            params={"subject_id": "u", "q": "hello", "semantic": "true"},
        )
    assert resp.status_code == 200
    assert resp.json()["search_mode"] == "semantic"
