"""Integration tests for semantic search and embedding-enhanced context."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _ingest(client: AsyncClient, subject_id: str, payload: dict) -> dict:
    resp = await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "source": "test",
            "type": "conversation",
            "payload": payload,
        },
    )
    assert resp.status_code == 201
    return resp.json()


async def _compile(client: AsyncClient, subject_id: str) -> dict:
    resp = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert resp.status_code == 200
    return resp.json()


async def _cleanup(client: AsyncClient, subject_id: str):
    await client.delete(f"/v1/subjects/{subject_id}")


# ---------------------------------------------------------------------------
# Embedding generation during compilation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_compile_generates_embeddings(client: AsyncClient, subject_id: str):
    """Compiled memories should have embeddings when provider is enabled (stub by default)."""
    await _ingest(
        client,
        subject_id,
        {"messages": [{"role": "user", "content": "My name is Alice and I work at Globex."}]},
    )
    result = await _compile(client, subject_id)
    assert result["memories_created"] > 0

    # Check timeline to see memories exist
    resp = await client.get("/v1/timeline", params={"subject_id": subject_id})
    timeline = resp.json()
    assert len(timeline["memories"]) > 0

    await _cleanup(client, subject_id)


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_semantic_search_returns_results(client: AsyncClient, subject_id: str):
    """Semantic search should return memories when embeddings exist."""
    await _ingest(
        client,
        subject_id,
        {"messages": [{"role": "user", "content": "My name is Bob and I work at Initech."}]},
    )
    await _compile(client, subject_id)

    resp = await client.get(
        "/v1/memories/search",
        params={
            "subject_id": subject_id,
            "q": "who works at Initech",
            "semantic": "true",
        },
    )
    assert resp.status_code == 200
    results = resp.json()["memories"]
    assert len(results) > 0

    await _cleanup(client, subject_id)


@pytest.mark.anyio
async def test_semantic_search_respects_kind_filter(client: AsyncClient, subject_id: str):
    """Semantic search with kind filter should only return that kind."""
    await _ingest(
        client,
        subject_id,
        {"messages": [{"role": "user", "content": "My name is Carol and I live in Paris."}]},
    )
    await _compile(client, subject_id)

    resp = await client.get(
        "/v1/memories/search",
        params={
            "subject_id": subject_id,
            "q": "where does Carol live",
            "semantic": "true",
            "kind": "profile_fact",
        },
    )
    assert resp.status_code == 200
    results = resp.json()["memories"]
    assert all(m["kind"] == "profile_fact" for m in results)

    await _cleanup(client, subject_id)


@pytest.mark.anyio
async def test_text_search_still_works(client: AsyncClient, subject_id: str):
    """Non-semantic search should still work as before (text ILIKE)."""
    await _ingest(
        client,
        subject_id,
        {
            "messages": [
                {"role": "user", "content": "My name is Dave and I work at Wayne Enterprises."}
            ]
        },
    )
    await _compile(client, subject_id)

    # Without semantic flag — classic text search
    resp = await client.get(
        "/v1/memories/search",
        params={
            "subject_id": subject_id,
            "q": "Wayne",
        },
    )
    assert resp.status_code == 200
    results = resp.json()["memories"]
    assert len(results) > 0
    assert any("Wayne" in m["content"] for m in results)

    await _cleanup(client, subject_id)


@pytest.mark.anyio
async def test_semantic_false_uses_text_search(client: AsyncClient, subject_id: str):
    """Explicitly setting semantic=false should use text search."""
    await _ingest(
        client, subject_id, {"messages": [{"role": "user", "content": "My name is Eve."}]}
    )
    await _compile(client, subject_id)

    resp = await client.get(
        "/v1/memories/search",
        params={
            "subject_id": subject_id,
            "q": "Eve",
            "semantic": "false",
        },
    )
    assert resp.status_code == 200
    results = resp.json()["memories"]
    assert len(results) > 0

    await _cleanup(client, subject_id)


# ---------------------------------------------------------------------------
# Context with semantic ranking
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_context_assembly_with_embeddings(client: AsyncClient, subject_id: str):
    """Context assembly should work with semantic scoring."""
    await _ingest(
        client,
        subject_id,
        {
            "messages": [
                {"role": "user", "content": "My name is Frank and I work at Stark Industries."},
                {"role": "assistant", "content": "Welcome Frank!"},
            ]
        },
    )
    await _ingest(
        client,
        subject_id,
        {
            "messages": [
                {"role": "user", "content": "I had a billing issue last week."},
                {"role": "assistant", "content": "I'll look into that for you."},
            ]
        },
    )
    await _compile(client, subject_id)

    resp = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "Help with billing inquiry",
            "max_tokens": 500,
        },
    )
    assert resp.status_code == 200
    ctx = resp.json()
    assert ctx["token_estimate"] > 0
    assert ctx["token_estimate"] <= 500
    assert "assembled_context" in ctx

    await _cleanup(client, subject_id)


# ---------------------------------------------------------------------------
# Graceful fallback
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_context_works_without_embeddings(client: AsyncClient, subject_id: str):
    """Context should still work when no embeddings exist (pre-existing data)."""
    # Ingest but don't compile — no memories, no embeddings
    await _ingest(client, subject_id, {"messages": [{"role": "user", "content": "Hello world."}]})

    resp = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "Greet the user",
        },
    )
    assert resp.status_code == 200
    ctx = resp.json()
    assert ctx["token_estimate"] > 0
    assert len(ctx["facts"]) == 0  # no compiled memories

    await _cleanup(client, subject_id)


# ---------------------------------------------------------------------------
# Candidate-pool union — regression guard against the recency-only selector
# ---------------------------------------------------------------------------
#
# Before this fix, `/v1/context` selected candidates by `created_at DESC
# LIMIT 50` per kind, then computed semantic scores against ONLY those
# candidates. Memories outside the recency window — but with high
# semantic relevance — were never scored. Live failure observed against
# statewave-support-docs: the GPU-relevant profile_facts sit at positions
# 127-133 of 160 by recency and were locked out of the candidate pool no
# matter how good their cosine match.
#
# This test ingests >50 facts (so the GPU-relevant fact is pushed out
# of the recency window), then asks a /v1/context query specifically
# about GPUs. The fact MUST appear in the assembled context — proving
# the semantic candidates entered the pool.


@pytest.mark.anyio
async def test_context_includes_semantically_relevant_facts_outside_recency_window(
    client: AsyncClient, subject_id: str
):
    """A semantically-relevant memory MUST be retrievable even when it's
    older than the top-50-by-recency cutoff."""
    # 1. Ingest the load-bearing memory FIRST so it ages out
    await _ingest(
        client,
        subject_id,
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Statewave's API process is CPU-only and does not require a GPU.",
                },
                {"role": "assistant", "content": "Acknowledged."},
            ]
        },
    )
    # 2. Compile so it produces a memory
    await _compile(client, subject_id)

    # 3. Now flood the subject with 60 unrelated memories so the GPU
    #    fact is pushed out of the recency-50 window for profile_facts.
    for i in range(60):
        await _ingest(
            client,
            subject_id,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"My favourite color today is filler-color-{i:03d}.",
                    }
                ]
            },
        )
    await _compile(client, subject_id)

    # 4. Ask a query that semantically targets the early memory.
    resp = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "Do I need a GPU for Statewave?",
            "max_tokens": 800,
        },
    )
    assert resp.status_code == 200
    ctx = resp.json()

    # 5. The assembled context must contain the GPU/CPU-only signal.
    #    Pre-fix this would fail: the GPU memory was in semantic_results
    #    but never entered the candidate pool, so it never scored, so it
    #    never made it into the bundle.
    assembled_lower = (ctx.get("assembled_context") or "").lower()
    facts_blob = " ".join(
        (m.get("content") or "").lower() for m in ctx.get("facts", [])
    )
    haystack = assembled_lower + " " + facts_blob
    assert (
        "cpu-only" in haystack or "gpu" in haystack
    ), (
        "The GPU-relevant memory was ingested first (so it's old) and is "
        "pushed out of the recency-50 candidate window by 60 newer fillers. "
        "If the semantic candidates aren't being unioned into the pool, "
        "this assertion fails — exact regression we're guarding against."
    )

    await _cleanup(client, subject_id)
