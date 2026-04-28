"""Edge-case integration tests — idempotency, degradation, ordering, filtering."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _ingest(
    client: AsyncClient, subject_id: str, payload: dict, source: str = "test"
) -> dict:
    resp = await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "source": source,
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
# Idempotent compilation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_compile_twice_is_idempotent(client: AsyncClient, subject_id: str):
    """Compiling the same subject twice should not create duplicate memories."""
    await _ingest(
        client,
        subject_id,
        {"messages": [{"role": "user", "content": "My name is Bob and I work at Initech."}]},
    )

    first = await _compile(client, subject_id)
    assert first["memories_created"] > 0

    second = await _compile(client, subject_id)
    assert second["memories_created"] == 0, "Second compile should produce zero new memories"

    # Total memories should equal first compile
    resp = await client.get("/v1/timeline", params={"subject_id": subject_id})
    timeline = resp.json()
    assert len(timeline["memories"]) == first["memories_created"]

    await _cleanup(client, subject_id)


@pytest.mark.anyio
async def test_compile_after_new_episode(client: AsyncClient, subject_id: str):
    """Adding a new episode after compile should produce new memories on recompile."""
    await _ingest(
        client, subject_id, {"messages": [{"role": "user", "content": "My name is Carol."}]}
    )
    first = await _compile(client, subject_id)
    count_1 = first["memories_created"]

    await _ingest(
        client, subject_id, {"messages": [{"role": "user", "content": "I prefer dark mode."}]}
    )
    second = await _compile(client, subject_id)
    assert second["memories_created"] > 0, "New episode should produce new memories"

    resp = await client.get("/v1/timeline", params={"subject_id": subject_id})
    total_memories = len(resp.json()["memories"])
    assert total_memories == count_1 + second["memories_created"]

    await _cleanup(client, subject_id)


# ---------------------------------------------------------------------------
# Token budget degradation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_very_small_budget_does_not_crash(client: AsyncClient, subject_id: str):
    """A budget of 1 token should return a valid (mostly empty) response, not an error."""
    await _ingest(
        client, subject_id, {"messages": [{"role": "user", "content": "My name is Dave."}]}
    )
    await _compile(client, subject_id)

    resp = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "help",
            "max_tokens": 1,
        },
    )
    assert resp.status_code == 200
    ctx = resp.json()
    # With 1 token budget, assembled context may just be truncated or minimal
    assert "assembled_context" in ctx
    assert ctx["token_estimate"] >= 0

    await _cleanup(client, subject_id)


@pytest.mark.anyio
async def test_context_without_compile(client: AsyncClient, subject_id: str):
    """Requesting context before any compilation should return an empty but valid bundle."""
    await _ingest(client, subject_id, {"messages": [{"role": "user", "content": "Hello world."}]})

    # No compile step — context should still work
    resp = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "greet the user",
        },
    )
    assert resp.status_code == 200
    ctx = resp.json()
    # Should have no facts or summaries, but may include raw episodes
    assert len(ctx["facts"]) == 0

    await _cleanup(client, subject_id)


# ---------------------------------------------------------------------------
# Empty / unknown payloads
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_empty_payload_does_not_crash(client: AsyncClient, subject_id: str):
    """Ingesting and compiling an empty payload should not error."""
    await _ingest(client, subject_id, {})
    result = await _compile(client, subject_id)
    assert result["memories_created"] == 0

    await _cleanup(client, subject_id)


@pytest.mark.anyio
async def test_unknown_payload_shape(client: AsyncClient, subject_id: str):
    """A payload with unrecognized keys should compile without error."""
    await _ingest(client, subject_id, {"foo": "bar", "nested": {"a": 1}})
    result = await _compile(client, subject_id)
    assert result["memories_created"] == 0

    await _cleanup(client, subject_id)


# ---------------------------------------------------------------------------
# Timeline ordering
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_timeline_chronological_order(client: AsyncClient, subject_id: str):
    """Timeline episodes should be in chronological (ascending) order."""
    for i in range(5):
        await _ingest(client, subject_id, {"text": f"Event number {i}"})

    resp = await client.get("/v1/timeline", params={"subject_id": subject_id})
    assert resp.status_code == 200
    episodes = resp.json()["episodes"]
    assert len(episodes) == 5

    timestamps = [ep["created_at"] for ep in episodes]
    assert timestamps == sorted(timestamps), "Episodes should be in ascending chronological order"

    await _cleanup(client, subject_id)


# ---------------------------------------------------------------------------
# Search filtering
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_search_filters_by_kind(client: AsyncClient, subject_id: str):
    """Search with kind filter should return only that kind."""
    await _ingest(
        client,
        subject_id,
        {
            "messages": [
                {"role": "user", "content": "My name is Eve and I work at Stark Industries."}
            ]
        },
    )
    await _compile(client, subject_id)

    # Search for profile_fact only
    resp = await client.get(
        "/v1/memories/search",
        params={
            "subject_id": subject_id,
            "kind": "profile_fact",
        },
    )
    assert resp.status_code == 200
    facts = resp.json()["memories"]
    assert len(facts) > 0
    assert all(m["kind"] == "profile_fact" for m in facts)

    # Search for episode_summary only
    resp = await client.get(
        "/v1/memories/search",
        params={
            "subject_id": subject_id,
            "kind": "episode_summary",
        },
    )
    summaries = resp.json()["memories"]
    assert len(summaries) > 0
    assert all(m["kind"] == "episode_summary" for m in summaries)

    # Search for a kind with no results
    resp = await client.get(
        "/v1/memories/search",
        params={
            "subject_id": subject_id,
            "kind": "procedure",
        },
    )
    assert resp.json()["memories"] == []

    await _cleanup(client, subject_id)


@pytest.mark.anyio
async def test_search_by_query(client: AsyncClient, subject_id: str):
    """Text search should filter memories by content."""
    await _ingest(
        client,
        subject_id,
        {
            "messages": [
                {"role": "user", "content": "My name is Frank and I work at Wayne Enterprises."}
            ]
        },
    )
    await _compile(client, subject_id)

    resp = await client.get(
        "/v1/memories/search",
        params={
            "subject_id": subject_id,
            "q": "Wayne",
        },
    )
    results = resp.json()["memories"]
    assert len(results) > 0
    assert any("Wayne" in m["content"] for m in results)

    # Search for something not present
    resp = await client.get(
        "/v1/memories/search",
        params={
            "subject_id": subject_id,
            "q": "xyznonexistent",
        },
    )
    assert resp.json()["memories"] == []

    await _cleanup(client, subject_id)


# ---------------------------------------------------------------------------
# Healthz / Readyz
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_healthz(client: AsyncClient):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_readyz(client: AsyncClient):
    resp = await client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


# ---------------------------------------------------------------------------
# Structured errors
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_validation_error_returns_structured_json(client: AsyncClient):
    """Missing required fields should return a structured error, not a raw 422."""
    resp = await client.post("/v1/episodes", json={})
    assert resp.status_code == 422
    body = resp.json()
    assert "error" in body
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["message"] == "Request validation failed"
    assert isinstance(body["error"]["details"], list)


@pytest.mark.anyio
async def test_request_id_in_response_header(client: AsyncClient):
    """Every response should include X-Request-ID."""
    resp = await client.get("/healthz")
    assert "x-request-id" in resp.headers


@pytest.mark.anyio
async def test_custom_request_id_propagated(client: AsyncClient):
    """If the client sends X-Request-ID, it should be echoed back."""
    resp = await client.get("/healthz", headers={"X-Request-ID": "test-req-42"})
    assert resp.headers["x-request-id"] == "test-req-42"


# ---------------------------------------------------------------------------
# Subject listing
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_list_subjects(client: AsyncClient, subject_id: str):
    """GET /v1/subjects returns known subjects with counts."""
    await _ingest(client, subject_id, {"messages": [{"role": "user", "content": "Hello"}]})
    await _compile(client, subject_id)

    resp = await client.get("/v1/subjects")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] > 0
    found = [s for s in data["subjects"] if s["subject_id"] == subject_id]
    assert len(found) == 1
    assert found[0]["episode_count"] >= 1
    assert found[0]["memory_count"] >= 1

    await _cleanup(client, subject_id)


@pytest.mark.anyio
async def test_list_subjects_empty(client: AsyncClient):
    """GET /v1/subjects with pagination returns valid shape."""
    resp = await client.get("/v1/subjects", params={"limit": 1, "offset": 99999})
    assert resp.status_code == 200
    data = resp.json()
    assert data["subjects"] == []
    assert data["total"] == 0
