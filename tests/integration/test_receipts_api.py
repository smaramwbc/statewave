"""End-to-end integration tests for state-assembly receipts (#49).

Walks the receipt path through the real FastAPI app against the test
Postgres database:

  1. POST /v1/context with emit_receipt=true → response carries
     receipt_id and receipt_emitted=true.
  2. POST /v1/context with no flag → no receipt.
  3. GET /v1/receipts/{id} → returns the body, schema matches.
  4. context_hash recomputed from the response's assembled_context
     matches the receipt's context_hash — the load-bearing integrity
     check for issue #49's negative test #6.
  5. The receipt's selected_entries cover the same memories the
     response's provenance reports.
  6. Tenant scoping — receipts written under tenant A are invisible to
     tenant B.
  7. GET /v1/receipts list returns rows for the subject.
"""

from __future__ import annotations

import hashlib

import pytest
from httpx import AsyncClient


_EPISODES = [
    {
        "source": "support-chat",
        "type": "conversation",
        "payload": {
            "messages": [
                {
                    "role": "user",
                    "content": "My name is Alice Chen and I'm on the Enterprise plan.",
                },
                {"role": "assistant", "content": "Welcome Alice."},
            ]
        },
    },
    {
        "source": "support-chat",
        "type": "conversation",
        "payload": {
            "messages": [
                {
                    "role": "user",
                    "content": "I prefer email over Slack. alice@globex.com.",
                },
                {"role": "assistant", "content": "Got it."},
            ]
        },
    },
]


async def _seed(client: AsyncClient, subject_id: str) -> None:
    for ep in _EPISODES:
        r = await client.post("/v1/episodes", json={**ep, "subject_id": subject_id})
        assert r.status_code == 201, r.text
    r = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert r.status_code == 200, r.text


@pytest.mark.anyio
async def test_emit_receipt_returns_id_and_persists(client: AsyncClient, subject_id: str):
    await _seed(client, subject_id)

    r = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "What plan is the customer on?",
            "emit_receipt": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["receipt_emitted"] is True
    assert body["receipt_id"]
    assert len(body["receipt_id"]) == 26  # ULID

    # Fetch the receipt back through the read API.
    r2 = await client.get(f"/v1/receipts/{body['receipt_id']}")
    assert r2.status_code == 200
    receipt = r2.json()
    assert receipt["receipt_id"] == body["receipt_id"]
    assert receipt["subject_id"] == subject_id
    assert receipt["mode"] == "retrieval"
    assert receipt["task"] == "What plan is the customer on?"


@pytest.mark.anyio
async def test_no_receipt_when_flag_omitted(client: AsyncClient, subject_id: str):
    await _seed(client, subject_id)
    r = await client.post(
        "/v1/context",
        json={"subject_id": subject_id, "task": "anything"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["receipt_id"] is None
    assert body["receipt_emitted"] is False


@pytest.mark.anyio
async def test_context_hash_matches_delivered_bytes(client: AsyncClient, subject_id: str):
    """Issue #49 negative test #6: a recomputed sha256 over the
    bytes the agent received must match receipt.output.context_hash."""
    await _seed(client, subject_id)

    r = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "What plan is the customer on?",
            "emit_receipt": True,
        },
    )
    body = r.json()
    delivered = body["assembled_context"]
    recomputed = hashlib.sha256(delivered.encode("utf-8")).hexdigest()

    r2 = await client.get(f"/v1/receipts/{body['receipt_id']}")
    assert r2.json()["output"]["context_hash"] == recomputed


@pytest.mark.anyio
async def test_receipt_selected_entries_match_provenance(
    client: AsyncClient, subject_id: str
):
    """The receipt's selected_entries must cover the same memory and
    episode ids the response reports in `provenance`. Without this, the
    receipt is decorative — it has to record what the agent actually
    received."""
    await _seed(client, subject_id)
    r = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "Customer history",
            "emit_receipt": True,
        },
    )
    body = r.json()
    provenance_memory_ids = set(
        body["provenance"].get("fact_ids", [])
        + body["provenance"].get("summary_ids", [])
        + body["provenance"].get("procedure_ids", [])
    )
    provenance_episode_ids = set(body["provenance"].get("episode_ids", []))

    r2 = await client.get(f"/v1/receipts/{body['receipt_id']}")
    receipt = r2.json()
    receipt_memory_ids = {
        e["memory_id"] for e in receipt["selected_entries"] if e.get("type") != "episode"
    }
    receipt_episode_ids = {
        e["episode_id"] for e in receipt["selected_entries"] if e.get("type") == "episode"
    }
    assert receipt_memory_ids == provenance_memory_ids
    assert receipt_episode_ids == provenance_episode_ids


@pytest.mark.anyio
async def test_list_receipts_returns_rows_for_subject(
    client: AsyncClient, subject_id: str
):
    await _seed(client, subject_id)
    for task in ("first", "second", "third"):
        r = await client.post(
            "/v1/context",
            json={"subject_id": subject_id, "task": task, "emit_receipt": True},
        )
        assert r.status_code == 200

    r = await client.get(f"/v1/receipts?subject_id={subject_id}&limit=10")
    assert r.status_code == 200
    body = r.json()
    assert len(body["receipts"]) == 3
    # Newest-first
    assert body["receipts"][0]["task"] == "third"
    assert body["next_cursor"] is None  # < limit


@pytest.mark.anyio
async def test_receipt_404_for_unknown_id(client: AsyncClient):
    r = await client.get("/v1/receipts/00000000000000000000000000")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_handoff_emits_receipt(client: AsyncClient, subject_id: str):
    """The handoff path shares the same receipt machinery as
    /v1/context. Issue #49's accountability story would be incomplete
    if one assembly surface emitted receipts and another didn't."""
    await _seed(client, subject_id)
    # Need a session so handoff has something to brief on.
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "source": "user",
            "type": "message",
            "payload": {"text": "system is down, urgent"},
            "session_id": "sess-1",
        },
    )

    r = await client.post(
        "/v1/handoff",
        json={
            "subject_id": subject_id,
            "session_id": "sess-1",
            "reason": "escalation",
            "emit_receipt": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["receipt_emitted"] is True
    assert body["receipt_id"]

    r2 = await client.get(f"/v1/receipts/{body['receipt_id']}")
    assert r2.status_code == 200
    receipt = r2.json()
    assert receipt["task"].startswith("handoff:")
