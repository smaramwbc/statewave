"""Golden-path integration test: Record → Compile → Context → Govern.

This single test walks through the entire Statewave lifecycle for one subject,
asserting the product guarantees at each step.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Test data — a realistic support-agent scenario
# ---------------------------------------------------------------------------

_EPISODES = [
    {
        "source": "support-chat",
        "type": "conversation",
        "payload": {
            "messages": [
                {"role": "user", "content": "My name is Alice Chen and I work at Globex Corporation. I am on the Enterprise plan."},
                {"role": "assistant", "content": "Welcome Alice! How can I help you today?"},
            ]
        },
    },
    {
        "source": "support-chat",
        "type": "conversation",
        "payload": {
            "messages": [
                {"role": "user", "content": "I prefer email notifications over Slack. My email is alice@globex.com."},
                {"role": "assistant", "content": "Got it, I have updated your preference."},
            ]
        },
    },
    {
        "source": "support-chat",
        "type": "conversation",
        "payload": {
            "messages": [
                {"role": "user", "content": "We had a billing issue last week — we were double-charged for the March invoice."},
                {"role": "assistant", "content": "I see the double charge. I have initiated a refund for the duplicate payment."},
            ]
        },
    },
    {
        "source": "support-chat",
        "type": "conversation",
        "payload": {
            "messages": [
                {"role": "user", "content": "Can you help me upgrade our team from 5 to 20 seats?"},
                {"role": "assistant", "content": "Sure, I have upgraded your team to 20 seats. Your next invoice will reflect the change."},
            ]
        },
    },
]


# ---------------------------------------------------------------------------
# Golden path
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_full_lifecycle(client: AsyncClient, subject_id: str):
    """Record → Compile → Context → Delete — the complete Statewave loop."""

    # ── 1. Record episodes ────────────────────────────────────────────────
    episode_ids: list[str] = []
    for ep in _EPISODES:
        resp = await client.post("/v1/episodes", json={**ep, "subject_id": subject_id})
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["subject_id"] == subject_id
        assert data["id"]
        episode_ids.append(data["id"])

    assert len(episode_ids) == 4

    # ── 2. Compile memories ───────────────────────────────────────────────
    resp = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert resp.status_code == 200
    compile_data = resp.json()
    assert compile_data["memories_created"] > 0

    memories = compile_data["memories"]
    kinds = {m["kind"] for m in memories}
    assert "episode_summary" in kinds, "Should produce episode summaries"
    assert "profile_fact" in kinds, "Should extract profile facts"

    # Verify provenance — every memory must trace back to a source episode
    for mem in memories:
        assert len(mem["source_episode_ids"]) > 0, f"Memory {mem['id']} has no provenance"
        for src_id in mem["source_episode_ids"]:
            assert src_id in episode_ids, f"Provenance {src_id} not in recorded episodes"

    # Verify specific facts were extracted
    fact_contents = [m["content"] for m in memories if m["kind"] == "profile_fact"]
    all_facts = " ".join(fact_contents).lower()
    assert "alice" in all_facts, "Should extract user's name"
    assert "globex" in all_facts, "Should extract company"
    assert "enterprise" in all_facts, "Should extract plan"
    assert "email" in all_facts or "prefer" in all_facts, "Should extract notification preference"

    # ── 3. Context retrieval — full budget ────────────────────────────────
    resp = await client.post("/v1/context", json={
        "subject_id": subject_id,
        "task": "Customer is asking about their billing issue",
        "max_tokens": 500,
    })
    assert resp.status_code == 200
    ctx = resp.json()

    assert ctx["subject_id"] == subject_id
    assert ctx["task"] == "Customer is asking about their billing issue"
    assert ctx["token_estimate"] > 0
    assert ctx["token_estimate"] <= 500

    # assembled_context should contain structured sections
    text = ctx["assembled_context"]
    assert "## Task" in text
    assert "## About this user" in text

    # Identity facts must appear
    assert "Alice" in text
    assert "Globex" in text or "globex" in text
    assert "Enterprise" in text

    # Billing history should be present (highest relevance to task)
    assert "billing" in text.lower() or "double-charged" in text.lower()

    # Provenance must be populated
    assert len(ctx["provenance"]["fact_ids"]) > 0
    assert len(ctx["provenance"]["summary_ids"]) > 0

    # Episodes should be deduplicated — summaries cover them, so episodes list should be empty
    assert len(ctx["episodes"]) == 0, "Episodes covered by summaries should not appear"

    # ── 4. Context retrieval — tight budget ───────────────────────────────
    resp = await client.post("/v1/context", json={
        "subject_id": subject_id,
        "task": "Customer is asking about their billing issue",
        "max_tokens": 100,
    })
    assert resp.status_code == 200
    tight = resp.json()
    assert tight["token_estimate"] <= 100

    # Even with tight budget, the billing-relevant info should survive
    tight_text = tight["assembled_context"]
    assert "billing" in tight_text.lower() or "double" in tight_text.lower(), \
        "Task-relevant history should survive tight budget"

    # Fewer items should be included than the full-budget version
    tight_total = (
        len(tight["provenance"].get("fact_ids", []))
        + len(tight["provenance"].get("summary_ids", []))
        + len(tight["provenance"].get("episode_ids", []))
    )
    full_total = (
        len(ctx["provenance"]["fact_ids"])
        + len(ctx["provenance"]["summary_ids"])
        + len(ctx["provenance"]["episode_ids"])
    )
    assert tight_total <= full_total, "Tight budget should include fewer or equal items"

    # ── 5. Delete subject (Govern) ────────────────────────────────────────
    resp = await client.delete(f"/v1/subjects/{subject_id}")
    assert resp.status_code == 200
    del_data = resp.json()
    assert del_data["subject_id"] == subject_id
    assert del_data["episodes_deleted"] == 4
    assert del_data["memories_deleted"] == compile_data["memories_created"]

    # ── 6. Verify subject is truly gone ───────────────────────────────────
    resp = await client.get("/v1/timeline", params={"subject_id": subject_id})
    assert resp.status_code == 200
    timeline = resp.json()
    assert len(timeline["episodes"]) == 0, "Episodes should be deleted"
    assert len(timeline["memories"]) == 0, "Memories should be deleted"

    resp = await client.post("/v1/context", json={
        "subject_id": subject_id,
        "task": "anything",
    })
    assert resp.status_code == 200
    empty_ctx = resp.json()
    assert empty_ctx["token_estimate"] > 0  # task header has tokens
    assert len(empty_ctx["facts"]) == 0
    assert len(empty_ctx["episodes"]) == 0
