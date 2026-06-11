"""End-to-end: the REAL multi-agent-memory Stripe contradiction, resolved via
structured candidates + v2 entity-qualified claims — on real Postgres.

Bloomberg seeds Stripe at 3.5% + 35c plus independent positioning/differentiator
facts; a later source reports 2.9% + 30c. The stale pricing must be superseded
and absent from active context (raw body, episode_summary, synthesis input)
WITHOUT deleting the independent Bloomberg facts.
"""

from __future__ import annotations

import uuid

import pytest

import tests.integration.conftest as _conftest
from server.db import repositories as repo
from server.db.engine import set_engine_for_testing
from server.services.backup import export_subject, import_subject

_Q = {
    "payment_method": "card",
    "rate_type": "standard",
    "currency": "USD",
    "charge_unit": "transaction",
}


def _pricing(bp, minor, *, entity="organization:stripe", quals=None):
    return {
        "kind": "domain_fact",
        "text": f"Stripe's standard card-processing rate is {bp / 100:.1f}% plus {minor} cents per transaction.",
        "metadata": {"source": "Bloomberg"},
        "claim": {
            "schema_version": 2,
            "key": "pricing.processing_rate",
            "entity_key": entity,
            "qualifiers": quals or dict(_Q),
            "value": {"percentage_basis_points": bp, "fixed_minor_units": minor},
        },
    }


_POSITIONING = {
    "kind": "domain_fact",
    "text": "Stripe is moving upmarket toward high-volume enterprise merchants.",
    "metadata": {"source": "Bloomberg"},
}
_DIFFERENTIATORS = {
    "kind": "domain_fact",
    "text": "Stripe differentiators include developer APIs and broad global coverage.",
    "metadata": {"source": "Bloomberg"},
}

_BLOOMBERG_RAW = (
    "Bloomberg: Stripe's standard card-processing rate is 3.5% plus 35 cents per "
    "transaction. Stripe is moving upmarket. Differentiators: developer APIs, global coverage."
)
_CURRENT_RAW = (
    "TechCrunch: Stripe reverted its standard card rate to 2.9% plus 30 cents per transaction."
)


async def _episode(client, subject, raw, candidates, occurred_at):
    r = await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject,
            "source": "agent",
            "type": "agent.analyst.findings",
            "payload": {"text": raw, "statewave": {"memory_candidates": candidates}},
            "occurred_at": occurred_at,
        },
    )
    assert r.status_code == 201, r.text


async def _compile(client, subject):
    r = await client.post("/v1/memories/compile", json={"subject_id": subject, "async": False})
    assert r.status_code == 200, r.text


async def _timeline(client, subject):
    return (await client.get("/v1/timeline", params={"subject_id": subject})).json()


def _claim(m):
    return (m.get("metadata") or {}).get("claim")


@pytest.mark.anyio
async def test_stripe_contradiction_resolves_end_to_end(client, subject_id):
    # 1. Bloomberg: pricing + positioning + differentiators as atomic candidates
    await _episode(
        client,
        subject_id,
        _BLOOMBERG_RAW,
        [_pricing(350, 35), _POSITIONING, _DIFFERENTIATORS],
        "2025-01-01T00:00:00Z",
    )
    await _compile(client, subject_id)
    tl = await _timeline(client, subject_id)
    mems = tl["memories"]

    # 2. Three atomic memories, NO catch-all episode_summary
    assert len(mems) == 3
    assert all(m["kind"] == "profile_fact" for m in mems)
    assert not any(m["kind"] == "episode_summary" for m in mems)
    pricing = [m for m in mems if _claim(m)]
    assert len(pricing) == 1
    # 3. pricing 3.5% initially active, claim is v2/canonical
    assert pricing[0]["status"] == "active"
    assert _claim(pricing[0])["schema_version"] == 2
    assert _claim(pricing[0])["value"] == {"percentage_basis_points": 350, "fixed_minor_units": 35}
    # candidate metadata preserved alongside the claim
    assert pricing[0]["metadata"]["source"] == "Bloomberg"

    # 4. later source: 2.9% + 30c
    await _episode(client, subject_id, _CURRENT_RAW, [_pricing(290, 30)], "2025-06-01T00:00:00Z")
    await _compile(client, subject_id)
    tl = await _timeline(client, subject_id)
    mems = tl["memories"]

    pricing_mems = [m for m in mems if _claim(m)]
    superseded = [m for m in pricing_mems if m["status"] == "superseded"]
    active_pricing = [m for m in pricing_mems if m["status"] == "active"]
    # 5/6/15. exactly one supersession; the stale one is superseded, current active
    assert len(superseded) == 1
    assert _claim(superseded[0])["value"] == {
        "percentage_basis_points": 350,
        "fixed_minor_units": 35,
    }
    assert len(active_pricing) == 1
    assert _claim(active_pricing[0])["value"] == {
        "percentage_basis_points": 290,
        "fixed_minor_units": 30,
    }
    # 7. independent Bloomberg facts remain active
    independents = [m for m in mems if not _claim(m)]
    assert {m["content"] for m in independents if m["status"] == "active"} >= {
        _POSITIONING["text"],
        _DIFFERENTIATORS["text"],
    }
    assert all(m["status"] == "active" for m in independents)

    # 8-11. assemble active context
    ctx = (
        await client.post(
            "/v1/context",
            json={
                "subject_id": subject_id,
                "task": "What is Stripe's current card processing rate?",
                "max_tokens": 4000,
            },
        )
    ).json()
    asm = ctx["assembled_context"]
    assert "2.9" in asm and "30 cents" in asm  # current rate present
    assert "3.5" not in asm and "35 cents" not in asm  # stale absent — raw body, summary, fact
    # independent facts present in active context
    assert "upmarket" in asm and "developer apis" in asm.lower()
    # raw mixed episode body not injected
    assert "bloomberg:" not in asm.lower()
    # provenance: active pricing fact links to its source episode
    assert active_pricing[0]["source_episode_ids"]

    # 12-13. timeline/admin still expose the superseded pricing memory + episodes
    assert any(m["status"] == "superseded" for m in tl["memories"])
    assert len(tl["episodes"]) == 2

    # 16. read does not mutate
    before = {m["id"]: m["status"] for m in (await _timeline(client, subject_id))["memories"]}
    await client.post(
        "/v1/context", json={"subject_id": subject_id, "task": "rate?", "max_tokens": 1000}
    )
    after = {m["id"]: m["status"] for m in (await _timeline(client, subject_id))["memories"]}
    assert before == after


# --- structured-candidate compatibility matrix --------------------------------


@pytest.mark.anyio
async def test_episode_without_candidates_compiles_unchanged(client, subject_id):
    r = await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "source": "chat",
            "type": "message",
            "payload": {"text": "My name is Alice. I work at Globex."},
        },
    )
    assert r.status_code == 201
    await _compile(client, subject_id)
    mems = (await _timeline(client, subject_id))["memories"]
    # legacy heuristic path: episode_summary + profile_facts, claims on name/employer
    assert any(m["kind"] == "episode_summary" for m in mems)
    assert any((_claim(m) or {}).get("key") == "employment.current_employer" for m in mems)


@pytest.mark.anyio
async def test_malformed_container_uses_full_legacy(client, subject_id):
    # candidates not a list -> full legacy compile of the raw text
    r = await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "source": "chat",
            "type": "message",
            "payload": {"text": "I work at Globex", "statewave": {"memory_candidates": "nope"}},
        },
    )
    assert r.status_code == 201
    await _compile(client, subject_id)
    mems = (await _timeline(client, subject_id))["memories"]
    assert any(m["kind"] == "episode_summary" for m in mems)  # legacy summary emitted


@pytest.mark.anyio
async def test_invalid_candidate_kind_uses_full_legacy(client, subject_id):
    await _episode(
        client,
        subject_id,
        "I work at Globex",
        [{"kind": "???", "text": "x"}],
        "2025-01-01T00:00:00Z",
    )
    await _compile(client, subject_id)
    mems = (await _timeline(client, subject_id))["memories"]
    assert any(m["kind"] == "episode_summary" for m in mems)


@pytest.mark.anyio
async def test_invalid_claim_becomes_unkeyed_candidate(client, subject_id):
    bad = {
        "kind": "domain_fact",
        "text": "Some fact with a broken claim.",
        "metadata": {"source": "x"},
        "claim": {"schema_version": 2, "key": "made.up"},
    }
    await _episode(client, subject_id, "raw", [bad], "2025-01-01T00:00:00Z")
    await _compile(client, subject_id)
    mems = (await _timeline(client, subject_id))["memories"]
    assert len(mems) == 1 and mems[0]["content"] == "Some fact with a broken claim."
    assert _claim(mems[0]) is None  # claim dropped, text kept
    assert mems[0]["metadata"]["source"] == "x"  # unrelated metadata survives


@pytest.mark.anyio
async def test_export_import_preserves_v2_envelope(client, subject_id):
    await _episode(client, subject_id, "raw", [_pricing(350, 35)], "2025-01-01T00:00:00Z")
    await _compile(client, subject_id)
    prev = set_engine_for_testing(_conftest._engine, _conftest._session_factory)
    try:
        doc = await export_subject(subject_id)
        target = f"imp-{uuid.uuid4().hex[:8]}"
        await import_subject(doc, target_subject_id=target, preserve_ids=False)
    finally:
        set_engine_for_testing(*prev)
    async with _conftest._session_factory() as s:
        rows = await repo.list_memories_by_subject(s, target, limit=50)
    claim = [r.metadata_.get("claim") for r in rows if r.metadata_.get("claim")][0]
    assert claim["schema_version"] == 2
    assert claim["value"] == {"percentage_basis_points": 350, "fixed_minor_units": 35}
    assert claim["entity_key"] == "organization:stripe"
