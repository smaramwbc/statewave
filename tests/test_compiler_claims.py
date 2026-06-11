"""Compiler claim emission — heuristic + LLM.

Optimized for near-zero wrongful authoritative claims: a memory without a
confidently recognized claim is emitted exactly as before (no claim metadata).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from unittest.mock import AsyncMock, patch

from server.db.tables import EpisodeRow
from server.services.compilers.heuristic import HeuristicCompiler
from server.services.compilers.llm import LLMCompiler, _llm_claim_metadata, _safe_dt
from server.services.conflicts import resolve_conflicts

_NOW = datetime.now(timezone.utc)


def _ep(text: str) -> EpisodeRow:
    return EpisodeRow(
        id=uuid.uuid4(),
        subject_id="user-1",
        source="test",
        type="conversation",
        payload={"text": text},
        metadata_={},
        provenance={},
        created_at=_NOW,
    )


def _facts(text):
    return [m for m in HeuristicCompiler().compile([_ep(text)]) if m.kind == "profile_fact"]


def _claims(text):
    return [m.metadata_.get("claim") for m in _facts(text)]


# --- heuristic positives ------------------------------------------------------


def test_heuristic_name_claim():
    (f,) = _facts("My name is Alice Chen")
    assert f.content == "My name is Alice Chen"  # text unchanged
    assert f.metadata_["claim"]["key"] == "identity.name"
    assert f.metadata_["claim"]["value"] == "alice chen"
    assert f.metadata_["claim"]["scope"] == "single"
    assert f.metadata_["claim"]["source"] == "heuristic"
    assert f.metadata_["claim"]["schema_version"] == 1


def test_heuristic_employer_claim():
    (f,) = _facts("I work at Globex")
    assert f.metadata_["claim"]["key"] == "employment.current_employer"
    assert f.metadata_["claim"]["value"] == "globex"


def test_heuristic_home_claim():
    (f,) = _facts("I live in Berlin")
    assert f.metadata_["claim"]["key"] == "location.current_home"
    assert f.metadata_["claim"]["value"] == "berlin"


def test_heuristic_no_temporal_invented():
    (f,) = _facts("I work at Globex")
    assert "valid_from" not in f.metadata_["claim"]
    assert "valid_to" not in f.metadata_["claim"]


# --- heuristic negatives: fact emitted, NO claim ------------------------------


@pytest.mark.parametrize(
    "text",
    [
        'She said, "I work at Acme".',  # reported speech / quoted
        "I'm at the gym",  # i'm at -> not employer
        "I'm from Berlin",  # origin -> not current home
        "I am Bob",  # i am -> not a name claim
        "I use Stripe",  # generic tool use -> unkeyed
        "I prefer email",  # preference -> unkeyed
        "I work at Acme but might leave soon",  # uncertain
    ],
)
def test_heuristic_emits_fact_without_claim(text):
    # The memory is still extracted (behavior unchanged); it just carries no claim.
    facts = _facts(text)
    assert facts, "expected the fact to still be emitted"
    assert all(f.metadata_ == {} for f in facts)


@pytest.mark.parametrize(
    "text",
    [
        "My employer is not Acme.",
        "I might work at Acme.",
        "Alice works at Acme.",
        "I previously worked at Acme.",
        "I am considering moving to Berlin.",
    ],
)
def test_heuristic_never_claims_negated_hypothetical_historical_thirdparty(text):
    assert all(c is None for c in _claims(text))


def test_heuristic_generic_stripe_is_not_a_billing_claim():
    assert all(c is None for c in _claims("I use Stripe and I prefer PayPal"))


def test_heuristic_preserves_unrelated_nothing_lost():
    # No claim case keeps metadata exactly {} (no spurious keys).
    facts = _facts("I use Stripe")
    assert facts and all(f.metadata_ == {} for f in facts)


# --- LLM claim validation (registry is authoritative, model is untrusted) -----


def test_llm_valid_canonical_claim():
    md = _llm_claim_metadata({"claim": {"key": "employer", "value": "Globex"}})
    assert md["claim"]["key"] == "employment.current_employer"
    assert md["claim"]["value"] == "globex"
    assert md["claim"]["source"] == "llm"


def test_llm_alias_canonicalized():
    md = _llm_claim_metadata({"claim": {"key": "works_at", "value": "Acme"}})
    assert md["claim"]["key"] == "employment.current_employer"


def test_llm_proposed_scope_overridden_by_registry():
    # Model lies (scope=single) on a multi-valued key — registry wins.
    md = _llm_claim_metadata(
        {"claim": {"key": "payment.processors_used", "value": "Stripe", "scope": "single"}}
    )
    assert md["claim"]["scope"] == "multi"


def test_llm_unknown_key_omitted():
    assert _llm_claim_metadata({"claim": {"key": "shoe_size", "value": "10"}}) is None


def test_llm_malformed_omitted():
    assert _llm_claim_metadata({"claim": {"key": 123, "value": "x"}}) is None
    assert _llm_claim_metadata({"claim": "nope"}) is None
    assert _llm_claim_metadata({}) is None


def test_llm_temporal_parsed_when_present():
    md = _llm_claim_metadata(
        {"claim": {"key": "employer", "value": "Globex", "valid_from": "2025-01-01"}}
    )
    assert md["claim"]["valid_from"].startswith("2025-01-01")
    assert _safe_dt("not-a-date") is None
    assert _safe_dt("2025-01-01T00:00:00Z") == datetime(2025, 1, 1, tzinfo=timezone.utc)


# --- LLM integration: count preserved, claims validated, one bad one tolerated -


def _compiler() -> LLMCompiler:
    c = LLMCompiler.__new__(LLMCompiler)
    c._model, c._api_key, c._client = "gpt-4o-mini", "k", None
    return c


@pytest.mark.asyncio
async def test_llm_compile_attaches_valid_drops_invalid_keeps_count():
    compiler = _compiler()
    resp = [
        {
            "kind": "profile_fact",
            "content": "Works at Globex",
            "summary": "s",
            "confidence": 0.9,
            "episode_index": 0,
            "claim": {"key": "employer", "value": "Globex"},
        },
        {
            "kind": "profile_fact",
            "content": "Shoe size 10",
            "summary": "s",
            "confidence": 0.8,
            "episode_index": 0,
            "claim": {"key": "shoe_size", "value": "10"},
        },  # unknown -> dropped
        {
            "kind": "profile_fact",
            "content": "Likes hiking",
            "summary": "s",
            "confidence": 0.7,
            "episode_index": 0,
            "claim": "garbage",
        },  # malformed -> dropped
        {
            "kind": "episode_summary",
            "content": "A chat happened",
            "summary": "s",
            "confidence": 0.8,
            "episode_index": 0,
        },  # no claim
    ]
    ep = _ep("conversation")
    with patch.object(compiler, "_call_llm_async", new_callable=AsyncMock, return_value=resp):
        mems = await compiler.compile_async([ep])

    assert len(mems) == 4  # nothing discarded
    by_content = {m.content: m for m in mems}
    assert by_content["Works at Globex"].metadata_["claim"]["key"] == "employment.current_employer"
    # compiler/model preserved everywhere; bad claims simply absent
    assert all(
        m.metadata_.get("compiler") == "llm" and m.metadata_.get("model") == "gpt-4o-mini"
        for m in mems
    )
    assert "claim" not in by_content["Shoe size 10"].metadata_
    assert "claim" not in by_content["Likes hiking"].metadata_
    assert "claim" not in by_content["A chat happened"].metadata_


# --- compiler-generated claims drive the resolver -----------------------------


async def _resolve(memories):
    with patch("server.services.conflicts.repo") as mock_repo:
        mock_repo.list_active_memories_by_subject = AsyncMock(return_value=memories)
        mock_repo.mark_memories_superseded = AsyncMock()
        return await resolve_conflicts(AsyncMock(), "user-1")


def _compiled_fact(text, *, created_days):
    (f,) = _facts(text)
    f.created_at = _NOW - __import__("datetime").timedelta(days=created_days)
    return f


@pytest.mark.asyncio
async def test_compiler_claims_supersede_current_employer():
    acme = _compiled_fact("I work at Acme", created_days=10)
    globex = _compiled_fact("I work at Globex", created_days=0)
    result = await _resolve([acme, globex])
    assert acme.id in result and globex.id not in result


@pytest.mark.asyncio
async def test_compiler_unclaimed_memories_keep_legacy_behavior():
    # Generic tool usage -> no claims -> distinct facts coexist.
    a = _compiled_fact("I use Stripe", created_days=10)
    b = _compiled_fact("I use PayPal", created_days=0)
    assert await _resolve([a, b]) == []


@pytest.mark.asyncio
async def test_compiler_uncertain_text_gains_no_supersession():
    # "I am Bob" is emitted but unkeyed; pairing it with a name claim does not
    # let it participate in claim supersession (mixed -> legacy, low overlap).
    bob = _compiled_fact("I am Bob", created_days=10)
    (alice,) = _facts("My name is Alice Chen")
    alice.created_at = _NOW
    assert await _resolve([bob, alice]) == []
