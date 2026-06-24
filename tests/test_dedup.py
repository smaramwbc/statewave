"""Tests for compile-time near-duplicate dedup (server/services/dedup.py).

Pure unit tests of the collapse logic + safety gates. Tiers 1-2 need no DB; the
semantic pass uses a fake embedding provider so cosine behaviour is deterministic.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from server.db.tables import MemoryRow
from server.services import dedup


def _mem(content: str, *, kind: str = "profile_fact", days_ago: int = 0,
         conf: float = 0.8, mid: uuid.UUID | None = None) -> MemoryRow:
    when = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return MemoryRow(
        id=mid or uuid.uuid4(),
        subject_id="user-1",
        kind=kind,
        content=content,
        summary=content[:200],
        confidence=conf,
        valid_from=when,
        source_episode_ids=[uuid.uuid4()],
        metadata_={},
        status="active",
    )


class _FakeProvider:
    """Maps each text to a fixed vector so cosine is deterministic in tests."""
    provides_semantic_similarity = True

    def __init__(self, vectors: dict[str, list[float]]):
        self._v = vectors

    async def embed_texts(self, texts):
        return [self._v[t] for t in texts]


# ── Pass 1: exact normalized-text collapse (no embeddings) ──────────────────

@pytest.mark.asyncio
async def test_exact_normalized_duplicates_collapse():
    # Same fact, differing only in case / whitespace / trailing punctuation —
    # the windowing-overlap signature. Semantic pass off to isolate Pass 1.
    a = _mem("Alice lives in Berlin", days_ago=5)
    b = _mem("alice   lives in berlin.", days_ago=2)
    c = _mem("Alice owns a red bike", days_ago=1)
    with patch.object(dedup.settings, "dedup_semantic_enabled", False):
        out = await dedup.dedup_candidates([a, b, c])
    contents = {m.content for m in out}
    assert len(out) == 2                       # a/b collapsed, c kept
    assert "Alice owns a red bike" in contents


@pytest.mark.asyncio
async def test_merge_unions_provenance_and_keeps_earliest_date():
    e1, e2 = uuid.uuid4(), uuid.uuid4()
    a = _mem("Bob works at Globex", days_ago=10)
    a.source_episode_ids = [e1]
    b = _mem("bob works at globex", days_ago=2)
    b.source_episode_ids = [e2]
    with patch.object(dedup.settings, "dedup_semantic_enabled", False):
        out = await dedup.dedup_candidates([a, b])
    assert len(out) == 1
    survivor = out[0]
    assert set(survivor.source_episode_ids) == {e1, e2}     # provenance preserved
    assert survivor.valid_from == a.valid_from              # earliest origin kept


# ── Pass 2: semantic clustering + the three safety gates ────────────────────

@pytest.mark.asyncio
async def test_word_order_variant_merges():
    # Same fact, reordered with stopword/punct differences (the windowing-
    # reformat case): identical significant content words + high cosine -> merge.
    a = _mem("Alice bought purple running shoes", days_ago=3)
    b = _mem("purple running shoes were bought by Alice", days_ago=1)
    prov = _FakeProvider({a.content: [1.0, 0.0], b.content: [0.99, 0.01]})
    out = await dedup.dedup_candidates([a, b], provider=prov)
    assert len(out) == 1


@pytest.mark.asyncio
async def test_number_guard_keeps_knowledge_update_pairs_distinct():
    # Same surface form, DIFFERENT value: must NOT merge (reconcile's job).
    a = _mem("API averages 250ms response time", days_ago=5)
    b = _mem("API averages 90ms response time", days_ago=1)
    # Force high cosine so only the number guard can save us.
    prov = _FakeProvider({a.content: [1.0, 0.0], b.content: [1.0, 0.0]})
    out = await dedup.dedup_candidates([a, b], provider=prov)
    assert len(out) == 2                        # number-set differs -> kept apart


@pytest.mark.asyncio
async def test_lexical_gate_blocks_topically_close_distinct_facts():
    # High cosine, no distinguishing number, but low token overlap ("Mia" vs
    # "Mary") -> the Jaccard gate must keep them separate.
    a = _mem("User has a daughter named Mia", days_ago=3)
    b = _mem("User has a daughter named Mary", days_ago=1)
    prov = _FakeProvider({a.content: [1.0, 0.0], b.content: [0.999, 0.0]})
    out = await dedup.dedup_candidates([a, b], provider=prov)
    assert len(out) == 2


@pytest.mark.asyncio
async def test_kind_gating_prevents_cross_kind_merge():
    # Identical text, different kind -> neither Pass 1 (keyed on kind+content)
    # nor Pass 2 (kind-gated) may merge them.
    a = _mem("Project uses Postgres for storage", kind="profile_fact", days_ago=3)
    b = _mem("Project uses Postgres for storage", kind="episode_summary", days_ago=1)
    prov = _FakeProvider({a.content: [1.0, 0.0], b.content: [1.0, 0.0]})
    out = await dedup.dedup_candidates([a, b], provider=prov)
    assert len(out) == 2                        # different kinds never merge


@pytest.mark.asyncio
async def test_semantic_survivors_get_embeddings_stapled():
    a = _mem("Dana plays the cello", days_ago=2)
    b = _mem("Dana enjoys astronomy", days_ago=1)
    prov = _FakeProvider({a.content: [1.0, 0.0], b.content: [0.0, 1.0]})  # orthogonal
    out = await dedup.dedup_candidates([a, b], provider=prov)
    assert len(out) == 2
    assert all(m.embedding is not None for m in out)   # reused for .embedding col


# ── Fail-open / guards ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stub_provider_runs_exact_pass_only():
    class _Stub:
        provides_semantic_similarity = False
        async def embed_texts(self, texts):  # pragma: no cover - must not be called
            raise AssertionError("stub must not be embedded")
    a = _mem("X happened", days_ago=2)
    b = _mem("x happened.", days_ago=1)          # exact-normalized dup
    c = _mem("Totally different fact", days_ago=1)
    out = await dedup.dedup_candidates([a, b, c], provider=_Stub())
    assert len(out) == 2                          # exact pass still collapsed a/b


@pytest.mark.asyncio
async def test_embed_failure_falls_back_to_exact_pass():
    a = _mem("alpha fact one", days_ago=2)
    b = _mem("beta fact two", days_ago=1)
    prov = _FakeProvider({})
    prov.embed_texts = AsyncMock(side_effect=RuntimeError("boom"))
    out = await dedup.dedup_candidates([a, b], provider=prov)
    assert len(out) == 2                          # no crash; exact-pass survivors


@pytest.mark.asyncio
async def test_disabled_returns_unchanged():
    a = _mem("same", days_ago=2)
    b = _mem("same", days_ago=1)
    with patch.object(dedup.settings, "dedup_compile_enabled", False):
        out = await dedup.dedup_candidates([a, b])
    assert len(out) == 2


@pytest.mark.asyncio
async def test_large_batch_skipped():
    cands = [_mem(f"fact {i}", days_ago=i % 30) for i in range(50)]
    with patch.object(dedup.settings, "dedup_max_candidates", 10):
        out = await dedup.dedup_candidates(cands)
    assert len(out) == 50                         # untouched above the ceiling
