"""Batched entity extraction (issue #380).

Per-memory extraction was one LLM call per compiled memory — ~745 calls
for the 411-episode docs pack, ~60% of all provider round-trips, and the
dominant amplifier when provider latency degrades. Batching cuts that to
~1/batch_size. Pinned invariants:

* per-entity validation rules are SHARED between the single and batched
  parsers (one `_entities_from_list`), so quality cannot drift;
* a structurally unusable batched response falls back to per-fact calls
  (which themselves fail-open to []); a missing index in a well-formed
  response yields [] for that fact only;
* the result is always index-aligned with the input;
* `populate_entities_for_memories` groups by `entity_extract_batch_size`;
* the admin import endpoint rebuilds the entity store only when asked
  (export documents don't carry subject_entities — before this, every
  docs-pack swap left the live subject with zero entity rows).
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from server.core.config import settings
from server.services import entity_extraction as ee

pytestmark = pytest.mark.asyncio


def _batched_response(mapping: dict) -> str:
    return json.dumps({"results": mapping})


async def test_batch_returns_index_aligned_lists():
    resp = _batched_response(
        {
            "0": [{"text": "Alice", "kind": "PERSON"}],
            "1": [{"text": "Acme Corp", "kind": "ORG"}, {"text": "Chicago", "kind": "GPE"}],
        }
    )
    with patch.object(ee, "acomplete", new=AsyncMock(return_value=resp)):
        out = await ee.extract_entities_batch(["fact a", "fact b"])
    assert [len(lst) for lst in out] == [1, 2]
    assert out[0][0].text == "Alice" and out[0][0].kind == "PERSON"
    assert out[1][1].normalized == "chicago"


async def test_batch_missing_index_yields_empty_for_that_fact_only():
    resp = _batched_response({"1": [{"text": "Alice", "kind": "PERSON"}]})
    with patch.object(ee, "acomplete", new=AsyncMock(return_value=resp)):
        out = await ee.extract_entities_batch(["fact a", "fact b"])
    assert out[0] == [] and len(out[1]) == 1


async def test_batch_structural_failure_falls_back_to_per_fact():
    single = AsyncMock(side_effect=[["A"], ["B"]])
    with patch.object(ee, "acomplete", new=AsyncMock(return_value="not json at all")), patch.object(
        ee, "extract_entities", new=single
    ):
        out = await ee.extract_entities_batch(["fact a", "fact b"])
    assert out == [["A"], ["B"]]
    assert single.await_count == 2


async def test_batch_provider_error_returns_empty_without_per_fact_storm():
    """A dead provider must NOT trigger batch_size sequential per-fact
    calls under the concurrency slot — each would burn the same timeout
    against the same dead provider. Entities are best-effort: empty."""
    single = AsyncMock()
    with patch.object(
        ee, "acomplete", new=AsyncMock(side_effect=RuntimeError("provider down"))
    ), patch.object(ee, "extract_entities", new=single):
        out = await ee.extract_entities_batch(["fact a", "fact b"])
    assert out == [[], []]
    single.assert_not_awaited()


async def test_batch_framing_collapses_multiline_content():
    """Fact content with embedded newlines (procedure steps like
    '2: run alembic') must not mimic the index framing."""
    captured = {}

    async def fake_acomplete(messages, **kwargs):
        captured["user"] = messages[1]["content"]
        return json.dumps({"results": {"0": [], "1": []}})

    with patch.object(ee, "acomplete", new=fake_acomplete):
        await ee.extract_entities_batch(["step one\n2: run alembic", "plain fact"])

    facts_block = captured["user"].split("FACTS")[1]
    assert "\n2: run alembic" not in facts_block, "newline framing must be collapsed"
    assert "FACT 0: step one 2: run alembic" in facts_block


async def test_single_fact_uses_single_path():
    single = AsyncMock(return_value=["X"])
    batched_call = AsyncMock()
    with patch.object(ee, "extract_entities", new=single), patch.object(
        ee, "acomplete", new=batched_call
    ):
        out = await ee.extract_entities_batch(["only fact"])
    assert out == [["X"]]
    batched_call.assert_not_awaited()


async def test_batched_parser_shares_validation_rules_with_single():
    """The dedup/normalization rules must be literally the same code path:
    duplicate normalized forms collapse, blank text drops, kind uppercases."""
    items = [
        {"text": "Alice", "kind": "person"},
        {"text": "alice", "kind": "PERSON"},  # duplicate after normalization
        {"text": "   ", "kind": "ORG"},  # blank
        {"text": "Acme", "kind": ""},  # kind missing → None
    ]
    via_batch = ee._entities_from_list(items)
    via_single = ee._parse_entities(json.dumps({"entities": items}))
    assert [(e.text, e.normalized, e.kind) for e in via_batch] == [
        (e.text, e.normalized, e.kind) for e in via_single
    ]
    assert via_batch[0].kind == "PERSON"
    assert via_batch[1].kind is None


# ---------------------------------------------------------------------------
# populate_entities_for_memories: grouping
# ---------------------------------------------------------------------------


async def test_populate_groups_by_batch_size(monkeypatch):
    from server.services import entities as ent

    memories = [
        ent.MemoryForEntities(id=uuid.uuid4(), content=f"fact {i}") for i in range(25)
    ]
    group_sizes: list[int] = []

    async def fake_batch(texts):
        group_sizes.append(len(texts))
        return [[] for _ in texts]

    monkeypatch.setattr(ent, "extract_entities_batch", fake_batch)
    with patch.object(settings, "entity_extract_batch_size", 10):
        touched = await ent.populate_entities_for_memories(
            AsyncMock(), memories, subject_id="subj", tenant_id=None
        )
    assert touched == 0
    assert sorted(group_sizes, reverse=True) == [10, 10, 5]


# ---------------------------------------------------------------------------
# rebuild-entities endpoint (deliberately separate from /admin/import: an
# inline rebuild made the import outlive client timeouts, and a retried
# already-committed import with preserve_ids=False duplicates the subject)
# ---------------------------------------------------------------------------


async def test_rebuild_endpoint_populates_from_active_memories(monkeypatch):
    from server.api import admin as api_admin

    class _Row:
        def __init__(self, content):
            self.id = uuid.uuid4()
            self.content = content

    rows = [_Row("fact a"), _Row("fact b")]
    seen = {}

    async def fake_list(_session, subject_id, *, tenant_id, limit):
        seen["limit"] = limit
        return rows

    async def fake_populate(_session, memories, *, subject_id, tenant_id):
        seen["memories"] = len(memories)
        return 5

    class _Ctx:
        async def __aenter__(self):
            return AsyncMock()

        async def __aexit__(self, *a):
            return None

    import server.db.engine as engine_module
    import server.db.repositories as repo_module
    import server.services.entities as ent_module

    monkeypatch.setattr(engine_module, "get_session_factory", lambda: lambda: _Ctx())
    monkeypatch.setattr(repo_module, "list_active_memories_by_subject", fake_list)
    monkeypatch.setattr(ent_module, "populate_entities_for_memories", fake_populate)

    result = await api_admin.rebuild_entities_endpoint("subj", tenant_id=None)
    assert result == {"subject_id": "subj", "entities_rebuilt": 5, "enabled": True}
    assert seen["memories"] == 2
    assert seen["limit"] == 10_000


async def test_rebuild_endpoint_is_noop_when_population_disabled(monkeypatch):
    from server.api import admin as api_admin

    with patch.object(settings, "entity_population_enabled", False):
        result = await api_admin.rebuild_entities_endpoint("subj", tenant_id=None)
    assert result["entities_rebuilt"] == 0 and result["enabled"] is False


async def test_import_request_has_no_rebuild_flag():
    """Pin the review outcome: the rebuild must never ride inside the
    import request again."""
    from server.api import admin as api_admin

    assert "rebuild_entities" not in api_admin.ImportSubjectRequest.model_fields
