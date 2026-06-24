"""Tests for the LLM reranker (server/services/reranker.py)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from server.db.tables import MemoryRow
from server.services import reranker


def _mem(content: str) -> MemoryRow:
    return MemoryRow(
        id=uuid.uuid4(), subject_id="s", kind="profile_fact", content=content,
        summary=content[:200], confidence=0.8,
        valid_from=datetime.now(timezone.utc), source_episode_ids=[uuid.uuid4()],
        metadata_={}, status="active",
    )


@pytest.mark.asyncio
async def test_rerank_reorders_and_truncates_to_top_n():
    cands = [_mem(f"fact {i}") for i in range(6)]
    # Model says the most relevant are 4, 1, 5 (in that order).
    llm = AsyncMock(return_value={"ranked": [4, 1, 5]})
    with patch.object(reranker.llm_adapter, "acomplete_json", llm):
        out = await reranker.rerank_memories("q", cands, top_n=3)
    assert [m.content for m in out] == ["fact 4", "fact 1", "fact 5"]


@pytest.mark.asyncio
async def test_rerank_backfills_when_model_returns_too_few():
    cands = [_mem(f"fact {i}") for i in range(6)]
    llm = AsyncMock(return_value={"ranked": [2]})        # only one index
    with patch.object(reranker.llm_adapter, "acomplete_json", llm):
        out = await reranker.rerank_memories("q", cands, top_n=4)
    assert len(out) == 4                                  # backfilled to top_n
    assert out[0].content == "fact 2"                     # model's pick first
    assert len({m.content for m in out}) == 4             # no dupes


@pytest.mark.asyncio
async def test_rerank_ignores_out_of_range_indices():
    cands = [_mem(f"fact {i}") for i in range(4)]
    llm = AsyncMock(return_value={"ranked": [99, 1, -1, 0]})
    with patch.object(reranker.llm_adapter, "acomplete_json", llm):
        out = await reranker.rerank_memories("q", cands, top_n=2)
    assert [m.content for m in out] == ["fact 1", "fact 0"]


@pytest.mark.asyncio
async def test_rerank_failopen_on_llm_error():
    cands = [_mem(f"fact {i}") for i in range(6)]
    llm = AsyncMock(side_effect=RuntimeError("boom"))
    with patch.object(reranker.llm_adapter, "acomplete_json", llm):
        out = await reranker.rerank_memories("q", cands, top_n=3)
    assert [m.content for m in out] == ["fact 0", "fact 1", "fact 2"]  # hybrid order


@pytest.mark.asyncio
async def test_rerank_skips_llm_when_pool_not_larger_than_top_n():
    cands = [_mem(f"fact {i}") for i in range(3)]
    llm = AsyncMock(return_value={"ranked": [2, 1, 0]})
    with patch.object(reranker.llm_adapter, "acomplete_json", llm):
        out = await reranker.rerank_memories("q", cands, top_n=5)
    assert len(out) == 3
    llm.assert_not_called()                               # nothing to filter
