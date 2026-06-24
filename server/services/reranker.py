"""LLM reranker — surface the precise answer fact to the top of retrieval.

Across all three public long-term-memory benchmarks the same pattern held: once
full-conversation compile makes every fact PRESENT, single-fact-lookup questions
(LoCoMo single_hop, BEAM information_extraction, LongMemEval single_session_*)
still miss because the one answer memory ranks mid-pack among 300-500 distinct
distractors. Neither entity-boost nor top_k tuning fixes it — the right fact
simply isn't near the top of the hybrid order. A reranker is the principled fix:
retrieve a generous candidate pool by hybrid similarity, then have an LLM score
each candidate's relevance to the question and reorder, keeping the best `top_n`
for the answer model.

Provider-neutral: routes through the central adapter with the compile model
(gpt-4o-mini in the equal-token bench config), so it stays GPT-only and adds no
new dependency. Fail-open: any error returns the original hybrid order truncated
to `top_n`, so a reranker failure can never drop retrieval below the un-reranked
baseline.
"""

from __future__ import annotations

from typing import Sequence

import structlog

from server.core.config import settings
from server.db.tables import MemoryRow
from server.services import llm as llm_adapter

logger = structlog.stdlib.get_logger()

_RERANK_SYSTEM_PROMPT = """\
You are a memory-retrieval reranker. Given a user QUESTION and a numbered list of \
candidate MEMORIES, return the indices of the memories MOST RELEVANT to answering \
the question, ordered most-relevant FIRST.

Rules:
- A memory is relevant if it contains a fact, value, date, name, place, preference, \
event, or statement the question asks about — including partial or supporting \
evidence, and facts needed to reason across multiple memories.
- Order strictly by how directly each memory helps answer THIS question.
- Include every plausibly-relevant memory; omit only clearly-irrelevant ones.
- Return AT MOST {top_n} indices.
- Return ONLY a JSON object: {{"ranked": [<indices, most relevant first>]}}."""


def _compile_model() -> str:
    return getattr(settings, "litellm_compile_model", None) or settings.litellm_model


def _rerank_model() -> str:
    # Reranking is relevance judgment — a stronger model than the compile model
    # is worth the single call/query. Falls back to the compile model.
    return getattr(settings, "rerank_model", "") or _compile_model()


async def rerank_memories(
    query: str,
    candidates: Sequence[MemoryRow],
    top_n: int,
    *,
    model: str | None = None,
) -> list[MemoryRow]:
    """Reorder `candidates` by LLM-judged relevance to `query`, return top_n.

    Fail-open: returns ``list(candidates)[:top_n]`` (original order) on any
    problem, so reranking never makes retrieval worse than the input order."""
    cands = list(candidates)
    if not query or len(cands) <= 1 or len(cands) <= top_n:
        # Nothing to filter — reordering alone rarely helps the answer model and
        # isn't worth an LLM call.
        return cands[:top_n]
    if len(cands) > settings.rerank_max_pool:
        cands = cands[: settings.rerank_max_pool]

    listing = "\n".join(f"{i}: {c.content}" for i, c in enumerate(cands))
    user = f"QUESTION: {query}\n\nMEMORIES:\n{listing}"
    try:
        parsed = await llm_adapter.acomplete_json(
            [
                {"role": "system", "content": _RERANK_SYSTEM_PROMPT.format(top_n=top_n)},
                {"role": "user", "content": user},
            ],
            model=model or _rerank_model(),
            temperature=0.0,
            max_tokens=min(settings.litellm_compile_max_tokens, 2000),
        )
    except Exception:
        logger.warning("rerank_llm_failed", exc_info=True)
        return cands[:top_n]

    ranked = parsed.get("ranked") if isinstance(parsed, dict) else None
    if not isinstance(ranked, list) or not ranked:
        return cands[:top_n]

    seen: set[int] = set()
    ordered: list[MemoryRow] = []
    for idx in ranked:
        if isinstance(idx, int) and 0 <= idx < len(cands) and idx not in seen:
            seen.add(idx)
            ordered.append(cands[idx])
        if len(ordered) >= top_n:
            break
    # Backfill from the original hybrid order so we always return up to top_n
    # even if the model returned too few indices (fail-safe, never under-fill).
    if len(ordered) < top_n:
        for i, c in enumerate(cands):
            if i not in seen:
                ordered.append(c)
                if len(ordered) >= top_n:
                    break

    logger.info("rerank_done", pool=len(cands), returned=len(ordered), top_n=top_n)
    return ordered[:top_n]
