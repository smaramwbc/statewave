"""Context assembly service — builds ranked, token-bounded context bundles.

Strategy:
- Score every memory and episode by kind priority + recency + task relevance.
- Assemble text incrementally, highest-scored items first.
- Stop adding items when the token budget is exhausted.
- The task header is always included (reserved budget).
- Provenance tracks exactly which items made it into the bundle.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import tiktoken
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.config import settings
from server.db import repositories as repo
from server.schemas.responses import ContextBundleResponse, EpisodeResponse, MemoryResponse
from server.services.compilers.heuristic import extract_payload_text

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

_KIND_PRIORITY: dict[str, float] = {
    "profile_fact": 10.0,
    "procedure": 8.0,
    "episode_summary": 5.0,
}
_EPISODE_PRIORITY = 3.0
_RECENCY_MAX = 5.0
_RELEVANCE_MAX = 5.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def assemble_context(
    session: AsyncSession,
    subject_id: str,
    task: str,
    max_tokens: int | None = None,
) -> ContextBundleResponse:
    """Build a deterministic, token-bounded, ranked context bundle."""
    budget = max_tokens or settings.default_max_context_tokens
    enc = tiktoken.get_encoding(settings.tiktoken_model)

    # -- Fetch candidates ---------------------------------------------------
    fact_rows = await repo.search_memories(session, subject_id, kind="profile_fact", limit=50)
    procedure_rows = await repo.search_memories(session, subject_id, kind="procedure", limit=20)
    summary_rows = await repo.search_memories(session, subject_id, kind="episode_summary", limit=30)
    episode_rows = await repo.list_episodes_by_subject(session, subject_id, limit=30)

    # -- Score all candidates ------------------------------------------------
    task_tokens = _tokenize_for_relevance(task)

    scored: list[_ScoredItem] = []
    all_memory_rows = list(fact_rows) + list(procedure_rows) + list(summary_rows)
    if all_memory_rows:
        ts_range = _timestamp_range([r.created_at for r in all_memory_rows])
        for row in all_memory_rows:
            score = (
                _KIND_PRIORITY.get(row.kind, 1.0)
                + _recency_score(row.created_at, ts_range)
                + _relevance_score(row.content, task_tokens)
            )
            scored.append(_ScoredItem(
                score=score,
                kind="memory",
                memory_row=row,
                text=f"- {row.content}",
                section=_section_for_kind(row.kind),
            ))

    if episode_rows:
        ep_ts_range = _timestamp_range([r.created_at for r in episode_rows])
        for row in episode_rows:
            ep_text = _short_episode_text(row.payload, row.source, row.type)
            content_text = extract_payload_text(row.payload)
            score = (
                _EPISODE_PRIORITY
                + _recency_score(row.created_at, ep_ts_range)
                + _relevance_score(content_text, task_tokens)
            )
            scored.append(_ScoredItem(
                score=score,
                kind="episode",
                episode_row=row,
                text=f"- {ep_text}",
                section="episodes",
            ))

    # Sort descending by score (stable sort preserves insertion order for ties)
    scored.sort(key=lambda s: s.score, reverse=True)

    # -- Assemble within budget ---------------------------------------------
    task_header = f"## Task\n{task}\n"
    budget_used = len(enc.encode(task_header))

    included_facts: list[MemoryResponse] = []
    included_procedures: list[MemoryResponse] = []
    included_episodes: list[EpisodeResponse] = []
    section_lines: dict[str, list[str]] = {
        "facts": [],
        "procedures": [],
        "summaries": [],
        "episodes": [],
    }

    for item in scored:
        item_tokens = len(enc.encode(item.text))
        # Reserve ~10 tokens for section headers
        if budget_used + item_tokens + 10 > budget:
            continue  # skip this item, try smaller ones

        budget_used += item_tokens
        section_lines[item.section].append(item.text)

        if item.kind == "memory":
            resp = _memory_response(item.memory_row)
            if item.memory_row.kind == "profile_fact":
                included_facts.append(resp)
            elif item.memory_row.kind == "procedure":
                included_procedures.append(resp)
            # episode_summary memories go into facts list for response compat
            elif item.memory_row.kind == "episode_summary":
                included_facts.append(resp)
        elif item.kind == "episode":
            included_episodes.append(_episode_response(item.episode_row))

    # -- Render final text --------------------------------------------------
    parts: list[str] = [task_header]
    if section_lines["facts"] or section_lines["summaries"]:
        parts.append("## Known facts")
        parts.extend(section_lines["facts"])
        parts.extend(section_lines["summaries"])
        parts.append("")
    if section_lines["procedures"]:
        parts.append("## Procedures")
        parts.extend(section_lines["procedures"])
        parts.append("")
    if section_lines["episodes"]:
        parts.append("## Recent episodes")
        parts.extend(section_lines["episodes"])
        parts.append("")

    assembled = "\n".join(parts)
    token_estimate = len(enc.encode(assembled))

    provenance = {
        "fact_ids": [str(f.id) for f in included_facts],
        "procedure_ids": [str(p.id) for p in included_procedures],
        "episode_ids": [str(e.id) for e in included_episodes],
    }

    return ContextBundleResponse(
        subject_id=subject_id,
        task=task,
        facts=included_facts,
        episodes=included_episodes,
        procedures=included_procedures,
        provenance=provenance,
        assembled_context=assembled,
        token_estimate=token_estimate,
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _tokenize_for_relevance(text: str) -> set[str]:
    """Lowercase word tokens for relevance scoring."""
    return set(text.lower().split())


def _relevance_score(content: str, task_tokens: set[str]) -> float:
    """Word-overlap relevance: 0 to _RELEVANCE_MAX."""
    if not task_tokens or not content:
        return 0.0
    content_tokens = set(content.lower().split())
    overlap = len(task_tokens & content_tokens)
    # Normalize by task token count
    ratio = overlap / len(task_tokens)
    return min(ratio * _RELEVANCE_MAX, _RELEVANCE_MAX)


def _timestamp_range(
    timestamps: list[datetime],
) -> tuple[float, float]:
    """Return (min_ts, max_ts) as unix floats."""
    ts_floats = [t.timestamp() for t in timestamps if t]
    if not ts_floats:
        return (0.0, 0.0)
    return (min(ts_floats), max(ts_floats))


def _recency_score(
    created_at: datetime | None,
    ts_range: tuple[float, float],
) -> float:
    """Linear recency: most recent = _RECENCY_MAX, oldest = 0."""
    if not created_at:
        return 0.0
    ts_min, ts_max = ts_range
    if ts_max == ts_min:
        return _RECENCY_MAX  # single item
    t = created_at.timestamp()
    ratio = (t - ts_min) / (ts_max - ts_min)
    return ratio * _RECENCY_MAX


def _section_for_kind(kind: str) -> str:
    if kind == "profile_fact":
        return "facts"
    if kind == "procedure":
        return "procedures"
    return "summaries"


def _short_episode_text(payload: dict, source: str, type_: str) -> str:
    """Render an episode as a single line for context text."""
    text = extract_payload_text(payload)
    if text:
        return f"[{source}/{type_}] {text[:150]}"
    return f"[{source}/{type_}] (no text content)"


# ---------------------------------------------------------------------------
# Row → response converters
# ---------------------------------------------------------------------------

def _memory_response(row: Any) -> MemoryResponse:
    return MemoryResponse(
        id=row.id,
        subject_id=row.subject_id,
        kind=row.kind,
        content=row.content,
        summary=row.summary,
        confidence=row.confidence,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        source_episode_ids=row.source_episode_ids or [],
        metadata=row.metadata_,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _episode_response(row: Any) -> EpisodeResponse:
    return EpisodeResponse(
        id=row.id,
        subject_id=row.subject_id,
        source=row.source,
        type=row.type,
        payload=row.payload,
        metadata=row.metadata_,
        provenance=row.provenance,
        created_at=row.created_at,
    )


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

class _ScoredItem:
    """An item (memory or episode) with its computed relevance score."""

    __slots__ = ("score", "kind", "memory_row", "episode_row", "text", "section")

    def __init__(
        self,
        score: float,
        kind: str,
        text: str,
        section: str,
        memory_row: Any = None,
        episode_row: Any = None,
    ) -> None:
        self.score = score
        self.kind = kind
        self.memory_row = memory_row
        self.episode_row = episode_row
        self.text = text
        self.section = section
