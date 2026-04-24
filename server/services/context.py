"""Context assembly service — builds ContextBundleResponse for downstream LLMs."""

from __future__ import annotations

import tiktoken
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.config import settings
from server.db import repositories as repo
from server.schemas.responses import ContextBundleResponse, EpisodeResponse, MemoryResponse


async def assemble_context(
    session: AsyncSession,
    subject_id: str,
    task: str,
    max_tokens: int | None = None,
) -> ContextBundleResponse:
    """Build a deterministic, token-bounded context bundle."""
    max_tokens = max_tokens or settings.default_max_context_tokens

    # Fetch active memories
    facts_rows = await repo.search_memories(session, subject_id, kind="profile_fact", limit=50)
    procedure_rows = await repo.search_memories(session, subject_id, kind="procedure", limit=20)
    episode_rows = await repo.list_episodes_by_subject(session, subject_id, limit=30)

    facts = [_memory_response(r) for r in facts_rows]
    procedures = [_memory_response(r) for r in procedure_rows]
    episodes = [_episode_response(r) for r in episode_rows]

    # Assemble text
    parts: list[str] = []
    parts.append(f"## Task\n{task}\n")
    if facts:
        parts.append("## Known facts")
        for f in facts:
            parts.append(f"- {f.content}")
        parts.append("")
    if procedures:
        parts.append("## Procedures")
        for p in procedures:
            parts.append(f"- {p.content}")
        parts.append("")
    if episodes:
        parts.append("## Recent episodes")
        for e in episodes:
            parts.append(f"- [{e.source}/{e.type}] {_short_payload(e.payload)}")
        parts.append("")

    assembled = "\n".join(parts)

    # Token estimate
    enc = tiktoken.get_encoding(settings.tiktoken_model)
    token_estimate = len(enc.encode(assembled))

    # Provenance
    provenance = {
        "fact_ids": [str(f.id) for f in facts],
        "procedure_ids": [str(p.id) for p in procedures],
        "episode_ids": [str(e.id) for e in episodes],
    }

    return ContextBundleResponse(
        subject_id=subject_id,
        task=task,
        facts=facts,
        episodes=episodes,
        procedures=procedures,
        provenance=provenance,
        assembled_context=assembled,
        token_estimate=token_estimate,
    )


# ---------------------------------------------------------------------------
# Row → response helpers
# ---------------------------------------------------------------------------

def _memory_response(row) -> MemoryResponse:
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


def _episode_response(row) -> EpisodeResponse:
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


def _short_payload(payload: dict) -> str:
    text = payload.get("text", "") or payload.get("content", "")
    if isinstance(text, str) and text:
        return text[:120]
    return str(payload)[:120]
