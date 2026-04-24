"""Heuristic memory compiler — v1 implementation.

Extracts profile facts and episode summaries from chat-like episode payloads.
Designed to be replaced by LLM-backed compilation in later versions.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime

from server.db.tables import EpisodeRow, MemoryRow


def compile_memories_from_episodes(episodes: list[EpisodeRow]) -> list[MemoryRow]:
    """Produce memory rows from a batch of episodes (heuristic v1)."""
    memories: list[MemoryRow] = []

    for ep in episodes:
        # Episode summary — always create one per episode
        summary_text = _summarise_payload(ep.payload)
        if summary_text:
            memories.append(
                MemoryRow(
                    id=uuid.uuid4(),
                    subject_id=ep.subject_id,
                    kind="episode_summary",
                    content=summary_text,
                    summary=summary_text[:200],
                    confidence=0.8,
                    valid_from=ep.created_at or datetime.utcnow(),
                    source_episode_ids=[ep.id],
                    metadata_={},
                    status="active",
                )
            )

        # Profile fact extraction — simple heuristic
        facts = _extract_profile_facts(ep.payload)
        for fact in facts:
            memories.append(
                MemoryRow(
                    id=uuid.uuid4(),
                    subject_id=ep.subject_id,
                    kind="profile_fact",
                    content=fact,
                    summary=fact[:200],
                    confidence=0.6,
                    valid_from=ep.created_at or datetime.utcnow(),
                    source_episode_ids=[ep.id],
                    metadata_={},
                    status="active",
                )
            )

    return memories


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:my name is|i'm|i am)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", re.IGNORECASE),
    re.compile(r"(?:i work at|i'm at|i am at)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:i live in|i'm from|i am from)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:i use|i prefer|my favorite)\s+(.+?)(?:\.|$)", re.IGNORECASE),
]


def _extract_profile_facts(payload: dict) -> list[str]:
    """Extract simple profile facts from message-like payloads."""
    text = _payload_text(payload)
    if not text:
        return []
    facts: list[str] = []
    for pattern in _FACT_PATTERNS:
        for m in pattern.finditer(text):
            facts.append(m.group(0).strip().rstrip("."))
    return facts


def _summarise_payload(payload: dict) -> str:
    """Create a short textual summary of the payload."""
    text = _payload_text(payload)
    if not text:
        return ""
    # Truncate for v1
    return text[:500]


def _payload_text(payload: dict) -> str:
    """Best-effort extraction of text from various payload shapes."""
    # {"messages": [...]} shape
    if "messages" in payload:
        parts: list[str] = []
        for msg in payload["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
    # {"text": "..."} shape
    if "text" in payload:
        return str(payload["text"])
    # {"content": "..."} shape
    if "content" in payload:
        return str(payload["content"])
    return ""
