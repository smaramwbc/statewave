"""Heuristic memory compiler — regex/pattern-based extraction.

Extracts profile facts and episode summaries from chat-like episode payloads.
No external dependencies. Suitable for local dev and as the default fallback.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Sequence

from server.db.tables import EpisodeRow, MemoryRow


class HeuristicCompiler:
    """Pattern-based memory compiler. Implements BaseCompiler protocol."""

    def compile(self, episodes: Sequence[EpisodeRow]) -> list[MemoryRow]:
        memories: list[MemoryRow] = []
        for ep in episodes:
            memories.extend(self._compile_episode(ep))
        return memories

    def _compile_episode(self, ep: EpisodeRow) -> list[MemoryRow]:
        results: list[MemoryRow] = []
        text = extract_payload_text(ep.payload)
        if not text:
            return results

        # Episode summary
        results.append(
            MemoryRow(
                id=uuid.uuid4(),
                subject_id=ep.subject_id,
                kind="episode_summary",
                content=text[:500],
                summary=text[:200],
                confidence=0.8,
                valid_from=ep.created_at or datetime.now(timezone.utc),
                source_episode_ids=[ep.id],
                metadata_={},
                status="active",
            )
        )

        # Profile facts
        for fact in _extract_profile_facts(text):
            results.append(
                MemoryRow(
                    id=uuid.uuid4(),
                    subject_id=ep.subject_id,
                    kind="profile_fact",
                    content=fact,
                    summary=fact[:200],
                    confidence=0.6,
                    valid_from=ep.created_at or datetime.now(timezone.utc),
                    source_episode_ids=[ep.id],
                    metadata_={},
                    status="active",
                )
            )

        return results


# ---------------------------------------------------------------------------
# Shared payload text extraction (usable by any compiler)
# ---------------------------------------------------------------------------

def extract_payload_text(payload: dict) -> str:
    """Best-effort text extraction from various payload shapes."""
    if "messages" in payload:
        parts: list[str] = []
        for msg in payload["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)
    if "text" in payload:
        return str(payload["text"])
    if "content" in payload:
        return str(payload["content"])
    return ""


# ---------------------------------------------------------------------------
# Fact extraction patterns
# ---------------------------------------------------------------------------

_FACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:my name is|i'm|i am)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", re.IGNORECASE),
    re.compile(r"(?:i work at|i'm at|i am at)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:i live in|i'm from|i am from)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"(?:i use|i prefer|my favorite)\s+(.+?)(?:\.|$)", re.IGNORECASE),
]


def _extract_profile_facts(text: str) -> list[str]:
    facts: list[str] = []
    for pattern in _FACT_PATTERNS:
        for m in pattern.finditer(text):
            facts.append(m.group(0).strip().rstrip("."))
    return facts
