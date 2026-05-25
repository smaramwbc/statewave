"""Heuristic memory compiler — regex/pattern-based extraction.

Extracts profile facts and episode summaries from chat-like episode payloads.
No external dependencies. Used by the default deploy
(STATEWAVE_COMPILER_TYPE=heuristic) and by tests that don't want to mock
an LLM. For higher-quality memories on real workloads, set
STATEWAVE_COMPILER_TYPE=llm with a configured LiteLLM model.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Sequence

from server.core.config import settings
from server.db.tables import EpisodeRow, MemoryRow
from server.services.auto_labeling import apply_suggestions
from server.services.memory_ttl import compute_valid_to


class HeuristicCompiler:
    """Pattern-based memory compiler. Implements BaseCompiler protocol."""

    def compile(self, episodes: Sequence[EpisodeRow]) -> list[MemoryRow]:
        memories: list[MemoryRow] = []
        for ep in episodes:
            memories.extend(self._compile_episode(ep))
        # Auto-labeling runs post-construction: detectors only need
        # `content`, and running them once at the end keeps a single
        # apply path that's easy to test in isolation. Gated on the
        # global flag so a v0.9 upgrade is a no-op for existing tenants.
        if settings.auto_labeling_enabled and memories:
            apply_suggestions(memories)
        return memories

    def _compile_episode(self, ep: EpisodeRow) -> list[MemoryRow]:
        results: list[MemoryRow] = []
        text = extract_payload_text(ep.payload)
        if not text:
            return results

        ttl = settings.kind_ttl_days
        ep_valid_from = episode_valid_from(ep)

        # Episode summary
        results.append(
            MemoryRow(
                id=uuid.uuid4(),
                subject_id=ep.subject_id,
                kind="episode_summary",
                content=text[:500],
                summary=text[:200],
                confidence=0.8,
                valid_from=ep_valid_from,
                valid_to=compute_valid_to("episode_summary", ep_valid_from, ttl),
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
                    valid_from=ep_valid_from,
                    valid_to=compute_valid_to("profile_fact", ep_valid_from, ttl),
                    source_episode_ids=[ep.id],
                    metadata_={},
                    status="active",
                )
            )

        return results


# ---------------------------------------------------------------------------
# Shared temporal anchor (usable by any compiler)
# ---------------------------------------------------------------------------


def episode_valid_from(ep: EpisodeRow) -> datetime:
    """Return the best-effort temporal anchor for memories derived from
    this episode.

    Priority:
      1. `payload.event_time` — explicit override set by the client
         (e.g. a connector replaying historical data sets this to the
         actual event date, not the POST date).
      2. `payload.messages[0].timestamp` — the first-message timestamp
         on chat-shaped payloads. This is what LoCoMo, Slack, Zendesk
         etc emit naturally; preserving it as `valid_from` makes
         "when did X happen?" answerable from the resulting memory.
      3. `ep.created_at` — when the episode was POSTed (the previous
         hardcoded default).
      4. `now()` — last resort if everything else is missing.

    Returning a real event time instead of the POST time keeps
    Statewave's bi-temporal validity story honest: a memory whose
    facts were true in May 2023 shouldn't carry `valid_from=today`.
    """
    payload = ep.payload or {}

    explicit = payload.get("event_time")
    if isinstance(explicit, str) and explicit:
        parsed = _parse_event_time(explicit)
        if parsed is not None:
            return parsed

    messages = payload.get("messages")
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        ts = messages[0].get("timestamp")
        if isinstance(ts, str) and ts:
            parsed = _parse_event_time(ts)
            if parsed is not None:
                return parsed

    return ep.created_at or datetime.now(timezone.utc)


def _parse_event_time(value: str) -> datetime | None:
    """Parse a small set of supported datetime formats. Returns None on
    failure rather than raising — callers fall back to ep.created_at.

    Supported:
      - ISO 8601 (`2023-05-08T13:56:00`, `2023-05-08T13:56:00Z`,
        `2023-05-08T13:56:00+00:00`)
      - LoCoMo's idiomatic format (`1:56 pm on 8 May, 2023`)

    Anything else returns None; we'd rather fall back than guess.
    """
    s = value.strip()
    if not s:
        return None

    # ISO 8601 first — handles the connector / replay path.
    try:
        iso = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if iso.tzinfo is None:
            iso = iso.replace(tzinfo=timezone.utc)
        return iso
    except ValueError:
        pass

    # LoCoMo format: "1:56 pm on 8 May, 2023". %I tolerates both
    # zero-padded and single-digit hours on CPython; %d does the same
    # for days. Returned datetime is naive — promote to UTC so it
    # round-trips through Postgres's `timestamp with time zone` cleanly.
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M%p on %d %B, %Y"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    return None


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
    # Name pattern: case-sensitive capture so we only grab proper nouns
    re.compile(r"(?i:my name is|i'm|i am)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"),
    re.compile(r"(?:i work at|i'm at|i am at)\s+(.+?)(?:\.|,|\n|$)", re.IGNORECASE),
    re.compile(r"(?:i am on|i'm on)\s+(.+?)(?:\.|,|\n|$)", re.IGNORECASE),
    re.compile(r"(?:i live in|i'm from|i am from)\s+(.+?)(?:\.|,|\n|$)", re.IGNORECASE),
    re.compile(r"(?:i use|i prefer|my favorite)\s+(.+?)(?:\.|,|\n|$)", re.IGNORECASE),
]


def _extract_profile_facts(text: str) -> list[str]:
    facts: list[str] = []
    for pattern in _FACT_PATTERNS:
        for m in pattern.finditer(text):
            facts.append(m.group(0).strip().rstrip(".").rstrip(","))
    return facts
