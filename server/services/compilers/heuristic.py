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
from server.services.claims import build_claim_envelope
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

        # Profile facts. Text/kind/count/order are unchanged; a conservatively
        # recognized single-valued claim (if any) is attached additively in
        # metadata_, otherwise metadata_ stays {} exactly as before.
        for fact, claim_md in _extract_facts_with_claims(text):
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
                    metadata_=claim_md or {},
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


# Pattern index -> (canonical claim key, required trigger prefix). ONLY the
# three cleanly single-valued triggers get a claim. The required prefix is
# deliberately TIGHTER than the extraction regex, dropping every ambiguous
# alternative the regex also accepts:
#   - "i'm"/"i am" (could be a possessive or place: "I am Bob's friend") — only
#     the explicit "my name is" is a confident self-name assertion;
#   - "i'm at"/"i am at" (a location, not an employer: "I'm at the gym") — only
#     "i work at" reliably means current employer;
#   - "i'm from"/"i am from" (ORIGIN, not current home) — only "i live in".
# The team/group pattern (idx 2) and the packed use/prefer/favorite pattern
# (idx 4) have no canonical single-valued key and never emit a claim. We do NOT
# emit billing.primary_payment_processor here: generic "I use Stripe" must stay
# unkeyed (the LLM compiler may key it only on explicit "primary/current" language).
_CLAIMABLE_PATTERNS: dict[int, tuple[str, str]] = {
    0: ("identity.name", "my name is"),
    1: ("employment.current_employer", "i work at"),
    3: ("location.current_home", "i live in"),
}

# A first-person assertion carrying any of these is not a confident CURRENT
# single-valued fact (negation, uncertainty, or explicit history). When present
# in the clause we emit the memory exactly as before, WITHOUT a claim.
_CLAIM_BLOCKERS = re.compile(
    r"\b(?:not|never|no\s+longer|n't|"
    r"might|maybe|may|could|would|considering|thinking|planning|hop(?:e|ing)|"
    r"want\s+to|plan\s+to|"
    r"used\s+to|previously|formerly|former|ex|until|ago|left|past|if)\b",
    re.IGNORECASE,
)


def _claim_blocked(text: str, m: "re.Match[str]") -> bool:
    """True if the match sits in reported speech (double quotes) or its clause
    carries a negation / uncertainty / history marker — i.e. not a confident
    current fact. Omission is always preferred to a wrong authoritative claim."""
    before = text[: m.start()]
    if before.count('"') % 2 == 1:
        return True
    if before.count("“") > before.count("”"):  # smart quotes
        return True
    window = text[max(0, m.start() - 48) : m.end()]
    return bool(_CLAIM_BLOCKERS.search(window))


def _extract_facts_with_claims(text: str) -> list[tuple[str, dict | None]]:
    """Each extracted fact text, paired with an optional claim envelope.

    Claim emission is conservative by design: only the three tight single-valued
    triggers, only the registry-canonical value, only when no negation / quote /
    history / uncertainty marker is present. Everything else -> (fact, None),
    emitted exactly as before.
    """
    out: list[tuple[str, dict | None]] = []
    for idx, pattern in enumerate(_FACT_PATTERNS):
        spec = _CLAIMABLE_PATTERNS.get(idx)
        for m in pattern.finditer(text):
            fact = m.group(0).strip().rstrip(".").rstrip(",")
            claim_md: dict | None = None
            if spec is not None:
                key, trigger = spec
                value = (m.group(1) or "").strip().rstrip(".").rstrip(",")
                if value and fact.lower().startswith(trigger) and not _claim_blocked(text, m):
                    # build_claim_envelope re-validates against the registry and
                    # returns None for anything it cannot confidently key.
                    claim_md = build_claim_envelope(key, value, source="heuristic")
            out.append((fact, claim_md))
    return out


def _extract_profile_facts(text: str) -> list[str]:
    """Unchanged public surface: the extracted fact strings (group(0)) only."""
    return [fact for fact, _ in _extract_facts_with_claims(text)]
