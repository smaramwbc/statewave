"""Context assembly service — builds ranked, token-bounded context bundles.

Strategy:
- Score every memory and episode by kind priority + recency + relevance.
- When embeddings are available, use semantic similarity for relevance.
- Otherwise, fall back to word-overlap relevance.
- Assemble text incrementally, highest-scored items first.
- Stop adding items when the token budget is exhausted.
- The task header is always included (reserved budget).
- Provenance tracks exactly which items made it into the bundle.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
import tiktoken
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.config import settings
from server.db import repositories as repo
from server.schemas.responses import (
    ContextBundleResponse,
    EpisodeResponse,
    MemoryResponse,
    SessionInfo,
)
from server.services.compilers.heuristic import extract_payload_text
from server.services.embeddings import get_provider as get_embedding_provider

logger = structlog.stdlib.get_logger()

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
_SEMANTIC_MAX = 8.0  # Semantic relevance has higher signal than word-overlap
_TEMPORAL_VALID_BONUS = (
    3.0  # Bonus for memories currently valid (valid_to is None or in the future)
)
_TEMPORAL_EXPIRED_PENALTY = -4.0  # Penalty for memories whose valid_to is in the past
_SESSION_BOOST = 6.0  # Bonus for episodes belonging to the current active session
_RESOLVED_SESSION_PENALTY = -5.0  # Penalty for episodes in already-resolved sessions

# Support-agent-specific scoring signals
_OPEN_ISSUE_BOOST = 4.0  # Bonus for episodes in sessions with open/unresolved issues
_ACTION_STEP_BOOST = 2.0  # Bonus for agent/assistant/tool episodes (what was tried)
_URGENCY_BOOST = 2.0  # Bonus for episodes containing urgency keywords
_IDLE_CHATTER_PENALTY = -2.0  # Penalty for very short, low-signal content

# Repeat-issue detection
_REPEAT_ISSUE_BOOST = 4.0  # Boost for resolved-session episodes when issue recurs
_REPEAT_RESOLVED_BOOST = 6.0  # Extra boost when prior session has a useful resolution summary
_REPEAT_OVERLAP_THRESHOLD = 0.3  # Minimum keyword overlap ratio to trigger repeat detection

# Urgency keywords (case-insensitive match)
_URGENCY_KEYWORDS = frozenset(
    [
        "urgent",
        "critical",
        "blocked",
        "deadline",
        "asap",
        "emergency",
        "breaking",
        "outage",
        "downtime",
        "compliance",
        "sla",
        "escalat",
        "immediately",
        "production down",
        "p0",
        "p1",
        "severity",
    ]
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def assemble_context(
    session: AsyncSession,
    subject_id: str,
    task: str,
    max_tokens: int | None = None,
    tenant_id: str | None = None,
    session_id: str | None = None,
) -> ContextBundleResponse:
    """Build a deterministic, token-bounded, ranked context bundle."""
    budget = max_tokens or settings.default_max_context_tokens
    enc = tiktoken.get_encoding(settings.tiktoken_model)

    # -- Fetch candidates ---------------------------------------------------
    fact_rows = await repo.search_memories(
        session, subject_id, tenant_id=tenant_id, kind="profile_fact", limit=50
    )
    procedure_rows = await repo.search_memories(
        session, subject_id, tenant_id=tenant_id, kind="procedure", limit=20
    )
    summary_rows = await repo.search_memories(
        session, subject_id, tenant_id=tenant_id, kind="episode_summary", limit=30
    )
    episode_rows = await repo.list_episodes_by_subject(
        session, subject_id, tenant_id=tenant_id, limit=30
    )

    # -- Prepare semantic scoring -------------------------------------------
    semantic_scores: dict[uuid.UUID, float] = {}
    provider = get_embedding_provider()
    if provider:
        try:
            task_embedding = await provider.embed_query(task)
            # Get semantic scores for all memories in one query
            semantic_results = await repo.search_memories_by_embedding(
                session,
                subject_id,
                task_embedding,
                tenant_id=tenant_id,
                limit=100,
            )
            for row, distance in semantic_results:
                # Convert cosine distance [0, 2] to similarity score [0, SEMANTIC_MAX]
                # distance 0 = identical → max score, distance 2 = opposite → 0
                similarity = max(0.0, 1.0 - distance)
                semantic_scores[row.id] = similarity * _SEMANTIC_MAX
            logger.debug("semantic_scores_computed", count=len(semantic_scores))
        except Exception:
            logger.warning("semantic_scoring_failed_using_word_overlap", exc_info=True)

    use_semantic = len(semantic_scores) > 0

    # -- Score all candidates ------------------------------------------------
    task_tokens = _tokenize_for_relevance(task)

    scored: list[_ScoredItem] = []
    all_memory_rows = list(fact_rows) + list(procedure_rows) + list(summary_rows)
    if all_memory_rows:
        ts_range = _timestamp_range([r.created_at for r in all_memory_rows])
        for row in all_memory_rows:
            # Use semantic score when available, else word-overlap
            if use_semantic and row.id in semantic_scores:
                relevance = semantic_scores[row.id]
            else:
                relevance = _relevance_score(row.content, task_tokens)

            score = (
                _KIND_PRIORITY.get(row.kind, 1.0)
                + _recency_score(row.created_at, ts_range)
                + relevance
                + _temporal_score(row.valid_from, row.valid_to)
            )
            scored.append(
                _ScoredItem(
                    score=score,
                    kind="memory",
                    memory_row=row,
                    text=_render_memory_line(row),
                    section=_section_for_kind(row.kind),
                )
            )

    if episode_rows:
        # Collect episode IDs already covered by included summaries
        covered_episode_ids: set[str] = set()
        for row in summary_rows:
            if row.source_episode_ids:
                covered_episode_ids.update(str(eid) for eid in row.source_episode_ids)

        # Fetch resolved sessions to penalize their episodes
        resolved_sessions = await repo.get_resolved_session_ids(
            session, subject_id, tenant_id=tenant_id
        )
        open_sessions = await repo.get_open_session_ids(session, subject_id, tenant_id=tenant_id)

        # -- Repeat-issue detection -----------------------------------------
        # Build keyword set for current session episodes
        current_session_texts: list[str] = []
        resolved_session_texts: dict[str, list[str]] = {}
        for row in episode_rows:
            ep_session = getattr(row, "session_id", None)
            text = extract_payload_text(row.payload)
            if not text:
                continue
            if session_id and ep_session == session_id:
                current_session_texts.append(text)
            elif ep_session and ep_session in resolved_sessions:
                resolved_session_texts.setdefault(ep_session, []).append(text)

        current_keywords = _extract_issue_keywords(" ".join(current_session_texts))

        # Also include task text in current keywords (the task describes the issue)
        current_keywords |= _extract_issue_keywords(task)

        # Compute overlap for each resolved session + check resolution summaries
        repeat_issue_sessions: dict[str, float] = {}  # session_id → boost score
        if current_keywords:
            # Fetch resolution summaries for resolved sessions
            resolution_rows = await repo.list_resolutions(
                session, subject_id, tenant_id=tenant_id, status="resolved", limit=20
            )
            resolution_summaries: dict[str, str] = {
                r.session_id: r.resolution_summary or ""
                for r in resolution_rows
                if r.resolution_summary
            }

            for sid, texts in resolved_session_texts.items():
                prior_keywords = _extract_issue_keywords(" ".join(texts))
                # Include resolution summary keywords
                if sid in resolution_summaries:
                    prior_keywords |= _extract_issue_keywords(resolution_summaries[sid])
                overlap = _session_keyword_overlap(current_keywords, prior_keywords)
                if overlap >= _REPEAT_OVERLAP_THRESHOLD:
                    # Higher boost if there's a useful resolution summary
                    if sid in resolution_summaries:
                        repeat_issue_sessions[sid] = _REPEAT_RESOLVED_BOOST
                    else:
                        repeat_issue_sessions[sid] = _REPEAT_ISSUE_BOOST

        ep_ts_range = _timestamp_range([r.created_at for r in episode_rows])
        for row in episode_rows:
            if str(row.id) in covered_episode_ids:
                continue  # already represented by a summary
            ep_text = _short_episode_text(row.payload, row.source, row.type)
            content_text = extract_payload_text(row.payload)
            score = (
                _EPISODE_PRIORITY
                + _recency_score(row.created_at, ep_ts_range)
                + _relevance_score(content_text, task_tokens)
            )
            # Boost episodes belonging to the active session
            if session_id and getattr(row, "session_id", None) == session_id:
                score += _SESSION_BOOST
            # Penalize episodes from resolved sessions
            ep_session = getattr(row, "session_id", None)
            if ep_session and ep_session in resolved_sessions:
                score += _RESOLVED_SESSION_PENALTY
            # Repeat-issue boost (counteracts resolved penalty for recurring issues)
            if ep_session and ep_session in repeat_issue_sessions:
                score += repeat_issue_sessions[ep_session]
            # Boost episodes from sessions with open issues
            if ep_session and ep_session in open_sessions:
                score += _OPEN_ISSUE_BOOST
            # Boost action/step episodes (agent, assistant, tool, system)
            if row.source in ("assistant", "agent", "tool", "system") or row.type in (
                "action",
                "tool_call",
                "resolution_attempt",
            ):
                score += _ACTION_STEP_BOOST
            # Urgency keyword detection
            if content_text and _has_urgency(content_text):
                score += _URGENCY_BOOST
            # Idle chatter penalty (very short, low-signal content)
            if content_text and len(content_text.strip()) < 20:
                score += _IDLE_CHATTER_PENALTY
            scored.append(
                _ScoredItem(
                    score=score,
                    kind="episode",
                    episode_row=row,
                    text=f"- {ep_text}",
                    section="episodes",
                )
            )

    # Sort descending by score (stable sort preserves insertion order for ties)
    scored.sort(key=lambda s: s.score, reverse=True)

    # -- Assemble within budget ---------------------------------------------
    task_header = f"## Task\n{task}\n"
    budget_used = len(enc.encode(task_header))

    included_facts: list[MemoryResponse] = []
    included_summaries: list[MemoryResponse] = []
    included_procedures: list[MemoryResponse] = []
    included_episodes: list[EpisodeResponse] = []
    section_lines: dict[str, list[str]] = {
        "facts": [],
        "procedures": [],
        "history": [],
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
            elif item.memory_row.kind == "episode_summary":
                included_summaries.append(resp)
        elif item.kind == "episode":
            included_episodes.append(_episode_response(item.episode_row))

    # -- Render final text --------------------------------------------------
    # Order: task → facts → procedures → history → episodes
    # This mirrors how a human would brief someone: who is this person,
    # what do they need, what happened recently, raw detail if room.
    parts: list[str] = [task_header]
    if section_lines["facts"]:
        parts.append("## About this user")
        parts.extend(section_lines["facts"])
        parts.append("")
    if section_lines["procedures"]:
        parts.append("## Procedures")
        parts.extend(section_lines["procedures"])
        parts.append("")
    if section_lines["history"]:
        parts.append("## Recent history")
        parts.extend(section_lines["history"])
        parts.append("")
    if section_lines["episodes"]:
        parts.append("## Recent interactions")
        # Group by session when a current session is specified
        if session_id:
            current_lines: list[str] = []
            other_lines: list[str] = []
            for ep in included_episodes:
                line = f"- {_short_episode_text(ep.payload, ep.source, ep.type)}"
                if ep.session_id == session_id:
                    current_lines.append(line)
                else:
                    other_lines.append(line)
            if current_lines:
                parts.append(f"### Current session ({session_id})")
                parts.extend(current_lines)
            if other_lines:
                parts.append("### Previous interactions")
                parts.extend(other_lines)
        else:
            parts.extend(section_lines["episodes"])
        parts.append("")

    assembled = "\n".join(parts)
    token_estimate = len(enc.encode(assembled))

    provenance = {
        "fact_ids": [str(f.id) for f in included_facts],
        "summary_ids": [str(s.id) for s in included_summaries],
        "procedure_ids": [str(p.id) for p in included_procedures],
        "episode_ids": [str(e.id) for e in included_episodes],
    }

    # Build session info from included episodes
    session_map: dict[str, list[EpisodeResponse]] = {}
    for ep in included_episodes:
        sid = ep.session_id or "__none__"
        session_map.setdefault(sid, []).append(ep)
    sessions: list[SessionInfo] = []
    for sid, eps in session_map.items():
        if sid == "__none__":
            continue
        times = [e.created_at for e in eps if e.created_at]
        sessions.append(
            SessionInfo(
                session_id=sid,
                episode_count=len(eps),
                first_at=min(times) if times else None,
                last_at=max(times) if times else None,
            )
        )

    # Include summaries in the facts response list for backward compatibility
    all_facts = included_facts + included_summaries

    return ContextBundleResponse(
        subject_id=subject_id,
        task=task,
        facts=all_facts,
        episodes=included_episodes,
        procedures=included_procedures,
        provenance=provenance,
        assembled_context=assembled,
        token_estimate=token_estimate,
        sessions=sessions,
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# Common stopwords to exclude from issue keyword extraction
_STOPWORDS = frozenset(
    "i me my we our you your he she it they them their this that "
    "is am are was were be been being have has had do does did will would "
    "shall should can could may might must a an the and but or nor for "
    "so yet at by from in into of on to up with as if then else when "
    "where how what which who whom why all any each every no not only "
    "very just also too hi hello thanks thank okay ok yes please hey "
    "sure got get let know think want need help".split()
)


def _extract_issue_keywords(text: str) -> set[str]:
    """Extract meaningful issue keywords from text (lowercase, stopwords removed)."""
    if not text:
        return set()
    words = set(text.lower().split())
    return words - _STOPWORDS


def _session_keyword_overlap(current_keywords: set[str], prior_keywords: set[str]) -> float:
    """Compute overlap ratio of current keywords found in prior keywords."""
    if not current_keywords or not prior_keywords:
        return 0.0
    overlap = len(current_keywords & prior_keywords)
    return overlap / len(current_keywords)


def _tokenize_for_relevance(text: str) -> set[str]:
    """Lowercase word tokens for relevance scoring."""
    return set(text.lower().split())


def _has_urgency(text: str) -> bool:
    """Check if text contains urgency/severity keywords."""
    lower = text.lower()
    return any(kw in lower for kw in _URGENCY_KEYWORDS)


def _relevance_score(content: str, task_tokens: set[str]) -> float:
    """Word-overlap relevance: 0 to _RELEVANCE_MAX."""
    if not task_tokens or not content:
        return 0.0
    content_tokens = set(content.lower().split())
    overlap = len(task_tokens & content_tokens)
    # Normalize by task token count
    ratio = overlap / len(task_tokens)
    return min(ratio * _RELEVANCE_MAX, _RELEVANCE_MAX)


def _temporal_score(
    valid_from: datetime | None,
    valid_to: datetime | None,
) -> float:
    """Score boost/penalty based on temporal validity window.

    - Memory with no valid_to (still current): gets a bonus.
    - Memory whose valid_to is in the future: gets a bonus.
    - Memory whose valid_to is in the past (expired/superseded): gets a penalty.
    """
    now = datetime.now(timezone.utc)
    if valid_to is None:
        return _TEMPORAL_VALID_BONUS  # still active / no expiry
    if valid_to.tzinfo is None:
        valid_to = valid_to.replace(tzinfo=timezone.utc)
    if valid_to > now:
        return _TEMPORAL_VALID_BONUS  # still within validity window
    return _TEMPORAL_EXPIRED_PENALTY  # expired


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
    if kind == "episode_summary":
        return "history"
    return "history"


def _short_episode_text(payload: dict, source: str, type_: str) -> str:
    """Render an episode as a single line for context text."""
    text = extract_payload_text(payload)
    if text:
        return f"[{source}/{type_}] {text[:150]}"
    return f"[{source}/{type_}] (no text content)"


def _render_memory_line(row: Any) -> str:
    """Render a memory as a clean context line, appropriate for its kind."""
    if row.kind == "episode_summary":
        # Summaries contain raw "role: content" lines — condense into a readable note
        return f"- {_clean_summary(row.content)}"
    return f"- {row.content}"


def _clean_summary(text: str) -> str:
    """Turn raw 'user: X\\nassistant: Y' into a readable history note."""
    import re

    lines = text.strip().split("\n")
    parts: list[str] = []
    for line in lines:
        # Strip role prefix
        cleaned = re.sub(r"^(user|assistant|system):\s*", "", line.strip())
        if cleaned:
            parts.append(cleaned)
    if len(parts) <= 2:
        return " → ".join(parts)
    return parts[0] + " → " + parts[-1]  # first user msg → last assistant response


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
        session_id=getattr(row, "session_id", None),
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
