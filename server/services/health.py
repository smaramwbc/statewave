"""Customer health scoring for support workflows.

Computes a deterministic health score (0-100) and state (healthy/watch/at_risk)
from signals already tracked by Statewave: resolution history, episode patterns,
urgency indicators, and issue recency.

Design principles:
- Support-focused: measures how well a customer's support needs are being met
- Explainable: every factor that affects the score is returned with its contribution
- Deterministic: same data always produces same score
- Compact: no ML, no stored state, computed on demand
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Starting score — degraded by negative signals, not boosted above 100
_BASE_SCORE = 100

# Penalties
_UNRESOLVED_ISSUE_PENALTY = 15  # Per open/unresolved session
_UNRESOLVED_CAP = 45  # Max penalty from unresolved issues

_REPEATED_ISSUE_PENALTY = 20  # If 2+ sessions share issue patterns
_ESCALATION_PENALTY = 10  # Per episode with urgency markers (capped)
_ESCALATION_CAP = 20

_IDLE_OPEN_PENALTY = 15  # Open issue with no activity in 7+ days

# SLA-based penalties
_SLA_BREACH_PENALTY = 10  # Per session with resolution SLA breach (capped)
_SLA_BREACH_CAP = 20
_SLOW_FIRST_RESPONSE_PENALTY = 5  # Avg first response > 10 min

# Bonuses (can recover score toward 100)
_RECENT_RESOLUTION_BONUS = 10  # A session was resolved in last 7 days
_HIGH_RESOLUTION_RATE_BONUS = 10  # >80% sessions resolved

# Thresholds
_HEALTHY_THRESHOLD = 70
_WATCH_THRESHOLD = 40

# Time windows
_RECENCY_DAYS = 7
_IDLE_DAYS = 7

# Urgency keywords (reuse from context module)
_URGENCY_KEYWORDS = frozenset(
    [
        "urgent",
        "critical",
        "asap",
        "emergency",
        "escalat",
        "broken",
        "down",
        "outage",
        "blocker",
        "p0",
        "p1",
        "sev1",
        "sev2",
    ]
)


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass
class HealthFactor:
    """A single factor contributing to the health score."""

    signal: str
    impact: int  # positive = bonus, negative = penalty
    detail: str


@dataclass
class HealthResult:
    """Customer health scoring result."""

    subject_id: str
    score: int  # 0-100
    state: str  # healthy | watch | at_risk
    factors: list[HealthFactor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


async def compute_health(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
) -> HealthResult:
    """Compute health score for a subject based on support history."""
    now = datetime.now(timezone.utc)

    # Gather data
    resolutions = await repo.list_resolutions(session, subject_id, tenant_id=tenant_id, limit=200)
    episodes = await repo.list_episodes_by_subject(
        session, subject_id, tenant_id=tenant_id, limit=200
    )

    score = _BASE_SCORE
    factors: list[HealthFactor] = []

    # --- Signal 1: Unresolved issues ---
    open_sessions = [r for r in resolutions if r.status == "open"]
    if open_sessions:
        penalty = min(len(open_sessions) * _UNRESOLVED_ISSUE_PENALTY, _UNRESOLVED_CAP)
        score -= penalty
        factors.append(
            HealthFactor(
                signal="unresolved_issues",
                impact=-penalty,
                detail=f"{len(open_sessions)} open session(s)",
            )
        )

    # --- Signal 2: Repeated issue patterns ---
    # Detect if multiple sessions exist with same issue keywords
    resolved_sessions = [r for r in resolutions if r.status == "resolved"]
    open_session_ids = {r.session_id for r in open_sessions}

    # Check if any open session shares keywords with a prior resolved session
    # (indicates a recurring problem)
    if open_sessions and resolved_sessions:
        open_texts = _collect_session_texts(episodes, open_session_ids)
        resolved_texts = _collect_session_texts(episodes, {r.session_id for r in resolved_sessions})
        if open_texts and resolved_texts and _has_keyword_overlap(open_texts, resolved_texts):
            score -= _REPEATED_ISSUE_PENALTY
            factors.append(
                HealthFactor(
                    signal="repeated_issues",
                    impact=-_REPEATED_ISSUE_PENALTY,
                    detail="Open issues resemble previously resolved ones",
                )
            )

    # --- Signal 3: Escalation / urgency markers ---
    escalation_count = sum(1 for ep in episodes if _ep_has_urgency(ep))
    if escalation_count > 0:
        penalty = min(escalation_count * _ESCALATION_PENALTY, _ESCALATION_CAP)
        score -= penalty
        factors.append(
            HealthFactor(
                signal="escalations",
                impact=-penalty,
                detail=f"{escalation_count} episode(s) with urgency markers",
            )
        )

    # --- Signal 4: Idle open issues (no activity in 7+ days) ---
    if open_sessions:
        # Find most recent episode for each open session
        for res in open_sessions:
            session_eps = [e for e in episodes if e.session_id == res.session_id]
            if session_eps:
                latest = max(session_eps, key=lambda e: e.created_at)
                days_idle = (now - latest.created_at).days
                if days_idle >= _IDLE_DAYS:
                    score -= _IDLE_OPEN_PENALTY
                    factors.append(
                        HealthFactor(
                            signal="idle_open_issue",
                            impact=-_IDLE_OPEN_PENALTY,
                            detail=f"Session {res.session_id} idle for {days_idle} days",
                        )
                    )
                    break  # Only penalize once for idle

    # --- Signal 5: Recent resolution (bonus) ---
    recent_resolutions = [
        r
        for r in resolved_sessions
        if r.resolved_at and (now - r.resolved_at).days <= _RECENCY_DAYS
    ]
    if recent_resolutions:
        score += _RECENT_RESOLUTION_BONUS
        factors.append(
            HealthFactor(
                signal="recent_resolution",
                impact=_RECENT_RESOLUTION_BONUS,
                detail=f"{len(recent_resolutions)} session(s) resolved in last {_RECENCY_DAYS} days",
            )
        )

    # --- Signal 6: High resolution rate (bonus) ---
    total_sessions = len(resolutions)
    if total_sessions >= 2:
        rate = len(resolved_sessions) / total_sessions
        if rate >= 0.8:
            score += _HIGH_RESOLUTION_RATE_BONUS
            factors.append(
                HealthFactor(
                    signal="high_resolution_rate",
                    impact=_HIGH_RESOLUTION_RATE_BONUS,
                    detail=f"{rate:.0%} sessions resolved ({len(resolved_sessions)}/{total_sessions})",
                )
            )

    # --- Signal 7: SLA resolution breaches ---
    # Check resolved sessions where resolution took > 24h (breach proxy)
    sla_breach_count = 0
    for r in resolved_sessions:
        if r.resolved_at:
            # Find first user episode in that session
            session_eps = [e for e in episodes if e.session_id == r.session_id]
            user_eps = [e for e in session_eps if getattr(e, "source", "") in ("user", "chat", "support-chat")]
            if user_eps:
                first_user = min(user_eps, key=lambda e: e.created_at)
                resolution_hours = (r.resolved_at - first_user.created_at).total_seconds() / 3600
                if resolution_hours > 24:
                    sla_breach_count += 1

    if sla_breach_count > 0:
        penalty = min(sla_breach_count * _SLA_BREACH_PENALTY, _SLA_BREACH_CAP)
        score -= penalty
        factors.append(
            HealthFactor(
                signal="sla_resolution_breaches",
                impact=-penalty,
                detail=f"{sla_breach_count} session(s) exceeded 24h resolution SLA",
            )
        )

    # --- Signal 8: Slow average first response ---
    response_times: list[float] = []
    for r in resolutions:
        session_eps = [e for e in episodes if e.session_id == r.session_id]
        session_eps_sorted = sorted(session_eps, key=lambda e: e.created_at)
        first_user = next(
            (e for e in session_eps_sorted if getattr(e, "source", "") in ("user", "chat", "support-chat")),
            None,
        )
        first_agent = next(
            (e for e in session_eps_sorted if getattr(e, "source", "") in ("assistant", "agent", "system", "tool")
             and first_user and e.created_at >= first_user.created_at),
            None,
        )
        if first_user and first_agent:
            response_times.append((first_agent.created_at - first_user.created_at).total_seconds())

    if response_times:
        avg_response_min = (sum(response_times) / len(response_times)) / 60
        if avg_response_min > 10:
            score -= _SLOW_FIRST_RESPONSE_PENALTY
            factors.append(
                HealthFactor(
                    signal="slow_first_response",
                    impact=-_SLOW_FIRST_RESPONSE_PENALTY,
                    detail=f"Avg first response {avg_response_min:.1f} min (threshold: 10 min)",
                )
            )

    # Clamp score
    score = max(0, min(100, score))

    # Determine state
    if score >= _HEALTHY_THRESHOLD:
        state = "healthy"
    elif score >= _WATCH_THRESHOLD:
        state = "watch"
    else:
        state = "at_risk"

    return HealthResult(
        subject_id=subject_id,
        score=score,
        state=state,
        factors=factors,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_session_texts(episodes, session_ids: set[str]) -> str:
    """Concatenate message content from episodes belonging to given sessions."""
    texts = []
    for ep in episodes:
        if ep.session_id in session_ids:
            payload = ep.payload or {}
            messages = payload.get("messages", [])
            for msg in messages:
                if isinstance(msg, dict) and msg.get("content"):
                    texts.append(msg["content"])
    return " ".join(texts)


def _has_keyword_overlap(text_a: str, text_b: str) -> bool:
    """Check if two text blobs share meaningful keywords (simple overlap)."""
    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "it",
        "to",
        "of",
        "and",
        "or",
        "for",
        "in",
        "on",
        "at",
        "by",
        "my",
        "i",
        "we",
        "you",
        "this",
        "that",
        "with",
    }
    words_a = {w for w in text_a.lower().split() if len(w) > 2 and w not in stopwords}
    words_b = {w for w in text_b.lower().split() if len(w) > 2 and w not in stopwords}
    if not words_a or not words_b:
        return False
    smaller = min(len(words_a), len(words_b))
    overlap = len(words_a & words_b)
    return (overlap / smaller) >= 0.3


def _ep_has_urgency(ep) -> bool:
    """Check if an episode contains urgency keywords."""
    payload = ep.payload or {}
    messages = payload.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict) and msg.get("content"):
            lower = msg["content"].lower()
            if any(kw in lower for kw in _URGENCY_KEYWORDS):
                return True
    return False
