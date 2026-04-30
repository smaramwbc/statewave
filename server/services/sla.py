"""Support SLA tracking — computes response and resolution time metrics.

Derives SLA signals from episode timestamps and resolution records:
- Time to first response (per session)
- Time to resolution (per session)
- Open issue duration (for unresolved sessions)
- Breach flags against configurable thresholds

Computed on demand from existing data. No stored state needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo

# Default thresholds
DEFAULT_FIRST_RESPONSE_THRESHOLD = timedelta(minutes=5)
DEFAULT_RESOLUTION_THRESHOLD = timedelta(hours=24)


@dataclass
class SessionSLA:
    """SLA metrics for a single support session."""

    session_id: str
    status: str  # open | resolved
    first_message_at: datetime | None = None
    first_response_at: datetime | None = None
    resolved_at: datetime | None = None
    first_response_seconds: float | None = None
    resolution_seconds: float | None = None
    open_duration_seconds: float | None = None
    first_response_breached: bool = False
    resolution_breached: bool = False


@dataclass
class SLASummary:
    """Subject-level SLA summary."""

    subject_id: str
    total_sessions: int = 0
    resolved_sessions: int = 0
    open_sessions: int = 0
    avg_first_response_seconds: float | None = None
    avg_resolution_seconds: float | None = None
    first_response_breach_count: int = 0
    resolution_breach_count: int = 0
    sessions: list[SessionSLA] = field(default_factory=list)


async def compute_sla(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    first_response_threshold: timedelta = DEFAULT_FIRST_RESPONSE_THRESHOLD,
    resolution_threshold: timedelta = DEFAULT_RESOLUTION_THRESHOLD,
) -> SLASummary:
    """Compute SLA metrics for a subject from episode and resolution data."""
    now = datetime.now(timezone.utc)

    episodes = await repo.list_episodes_by_subject(
        session, subject_id, tenant_id=tenant_id, limit=500
    )
    resolutions = await repo.list_resolutions(session, subject_id, tenant_id=tenant_id, limit=200)

    # Index resolutions by session_id
    resolution_map: dict[str, object] = {}
    for r in resolutions:
        resolution_map[r.session_id] = r

    # Group episodes by session
    session_episodes: dict[str, list] = {}
    for ep in episodes:
        sid = getattr(ep, "session_id", None)
        if sid:
            session_episodes.setdefault(sid, []).append(ep)

    # Compute per-session SLA
    session_slas: list[SessionSLA] = []
    response_times: list[float] = []
    resolution_times: list[float] = []
    fr_breach_count = 0
    res_breach_count = 0

    for sid, eps in session_episodes.items():
        eps_sorted = sorted(eps, key=lambda e: e.created_at)
        first_ep = eps_sorted[0] if eps_sorted else None
        resolution = resolution_map.get(sid)
        status = resolution.status if resolution else "unknown"

        sla = SessionSLA(session_id=sid, status=status)

        # First user message (various source names for user input)
        user_sources = ("user", "chat", "support-chat", "support_chat", "customer", "manual_input")
        first_user = next(
            (e for e in eps_sorted if getattr(e, "source", "") in user_sources),
            None,
        )
        # First agent/assistant response
        agent_sources = ("assistant", "agent", "system", "tool", "support", "staff")
        first_response = next(
            (
                e
                for e in eps_sorted
                if getattr(e, "source", "") in agent_sources
                and (first_user is None or e.created_at >= first_user.created_at)
            ),
            None,
        )

        # Use first_user timestamp, or fall back to first episode in session
        if first_user:
            sla.first_message_at = first_user.created_at
        elif first_ep:
            sla.first_message_at = first_ep.created_at

        if first_response:
            sla.first_response_at = first_response.created_at

        # Time to first response
        if first_user and first_response:
            delta = (first_response.created_at - first_user.created_at).total_seconds()
            sla.first_response_seconds = delta
            response_times.append(delta)
            if delta > first_response_threshold.total_seconds():
                sla.first_response_breached = True
                fr_breach_count += 1

        # Time to resolution
        if resolution and getattr(resolution, "resolved_at", None) and first_user:
            delta = (resolution.resolved_at - first_user.created_at).total_seconds()
            sla.resolution_seconds = delta
            sla.resolved_at = resolution.resolved_at
            resolution_times.append(delta)
            if delta > resolution_threshold.total_seconds():
                sla.resolution_breached = True
                res_breach_count += 1

        # Open duration (for unresolved)
        if status == "open" and first_user:
            sla.open_duration_seconds = (now - first_user.created_at).total_seconds()

        session_slas.append(sla)

    # Summary
    return SLASummary(
        subject_id=subject_id,
        total_sessions=len(session_slas),
        resolved_sessions=sum(1 for s in session_slas if s.status == "resolved"),
        open_sessions=sum(1 for s in session_slas if s.status == "open"),
        avg_first_response_seconds=(
            sum(response_times) / len(response_times) if response_times else None
        ),
        avg_resolution_seconds=(
            sum(resolution_times) / len(resolution_times) if resolution_times else None
        ),
        first_response_breach_count=fr_breach_count,
        resolution_breach_count=res_breach_count,
        sessions=session_slas,
    )
