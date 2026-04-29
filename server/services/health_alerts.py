"""Proactive health alerts — emits webhooks on health state transitions.

Fires `subject.health_degraded` when a subject's health state worsens:
  healthy → watch, watch → at_risk, healthy → at_risk

Fires `subject.health_improved` when a subject's health state recovers:
  at_risk → watch, watch → healthy, at_risk → healthy

Deduplication: compares against last-cached state. Unchanged states
do not trigger any events.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.services import webhooks
from server.services.health import HealthResult

# State severity ordering (higher = worse)
_STATE_SEVERITY = {"healthy": 0, "watch": 1, "at_risk": 2}


async def check_and_alert(
    session: AsyncSession,
    health: HealthResult,
    *,
    tenant_id: str | None = None,
) -> str | None:
    """Check if health changed and fire appropriate webhook.

    Returns the event name fired ('subject.health_degraded' or
    'subject.health_improved'), or None if no event was emitted.
    """
    current_severity = _STATE_SEVERITY.get(health.state, 0)

    # Get cached previous state
    cached = await repo.get_health_cache(session, health.subject_id)
    previous_state = cached.last_state if cached else "healthy"
    previous_severity = _STATE_SEVERITY.get(previous_state, 0)

    # Update cache regardless
    await repo.upsert_health_cache(
        session,
        health.subject_id,
        health.state,
        health.score,
        tenant_id=tenant_id,
    )

    # No change — no event
    if current_severity == previous_severity:
        return None

    # Determine event type
    if current_severity > previous_severity:
        event = "subject.health_degraded"
    else:
        # Improvement — but only fire if there was a cached prior worse state
        # (first-time healthy subjects with no cache should not emit)
        if cached is None:
            return None
        event = "subject.health_improved"

    # Fire webhook
    payload = {
        "subject_id": health.subject_id,
        "previous_state": previous_state,
        "current_state": health.state,
        "score": health.score,
        "factors": [
            {"signal": f.signal, "impact": f.impact, "detail": f.detail} for f in health.factors[:3]
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    await webhooks.fire(event, payload, db=session)
    return event
