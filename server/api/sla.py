"""SLA tracking endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.dependencies import get_tenant_id
from server.db.engine import get_session
from server.schemas.responses import SessionSLAResponse, SLASummaryResponse
from server.services.sla import compute_sla

router = APIRouter(prefix="/v1/subjects", tags=["subjects"])


@router.get(
    "/{subject_id}/sla",
    response_model=SLASummaryResponse,
    summary="Get SLA metrics for a subject",
)
async def get_sla(
    subject_id: str,
    first_response_threshold_minutes: float = Query(5.0, ge=0),
    resolution_threshold_hours: float = Query(24.0, ge=0),
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    """Compute SLA metrics (first response time, resolution time, breach flags)."""
    from datetime import timedelta

    result = await compute_sla(
        session,
        subject_id,
        tenant_id=tenant_id,
        first_response_threshold=timedelta(minutes=first_response_threshold_minutes),
        resolution_threshold=timedelta(hours=resolution_threshold_hours),
    )
    return SLASummaryResponse(
        subject_id=result.subject_id,
        total_sessions=result.total_sessions,
        resolved_sessions=result.resolved_sessions,
        open_sessions=result.open_sessions,
        avg_first_response_seconds=result.avg_first_response_seconds,
        avg_resolution_seconds=result.avg_resolution_seconds,
        first_response_breach_count=result.first_response_breach_count,
        resolution_breach_count=result.resolution_breach_count,
        sessions=[
            SessionSLAResponse(
                session_id=s.session_id,
                status=s.status,
                first_message_at=s.first_message_at.isoformat() if s.first_message_at else None,
                first_response_at=s.first_response_at.isoformat() if s.first_response_at else None,
                resolved_at=s.resolved_at.isoformat() if s.resolved_at else None,
                first_response_seconds=s.first_response_seconds,
                resolution_seconds=s.resolution_seconds,
                open_duration_seconds=s.open_duration_seconds,
                first_response_breached=s.first_response_breached,
                resolution_breached=s.resolution_breached,
            )
            for s in result.sessions
        ],
    )
