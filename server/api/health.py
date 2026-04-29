"""Customer health scoring endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.dependencies import get_tenant_id
from server.db.engine import get_session
from server.schemas.responses import HealthFactorResponse, HealthResponse
from server.services.health import compute_health
from server.services.health_alerts import check_and_alert

router = APIRouter(prefix="/v1/subjects", tags=["subjects"])


@router.get(
    "/{subject_id}/health",
    response_model=HealthResponse,
    summary="Get customer health score",
)
async def get_health(
    subject_id: str,
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    """Compute support health score for a subject based on resolution and episode history."""
    result = await compute_health(session, subject_id, tenant_id=tenant_id)
    await check_and_alert(session, result, tenant_id=tenant_id)
    await session.commit()
    return HealthResponse(
        subject_id=result.subject_id,
        score=result.score,
        state=result.state,
        factors=[
            HealthFactorResponse(signal=f.signal, impact=f.impact, detail=f.detail)
            for f in result.factors
        ],
    )
