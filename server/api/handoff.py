"""Handoff context pack route."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.dependencies import get_tenant_id
from server.db.engine import get_session
from server.schemas.requests import HandoffRequest
from server.schemas.responses import HandoffResponse
from server.services.handoff import assemble_handoff

router = APIRouter(tags=["handoff"])


@router.post(
    "/v1/handoff",
    response_model=HandoffResponse,
    summary="Generate a handoff context pack",
)
async def create_handoff(
    body: HandoffRequest,
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    """Generate a compact handoff brief for escalation, shift change, or agent transfer."""
    return await assemble_handoff(
        session,
        body.subject_id,
        body.session_id,
        reason=body.reason,
        max_tokens=body.max_tokens,
        tenant_id=tenant_id,
    )
