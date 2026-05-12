"""Handoff context pack route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.dependencies import get_tenant_id
from server.db.engine import get_session
from server.schemas.requests import HandoffRequest
from server.schemas.responses import HandoffResponse
from server.services.handoff import assemble_handoff
from server.services.receipts import load_tenant_receipt_config

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
    """Generate a compact handoff brief for escalation, shift change, or agent transfer.

    Same caller-identity gate as /v1/context: when the tenant config
    sets `require_caller_identity: true`, both `caller_id` and
    `caller_type` are mandatory."""
    tenant_config = await load_tenant_receipt_config(session, tenant_id)
    if tenant_config.get("require_caller_identity") and (
        not body.caller_id or not body.caller_type
    ):
        raise HTTPException(
            status_code=401,
            detail=(
                "tenant config requires caller_id and caller_type on every "
                "handoff call"
            ),
        )
    return await assemble_handoff(
        session,
        body.subject_id,
        body.session_id,
        reason=body.reason,
        max_tokens=body.max_tokens,
        tenant_id=tenant_id,
        emit_receipt=body.emit_receipt,
        query_id=body.query_id,
        task_id=body.task_id,
        parent_receipt_id=body.parent_receipt_id,
        caller_id=body.caller_id,
        caller_type=body.caller_type,
    )
