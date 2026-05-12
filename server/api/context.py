"""Context assembly route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.engine import get_session
from server.schemas.requests import GetContextRequest
from server.schemas.responses import ContextBundleResponse
from server.services.context import assemble_context
from server.services.receipts import load_tenant_receipt_config
from server.core.tracing import span
from server.core.dependencies import get_tenant_id

router = APIRouter(tags=["context"])


@router.post("/v1/context", response_model=ContextBundleResponse, summary="Assemble context bundle")
async def get_context(
    body: GetContextRequest,
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    """Build a ranked, token-bounded context bundle for an AI task. Returns identity facts, recent history, and raw episodes within the token budget.

    When `emit_receipt=true` (or the tenant's receipts config is `always`),
    the response includes a `receipt_id` pointing at an immutable
    state-assembly receipt — see docs/state-assembly-receipts.md.

    When the tenant config sets `require_caller_identity: true`, both
    `caller_id` and `caller_type` are mandatory — missing values
    return 401. This is the lever compliance-grade tenants flip to
    make policy enforcement non-bypassable."""
    tenant_config = await load_tenant_receipt_config(session, tenant_id)
    if tenant_config.get("require_caller_identity") and (
        not body.caller_id or not body.caller_type
    ):
        raise HTTPException(
            status_code=401,
            detail=(
                "tenant config requires caller_id and caller_type on every "
                "context assembly call"
            ),
        )
    with span("assemble_context", {"subject_id": body.subject_id, "task": body.task}):
        return await assemble_context(
            session,
            body.subject_id,
            body.task,
            max_tokens=body.max_tokens,
            tenant_id=tenant_id,
            session_id=body.session_id,
            emit_receipt=body.emit_receipt,
            query_id=body.query_id,
            task_id=body.task_id,
            parent_receipt_id=body.parent_receipt_id,
            caller_id=body.caller_id,
            caller_type=body.caller_type,
        )
