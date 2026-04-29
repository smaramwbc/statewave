"""Context assembly route."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.engine import get_session
from server.schemas.requests import GetContextRequest
from server.schemas.responses import ContextBundleResponse
from server.services.context import assemble_context
from server.core.tracing import span
from server.core.dependencies import get_tenant_id

router = APIRouter(tags=["context"])


@router.post("/v1/context", response_model=ContextBundleResponse, summary="Assemble context bundle")
async def get_context(
    body: GetContextRequest,
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    """Build a ranked, token-bounded context bundle for an AI task. Returns identity facts, recent history, and raw episodes within the token budget."""
    with span("assemble_context", {"subject_id": body.subject_id, "task": body.task}):
        return await assemble_context(
            session,
            body.subject_id,
            body.task,
            max_tokens=body.max_tokens,
            tenant_id=tenant_id,
            session_id=body.session_id,
        )
