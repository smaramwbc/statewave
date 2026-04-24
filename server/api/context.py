"""Context assembly route."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.engine import get_session
from server.schemas.requests import GetContextRequest
from server.schemas.responses import ContextBundleResponse
from server.services.context import assemble_context

router = APIRouter(tags=["context"])


@router.post("/v1/context", response_model=ContextBundleResponse, summary="Assemble context bundle")
async def get_context(
    body: GetContextRequest,
    session: AsyncSession = Depends(get_session),
):
    """Build a ranked, token-bounded context bundle for an AI task. Returns identity facts, recent history, and raw episodes within the token budget."""
    return await assemble_context(
        session,
        body.subject_id,
        body.task,
        max_tokens=body.max_tokens,
    )
