"""Timeline route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.db.engine import get_session
from server.schemas.responses import EpisodeResponse, MemoryResponse, TimelineResponse
from server.core.dependencies import get_tenant_id

router = APIRouter(tags=["timeline"])


@router.get("/v1/timeline", response_model=TimelineResponse, summary="Get subject timeline")
async def get_timeline(
    subject_id: str = Query(...),
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    # Fetch one extra row per collection to detect whether more pages
    # remain, without a separate COUNT query (#331); trim back to `limit`
    # before building the response.
    episodes = await repo.list_episodes_by_subject(
        session, subject_id, tenant_id=tenant_id, limit=limit + 1, offset=offset
    )
    memories = await repo.list_memories_by_subject(
        session, subject_id, tenant_id=tenant_id, limit=limit + 1, offset=offset
    )
    episodes_has_more = len(episodes) > limit
    memories_has_more = len(memories) > limit
    episodes = episodes[:limit]
    memories = memories[:limit]
    return TimelineResponse(
        subject_id=subject_id,
        episodes=[EpisodeResponse.from_row(e) for e in episodes],
        memories=[MemoryResponse.from_row(m) for m in memories],
        episodes_has_more=episodes_has_more,
        memories_has_more=memories_has_more,
    )