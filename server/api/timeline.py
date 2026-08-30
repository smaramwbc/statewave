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
    limit: int = Query(100, ge=1, le=200, description="Rows per collection."),
    offset: int = Query(
        0,
        ge=0,
        description=(
            "Rows to skip per collection, counted from the oldest end by default "
            "or from the newest end when `newest_first=true`."
        ),
    ),
    newest_first: bool = Query(
        False,
        description=(
            "Page from the most recent records instead of the oldest ones. "
            "Applies to both collections. Rows within a page are still returned "
            "in ascending chronological order; `offset` then counts back from the "
            "newest row, so `offset=limit` is the next-older page."
        ),
    ),
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    # With the default ascending order a subject past `limit` rows returns its
    # OLDEST rows and its recent history is unreachable through this route —
    # the wrong half to lose for a consumer asking "what has happened lately".
    # `newest_first` takes the window from the other end; the default stays
    # False so an existing client sees no change.
    #
    # Fetch one extra row per collection to detect whether more pages remain
    # without a separate COUNT query (#331). The repository returns ascending
    # rows in both modes, so the surplus row is the LAST one when paging from
    # the oldest end and the FIRST one when paging from the newest end — trim
    # accordingly, or newest_first would silently drop the newest row.
    episodes = await repo.list_episodes_by_subject(
        session,
        subject_id,
        tenant_id=tenant_id,
        limit=limit + 1,
        offset=offset,
        newest_first=newest_first,
    )
    memories = await repo.list_memories_by_subject(
        session,
        subject_id,
        tenant_id=tenant_id,
        limit=limit + 1,
        offset=offset,
        newest_first=newest_first,
    )
    episodes_has_more = len(episodes) > limit
    memories_has_more = len(memories) > limit
    if newest_first:
        episodes = episodes[1:] if episodes_has_more else episodes
        memories = memories[1:] if memories_has_more else memories
    else:
        episodes = episodes[:limit]
        memories = memories[:limit]
    return TimelineResponse(
        subject_id=subject_id,
        episodes=[EpisodeResponse.from_row(e) for e in episodes],
        memories=[MemoryResponse.from_row(m) for m in memories],
        episodes_has_more=episodes_has_more,
        memories_has_more=memories_has_more,
    )
