"""Timeline route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.db.engine import get_session
from server.schemas.responses import EpisodeResponse, MemoryResponse, TimelineResponse

router = APIRouter(tags=["timeline"])


@router.get("/v1/timeline", response_model=TimelineResponse, summary="Get subject timeline")
async def get_timeline(
    subject_id: str = Query(...),
    session: AsyncSession = Depends(get_session),
):
    episodes = await repo.list_episodes_by_subject(session, subject_id)
    memories = await repo.list_memories_by_subject(session, subject_id)
    return TimelineResponse(
        subject_id=subject_id,
        episodes=[
            EpisodeResponse(
                id=e.id,
                subject_id=e.subject_id,
                source=e.source,
                type=e.type,
                payload=e.payload,
                metadata=e.metadata_,
                provenance=e.provenance,
                created_at=e.created_at,
            )
            for e in episodes
        ],
        memories=[
            MemoryResponse(
                id=m.id,
                subject_id=m.subject_id,
                kind=m.kind,
                content=m.content,
                summary=m.summary,
                confidence=m.confidence,
                valid_from=m.valid_from,
                valid_to=m.valid_to,
                source_episode_ids=m.source_episode_ids or [],
                metadata=m.metadata_,
                status=m.status,
                created_at=m.created_at,
                updated_at=m.updated_at,
            )
            for m in memories
        ],
    )
