"""Episode routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.engine import get_session
from server.db.tables import EpisodeRow
from server.schemas.requests import CreateEpisodeRequest
from server.schemas.responses import EpisodeResponse

router = APIRouter(prefix="/v1/episodes", tags=["episodes"])


@router.post("", response_model=EpisodeResponse, status_code=201, summary="Ingest an episode")
async def create_episode(
    body: CreateEpisodeRequest,
    session: AsyncSession = Depends(get_session),
):
    """Record a raw interaction episode. Episodes are append-only and immutable."""
    row = EpisodeRow(
        subject_id=body.subject_id,
        source=body.source,
        type=body.type,
        payload=body.payload,
        metadata_=body.metadata,
        provenance=body.provenance,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return EpisodeResponse(
        id=row.id,
        subject_id=row.subject_id,
        source=row.source,
        type=row.type,
        payload=row.payload,
        metadata=row.metadata_,
        provenance=row.provenance,
        created_at=row.created_at,
    )
