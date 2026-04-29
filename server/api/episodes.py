"""Episode routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.db.engine import get_session
from server.db.tables import EpisodeRow
from server.schemas.requests import BatchCreateEpisodesRequest, CreateEpisodeRequest
from server.schemas.responses import BatchCreateEpisodesResponse, EpisodeResponse
from server.services import webhooks
from server.core.tracing import span
from server.core.dependencies import get_tenant_id

router = APIRouter(prefix="/v1/episodes", tags=["episodes"])


@router.post("", response_model=EpisodeResponse, status_code=201, summary="Ingest an episode")
async def create_episode(
    body: CreateEpisodeRequest,
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    """Record a raw interaction episode. Episodes are append-only and immutable."""
    row = EpisodeRow(
        subject_id=body.subject_id,
        tenant_id=tenant_id,
        session_id=body.session_id,
        source=body.source,
        type=body.type,
        payload=body.payload,
        metadata_=body.metadata,
        provenance=body.provenance,
    )
    await repo.insert_episode(session, row)
    await session.commit()
    await session.refresh(row)
    await webhooks.fire("episode.created", {"id": str(row.id), "subject_id": row.subject_id})
    return EpisodeResponse(
        id=row.id,
        subject_id=row.subject_id,
        source=row.source,
        type=row.type,
        payload=row.payload,
        metadata=row.metadata_,
        provenance=row.provenance,
        session_id=row.session_id,
        created_at=row.created_at,
    )


@router.post(
    "/batch",
    response_model=BatchCreateEpisodesResponse,
    status_code=201,
    summary="Ingest episodes in batch",
)
async def create_episodes_batch(
    body: BatchCreateEpisodesRequest,
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    """Record multiple episodes in a single request. Max 100 per call."""
    with span("create_episodes_batch", {"count": len(body.episodes)}):
        rows: list[EpisodeRow] = []
        for ep in body.episodes:
            row = EpisodeRow(
                subject_id=ep.subject_id,
                tenant_id=tenant_id,
                session_id=ep.session_id,
                source=ep.source,
                type=ep.type,
                payload=ep.payload,
                metadata_=ep.metadata,
                provenance=ep.provenance,
            )
            await repo.insert_episode(session, row)
            rows.append(row)
        await session.commit()
        for row in rows:
            await session.refresh(row)
        await webhooks.fire(
            "episodes.batch_created",
            {
                "count": len(rows),
                "subject_ids": list({r.subject_id for r in rows}),
            },
        )
        return BatchCreateEpisodesResponse(
            episodes_created=len(rows),
            episodes=[
                EpisodeResponse(
                    id=r.id,
                    subject_id=r.subject_id,
                    source=r.source,
                    type=r.type,
                    payload=r.payload,
                    metadata=r.metadata_,
                    provenance=r.provenance,
                    session_id=r.session_id,
                    created_at=r.created_at,
                )
                for r in rows
            ],
        )
