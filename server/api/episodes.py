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
    """Record a raw interaction episode. Episodes are append-only and immutable.

    `metadata` is stored and returned unchanged, and takes no part in retrieval
    or ranking. One key is read when the context is assembled:
    `metadata["outcome"]`, if it is an object of the form
    `{"status": "succeeded" | "failed", "reason": "..."}`, is appended to the
    episode's context line. Any other value under `outcome`, a bare string or
    an unknown status, is ignored. Everything else the model has to read
    belongs in the episode's `payload`, `source` or `type`.
    """
    # `occurred_at` is optional in the request: when None, the database column
    # server-defaults to now() (= ingest time), which matches the legacy behaviour.
    # Connectors that backfill historical data set this explicitly.
    row_kwargs: dict = dict(
        subject_id=body.subject_id,
        tenant_id=tenant_id,
        session_id=body.session_id,
        source=body.source,
        type=body.type,
        payload=body.payload,
        metadata_=body.metadata,
        provenance=body.provenance,
        # Idempotency key: a first-class request field, falling back to where the
        # connectors historically stashed it (metadata.idempotency_key) so older
        # clients de-dup too. Drives the idempotent insert in insert_episode.
        idempotency_key=body.idempotency_key or body.metadata.get("idempotency_key"),
    )
    if body.occurred_at is not None:
        row_kwargs["occurred_at"] = body.occurred_at
    row = EpisodeRow(**row_kwargs)
    # On an idempotency conflict this returns the existing episode (the local row
    # is expunged), so rebind before commit/refresh.
    row = await repo.insert_episode(session, row)
    await session.commit()
    await session.refresh(row)
    await webhooks.fire(
        "episode.created",
        {"id": str(row.id), "subject_id": row.subject_id},
        tenant_id=tenant_id,
    )
    return EpisodeResponse.from_row(row)


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
    """Record multiple episodes in a single request. Max 100 per call.

    Each episode's `metadata` is stored and returned unchanged, and takes no
    part in retrieval or ranking. One key is read when the context is
    assembled: `metadata["outcome"]`, if it is an object of the form
    `{"status": "succeeded" | "failed", "reason": "..."}`, is appended to the
    episode's context line. Any other value under `outcome`, a bare string or
    an unknown status, is ignored. Everything else the model has to read
    belongs in the episode's `payload`, `source` or `type`.
    """
    with span("create_episodes_batch", {"count": len(body.episodes)}):
        rows: list[EpisodeRow] = []
        for ep in body.episodes:
            row_kwargs: dict = dict(
                subject_id=ep.subject_id,
                tenant_id=tenant_id,
                session_id=ep.session_id,
                source=ep.source,
                type=ep.type,
                payload=ep.payload,
                metadata_=ep.metadata,
                provenance=ep.provenance,
                idempotency_key=ep.idempotency_key or ep.metadata.get("idempotency_key"),
            )
            if ep.occurred_at is not None:
                row_kwargs["occurred_at"] = ep.occurred_at
            row = EpisodeRow(**row_kwargs)
            # insert_episode returns the EXISTING row on an idempotency conflict,
            # so append what it returns (not the local row, which is expunged).
            rows.append(await repo.insert_episode(session, row))
        await session.commit()
        for row in rows:
            await session.refresh(row)
        await webhooks.fire(
            "episodes.batch_created",
            {
                "count": len(rows),
                "subject_ids": list({r.subject_id for r in rows}),
            },
            tenant_id=tenant_id,
        )
        return BatchCreateEpisodesResponse(
            episodes_created=len(rows),
            episodes=[EpisodeResponse.from_row(r) for r in rows],
        )