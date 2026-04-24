"""Memory routes — compile and search."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.db.engine import get_session
from server.schemas.requests import CompileMemoriesRequest
from server.schemas.responses import CompileMemoriesResponse, MemoryResponse, SearchMemoriesResponse
from server.services.compiler import compile_memories_from_episodes

router = APIRouter(prefix="/v1/memories", tags=["memories"])


@router.post("/compile", response_model=CompileMemoriesResponse)
async def compile_memories(
    body: CompileMemoriesRequest,
    session: AsyncSession = Depends(get_session),
):
    episodes = await repo.list_episodes_by_subject(session, body.subject_id)
    new_rows = compile_memories_from_episodes(list(episodes))
    for row in new_rows:
        session.add(row)
    await session.commit()
    # refresh
    for row in new_rows:
        await session.refresh(row)
    return CompileMemoriesResponse(
        subject_id=body.subject_id,
        memories_created=len(new_rows),
        memories=[_to_response(r) for r in new_rows],
    )


@router.get("/search", response_model=SearchMemoriesResponse)
async def search_memories(
    subject_id: str = Query(...),
    kind: str | None = Query(None),
    query: str | None = Query(None, alias="q"),
    limit: int = Query(20, le=100),
    session: AsyncSession = Depends(get_session),
):
    rows = await repo.search_memories(session, subject_id, kind=kind, query=query, limit=limit)
    return SearchMemoriesResponse(memories=[_to_response(r) for r in rows])


def _to_response(row) -> MemoryResponse:
    return MemoryResponse(
        id=row.id,
        subject_id=row.subject_id,
        kind=row.kind,
        content=row.content,
        summary=row.summary,
        confidence=row.confidence,
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        source_episode_ids=row.source_episode_ids or [],
        metadata=row.metadata_,
        status=row.status,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
