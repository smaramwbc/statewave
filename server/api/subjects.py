"""Subject management routes (delete-by-subject)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.db.engine import get_session
from server.schemas.responses import DeleteSubjectResponse, ListSubjectsResponse
from server.services import webhooks

router = APIRouter(prefix="/v1/subjects", tags=["subjects"])


@router.get("", response_model=ListSubjectsResponse, summary="List known subjects")
async def list_subjects(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """List all known subject IDs with episode and memory counts."""
    rows = await repo.list_subjects(session, limit=limit, offset=offset)
    return ListSubjectsResponse(
        subjects=rows,
        total=len(rows),
    )


@router.delete("/{subject_id}", response_model=DeleteSubjectResponse, summary="Delete all subject data")
async def delete_subject(
    subject_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Permanently delete all episodes and memories for a subject. This is irreversible."""
    ep_count = await repo.delete_episodes_by_subject(session, subject_id)
    mem_count = await repo.delete_memories_by_subject(session, subject_id)
    await session.commit()
    await webhooks.fire("subject.deleted", {
        "subject_id": subject_id,
        "episodes_deleted": ep_count,
        "memories_deleted": mem_count,
    })
    return DeleteSubjectResponse(
        subject_id=subject_id,
        episodes_deleted=ep_count,
        memories_deleted=mem_count,
    )
