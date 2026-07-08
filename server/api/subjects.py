"""Subject management routes (delete-by-subject)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.db.engine import get_session
from server.schemas.responses import DeleteSubjectResponse, ListSubjectsResponse
from server.services import webhooks
from server.core.dependencies import get_tenant_id

router = APIRouter(prefix="/v1/subjects", tags=["subjects"])


@router.get("", response_model=ListSubjectsResponse, summary="List known subjects")
async def list_subjects(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    """List all known subject IDs with episode and memory counts."""
    rows = await repo.list_subjects(session, tenant_id=tenant_id, limit=limit, offset=offset)
    total = await repo.count_subjects(session, tenant_id=tenant_id)
    return ListSubjectsResponse(
        subjects=rows,
        total=total,
    )


@router.delete(
    "/{subject_id}", response_model=DeleteSubjectResponse, summary="Delete all subject data"
)
async def delete_subject(
    subject_id: str,
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
):
    """Permanently delete all episodes and memories for a subject. This is irreversible."""
    ep_count = await repo.delete_episodes_by_subject(session, subject_id, tenant_id=tenant_id)
    mem_count = await repo.delete_memories_by_subject(session, subject_id, tenant_id=tenant_id)
    # "Permanently delete all subject data" must also reap the subject's
    # resolutions and health-cache row — there is no FK cascade. Leaving them
    # behind keeps open/resolved-session logic treating the deleted subject's
    # sessions as live and lets a stale health-cache row suppress/forge alerts
    # if the subject id is later reused.
    await repo.delete_resolutions_by_subject(session, subject_id, tenant_id=tenant_id)
    await repo.delete_health_cache_by_subject(session, subject_id, tenant_id=tenant_id)
    # Phase 2: subject_entities is OUT-OF-BAND from memories (no FK
    # since N entities can point at M memories, neither owns the other),
    # so the cascade has to be explicit here too. Without this, a
    # re-ingested subject would inherit stale entity rows pointing at
    # memory_ids that no longer exist — Phase 3 retrieval would surface
    # boost from ghost memories.
    await repo.delete_entities_by_subject(session, subject_id, tenant_id=tenant_id)
    await session.commit()
    # Only fire the webhook when something was actually deleted (issue #282):
    # deleting a missing subject (or the same subject twice) is a no-op, so a
    # subject.deleted event with zero counts would be a spurious deletion
    # signal to consumers (cache invalidation, audit, compliance records).
    # The HTTP response stays 200 with honest zero counts either way.
    if ep_count + mem_count > 0:
        await webhooks.fire(
            "subject.deleted",
            {
                "subject_id": subject_id,
                "episodes_deleted": ep_count,
                "memories_deleted": mem_count,
            },
            tenant_id=tenant_id,
        )
    return DeleteSubjectResponse(
        subject_id=subject_id,
        episodes_deleted=ep_count,
        memories_deleted=mem_count,
    )
