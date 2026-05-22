"""Memory template routes.

Templates are read-only declarative patterns (see
`server.services.templates`). `apply` validates caller-supplied field
values against a template and ingests an ordinary episode — the
templated episode then flows through the normal compile/context
pipeline, with `template_id` / `template_version` recorded for
provenance.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.dependencies import get_tenant_id
from server.db import repositories as repo
from server.db.engine import get_session
from server.db.tables import EpisodeRow
from server.schemas.requests import ApplyTemplateRequest
from server.schemas.responses import EpisodeResponse
from server.services import templates, webhooks

router = APIRouter(prefix="/v1/memory-templates", tags=["memory-templates"])


@router.get(
    "",
    response_model=templates.MemoryTemplateList,
    summary="List memory templates",
)
async def list_memory_templates() -> templates.MemoryTemplateList:
    """List every bundled memory template, with its full field schema."""
    return templates.MemoryTemplateList(templates=templates.list_templates())


@router.get(
    "/{template_id}",
    response_model=templates.MemoryTemplate,
    summary="Get a memory template",
)
async def get_memory_template(template_id: str) -> templates.MemoryTemplate:
    """Fetch one memory template by id — fields, types, and the content scaffold."""
    template = templates.get_template(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail=f"unknown memory template: {template_id}")
    return template


@router.post(
    "/{template_id}/apply",
    response_model=EpisodeResponse,
    status_code=201,
    summary="Apply a memory template",
)
async def apply_memory_template(
    template_id: str,
    body: ApplyTemplateRequest,
    session: AsyncSession = Depends(get_session),
    tenant_id: str | None = Depends(get_tenant_id),
) -> EpisodeResponse:
    """Validate field values against a template and ingest the resulting episode.

    The episode's `payload` carries the template id/version, the supplied
    field values, and the deterministically-rendered `content`;
    `metadata.template` records the id/version for provenance. The
    episode is otherwise identical to one created via `POST /v1/episodes`
    and compiles the same way.
    """
    template = templates.get_template(template_id)
    if template is None:
        raise HTTPException(status_code=404, detail=f"unknown memory template: {template_id}")

    try:
        payload = templates.render(template, body.values)
    except templates.TemplateError as exc:
        # Caller-supplied values failed the template's field schema.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    row = EpisodeRow(
        subject_id=body.subject_id,
        tenant_id=tenant_id,
        session_id=body.session_id,
        source="memory-template",
        type=template.episode_type,
        payload=payload,
        metadata_={"template": {"id": template.id, "version": template.version}},
        provenance={},
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
        occurred_at=row.occurred_at,
        created_at=row.created_at,
    )
