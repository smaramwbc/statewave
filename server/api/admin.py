"""Admin endpoints — operator introspection and advanced bootstrap capabilities."""

from __future__ import annotations

import uuid
from typing import Literal, Optional

import structlog
from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel

from server.core.config import settings
from server.schemas.requests import TenantConfigPatch
from server.schemas.responses import TenantConfigResponse
from server.services import webhooks

logger = structlog.stdlib.get_logger()

router = APIRouter(prefix="/admin", tags=["admin"])


def _like_escape(s: str) -> str:
    """Escape SQL LIKE metacharacters in *s* so it matches literally.

    ``%`` and ``_`` are wildcards in SQL LIKE patterns; ``\\`` is the escape
    prefix.  Callers must also pass ``escape="\\\\"`` to the SQLAlchemy
    ``.ilike()`` / ``.like()`` / ``.not_like()`` call so the DB knows which
    character introduces an escape sequence.
    """
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# ─── Response Models ─────────────────────────────────────────────────────────


class SubjectListItem(BaseModel):
    subject_id: str
    tenant_id: str | None
    episode_count: int
    memory_count: int
    session_count: int
    last_episode_at: str | None
    health_state: str | None
    health_score: int | None
    open_sessions: int


class SubjectListResponse(BaseModel):
    subjects: list[SubjectListItem]
    total: int
    limit: int
    offset: int


class SubjectSummary(BaseModel):
    episode_count: int
    memory_count: int
    session_count: int
    first_seen_at: str | None
    last_activity_at: str | None


class SubjectHealthSummary(BaseModel):
    score: int
    state: str
    factors: list[dict]


class SubjectSLASummary(BaseModel):
    total_sessions: int
    resolved_sessions: int
    open_sessions: int
    avg_first_response_seconds: float | None
    avg_resolution_seconds: float | None
    first_response_breach_count: int
    resolution_breach_count: int


class SubjectDetailResponse(BaseModel):
    subject_id: str
    tenant_id: str | None
    summary: SubjectSummary
    health: SubjectHealthSummary | None
    sla: SubjectSLASummary | None


class MemoryListItem(BaseModel):
    id: str
    kind: str
    content: str
    summary: str
    confidence: float
    status: str
    source_episode_ids: list[str]
    valid_from: str
    valid_to: str | None
    created_at: str
    # Governance fields (v0.9): exposed on subject-scoped listings so the
    # admin "open subject → view memories" flow can show labels in one
    # place — both already-authoritative and detector-suggested.
    # Empty list is the documented default; the policy evaluator reads
    # only ``sensitivity_labels`` so surfacing them here is non-destructive.
    sensitivity_labels: list[str]
    suggested_labels: list[str]


class MemoryListResponse(BaseModel):
    memories: list[MemoryListItem]
    total: int
    limit: int
    offset: int


class SuggestedLabelMemoryItem(BaseModel):
    """Memory row enriched with its auto-labeling suggestions.

    Surface for the admin review endpoint that powers the
    "promote suggested labels into authoritative sensitivity_labels"
    operator workflow (v0.9, issue #158). `sensitivity_labels` is
    included so the UI can show what's already authoritative next to
    what the detectors propose adding.
    """

    id: str
    subject_id: str
    tenant_id: str | None
    kind: str
    content: str
    summary: str
    suggested_labels: list[str]
    sensitivity_labels: list[str]
    created_at: str


class SuggestedLabelsListResponse(BaseModel):
    memories: list[SuggestedLabelMemoryItem]
    total: int
    limit: int
    offset: int
    catalogue: list[dict[str, str]]


class EpisodeListItem(BaseModel):
    id: str
    session_id: str | None
    source: str
    type: str
    payload: dict
    metadata: dict
    provenance: dict
    created_at: str


class EpisodeListResponse(BaseModel):
    episodes: list[EpisodeListItem]
    total: int
    limit: int
    offset: int


def _require_snapshots():
    """Guard: raise 404 if snapshots feature is disabled."""
    if not settings.enable_snapshots:
        raise HTTPException(status_code=404, detail="Not found")


# ─── Dashboard Aggregation ───


@router.get("/dashboard")
async def dashboard_overview():
    """Single aggregation endpoint for the admin dashboard.

    Returns system health, migration status, job stats, webhook stats,
    data counts, and subject health distribution in one request.
    """
    import asyncio

    from sqlalchemy import func, select

    from server.db import engine as engine_module
    from server.db.tables import CompileJobRow, EpisodeRow, MemoryRow
    from server.services.migrations import check_migration_status
    from server.services.readiness import run_readiness_checks

    async def _get_counts():
        async with engine_module.get_session_factory()() as session:
            episodes = await session.scalar(select(func.count()).select_from(EpisodeRow)) or 0
            memories = await session.scalar(select(func.count()).select_from(MemoryRow)) or 0
            subjects = (
                await session.scalar(select(func.count(func.distinct(EpisodeRow.subject_id)))) or 0
            )
            return {"episodes": episodes, "memories": memories, "subjects": subjects}

    async def _get_job_stats():
        async with engine_module.get_session_factory()() as session:
            rows = await session.execute(
                select(CompileJobRow.status, func.count()).group_by(CompileJobRow.status)
            )
            stats = {row[0]: row[1] for row in rows}
            return stats

    async def _get_health_distribution():
        """Get subject health score distribution from cache table.

        Also includes subjects without health data as 'unknown'.
        """
        from server.db.tables import Base, EpisodeRow

        # Check if health cache table exists in metadata
        if "subject_health_cache" not in Base.metadata.tables:
            return None
        try:
            from sqlalchemy import text

            async with engine_module.get_session_factory()() as session:
                # Get health state distribution from cache
                rows = await session.execute(
                    text(
                        "SELECT last_state, COUNT(*) FROM subject_health_cache GROUP BY last_state"
                    )
                )
                dist = {row[0]: row[1] for row in rows}

                # Get total distinct subjects from episodes
                total_subjects = (
                    await session.scalar(select(func.count(func.distinct(EpisodeRow.subject_id))))
                    or 0
                )

                # Calculate subjects without health data
                subjects_with_health = sum(dist.values())
                unknown_count = total_subjects - subjects_with_health

                if unknown_count > 0:
                    dist["unknown"] = unknown_count

                return dist
        except Exception:
            return None

    async def _get_readiness():
        from server.db.engine import get_engine

        async with get_engine().connect() as conn:
            return await run_readiness_checks(conn)

    # Run all queries concurrently
    readiness, migration, counts, job_stats, webhook_stats, health_dist = await asyncio.gather(
        _get_readiness(),
        check_migration_status(),
        _get_counts(),
        _get_job_stats(),
        webhooks.get_delivery_stats(),
        _get_health_distribution(),
    )

    return {
        "readiness": {
            "status": readiness.status,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail, "latency_ms": c.latency_ms}
                for c in readiness.checks
            ],
        },
        "migration": {
            "current_revision": migration.current_revision,
            "expected_head": migration.expected_head,
            "is_compatible": migration.is_compatible,
            "pending_count": migration.pending_count,
        },
        "counts": counts,
        "jobs": job_stats,
        "webhooks": webhook_stats,
        "health_distribution": health_dist,
    }


@router.get("/tenants")
async def list_tenants():
    """List all distinct tenant IDs in the system."""
    from sqlalchemy import distinct, select

    from server.db import engine as engine_module
    from server.db.tables import EpisodeRow

    async with engine_module.get_session_factory()() as session:
        result = await session.execute(
            select(distinct(EpisodeRow.tenant_id))
            .where(EpisodeRow.tenant_id.isnot(None))
            .order_by(EpisodeRow.tenant_id)
        )
        tenants = [row[0] for row in result.all()]
        return {"tenants": tenants}


# ─── Subject Explorer ────────────────────────────────────────────────────────


@router.get("/subjects", response_model=SubjectListResponse)
async def list_subjects_admin(
    search: str | None = Query(None, description="Search in subject_id"),
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    health_state: str | None = Query(None, description="Filter by health state"),
    has_open_sessions: bool | None = Query(None, description="Filter by open sessions"),
    sort_by: Literal["subject_id", "last_activity", "episode_count", "memory_count"] = Query(
        "last_activity", description="Sort field"
    ),
    sort_order: Literal["asc", "desc"] = Query("desc", description="Sort order"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List subjects with search, filtering, and aggregated stats for admin explorer."""

    from sqlalchemy import func, select

    from server.db import engine as engine_module
    from server.db.tables import (
        EpisodeRow,
        MemoryRow,
        ResolutionRow,
        SubjectHealthCacheRow,
    )

    async with engine_module.get_session_factory()() as session:
        # Build base subqueries for aggregation
        # Episode stats per subject
        ep_stats = (
            select(
                EpisodeRow.subject_id,
                EpisodeRow.tenant_id,
                func.count().label("episode_count"),
                func.max(EpisodeRow.created_at).label("last_episode_at"),
            )
            .group_by(EpisodeRow.subject_id, EpisodeRow.tenant_id)
            .subquery()
        )

        # Memory stats per subject
        mem_stats = (
            select(
                MemoryRow.subject_id,
                MemoryRow.tenant_id,
                func.count().label("memory_count"),
            )
            .group_by(MemoryRow.subject_id, MemoryRow.tenant_id)
            .subquery()
        )

        # Session count per subject (distinct non-null session_ids from episodes)
        session_stats = (
            select(
                EpisodeRow.subject_id,
                EpisodeRow.tenant_id,
                func.count(func.distinct(EpisodeRow.session_id)).label("session_count"),
            )
            .where(EpisodeRow.session_id.isnot(None))
            .group_by(EpisodeRow.subject_id, EpisodeRow.tenant_id)
            .subquery()
        )

        # Open sessions per subject
        open_sessions = (
            select(
                ResolutionRow.subject_id,
                ResolutionRow.tenant_id,
                func.count().label("open_count"),
            )
            .where(ResolutionRow.status == "open")
            .group_by(ResolutionRow.subject_id, ResolutionRow.tenant_id)
            .subquery()
        )

        # Health cache
        health_cache = select(
            SubjectHealthCacheRow.subject_id,
            SubjectHealthCacheRow.tenant_id,
            SubjectHealthCacheRow.last_state,
            SubjectHealthCacheRow.last_score,
        ).subquery()

        def _same_subject_and_tenant(stats):
            return (ep_stats.c.subject_id == stats.c.subject_id) & (
                ep_stats.c.tenant_id.is_not_distinct_from(stats.c.tenant_id)
            )

        # Main query joining all
        stmt = select(
            ep_stats.c.subject_id,
            ep_stats.c.tenant_id,
            ep_stats.c.episode_count,
            func.coalesce(mem_stats.c.memory_count, 0).label("memory_count"),
            func.coalesce(session_stats.c.session_count, 0).label("session_count"),
            ep_stats.c.last_episode_at,
            health_cache.c.last_state.label("health_state"),
            health_cache.c.last_score.label("health_score"),
            func.coalesce(open_sessions.c.open_count, 0).label("open_sessions"),
        ).select_from(
            ep_stats.outerjoin(mem_stats, _same_subject_and_tenant(mem_stats))
            .outerjoin(session_stats, _same_subject_and_tenant(session_stats))
            .outerjoin(open_sessions, _same_subject_and_tenant(open_sessions))
            .outerjoin(health_cache, _same_subject_and_tenant(health_cache))
        )

        # Exclude internal subjects — escape the leading ``_`` so it is
        # matched literally, not as the single-character SQL wildcard.
        stmt = stmt.where(ep_stats.c.subject_id.not_like(r"\_snapshot/%", escape="\\"))
        stmt = stmt.where(ep_stats.c.subject_id.not_like(r"\_bootstrap_tmp/%", escape="\\"))

        # Apply filters
        if search:
            escaped = _like_escape(search)
            stmt = stmt.where(ep_stats.c.subject_id.ilike(f"%{escaped}%", escape="\\"))

        if tenant_id:
            stmt = stmt.where(ep_stats.c.tenant_id == tenant_id)

        if health_state:
            stmt = stmt.where(health_cache.c.last_state == health_state)

        if has_open_sessions is True:
            stmt = stmt.where(func.coalesce(open_sessions.c.open_count, 0) > 0)
        elif has_open_sessions is False:
            stmt = stmt.where(func.coalesce(open_sessions.c.open_count, 0) == 0)

        # Count total (before pagination)
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = await session.scalar(count_stmt) or 0

        # Apply sorting
        sort_column = {
            "subject_id": ep_stats.c.subject_id,
            "last_activity": ep_stats.c.last_episode_at,
            "episode_count": ep_stats.c.episode_count,
            "memory_count": func.coalesce(mem_stats.c.memory_count, 0),
        }.get(sort_by, ep_stats.c.last_episode_at)

        if sort_order == "desc":
            stmt = stmt.order_by(sort_column.desc().nulls_last())
        else:
            stmt = stmt.order_by(sort_column.asc().nulls_last())

        # Pagination
        stmt = stmt.limit(limit).offset(offset)

        result = await session.execute(stmt)
        rows = result.all()

        # Compute health for subjects that don't have cached values
        from server.db import repositories as repo
        from server.services.health import compute_health

        subjects = []
        for row in rows:
            health_state = row.health_state
            health_score = row.health_score

            # If no cached health, compute it now
            if health_state is None:
                try:
                    health_result = await compute_health(
                        session, row.subject_id, tenant_id=row.tenant_id
                    )
                    health_state = health_result.state
                    health_score = health_result.score
                    # Cache for future requests (best-effort, separate session)
                    try:
                        from server.db.engine import get_session_factory

                        async with get_session_factory()() as cache_session:
                            await repo.upsert_health_cache(
                                cache_session,
                                row.subject_id,
                                health_state,
                                health_score,
                                tenant_id=row.tenant_id,
                            )
                            await cache_session.commit()
                    except Exception:
                        pass
                except Exception:
                    pass

            subjects.append(
                SubjectListItem(
                    subject_id=row.subject_id,
                    tenant_id=row.tenant_id,
                    episode_count=row.episode_count,
                    memory_count=row.memory_count,
                    session_count=row.session_count,
                    last_episode_at=row.last_episode_at.isoformat()
                    if row.last_episode_at
                    else None,
                    health_state=health_state,
                    health_score=health_score,
                    open_sessions=row.open_sessions,
                )
            )

        return SubjectListResponse(
            subjects=subjects,
            total=total,
            limit=limit,
            offset=offset,
        )


@router.get("/subjects/{subject_id}", response_model=SubjectDetailResponse)
async def get_subject_detail(
    subject_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
):
    """Get detailed information about a specific subject for admin inspection."""
    from datetime import timedelta

    from sqlalchemy import func, select

    from server.db import engine as engine_module
    from server.db.tables import EpisodeRow, MemoryRow, ResolutionRow
    from server.services.health import compute_health
    from server.services.sla import compute_sla

    async with engine_module.get_session_factory()() as session:
        # Check subject exists
        ep_count_stmt = (
            select(func.count()).select_from(EpisodeRow).where(EpisodeRow.subject_id == subject_id)
        )
        if tenant_id:
            ep_count_stmt = ep_count_stmt.where(EpisodeRow.tenant_id == tenant_id)
        ep_count = await session.scalar(ep_count_stmt) or 0

        mem_count_stmt = (
            select(func.count()).select_from(MemoryRow).where(MemoryRow.subject_id == subject_id)
        )
        if tenant_id:
            mem_count_stmt = mem_count_stmt.where(MemoryRow.tenant_id == tenant_id)
        mem_count = await session.scalar(mem_count_stmt) or 0

        if ep_count == 0 and mem_count == 0:
            raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not found")

        # Get timestamps
        time_stmt = select(
            func.min(EpisodeRow.created_at).label("first_seen"),
            func.max(EpisodeRow.created_at).label("last_activity"),
        ).where(EpisodeRow.subject_id == subject_id)
        if tenant_id:
            time_stmt = time_stmt.where(EpisodeRow.tenant_id == tenant_id)
        time_result = await session.execute(time_stmt)
        time_row = time_result.one()

        # Session count — count distinct session_ids from both episodes and resolutions
        # First, from episodes (where session_id is not null)
        ep_session_stmt = select(func.count(func.distinct(EpisodeRow.session_id))).where(
            EpisodeRow.subject_id == subject_id,
            EpisodeRow.session_id.isnot(None),
        )
        if tenant_id:
            ep_session_stmt = ep_session_stmt.where(EpisodeRow.tenant_id == tenant_id)
        ep_session_count = await session.scalar(ep_session_stmt) or 0

        # Also from resolutions table
        res_session_stmt = select(func.count(func.distinct(ResolutionRow.session_id))).where(
            ResolutionRow.subject_id == subject_id
        )
        if tenant_id:
            res_session_stmt = res_session_stmt.where(ResolutionRow.tenant_id == tenant_id)
        res_session_count = await session.scalar(res_session_stmt) or 0

        # Use the higher of the two (they may overlap)
        session_count = max(ep_session_count, res_session_count)

        # Get tenant_id from the data if not specified
        actual_tenant_id = tenant_id
        if not actual_tenant_id:
            tenant_stmt = (
                select(EpisodeRow.tenant_id).where(EpisodeRow.subject_id == subject_id).limit(1)
            )
            actual_tenant_id = await session.scalar(tenant_stmt)

        summary = SubjectSummary(
            episode_count=ep_count,
            memory_count=mem_count,
            session_count=session_count,
            first_seen_at=time_row.first_seen.isoformat() if time_row.first_seen else None,
            last_activity_at=time_row.last_activity.isoformat() if time_row.last_activity else None,
        )

        # Health
        health_summary = None
        try:
            health_result = await compute_health(session, subject_id, tenant_id=tenant_id)
            health_summary = SubjectHealthSummary(
                score=health_result.score,
                state=health_result.state,
                factors=[
                    {"signal": f.signal, "impact": f.impact, "detail": f.detail}
                    for f in health_result.factors
                ],
            )
        except Exception:
            pass

        # Update health cache in background (separate session to avoid conflicts)
        if health_summary:
            try:
                from server.db.engine import get_session_factory
                from server.db import repositories as repo

                async with get_session_factory()() as cache_session:
                    await repo.upsert_health_cache(
                        cache_session,
                        subject_id,
                        health_summary.state,
                        health_summary.score,
                        tenant_id=tenant_id,
                    )
                    await cache_session.commit()
            except Exception:
                pass  # Cache update is best-effort

        # SLA
        sla_summary = None
        try:
            sla_result = await compute_sla(
                session,
                subject_id,
                tenant_id=tenant_id,
                first_response_threshold=timedelta(minutes=5),
                resolution_threshold=timedelta(hours=24),
            )
            sla_summary = SubjectSLASummary(
                total_sessions=sla_result.total_sessions,
                resolved_sessions=sla_result.resolved_sessions,
                open_sessions=sla_result.open_sessions,
                avg_first_response_seconds=sla_result.avg_first_response_seconds,
                avg_resolution_seconds=sla_result.avg_resolution_seconds,
                first_response_breach_count=sla_result.first_response_breach_count,
                resolution_breach_count=sla_result.resolution_breach_count,
            )
        except Exception:
            pass

        return SubjectDetailResponse(
            subject_id=subject_id,
            tenant_id=actual_tenant_id,
            summary=summary,
            health=health_summary,
            sla=sla_summary,
        )


@router.get("/subjects/{subject_id}/sla")
async def get_subject_sla(
    subject_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
):
    """Get SLA metrics and session list for a subject."""
    from datetime import timedelta

    from server.db import engine as engine_module
    from server.services.sla import compute_sla

    try:
        async with engine_module.get_session_factory()() as session:
            sla_result = await compute_sla(
                session,
                subject_id,
                tenant_id=tenant_id,
                first_response_threshold=timedelta(minutes=5),
                resolution_threshold=timedelta(hours=24),
            )
            return {
                "total_sessions": sla_result.total_sessions,
                "resolved_sessions": sla_result.resolved_sessions,
                "open_sessions": sla_result.open_sessions,
                "avg_first_response_seconds": sla_result.avg_first_response_seconds,
                "avg_resolution_seconds": sla_result.avg_resolution_seconds,
                "first_response_breach_count": sla_result.first_response_breach_count,
                "resolution_breach_count": sla_result.resolution_breach_count,
                "sessions": getattr(sla_result, "sessions", []),
            }
    except Exception:
        # Keep the response shape identical to the success branch — a typed
        # admin client reading avg_*/breach_count fields would otherwise get
        # missing keys on a transient failure while still seeing HTTP 200. Also
        # log it: silently swallowing makes a real DB error indistinguishable
        # from a genuinely empty subject on this operator-introspection endpoint.
        logger.warning("subject_sla_failed", subject_id=subject_id, exc_info=True)
        return {
            "total_sessions": 0,
            "resolved_sessions": 0,
            "open_sessions": 0,
            "avg_first_response_seconds": None,
            "avg_resolution_seconds": None,
            "first_response_breach_count": 0,
            "resolution_breach_count": 0,
            "sessions": [],
        }


@router.get("/subjects/{subject_id}/memories", response_model=MemoryListResponse)
async def list_subject_memories(
    subject_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    status: Literal["active", "superseded", "all"] = Query("all", description="Filter by status"),
    kind: str | None = Query(None, description="Filter by memory kind"),
    search: str | None = Query(None, description="Search in content, summary, and kind"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List memories for a subject with filtering, search, and pagination."""
    from sqlalchemy import func, or_, select

    from server.db import engine as engine_module
    from server.db.tables import MemoryRow

    async with engine_module.get_session_factory()() as session:
        base = select(MemoryRow).where(MemoryRow.subject_id == subject_id)
        if tenant_id:
            base = base.where(MemoryRow.tenant_id == tenant_id)
        if status != "all":
            base = base.where(MemoryRow.status == status)
        if kind:
            base = base.where(MemoryRow.kind == kind)
        if search:
            search_pattern = f"%{_like_escape(search)}%"
            base = base.where(
                or_(
                    MemoryRow.content.ilike(search_pattern, escape="\\"),
                    MemoryRow.summary.ilike(search_pattern, escape="\\"),
                    MemoryRow.kind.ilike(search_pattern, escape="\\"),
                )
            )

        # Count
        count_stmt = select(func.count()).select_from(base.subquery())
        total = await session.scalar(count_stmt) or 0

        # Get data
        stmt = base.order_by(MemoryRow.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(stmt)
        rows = result.scalars().all()

        memories = [
            MemoryListItem(
                id=str(m.id),
                kind=m.kind,
                content=m.content,
                summary=m.summary,
                confidence=m.confidence,
                status=m.status,
                source_episode_ids=[str(ep_id) for ep_id in (m.source_episode_ids or [])],
                valid_from=m.valid_from.isoformat(),
                valid_to=m.valid_to.isoformat() if m.valid_to else None,
                created_at=m.created_at.isoformat(),
                sensitivity_labels=list(m.sensitivity_labels or []),
                suggested_labels=list(m.suggested_labels or []),
            )
            for m in rows
        ]

        return MemoryListResponse(
            memories=memories,
            total=total,
            limit=limit,
            offset=offset,
        )


@router.get("/subjects/{subject_id}/episodes", response_model=EpisodeListResponse)
async def list_subject_episodes(
    subject_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    session_id: str | None = Query(None, description="Filter by session"),
    type: str | None = Query(None, description="Filter by episode type"),
    search: str | None = Query(None, description="Search in payload (JSON text), type, and source"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List episodes for a subject with filtering, search, and pagination."""
    from sqlalchemy import func, or_, select
    from sqlalchemy.dialects.postgresql import JSONB

    from server.db import engine as engine_module
    from server.db.tables import EpisodeRow

    async with engine_module.get_session_factory()() as session:
        base = select(EpisodeRow).where(EpisodeRow.subject_id == subject_id)
        if tenant_id:
            base = base.where(EpisodeRow.tenant_id == tenant_id)
        if session_id:
            base = base.where(EpisodeRow.session_id == session_id)
        if type:
            base = base.where(EpisodeRow.type == type)
        if search:
            search_pattern = f"%{_like_escape(search)}%"
            # Cast payload to text for searching
            base = base.where(
                or_(
                    EpisodeRow.payload.cast(JSONB).astext.ilike(search_pattern, escape="\\"),
                    EpisodeRow.type.ilike(search_pattern, escape="\\"),
                    EpisodeRow.source.ilike(search_pattern, escape="\\"),
                    EpisodeRow.session_id.ilike(search_pattern, escape="\\"),
                )
            )

        # Count
        count_stmt = select(func.count()).select_from(base.subquery())
        total = await session.scalar(count_stmt) or 0

        # Get data
        stmt = base.order_by(EpisodeRow.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(stmt)
        rows = result.scalars().all()

        episodes = [
            EpisodeListItem(
                id=str(e.id),
                session_id=e.session_id,
                source=e.source,
                type=e.type,
                payload=e.payload,
                metadata=e.metadata_,
                provenance=e.provenance,
                created_at=e.created_at.isoformat(),
            )
            for e in rows
        ]

        return EpisodeListResponse(
            episodes=episodes,
            total=total,
            limit=limit,
            offset=offset,
        )


@router.get(
    "/subjects/{subject_id}/episodes/{episode_id}/citing-memories",
    response_model=MemoryListResponse,
)
async def list_citing_memories(
    subject_id: str,
    episode_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List memories that cite (were derived from) a specific episode.

    This enables reverse provenance lookup: from an episode, find all
    memories that list it in their source_episode_ids.
    """
    import uuid as uuid_module

    from sqlalchemy import any_, func, select

    from server.db import engine as engine_module
    from server.db.tables import MemoryRow

    # Validate episode_id is a valid UUID
    try:
        episode_uuid = uuid_module.UUID(episode_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid episode_id format")

    async with engine_module.get_session_factory()() as session:
        # Find memories where episode_id is in source_episode_ids array
        base = select(MemoryRow).where(
            MemoryRow.subject_id == subject_id,
            episode_uuid == any_(MemoryRow.source_episode_ids),
        )
        if tenant_id:
            base = base.where(MemoryRow.tenant_id == tenant_id)

        # Count
        count_stmt = select(func.count()).select_from(base.subquery())
        total = await session.scalar(count_stmt) or 0

        # Get data ordered by created_at desc (newest first)
        stmt = base.order_by(MemoryRow.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(stmt)
        rows = result.scalars().all()

        memories = [
            MemoryListItem(
                id=str(m.id),
                kind=m.kind,
                content=m.content,
                summary=m.summary,
                confidence=m.confidence,
                status=m.status,
                source_episode_ids=[str(eid) for eid in (m.source_episode_ids or [])],
                valid_from=m.valid_from.isoformat(),
                valid_to=m.valid_to.isoformat() if m.valid_to else None,
                created_at=m.created_at.isoformat(),
                sensitivity_labels=list(m.sensitivity_labels or []),
                suggested_labels=list(m.suggested_labels or []),
            )
            for m in rows
        ]

        return MemoryListResponse(
            memories=memories,
            total=total,
            limit=limit,
            offset=offset,
        )


# ─── Retrieval simulator / activity / provenance ─────────────────────────────


class RetrievalSimulateItem(BaseModel):
    rank: int
    memory_id: str
    kind: str
    content: str
    summary: str
    confidence: float
    status: str
    created_at: str
    similarity: float
    cosine_distance: float
    estimated_tokens: int
    within_budget: bool


class RetrievalSimulateResponse(BaseModel):
    results: list[RetrievalSimulateItem]
    query: str
    tokens_used: int
    token_budget: int
    embedding_available: bool
    error: str | None


class ActivityDay(BaseModel):
    date: str
    episode_count: int
    memory_count: int


class ActivityResponse(BaseModel):
    days: list[ActivityDay]
    subject_id: str
    window_days: int


class ProvenanceEpisode(BaseModel):
    id: str
    source: str
    type: str
    payload: dict
    created_at: str


class ProvenanceMemory(BaseModel):
    id: str
    kind: str
    content: str
    summary: str
    confidence: float
    status: str
    created_at: str
    source_episode_ids: list[str]


class ProvenanceResponse(BaseModel):
    memory: ProvenanceMemory
    source_episodes: list[ProvenanceEpisode]
    sibling_memories: list[ProvenanceMemory]


@router.get(
    "/subjects/{subject_id}/retrieval-simulate",
    response_model=RetrievalSimulateResponse,
)
async def retrieval_simulate(
    subject_id: str,
    query: str = Query(..., min_length=1, description="Free-text query to simulate recall for"),
    limit: int = Query(default=15, ge=1, le=50),
    token_budget: int = Query(default=2000, ge=100, le=32000),
    tenant_id: str | None = Query(None),
):
    """Simulate retrieval for a query against a subject's memory.

    Embeds the query, scores every active memory by cosine similarity, and
    returns the ranked list with scores + a within_budget flag so the
    operator can see exactly which memories would be recalled — and which
    would be cut — for a given token budget.

    Returns embedding_available=False (and an empty results list) when the
    configured embedding_provider is the non-semantic stub, or when
    embeddings have not been generated for this subject yet.
    """
    from server.core.config import settings as cfg
    from server.db import engine as engine_module
    from server.db.repositories import search_memories_by_embedding
    from server.db.tables import MemoryRow
    from server.services import llm as llm_svc
    from sqlalchemy import select

    if cfg.embedding_provider == "stub":
        return RetrievalSimulateResponse(
            results=[],
            query=query,
            tokens_used=0,
            token_budget=token_budget,
            embedding_available=False,
            error=(
                "Retrieval simulation requires a real embedding provider. "
                "The stub provider returns random vectors and cannot rank "
                "memories by semantic similarity. Set "
                "STATEWAVE_EMBEDDING_PROVIDER=litellm and configure "
                "STATEWAVE_LITELLM_EMBEDDING_MODEL."
            ),
        )

    try:
        query_embedding = await llm_svc.aembed_query(query)
    except Exception as exc:  # noqa: BLE001
        return RetrievalSimulateResponse(
            results=[],
            query=query,
            tokens_used=0,
            token_budget=token_budget,
            embedding_available=False,
            error=f"Embedding failed: {exc}",
        )

    async with engine_module.get_session_factory()() as session:
        # Check whether any embeddings exist for this subject at all.
        has_emb = await session.scalar(
            select(MemoryRow.id)
            .where(MemoryRow.subject_id == subject_id)
            .where(MemoryRow.embedding.isnot(None))
            .limit(1)
        )
        if has_emb is None:
            return RetrievalSimulateResponse(
                results=[],
                query=query,
                tokens_used=0,
                token_budget=token_budget,
                embedding_available=False,
                error=(
                    "No embeddings found for this subject. Memories receive "
                    "embeddings during compilation. Run a compile job first."
                ),
            )

        ranked = await search_memories_by_embedding(
            session,
            subject_id,
            query_embedding,
            tenant_id=tenant_id,
            limit=limit,
        )

    items: list[RetrievalSimulateItem] = []
    tokens_used = 0
    for rank, (mem, distance) in enumerate(ranked, start=1):
        similarity = round(1.0 - distance / 2.0, 4)
        # ~4 chars per token is the standard LLM approximation
        est_tokens = max(1, len(mem.content) // 4)
        within_budget = (tokens_used + est_tokens) <= token_budget
        if within_budget:
            tokens_used += est_tokens
        items.append(
            RetrievalSimulateItem(
                rank=rank,
                memory_id=str(mem.id),
                kind=mem.kind,
                content=mem.content,
                summary=mem.summary,
                confidence=mem.confidence,
                status=mem.status,
                created_at=mem.created_at.isoformat(),
                similarity=similarity,
                cosine_distance=round(distance, 4),
                estimated_tokens=est_tokens,
                within_budget=within_budget,
            )
        )

    return RetrievalSimulateResponse(
        results=items,
        query=query,
        tokens_used=tokens_used,
        token_budget=token_budget,
        embedding_available=True,
        error=None,
    )


@router.get(
    "/subjects/{subject_id}/activity",
    response_model=ActivityResponse,
)
async def subject_activity(
    subject_id: str,
    days: int = Query(default=90, ge=7, le=365),
    tenant_id: str | None = Query(None),
):
    """Return episode + memory counts grouped by day for the past N days.

    The response covers every day in the window (zero-filled so the
    frontend can render a continuous calendar without filling gaps itself).
    Days are returned oldest-first so a chart can consume them directly.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import cast, func, select, text
    from sqlalchemy.types import Date

    from server.db import engine as engine_module
    from server.db.tables import EpisodeRow, MemoryRow

    since = datetime.now(tz=timezone.utc) - timedelta(days=days)

    async with engine_module.get_session_factory()() as session:
        # Episode counts per day
        ep_stmt = (
            select(
                cast(EpisodeRow.created_at, Date).label("day"),
                func.count().label("cnt"),
            )
            .where(EpisodeRow.subject_id == subject_id)
            .where(EpisodeRow.created_at >= since)
        )
        if tenant_id:
            ep_stmt = ep_stmt.where(EpisodeRow.tenant_id == tenant_id)
        ep_stmt = ep_stmt.group_by(text("day"))
        ep_rows = (await session.execute(ep_stmt)).all()

        # Memory counts per day
        mem_stmt = (
            select(
                cast(MemoryRow.created_at, Date).label("day"),
                func.count().label("cnt"),
            )
            .where(MemoryRow.subject_id == subject_id)
            .where(MemoryRow.created_at >= since)
        )
        if tenant_id:
            mem_stmt = mem_stmt.where(MemoryRow.tenant_id == tenant_id)
        mem_stmt = mem_stmt.group_by(text("day"))
        mem_rows = (await session.execute(mem_stmt)).all()

    ep_by_day: dict[str, int] = {str(r.day): r.cnt for r in ep_rows}
    mem_by_day: dict[str, int] = {str(r.day): r.cnt for r in mem_rows}

    # Zero-fill the full window so the calendar is continuous
    from datetime import date as date_type

    start = since.date()
    today = datetime.now(tz=timezone.utc).date()
    result_days: list[ActivityDay] = []
    cur = start
    while cur <= today:
        ds = str(cur)
        result_days.append(
            ActivityDay(
                date=ds,
                episode_count=ep_by_day.get(ds, 0),
                memory_count=mem_by_day.get(ds, 0),
            )
        )
        cur += timedelta(days=1)

    return ActivityResponse(
        days=result_days,
        subject_id=subject_id,
        window_days=days,
    )


@router.get(
    "/subjects/{subject_id}/memories/{memory_id}/provenance",
    response_model=ProvenanceResponse,
)
async def memory_provenance(
    subject_id: str,
    memory_id: str,
    tenant_id: str | None = Query(None),
):
    """Return full provenance for a single memory.

    Response contains:
    - memory: the memory itself
    - source_episodes: the episodes it was compiled from
    - sibling_memories: other memories that share ≥1 source episode
      (i.e. they were compiled from the same raw input, which helps
       spot duplication or explain why a memory was split/merged)
    """
    import uuid as uuid_module

    from sqlalchemy import any_, select

    from server.db import engine as engine_module
    from server.db.tables import EpisodeRow, MemoryRow

    try:
        mem_uuid = uuid_module.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory_id format")

    async with engine_module.get_session_factory()() as session:
        mem_result = await session.execute(
            select(MemoryRow).where(
                MemoryRow.id == mem_uuid,
                MemoryRow.subject_id == subject_id,
            )
        )
        mem = mem_result.scalar_one_or_none()
        if mem is None:
            raise HTTPException(status_code=404, detail="Memory not found")

        source_ep_ids: list[uuid_module.UUID] = list(mem.source_episode_ids or [])

        # Fetch the source episodes
        episodes: list[EpisodeRow] = []
        if source_ep_ids:
            ep_result = await session.execute(
                select(EpisodeRow).where(EpisodeRow.id.in_(source_ep_ids))
            )
            episodes = list(ep_result.scalars().all())

        # Sibling memories: share ≥1 source episode, are not this memory
        sibling_rows: list[MemoryRow] = []
        if source_ep_ids:
            for ep_id in source_ep_ids:
                sib_result = await session.execute(
                    select(MemoryRow).where(
                        MemoryRow.subject_id == subject_id,
                        MemoryRow.id != mem_uuid,
                        ep_id == any_(MemoryRow.source_episode_ids),
                    )
                )
                sibling_rows.extend(sib_result.scalars().all())
            # Deduplicate by id (a sibling may share multiple episodes)
            seen: set[uuid_module.UUID] = set()
            unique_siblings: list[MemoryRow] = []
            for s in sibling_rows:
                if s.id not in seen:
                    seen.add(s.id)
                    unique_siblings.append(s)
            sibling_rows = unique_siblings

    def _prov_memory(m: MemoryRow) -> ProvenanceMemory:
        return ProvenanceMemory(
            id=str(m.id),
            kind=m.kind,
            content=m.content,
            summary=m.summary,
            confidence=m.confidence,
            status=m.status,
            created_at=m.created_at.isoformat(),
            source_episode_ids=[str(eid) for eid in (m.source_episode_ids or [])],
        )

    return ProvenanceResponse(
        memory=_prov_memory(mem),
        source_episodes=[
            ProvenanceEpisode(
                id=str(ep.id),
                source=ep.source,
                type=ep.type,
                payload=ep.payload,
                created_at=ep.created_at.isoformat(),
            )
            for ep in episodes
        ],
        sibling_memories=[_prov_memory(s) for s in sibling_rows],
    )


# ─── Compiler Trace Inspector (feature #4) ───────────────────────────────────


class CompilerTraceEpisode(BaseModel):
    id: str
    source: str
    type: str
    payload: dict
    created_at: str
    text_preview: str


class CompilerTraceResponse(BaseModel):
    memory_id: str
    kind: str
    content: str
    summary: str
    confidence: float
    status: str
    created_at: str
    compiler: str
    model: str | None
    source_episode_count: int
    reconstructed_input: list[CompilerTraceEpisode]


@router.get(
    "/subjects/{subject_id}/memories/{memory_id}/compiler-trace",
    response_model=CompilerTraceResponse,
)
async def memory_compiler_trace(
    subject_id: str,
    memory_id: str,
    tenant_id: str | None = Query(None),
):
    """Reconstruct the compiler trace for a memory.

    The LLM compiler does not persist raw prompts/responses.  This
    endpoint reconstructs the trace from what IS stored: the source
    episodes that were fed in and the compiler/model metadata on the row.
    """
    import uuid as _uuid

    from server.db import engine as engine_module
    from server.db.repositories import get_episodes_by_ids
    from server.db.tables import MemoryRow
    from sqlalchemy import select

    async with engine_module.get_session_factory()() as session:
        try:
            mem_uuid = _uuid.UUID(memory_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid memory_id format")
        row = await session.scalar(
            select(MemoryRow)
            .where(MemoryRow.id == mem_uuid)
            .where(MemoryRow.subject_id == subject_id)
        )
        if row is None:
            raise HTTPException(status_code=404, detail="memory not found")

        episodes = []
        if row.source_episode_ids:
            episodes = list(await get_episodes_by_ids(session, row.source_episode_ids))

    meta = row.metadata_ or {}
    compiler = meta.get("compiler", "unknown")
    model = meta.get("model")

    def _ep_text(ep) -> str:
        pay = ep.payload or {}
        if isinstance(pay.get("text"), str):
            return pay["text"][:500]
        if isinstance(pay.get("content"), str):
            return pay["content"][:500]
        return str(pay)[:500]

    return CompilerTraceResponse(
        memory_id=str(row.id),
        kind=row.kind,
        content=row.content,
        summary=row.summary,
        confidence=row.confidence,
        status=row.status,
        created_at=row.created_at.isoformat(),
        compiler=compiler,
        model=model,
        source_episode_count=len(row.source_episode_ids or []),
        reconstructed_input=[
            CompilerTraceEpisode(
                id=str(ep.id),
                source=ep.source,
                type=ep.type,
                payload=ep.payload,
                created_at=ep.created_at.isoformat(),
                text_preview=_ep_text(ep),
            )
            for ep in episodes
        ],
    )


# ─── Memory Conflict Detector (feature #5) ───────────────────────────────────


class ConflictPair(BaseModel):
    memory_a_id: str
    memory_a_kind: str
    memory_a_content: str
    memory_b_id: str
    memory_b_kind: str
    memory_b_content: str
    similarity: float


class ConflictsResponse(BaseModel):
    pairs: list[ConflictPair]
    total_memories_checked: int
    embedding_available: bool
    error: str | None


@router.get(
    "/subjects/{subject_id}/conflicts",
    response_model=ConflictsResponse,
)
async def detect_memory_conflicts(
    subject_id: str,
    threshold: float = Query(default=0.85, ge=0.5, le=1.0),
    limit: int = Query(default=20, ge=1, le=100),
    tenant_id: str | None = Query(None),
):
    """Find memories that may conflict with or duplicate each other.

    Computes pairwise cosine similarity across active memories that have
    embeddings.  Pairs at or above `threshold` are returned ranked by
    similarity — candidates for deduplication or contradiction review.
    Capped at 150 memories for performance.
    """
    from server.core.config import settings as cfg
    from server.db import engine as engine_module
    from server.db.tables import MemoryRow
    from sqlalchemy import select

    if cfg.embedding_provider == "stub":
        return ConflictsResponse(
            pairs=[],
            total_memories_checked=0,
            embedding_available=False,
            error=(
                "Conflict detection requires real embeddings. "
                "Stub provider returns hash-based vectors."
            ),
        )

    async with engine_module.get_session_factory()() as session:
        stmt = (
            select(MemoryRow)
            .where(MemoryRow.subject_id == subject_id)
            .where(MemoryRow.status == "active")
            .where(MemoryRow.embedding.isnot(None))
            .order_by(MemoryRow.created_at.desc())
            .limit(150)
        )
        if tenant_id:
            stmt = stmt.where(MemoryRow.tenant_id == tenant_id)
        rows = (await session.execute(stmt)).scalars().all()

    if not rows:
        return ConflictsResponse(
            pairs=[],
            total_memories_checked=0,
            embedding_available=True,
            error=None,
        )

    import numpy as np

    ids = [str(r.id) for r in rows]
    kinds = [r.kind for r in rows]
    contents = [r.content for r in rows]
    raw_vecs = [r.embedding for r in rows]
    mat = np.array(
        [v if isinstance(v, list) else list(v) for v in raw_vecs], dtype=np.float32
    )
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    mat = mat / norms

    pairs: list[ConflictPair] = []
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(mat[i], mat[j]))
            if sim >= threshold:
                pairs.append(
                    ConflictPair(
                        memory_a_id=ids[i],
                        memory_a_kind=kinds[i],
                        memory_a_content=contents[i],
                        memory_b_id=ids[j],
                        memory_b_kind=kinds[j],
                        memory_b_content=contents[j],
                        similarity=round(sim, 4),
                    )
                )

    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return ConflictsResponse(
        pairs=pairs[:limit],
        total_memories_checked=len(rows),
        embedding_available=True,
        error=None,
    )


# ─── Memory Timeline Scrubber (feature #6) ───────────────────────────────────


class TimelineEvent(BaseModel):
    date: str
    memories_added: int
    cumulative_count: int


class TimelineMemory(BaseModel):
    id: str
    kind: str
    content_preview: str
    confidence: float
    status: str
    created_at: str


class MemoryTimelineResponse(BaseModel):
    events: list[TimelineEvent]
    snapshot_at: str | None
    memories_at_snapshot: list[TimelineMemory]
    subject_id: str


@router.get(
    "/subjects/{subject_id}/memory-timeline",
    response_model=MemoryTimelineResponse,
)
async def memory_timeline(
    subject_id: str,
    snapshot_at: str | None = Query(
        None, description="ISO-8601 datetime — omit for current state"
    ),
    tenant_id: str | None = Query(None),
):
    """Return a compile-event timeline for the memory scrubber.

    `events` gives one entry per calendar day when memories were created
    with a running cumulative count — use this to render the scrubber axis.

    `memories_at_snapshot` lists all non-tombstoned memories that existed
    at `snapshot_at` (defaults to now).
    """
    from datetime import datetime, timezone

    from sqlalchemy import cast, func, select, text
    from sqlalchemy.types import Date

    from server.db import engine as engine_module
    from server.db.tables import MemoryRow

    snap_dt: datetime | None = None
    if snapshot_at:
        try:
            snap_dt = datetime.fromisoformat(snapshot_at.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid snapshot_at — use ISO-8601."
            )

    async with engine_module.get_session_factory()() as session:
        day_stmt = (
            select(
                cast(MemoryRow.created_at, Date).label("day"),
                func.count().label("cnt"),
            )
            .where(MemoryRow.subject_id == subject_id)
        )
        if tenant_id:
            day_stmt = day_stmt.where(MemoryRow.tenant_id == tenant_id)
        day_stmt = day_stmt.group_by(text("day")).order_by(text("day"))
        day_rows = (await session.execute(day_stmt)).all()

        snap_stmt = (
            select(MemoryRow)
            .where(MemoryRow.subject_id == subject_id)
            .where(MemoryRow.status != "tombstoned")
            .order_by(MemoryRow.created_at.desc())
        )
        if tenant_id:
            snap_stmt = snap_stmt.where(MemoryRow.tenant_id == tenant_id)
        if snap_dt:
            snap_stmt = snap_stmt.where(MemoryRow.created_at <= snap_dt)
        snap_rows = (await session.execute(snap_stmt)).scalars().all()

    cumulative = 0
    events: list[TimelineEvent] = []
    for r in day_rows:
        cumulative += r.cnt
        events.append(
            TimelineEvent(
                date=str(r.day), memories_added=r.cnt, cumulative_count=cumulative
            )
        )

    return MemoryTimelineResponse(
        events=events,
        snapshot_at=snap_dt.isoformat() if snap_dt else None,
        memories_at_snapshot=[
            TimelineMemory(
                id=str(m.id),
                kind=m.kind,
                content_preview=m.content[:200],
                confidence=m.confidence,
                status=m.status,
                created_at=m.created_at.isoformat(),
            )
            for m in snap_rows
        ],
        subject_id=subject_id,
    )


# ─── Policy Sandbox (feature #7) ─────────────────────────────────────────────


class PolicySandboxRequest(BaseModel):
    yaml_content: str
    caller_id: str | None = None
    caller_type: str | None = None


class PolicySandboxResult(BaseModel):
    memory_id: str
    kind: str
    content_preview: str
    sensitivity_labels: list[str]
    action: str
    rule_id: str | None
    matched_labels: list[str]


class PolicySandboxResponse(BaseModel):
    results: list[PolicySandboxResult]
    total_memories: int
    allowed: int
    denied: int
    redacted: int
    error: str | None


@router.post(
    "/subjects/{subject_id}/policy-sandbox",
    response_model=PolicySandboxResponse,
)
async def policy_sandbox(
    subject_id: str,
    req: PolicySandboxRequest,
    tenant_id: str | None = Query(None),
):
    """Dry-run a YAML policy bundle against a subject's active memories.

    Returns each memory with its policy decision (allow/deny/redact) so
    operators can tune policy rules before applying them live.  The
    subject's live policy is never modified.
    """
    from server.db import engine as engine_module
    from server.db.repositories import list_active_memories_by_subject
    from server.services.policy import PolicyContext, PolicyError, evaluate_memory, load_bundle

    try:
        bundle = load_bundle(req.yaml_content)
    except PolicyError as exc:
        return PolicySandboxResponse(
            results=[],
            total_memories=0,
            allowed=0,
            denied=0,
            redacted=0,
            error=f"Policy YAML is invalid: {exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return PolicySandboxResponse(
            results=[],
            total_memories=0,
            allowed=0,
            denied=0,
            redacted=0,
            error=f"Failed to parse policy: {exc}",
        )

    async with engine_module.get_session_factory()() as session:
        rows = list(
            await list_active_memories_by_subject(session, subject_id, tenant_id=tenant_id)
        )

    context = PolicyContext(
        caller_id=req.caller_id,
        caller_type=req.caller_type,
        tenant_id=tenant_id,
    )

    results: list[PolicySandboxResult] = []
    allowed = denied = redacted = 0
    for mem in rows:
        decision = evaluate_memory(
            memory_labels=mem.sensitivity_labels or [],
            bundle=bundle,
            context=context,
        )
        if decision.action == "allow":
            allowed += 1
        elif decision.action == "deny":
            denied += 1
        else:
            redacted += 1
        results.append(
            PolicySandboxResult(
                memory_id=str(mem.id),
                kind=mem.kind,
                content_preview=mem.content[:200],
                sensitivity_labels=mem.sensitivity_labels or [],
                action=decision.action,
                rule_id=decision.rule_id,
                matched_labels=list(decision.matched_labels),
            )
        )

    return PolicySandboxResponse(
        results=results,
        total_memories=len(rows),
        allowed=allowed,
        denied=denied,
        redacted=redacted,
        error=None,
    )


# ─── Memory Cluster View (feature #8) ────────────────────────────────────────


class ClusterPoint(BaseModel):
    memory_id: str
    kind: str
    content_preview: str
    confidence: float
    status: str
    x: float
    y: float


class MemoryClustersResponse(BaseModel):
    points: list[ClusterPoint]
    total_memories: int
    embedding_available: bool
    error: str | None


@router.get(
    "/subjects/{subject_id}/memory-clusters",
    response_model=MemoryClustersResponse,
)
async def memory_clusters(
    subject_id: str,
    tenant_id: str | None = Query(None),
):
    """Project all memory embeddings to 2D via PCA for cluster visualisation.

    Returns (x, y) coordinates for each memory so the frontend can render
    a scatter plot showing how memories are distributed and whether natural
    clusters or outliers exist.
    """
    from server.core.config import settings as cfg
    from server.db import engine as engine_module
    from server.db.tables import MemoryRow
    from sqlalchemy import select

    if cfg.embedding_provider == "stub":
        return MemoryClustersResponse(
            points=[],
            total_memories=0,
            embedding_available=False,
            error=(
                "Cluster view requires real embeddings. "
                "Stub provider vectors are not semantic."
            ),
        )

    async with engine_module.get_session_factory()() as session:
        stmt = (
            select(MemoryRow)
            .where(MemoryRow.subject_id == subject_id)
            .where(MemoryRow.embedding.isnot(None))
            .order_by(MemoryRow.created_at.desc())
            .limit(500)
        )
        if tenant_id:
            stmt = stmt.where(MemoryRow.tenant_id == tenant_id)
        rows = (await session.execute(stmt)).scalars().all()

    if not rows:
        return MemoryClustersResponse(
            points=[],
            total_memories=0,
            embedding_available=True,
            error=None,
        )

    import numpy as np

    raw_vecs = [r.embedding for r in rows]
    mat = np.array(
        [v if isinstance(v, list) else list(v) for v in raw_vecs], dtype=np.float32
    )
    mat -= mat.mean(axis=0)
    if mat.shape[0] >= 2 and mat.shape[1] >= 2:
        _, _, Vt = np.linalg.svd(mat, full_matrices=False)
        coords = mat @ Vt[:2].T
    else:
        coords = np.zeros((len(rows), 2), dtype=np.float32)

    for dim in range(2):
        col = coords[:, dim]
        span = float(col.max() - col.min())
        if span > 0:
            coords[:, dim] = (col - col.min()) / span * 2 - 1

    return MemoryClustersResponse(
        points=[
            ClusterPoint(
                memory_id=str(r.id),
                kind=r.kind,
                content_preview=r.content[:150],
                confidence=r.confidence,
                status=r.status,
                x=round(float(coords[i, 0]), 4),
                y=round(float(coords[i, 1]), 4),
            )
            for i, r in enumerate(rows)
        ],
        total_memories=len(rows),
        embedding_available=True,
        error=None,
    )


# ─── Receipts + Regression Tester (feature #9) ───────────────────────────────


class AdminReceiptListItem(BaseModel):
    receipt_id: str
    as_of: str
    created_at: str
    mode: str
    context_size_bytes: int
    memory_count: int


class AdminReceiptListResponse(BaseModel):
    items: list[AdminReceiptListItem]
    total: int


class RegressionMemory(BaseModel):
    memory_id: str
    kind: str
    content_preview: str
    status: str
    created_at: str
    change: str


class RegressionResponse(BaseModel):
    receipt_id: str
    receipt_as_of: str
    stable: list[RegressionMemory]
    dropped: list[RegressionMemory]
    new_memories: list[RegressionMemory]


def _is_valid_uuid(s: str) -> bool:
    import uuid as _uuid

    try:
        _uuid.UUID(s)
        return True
    except ValueError:
        return False


@router.get(
    "/subjects/{subject_id}/admin-receipts",
    response_model=AdminReceiptListResponse,
)
async def list_subject_receipts_admin(
    subject_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    tenant_id: str | None = Query(None),
):
    """List state-assembly receipts for a subject (newest first).

    Used by the Receipts tab to populate the receipt picker for the
    regression tester.
    """
    from server.db import engine as engine_module
    from server.db.repositories import list_receipts

    async with engine_module.get_session_factory()() as session:
        rows = list(
            await list_receipts(session, subject_id, tenant_id=tenant_id, limit=limit)
        )

    items: list[AdminReceiptListItem] = []
    for r in rows:
        body = r.body if isinstance(r.body, dict) else {}
        selected = body.get("selected_entries", [])
        mem_count = sum(1 for e in selected if e.get("type") == "memory")
        items.append(
            AdminReceiptListItem(
                receipt_id=r.receipt_id,
                as_of=r.as_of.isoformat(),
                created_at=r.created_at.isoformat(),
                mode=r.mode or "standard",
                context_size_bytes=r.context_size_bytes or 0,
                memory_count=mem_count,
            )
        )

    return AdminReceiptListResponse(items=items, total=len(items))


@router.get(
    "/subjects/{subject_id}/admin-receipts/{receipt_id}/regression",
    response_model=RegressionResponse,
)
async def receipt_regression(
    subject_id: str,
    receipt_id: str,
    tenant_id: str | None = Query(None),
):
    """Diff the memory set from a historical receipt against current state.

    Returns which memories are still active (stable), which have since been
    tombstoned/superseded (dropped), and which new memories appeared after
    the receipt's as_of timestamp.
    """
    import uuid as _uuid

    from sqlalchemy import select

    from server.db import engine as engine_module
    from server.db.repositories import get_receipt_by_id
    from server.db.tables import MemoryRow

    async with engine_module.get_session_factory()() as session:
        receipt = await get_receipt_by_id(session, receipt_id, tenant_id=tenant_id)
        if receipt is None or receipt.subject_id != subject_id:
            raise HTTPException(status_code=404, detail="receipt not found")

        body = receipt.body if isinstance(receipt.body, dict) else {}
        selected = body.get("selected_entries", [])
        receipt_mem_ids: list[str] = [
            e["memory_id"]
            for e in selected
            if e.get("type") == "memory" and "memory_id" in e
        ]

        receipt_rows: list[MemoryRow] = []
        if receipt_mem_ids:
            uuids = [_uuid.UUID(mid) for mid in receipt_mem_ids if _is_valid_uuid(mid)]
            if uuids:
                stmt = select(MemoryRow).where(MemoryRow.id.in_(uuids))
                receipt_rows = list((await session.execute(stmt)).scalars().all())

        new_stmt = (
            select(MemoryRow)
            .where(MemoryRow.subject_id == subject_id)
            .where(MemoryRow.created_at > receipt.as_of)
            .where(MemoryRow.status == "active")
            .order_by(MemoryRow.created_at.asc())
            .limit(100)
        )
        if tenant_id:
            new_stmt = new_stmt.where(MemoryRow.tenant_id == tenant_id)
        receipt_uuid_set = {
            _uuid.UUID(mid) for mid in receipt_mem_ids if _is_valid_uuid(mid)
        }
        new_rows = [
            r
            for r in (await session.execute(new_stmt)).scalars().all()
            if r.id not in receipt_uuid_set
        ]

    def _regmem(r: MemoryRow, change: str) -> RegressionMemory:
        return RegressionMemory(
            memory_id=str(r.id),
            kind=r.kind,
            content_preview=r.content[:200],
            status=r.status,
            created_at=r.created_at.isoformat(),
            change=change,
        )

    stable = [_regmem(r, "stable") for r in receipt_rows if r.status == "active"]
    dropped = [
        _regmem(r, "tombstoned" if r.status == "tombstoned" else "superseded")
        for r in receipt_rows
        if r.status != "active"
    ]
    found_ids = {r.id for r in receipt_rows}
    for mid in receipt_mem_ids:
        if not _is_valid_uuid(mid):
            continue
        if _uuid.UUID(mid) not in found_ids:
            dropped.append(
                RegressionMemory(
                    memory_id=mid,
                    kind="unknown",
                    content_preview="(memory no longer exists in DB)",
                    status="deleted",
                    created_at="",
                    change="deleted",
                )
            )

    return RegressionResponse(
        receipt_id=receipt_id,
        receipt_as_of=receipt.as_of.isoformat(),
        stable=stable,
        dropped=dropped,
        new_memories=[_regmem(r, "new") for r in new_rows],
    )


# ─── Suggested-label review (auto-labeling, v0.9 #158) ───────────────────────


@router.get(
    "/memories/with-suggested-labels",
    response_model=SuggestedLabelsListResponse,
)
async def list_memories_with_suggested_labels(
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    subject_id: str | None = Query(None, description="Filter by subject"),
    label: str | None = Query(
        None,
        description=(
            "Filter to memories carrying this specific suggested label "
            "(e.g. `pii.email`). When omitted, lists every memory with at "
            "least one suggestion."
        ),
    ),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List memories with at least one auto-derived suggested label.

    Read-only review surface for the v0.9 auto-labeling pipeline (#158).
    The endpoint exists so operators can audit detector output and
    decide which suggestions to promote into authoritative
    ``sensitivity_labels``. v0.9 ships review only — promotion lands
    in a follow-up PR; the SDK / direct DB write path is the
    interim escape hatch.

    The endpoint does not require the feature flag to be enabled: an
    operator can flip the flag off, leave existing suggestions in
    place, and still see them here to triage.
    """
    from sqlalchemy import func, select

    from server.db import engine as engine_module
    from server.db.tables import MemoryRow
    from server.services.auto_labeling.detectors import label_catalogue

    async with engine_module.get_session_factory()() as session:
        # `array_length(col, 1) IS NOT NULL` is the cheap pg-native way to
        # filter to "non-empty array". The GIN index from migration 0022
        # makes the `&&`-overlap path that follows a millisecond hop.
        base = select(MemoryRow).where(func.array_length(MemoryRow.suggested_labels, 1).isnot(None))
        if tenant_id:
            base = base.where(MemoryRow.tenant_id == tenant_id)
        if subject_id:
            base = base.where(MemoryRow.subject_id == subject_id)
        if label:
            # Use the GIN-indexed overlap operator so a label filter
            # stays cheap even on millions of memories. We pass a
            # one-element list because `overlap` accepts an array.
            base = base.where(MemoryRow.suggested_labels.overlap([label]))

        count_stmt = select(func.count()).select_from(base.subquery())
        total = await session.scalar(count_stmt) or 0

        stmt = base.order_by(MemoryRow.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(stmt)
        rows = result.scalars().all()

        memories = [
            SuggestedLabelMemoryItem(
                id=str(m.id),
                subject_id=m.subject_id,
                tenant_id=m.tenant_id,
                kind=m.kind,
                content=m.content,
                summary=m.summary,
                suggested_labels=list(m.suggested_labels or []),
                sensitivity_labels=list(m.sensitivity_labels or []),
                created_at=m.created_at.isoformat(),
            )
            for m in rows
        ]

        return SuggestedLabelsListResponse(
            memories=memories,
            total=total,
            limit=limit,
            offset=offset,
            catalogue=label_catalogue(),
        )


# ─── Promote suggested labels → authoritative sensitivity_labels (v0.9 #160) ─


class PromoteLabelsRequest(BaseModel):
    labels: list[str]
    """Subset of the memory's current ``suggested_labels`` to promote.
    Every label in this list MUST already be present on the memory —
    promotion is strictly review-driven; the endpoint is not a backdoor
    for ad-hoc tenant-side label writes."""


class PromoteLabelsResponse(BaseModel):
    memory_id: str
    promoted: list[str]
    """Labels that moved from suggested → sensitivity on this call."""
    sensitivity_labels: list[str]
    """Memory's authoritative labels AFTER the promotion."""
    suggested_labels: list[str]
    """Memory's remaining suggestions AFTER the promotion (promoted
    labels are dropped from this list so they don't re-appear in
    the review queue)."""


@router.post(
    "/memories/{memory_id}/promote-labels",
    response_model=PromoteLabelsResponse,
)
async def promote_suggested_labels(
    memory_id: str,
    req: PromoteLabelsRequest,
    tenant_id: str | None = Query(None, description="Filter by tenant (defence-in-depth)"),
):
    """Promote a subset of a memory's ``suggested_labels`` into the
    authoritative ``sensitivity_labels`` column (v0.9 #160).

    Closes the loop on the auto-labeling story (#158): detectors
    stamp suggestions, an operator reviews them in the admin UI,
    and this endpoint is the explicit *commit* action that moves a
    suggestion into the column the policy evaluator actually reads.

    Contract:

      * Every label in ``req.labels`` MUST currently be in the
        memory's ``suggested_labels``. Ad-hoc label writes via this
        endpoint are rejected with 422 — the SDK is the path for
        tenant-side direct writes; this endpoint is review-only.
      * Promoted labels are appended to ``sensitivity_labels``
        (deduped + sorted) and REMOVED from ``suggested_labels`` so
        the review queue doesn't re-surface them. Idempotency: a
        second call with the same labels returns 422
        ``no_pending_suggestions``.
      * An audit entry is appended to ``memory.metadata.label_promotions``:
        ``{labels, promoted_at, promoted_by: null}``. ``promoted_by``
        is null in v0.9 — no admin identity layer exists yet; the
        TODO is tracked on the field. Time and what-was-promoted are
        captured today, who-promoted lands when admin identity does.

    Tenant scoped via the optional ``tenant_id`` query — defence in
    depth on top of the memory_id lookup so a misconfigured admin
    cookie can't cross-tenant-promote.
    """
    import uuid as uuid_module
    from datetime import datetime, timezone

    from sqlalchemy import select

    from server.db import engine as engine_module
    from server.db.tables import MemoryRow

    # Validate the request body before touching the DB.
    if not req.labels:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "promote_labels.empty",
                "message": "`labels` must be a non-empty list",
            },
        )
    if len(req.labels) != len(set(req.labels)):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "promote_labels.duplicate_labels",
                "message": "`labels` must not contain duplicates",
            },
        )

    try:
        memory_uuid = uuid_module.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory_id format")

    async with engine_module.get_session_factory()() as session:
        stmt = select(MemoryRow).where(MemoryRow.id == memory_uuid)
        if tenant_id:
            stmt = stmt.where(MemoryRow.tenant_id == tenant_id)
        result = await session.execute(stmt)
        memory = result.scalar_one_or_none()
        if memory is None:
            raise HTTPException(status_code=404, detail="memory not found")

        current_suggested = list(memory.suggested_labels or [])
        missing = [lbl for lbl in req.labels if lbl not in current_suggested]
        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "promote_labels.not_suggested",
                    "message": (
                        f"label(s) {sorted(missing)} are not in this memory's "
                        "suggested_labels; promotion is review-only — use the "
                        "SDK for direct tenant-side label writes"
                    ),
                },
            )

        # Compute the new state. Append + dedupe + sort so multiple
        # promotions converge to a stable list shape.
        new_sensitivity = sorted(set(memory.sensitivity_labels or []) | set(req.labels))
        new_suggested = sorted(set(current_suggested) - set(req.labels))

        # Audit trail in metadata. Append-only — never overwrite. Each
        # entry is self-contained so reading the row years later still
        # tells the full promotion history without joins. `promoted_by`
        # is null in v0.9; once admin identity lands (separate work)
        # the API will populate it.
        new_metadata = dict(memory.metadata_ or {})
        promotions = list(new_metadata.get("label_promotions") or [])
        promotions.append(
            {
                "labels": sorted(req.labels),
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "promoted_by": None,  # TODO: populate from admin identity once available
            }
        )
        new_metadata["label_promotions"] = promotions

        memory.sensitivity_labels = new_sensitivity
        memory.suggested_labels = new_suggested
        memory.metadata_ = new_metadata
        await session.commit()
        await session.refresh(memory)

        logger.info(
            "suggested_labels_promoted",
            memory_id=str(memory.id),
            tenant_id=memory.tenant_id,
            subject_id=memory.subject_id,
            promoted=sorted(req.labels),
        )

        return PromoteLabelsResponse(
            memory_id=str(memory.id),
            promoted=sorted(req.labels),
            sensitivity_labels=list(memory.sensitivity_labels or []),
            suggested_labels=list(memory.suggested_labels or []),
        )


# ─── Memory Evolution / Related Memories ─────────────────────────────────────


class RelatedMemoryItem(BaseModel):
    """A memory related to the target memory."""

    id: str
    kind: str
    content: str
    summary: str
    confidence: float
    status: str
    created_at: str
    relationship: str  # "supersedes" | "sibling" | "superseded_by"


class MemoryEvolutionResponse(BaseModel):
    """Response for memory evolution/related memories lookup."""

    memory_id: str
    status: str
    created_at: str
    superseding_memory: RelatedMemoryItem | None  # The memory that replaced this one
    superseded_memories: list[RelatedMemoryItem]  # Memories this one replaced
    sibling_memories: list[RelatedMemoryItem]  # Other memories from same sources
    source_episode_count: int


@router.get(
    "/subjects/{subject_id}/memories/{memory_id}/related",
    response_model=MemoryEvolutionResponse,
)
async def get_memory_related(
    subject_id: str,
    memory_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
):
    """Get memory evolution and related memories.

    Returns:
    - superseding_memory: If this memory is superseded, the active memory that replaced it
    - superseded_memories: If this memory is active, older memories it superseded
    - sibling_memories: Other memories derived from the same source episodes
    """
    import uuid as uuid_module

    from sqlalchemy import any_, or_, select

    from server.db import engine as engine_module
    from server.db.tables import MemoryRow

    # Validate memory_id is a valid UUID
    try:
        memory_uuid = uuid_module.UUID(memory_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid memory_id format")

    async with engine_module.get_session_factory()() as session:
        # Get the target memory
        stmt = select(MemoryRow).where(
            MemoryRow.id == memory_uuid,
            MemoryRow.subject_id == subject_id,
        )
        if tenant_id:
            stmt = stmt.where(MemoryRow.tenant_id == tenant_id)

        result = await session.execute(stmt)
        target = result.scalar_one_or_none()

        if not target:
            raise HTTPException(status_code=404, detail="Memory not found")

        superseding_memory = None
        superseded_memories: list[RelatedMemoryItem] = []
        sibling_memories: list[RelatedMemoryItem] = []

        # If this memory is superseded, find the active memory that replaced it
        if target.status == "superseded" and target.source_episode_ids:
            # Look for an active memory of the same kind with overlapping source episodes
            # that was created after this one
            superseder_stmt = (
                select(MemoryRow)
                .where(
                    MemoryRow.subject_id == subject_id,
                    MemoryRow.kind == target.kind,
                    MemoryRow.status == "active",
                    MemoryRow.created_at > target.created_at,
                    MemoryRow.id != target.id,
                )
                .order_by(MemoryRow.created_at.asc())
                .limit(1)
            )
            if tenant_id:
                superseder_stmt = superseder_stmt.where(MemoryRow.tenant_id == tenant_id)

            superseder_result = await session.execute(superseder_stmt)
            superseder = superseder_result.scalar_one_or_none()

            if superseder:
                superseding_memory = RelatedMemoryItem(
                    id=str(superseder.id),
                    kind=superseder.kind,
                    content=superseder.content,
                    summary=superseder.summary,
                    confidence=superseder.confidence,
                    status=superseder.status,
                    created_at=superseder.created_at.isoformat(),
                    relationship="supersedes",
                )

        # If this memory is active, find memories it superseded
        if target.status == "active":
            superseded_stmt = (
                select(MemoryRow)
                .where(
                    MemoryRow.subject_id == subject_id,
                    MemoryRow.kind == target.kind,
                    MemoryRow.status == "superseded",
                    MemoryRow.created_at < target.created_at,
                    MemoryRow.id != target.id,
                )
                .order_by(MemoryRow.created_at.desc())
                .limit(5)
            )
            if tenant_id:
                superseded_stmt = superseded_stmt.where(MemoryRow.tenant_id == tenant_id)

            superseded_result = await session.execute(superseded_stmt)
            for m in superseded_result.scalars().all():
                superseded_memories.append(
                    RelatedMemoryItem(
                        id=str(m.id),
                        kind=m.kind,
                        content=m.content,
                        summary=m.summary,
                        confidence=m.confidence,
                        status=m.status,
                        created_at=m.created_at.isoformat(),
                        relationship="superseded_by",
                    )
                )

        # Find sibling memories (same source episodes, different memory)
        if target.source_episode_ids:
            # Find memories that share at least one source episode
            sibling_conditions = [
                ep_id == any_(MemoryRow.source_episode_ids)
                for ep_id in target.source_episode_ids[:5]  # Limit to first 5 to avoid huge OR
            ]
            sibling_stmt = (
                select(MemoryRow)
                .where(
                    MemoryRow.subject_id == subject_id,
                    MemoryRow.id != target.id,
                    or_(*sibling_conditions),
                )
                .order_by(MemoryRow.created_at.desc())
                .limit(10)
            )
            if tenant_id:
                sibling_stmt = sibling_stmt.where(MemoryRow.tenant_id == tenant_id)

            sibling_result = await session.execute(sibling_stmt)
            for m in sibling_result.scalars().all():
                # Skip if already in superseding or superseded
                if superseding_memory and m.id == uuid_module.UUID(superseding_memory.id):
                    continue
                if any(s.id == str(m.id) for s in superseded_memories):
                    continue

                sibling_memories.append(
                    RelatedMemoryItem(
                        id=str(m.id),
                        kind=m.kind,
                        content=m.content,
                        summary=m.summary,
                        confidence=m.confidence,
                        status=m.status,
                        created_at=m.created_at.isoformat(),
                        relationship="sibling",
                    )
                )

        return MemoryEvolutionResponse(
            memory_id=str(target.id),
            status=target.status,
            created_at=target.created_at.isoformat(),
            superseding_memory=superseding_memory,
            superseded_memories=superseded_memories,
            sibling_memories=sibling_memories,
            source_episode_count=len(target.source_episode_ids),
        )


# ─── Session Timeline ─────────────────────────────────────────────────────────


class TimelineEpisodeEvent(BaseModel):
    """Episode event in a session timeline."""

    event_type: Literal["episode"] = "episode"
    id: str
    source: str
    type: str
    payload: dict
    metadata: dict
    provenance: dict
    created_at: str
    citing_memory_count: int


class TimelineResolutionEvent(BaseModel):
    """Resolution event in a session timeline."""

    event_type: Literal["resolution"] = "resolution"
    resolved_at: str
    status: str


class SessionTimelineResponse(BaseModel):
    """Session timeline with chronologically merged events."""

    session_id: str
    status: str
    first_message_at: str | None
    first_response_at: str | None
    resolved_at: str | None
    first_response_seconds: float | None
    resolution_seconds: float | None
    first_response_breached: bool
    resolution_breached: bool
    episode_count: int
    events: list[TimelineEpisodeEvent | TimelineResolutionEvent]


@router.get(
    "/subjects/{subject_id}/sessions/{session_id}/timeline",
    response_model=SessionTimelineResponse,
)
async def get_session_timeline(
    subject_id: str,
    session_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    limit: int = Query(200, ge=1, le=500, description="Max episodes to include"),
):
    """Get a chronological timeline of events for a session.

    Returns episodes in chronological order (oldest first), with resolution
    events interleaved at the correct timestamp. Each episode includes a
    citing_memory_count for quick provenance visibility.
    """
    from sqlalchemy import any_, func, select

    from server.db import engine as engine_module
    from server.db.tables import EpisodeRow, MemoryRow, ResolutionRow
    from server.services.sla import compute_sla

    async with engine_module.get_session_factory()() as session:
        # Get episodes for this session
        base = select(EpisodeRow).where(
            EpisodeRow.subject_id == subject_id,
            EpisodeRow.session_id == session_id,
        )
        if tenant_id:
            base = base.where(EpisodeRow.tenant_id == tenant_id)

        # Order chronologically (oldest first for timeline)
        stmt = base.order_by(EpisodeRow.created_at.asc()).limit(limit)
        result = await session.execute(stmt)
        episode_rows = result.scalars().all()

        # Get total count for this session
        count_stmt = select(func.count()).select_from(base.subquery())
        episode_count = await session.scalar(count_stmt) or 0

        # Get citing memory counts for all episode IDs in one query
        episode_ids = [e.id for e in episode_rows]
        citing_counts: dict[str, int] = {}

        if episode_ids:
            # Count memories that cite each episode
            for ep_id in episode_ids:
                count_q = select(func.count()).where(
                    MemoryRow.subject_id == subject_id,
                    ep_id == any_(MemoryRow.source_episode_ids),
                )
                if tenant_id:
                    count_q = count_q.where(MemoryRow.tenant_id == tenant_id)
                citing_counts[str(ep_id)] = await session.scalar(count_q) or 0

        # Get resolution for this session
        resolution_stmt = select(ResolutionRow).where(
            ResolutionRow.subject_id == subject_id,
            ResolutionRow.session_id == session_id,
        )
        if tenant_id:
            resolution_stmt = resolution_stmt.where(ResolutionRow.tenant_id == tenant_id)
        resolution_result = await session.execute(resolution_stmt)
        resolution = resolution_result.scalar_one_or_none()

        # Compute SLA metrics for this session
        sla_result = await compute_sla(session, subject_id, tenant_id=tenant_id)
        session_sla = next((s for s in sla_result.sessions if s.session_id == session_id), None)

        # Build chronological event list
        events: list[TimelineEpisodeEvent | TimelineResolutionEvent] = []

        resolution_inserted = False
        resolved_at = resolution.resolved_at if resolution else None

        for ep in episode_rows:
            # Insert resolution event at the right position
            if not resolution_inserted and resolved_at and ep.created_at > resolved_at:
                events.append(
                    TimelineResolutionEvent(
                        resolved_at=resolved_at.isoformat(),
                        status=resolution.status if resolution else "resolved",
                    )
                )
                resolution_inserted = True

            events.append(
                TimelineEpisodeEvent(
                    id=str(ep.id),
                    source=ep.source,
                    type=ep.type,
                    payload=ep.payload,
                    metadata=ep.metadata_,
                    provenance=ep.provenance,
                    created_at=ep.created_at.isoformat(),
                    citing_memory_count=citing_counts.get(str(ep.id), 0),
                )
            )

        # If resolution is after all episodes, append at end
        if not resolution_inserted and resolved_at:
            events.append(
                TimelineResolutionEvent(
                    resolved_at=resolved_at.isoformat(),
                    status=resolution.status if resolution else "resolved",
                )
            )

        return SessionTimelineResponse(
            session_id=session_id,
            status=session_sla.status if session_sla else ("resolved" if resolution else "open"),
            first_message_at=(
                session_sla.first_message_at.isoformat()
                if session_sla and session_sla.first_message_at
                else None
            ),
            first_response_at=(
                session_sla.first_response_at.isoformat()
                if session_sla and session_sla.first_response_at
                else None
            ),
            resolved_at=resolved_at.isoformat() if resolved_at else None,
            first_response_seconds=session_sla.first_response_seconds if session_sla else None,
            resolution_seconds=session_sla.resolution_seconds if session_sla else None,
            first_response_breached=session_sla.first_response_breached if session_sla else False,
            resolution_breached=session_sla.resolution_breached if session_sla else False,
            episode_count=episode_count,
            events=events,
        )


# ─── Usage Metering ───


@router.get("/usage")
async def usage_metering(tenant_id: str | None = None):
    """On-demand usage metrics for operator capacity planning.

    Returns counts for key operations across time windows (today, 7d, 30d, all-time).
    Optionally filtered by tenant_id.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import and_, func, select

    from server.db import engine as engine_module
    from server.db.tables import CompileJobRow, EpisodeRow, MemoryRow, WebhookEventRow

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)

    async with engine_module.get_session_factory()() as session:

        async def _count(table, ts_col, *, since=None):
            stmt = select(func.count()).select_from(table)
            conditions = []
            if tenant_id and hasattr(table, "tenant_id"):
                conditions.append(table.tenant_id == tenant_id)
            if since:
                conditions.append(ts_col >= since)
            if conditions:
                stmt = stmt.where(and_(*conditions))
            return await session.scalar(stmt) or 0

        # Episodes
        ep_today = await _count(EpisodeRow, EpisodeRow.created_at, since=today_start)
        ep_7d = await _count(EpisodeRow, EpisodeRow.created_at, since=seven_days_ago)
        ep_30d = await _count(EpisodeRow, EpisodeRow.created_at, since=thirty_days_ago)
        ep_total = await _count(EpisodeRow, EpisodeRow.created_at)

        # Memories compiled
        mem_today = await _count(MemoryRow, MemoryRow.created_at, since=today_start)
        mem_7d = await _count(MemoryRow, MemoryRow.created_at, since=seven_days_ago)
        mem_30d = await _count(MemoryRow, MemoryRow.created_at, since=thirty_days_ago)
        mem_total = await _count(MemoryRow, MemoryRow.created_at)

        # Compile jobs
        job_today = await _count(CompileJobRow, CompileJobRow.created_at, since=today_start)
        job_7d = await _count(CompileJobRow, CompileJobRow.created_at, since=seven_days_ago)
        job_30d = await _count(CompileJobRow, CompileJobRow.created_at, since=thirty_days_ago)
        job_total = await _count(CompileJobRow, CompileJobRow.created_at)

        # Webhooks
        wh_today = await _count(WebhookEventRow, WebhookEventRow.created_at, since=today_start)
        wh_7d = await _count(WebhookEventRow, WebhookEventRow.created_at, since=seven_days_ago)
        wh_30d = await _count(WebhookEventRow, WebhookEventRow.created_at, since=thirty_days_ago)
        wh_total = await _count(WebhookEventRow, WebhookEventRow.created_at)

        # Distinct subjects active in period
        async def _active_subjects(since=None):
            stmt = select(func.count(func.distinct(EpisodeRow.subject_id)))
            conditions = []
            if tenant_id:
                conditions.append(EpisodeRow.tenant_id == tenant_id)
            if since:
                conditions.append(EpisodeRow.created_at >= since)
            if conditions:
                stmt = stmt.where(and_(*conditions))
            return await session.scalar(stmt) or 0

        subj_7d = await _active_subjects(since=seven_days_ago)
        subj_30d = await _active_subjects(since=thirty_days_ago)
        subj_total = await _active_subjects()

    return {
        "period_start": today_start.isoformat(),
        "generated_at": now.isoformat(),
        "tenant_id": tenant_id,
        "episodes": {"today": ep_today, "7d": ep_7d, "30d": ep_30d, "total": ep_total},
        "memories": {"today": mem_today, "7d": mem_7d, "30d": mem_30d, "total": mem_total},
        "compile_jobs": {"today": job_today, "7d": job_7d, "30d": job_30d, "total": job_total},
        "webhooks": {"today": wh_today, "7d": wh_7d, "30d": wh_30d, "total": wh_total},
        "active_subjects": {"7d": subj_7d, "30d": subj_30d, "total": subj_total},
    }


# ─── Compile Jobs (operator introspection) ───


@router.get("/jobs")
async def list_compile_jobs(
    status: str | None = Query(
        None, description="Filter by status: pending, running, completed, failed"
    ),
    subject_id: str | None = Query(None, description="Filter by subject"),
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List compile jobs for operator debugging.

    Returns recent jobs ordered by creation time (newest first).
    """
    from server.services.compile_jobs_durable import list_jobs

    jobs, total = await list_jobs(
        status=status, subject_id=subject_id, tenant_id=tenant_id, limit=limit, offset=offset
    )
    return {"jobs": jobs, "total": total, "limit": limit, "offset": offset}


@router.delete("/jobs")
async def purge_compile_jobs(
    status: str | None = Query(None, description="Filter by terminal status: completed or failed"),
    subject_id: str | None = Query(None, description="Filter by subject"),
    tenant_id: str | None = Query(None, description="Filter by tenant"),
):
    """Bulk-delete terminal compile jobs matching the given filter.

    Refuses an empty filter (you must pass at least one of status, subject_id,
    tenant_id) and refuses non-terminal statuses — `pending`/`running` jobs
    may still be held by the worker.
    """
    from server.services.compile_jobs_durable import purge_jobs

    try:
        deleted = await purge_jobs(status=status, subject_id=subject_id, tenant_id=tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"deleted": deleted}


# ─── Tenant Audit ───


@router.get("/tenant-audit")
async def tenant_audit():
    """Report rows with NULL tenant_id — helps operators backfill after enabling tenants."""
    from sqlalchemy import func, select

    from server.db.engine import get_session_factory
    from server.db.tables import CompileJobRow, EpisodeRow, MemoryRow

    async with get_session_factory()() as session:
        ep_null = await session.scalar(
            select(func.count()).select_from(EpisodeRow).where(EpisodeRow.tenant_id.is_(None))
        )
        mem_null = await session.scalar(
            select(func.count()).select_from(MemoryRow).where(MemoryRow.tenant_id.is_(None))
        )
        jobs_null = await session.scalar(
            select(func.count()).select_from(CompileJobRow).where(CompileJobRow.tenant_id.is_(None))
        )

    return {
        "null_tenant_rows": {
            "episodes": ep_null or 0,
            "memories": mem_null or 0,
            "compile_jobs": jobs_null or 0,
        },
        "guidance": "Backfill with UPDATE <table> SET tenant_id = 'your-tenant' WHERE tenant_id IS NULL",
    }


# ─── Backup / Restore ───


class ImportSubjectRequest(BaseModel):
    document: dict
    target_subject_id: str | None = None
    target_tenant_id: str | None = None
    preserve_ids: bool = True


@router.get("/export/{subject_id}")
async def export_subject_endpoint(
    subject_id: str,
    tenant_id: str | None = Query(None, description="Scope export to tenant"),
):
    """Export all episodes and memories for a subject as a portable JSON document.

    The output includes a SHA-256 checksum for integrity verification.
    Use this to back up a subject before risky operations or to migrate
    between Statewave instances.
    """
    from server.services.backup import export_subject

    doc = await export_subject(subject_id, tenant_id=tenant_id)
    if doc["counts"]["episodes"] == 0 and doc["counts"]["memories"] == 0:
        raise HTTPException(status_code=404, detail=f"No data found for subject '{subject_id}'")
    return doc


@router.post("/import")
async def import_subject_endpoint(req: ImportSubjectRequest):
    """Import a previously exported subject document.

    Options:
    - target_subject_id: override subject_id (default: use original from export)
    - target_tenant_id: override tenant_id (default: use original from export)
    - preserve_ids: keep original UUIDs (default true; set false to generate new ones)

    Safety: validates format version and checksum before importing.
    """
    from server.services.backup import import_subject

    try:
        result = await import_subject(
            req.document,
            target_subject_id=req.target_subject_id,
            target_tenant_id=req.target_tenant_id,
            preserve_ids=req.preserve_ids,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── Webhooks ───


@router.get("/webhooks")
async def list_webhook_events(
    status: str | None = Query(
        None, description="Filter by status: pending, delivered, dead_letter"
    ),
    event_type: str | None = Query(None, description="Filter by event type"),
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List webhook events for operator debugging.

    Returns recent events ordered by creation time (newest first).
    """
    events, total = await webhooks.list_events(
        status=status, event_type=event_type, tenant_id=tenant_id, limit=limit, offset=offset
    )
    return {"events": events, "total": total, "limit": limit, "offset": offset}


@router.delete("/webhooks")
async def purge_webhook_events(
    status: str | None = Query(
        None, description="Filter by terminal status: delivered or dead_letter"
    ),
    event_type: str | None = Query(None, description="Filter by event type"),
    tenant_id: str | None = Query(None, description="Filter by tenant"),
):
    """Bulk-delete terminal webhook events matching the given filter.

    Refuses an empty filter and refuses non-terminal statuses — `pending`
    events may still be picked up by the delivery worker.
    """
    try:
        deleted = await webhooks.purge_events(
            status=status, event_type=event_type, tenant_id=tenant_id
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"deleted": deleted}


@router.get("/webhooks/stats")
async def webhook_stats(
    tenant_id: str | None = Query(None, description="Filter by tenant"),
):
    """Aggregate webhook delivery statistics (optionally filtered by tenant)."""
    return await webhooks.get_delivery_stats(tenant_id=tenant_id)


@router.get("/webhooks/{event_id}")
async def webhook_event_status(
    event_id: uuid.UUID,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
):
    """Get delivery status of a specific webhook event (optionally tenant-filtered)."""
    result = await webhooks.get_event_status(event_id, tenant_id=tenant_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Webhook event not found")
    return result


# ─── Subject Deletion (single + filtered bulk) ─────────────────────────────


class BulkDeleteFilter(BaseModel):
    """Selector for matching subjects to delete.

    At least one selector must be present so an empty body cannot become
    "delete everything" by accident. The selectors are AND-ed together,
    with one exception: `match_all=True` is a stand-alone explicit opt-in
    that disables the empty-filter guard so an operator can wipe an entire
    workspace on purpose. The frontend gates that behind a type-to-confirm
    phrase.
    """

    subject_id_prefix: str | None = None
    """Match subjects whose id starts with this prefix (e.g. 'demo_web_')."""

    older_than_days: int | None = None
    """Match subjects whose most recent episode is older than N days."""

    tenant_id: str | None = None
    """Restrict to a specific tenant. Combined with other filters via AND."""

    match_all: bool = False
    """Explicit opt-in to match every subject. Required when no other
    selector is set; ignored otherwise (other selectors take precedence)."""


class BulkDeleteSample(BaseModel):
    subject_id: str
    tenant_id: str | None
    episode_count: int
    memory_count: int
    last_episode_at: str | None


class BulkDeletePreview(BaseModel):
    matched: int
    sample: list[BulkDeleteSample]
    total_episodes: int
    total_memories: int


class BulkDeleteCommitRequest(BulkDeleteFilter):
    confirm: bool = False
    """Must be True. Refused otherwise."""

    expected_count: int
    """Optimistic-concurrency guard: must equal the current matched count.
    Prevents accidentally deleting more than the operator previewed when the
    set drifts between preview and commit.
    """


class BulkDeleteResult(BaseModel):
    deleted_subjects: int
    deleted_episodes: int
    deleted_memories: int
    failed: list[str]


def _filter_is_empty(f: BulkDeleteFilter) -> bool:
    """An empty filter is one with no selector AND no `match_all` opt-in.

    `match_all` is the explicit "yes I want everything" escape hatch. When
    set, the filter is no longer considered empty and the request proceeds
    against the unscoped subject set.
    """
    if f.match_all:
        return False
    return not f.subject_id_prefix and f.older_than_days is None and not f.tenant_id


async def _matching_subjects(
    f: BulkDeleteFilter,
) -> tuple[list[BulkDeleteSample], int, int]:
    """Resolve the filter to a concrete list of matching subjects.

    Returns (all_matches, total_eps, total_mems). The caller decides how many
    matches to surface in a response — the full list is needed for committing.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import func, select

    from server.db import engine as engine_module
    from server.db.tables import EpisodeRow, MemoryRow

    async with engine_module.get_session_factory()() as session:
        # Aggregate per (subject_id, tenant_id) from episodes — this is the
        # authoritative grouping used elsewhere in admin.
        ep_stmt = select(
            EpisodeRow.subject_id,
            EpisodeRow.tenant_id,
            func.count().label("ep_count"),
            func.max(EpisodeRow.created_at).label("last_episode_at"),
        ).group_by(EpisodeRow.subject_id, EpisodeRow.tenant_id)
        if f.subject_id_prefix:
            ep_stmt = ep_stmt.where(EpisodeRow.subject_id.like(f"{f.subject_id_prefix}%"))
        if f.tenant_id:
            ep_stmt = ep_stmt.where(EpisodeRow.tenant_id == f.tenant_id)
        if f.older_than_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=f.older_than_days)
            # The HAVING-style filter on the max(created_at) — apply via subquery.
            sub = ep_stmt.subquery()
            ep_stmt = select(
                sub.c.subject_id,
                sub.c.tenant_id,
                sub.c.ep_count,
                sub.c.last_episode_at,
            ).where(sub.c.last_episode_at < cutoff)

        rows = (await session.execute(ep_stmt)).all()

        # Memory counts per subject — separate query for clarity.
        mem_counts: dict[tuple[str, str | None], int] = {}
        if rows:
            mem_stmt = (
                select(
                    MemoryRow.subject_id,
                    MemoryRow.tenant_id,
                    func.count().label("mem_count"),
                )
                .where(MemoryRow.subject_id.in_([r.subject_id for r in rows]))
                .group_by(MemoryRow.subject_id, MemoryRow.tenant_id)
            )
            for m in (await session.execute(mem_stmt)).all():
                mem_counts[(m.subject_id, m.tenant_id)] = m.mem_count

    total_eps = sum(r.ep_count for r in rows)
    total_mems = sum(mem_counts.values())

    matches: list[BulkDeleteSample] = []
    for r in rows:
        matches.append(
            BulkDeleteSample(
                subject_id=r.subject_id,
                tenant_id=r.tenant_id,
                episode_count=r.ep_count,
                memory_count=mem_counts.get((r.subject_id, r.tenant_id), 0),
                last_episode_at=r.last_episode_at.isoformat() if r.last_episode_at else None,
            )
        )
    return matches, total_eps, total_mems


@router.delete("/subjects/{subject_id}")
async def delete_subject_admin(
    subject_id: str,
    tenant_id: str | None = Query(None, description="Restrict to tenant"),
):
    """Permanently delete all episodes and memories for a subject.

    This is the admin-tool equivalent of `DELETE /v1/subjects/{id}` — same
    cascade, same webhook, same irreversibility.
    """
    from server.db import engine as engine_module
    from server.db import repositories as repo

    async with engine_module.get_session_factory()() as session:
        ep_count = await repo.delete_episodes_by_subject(session, subject_id, tenant_id=tenant_id)
        mem_count = await repo.delete_memories_by_subject(session, subject_id, tenant_id=tenant_id)
        await repo.delete_resolutions_by_subject(session, subject_id, tenant_id=tenant_id)
        await repo.delete_health_cache_by_subject(session, subject_id, tenant_id=tenant_id)
        await session.commit()

    if ep_count == 0 and mem_count == 0:
        raise HTTPException(status_code=404, detail=f"Subject '{subject_id}' not found")

    await webhooks.fire(
        "subject.deleted",
        {
            "subject_id": subject_id,
            "episodes_deleted": ep_count,
            "memories_deleted": mem_count,
        },
        tenant_id=tenant_id,
    )
    return {
        "subject_id": subject_id,
        "episodes_deleted": ep_count,
        "memories_deleted": mem_count,
    }


@router.post("/subjects/preview-delete", response_model=BulkDeletePreview)
async def preview_bulk_delete(filter: BulkDeleteFilter):
    """Preview a filtered bulk delete without committing.

    Returns the match count, totals (episodes + memories), and a sample list
    so the operator can eyeball what they're about to wipe.
    """
    if _filter_is_empty(filter):
        raise HTTPException(
            status_code=400,
            detail=(
                "At least one filter must be set (subject_id_prefix, "
                "older_than_days, or tenant_id), or match_all=true to "
                "explicitly target every subject."
            ),
        )
    matches, total_eps, total_mems = await _matching_subjects(filter)
    return BulkDeletePreview(
        matched=len(matches),
        sample=matches[:20],
        total_episodes=total_eps,
        total_memories=total_mems,
    )


@router.post("/subjects/bulk-delete", response_model=BulkDeleteResult)
async def commit_bulk_delete(req: BulkDeleteCommitRequest):
    """Commit a previously previewed filtered bulk delete.

    Safety: requires `confirm: true` and `expected_count` matching the current
    match count. If subjects have been added or removed since the preview, the
    request is rejected with 409 — the operator must re-preview.
    """
    from server.db import engine as engine_module
    from server.db import repositories as repo

    if not req.confirm:
        raise HTTPException(status_code=400, detail="confirm must be true to commit a bulk delete")
    if _filter_is_empty(req):
        raise HTTPException(
            status_code=400,
            detail=(
                "At least one filter must be set (subject_id_prefix, "
                "older_than_days, or tenant_id), or match_all=true to "
                "explicitly target every subject."
            ),
        )

    # Recompute the match set against current state.
    matches, _, _ = await _matching_subjects(req)
    if len(matches) != req.expected_count:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Match count changed since preview "
                f"({len(matches)} now vs expected {req.expected_count}). Re-preview before committing."
            ),
        )

    deleted_subjects = 0
    deleted_eps = 0
    deleted_mems = 0
    failed: list[str] = []

    async with engine_module.get_session_factory()() as session:
        for s in matches:
            try:
                ep_n = await repo.delete_episodes_by_subject(
                    session, s.subject_id, tenant_id=s.tenant_id
                )
                mem_n = await repo.delete_memories_by_subject(
                    session, s.subject_id, tenant_id=s.tenant_id
                )
                await repo.delete_resolutions_by_subject(
                    session, s.subject_id, tenant_id=s.tenant_id
                )
                await repo.delete_health_cache_by_subject(
                    session, s.subject_id, tenant_id=s.tenant_id
                )
                await session.commit()
                deleted_subjects += 1
                deleted_eps += ep_n
                deleted_mems += mem_n
                # Fire one webhook per subject so downstream pipelines get the
                # same shape they get for /v1/subjects/{id} deletes.
                await webhooks.fire(
                    "subject.deleted",
                    {
                        "subject_id": s.subject_id,
                        "episodes_deleted": ep_n,
                        "memories_deleted": mem_n,
                    },
                    tenant_id=s.tenant_id,
                )
            except Exception:
                await session.rollback()
                failed.append(s.subject_id)

    return BulkDeleteResult(
        deleted_subjects=deleted_subjects,
        deleted_episodes=deleted_eps,
        deleted_memories=deleted_mems,
        failed=failed,
    )


# ─── Subject Snapshots (advanced bootstrap/admin, feature-flagged) ───


class RestoreSnapshotRequest(BaseModel):
    target_subject_id: str


class RestoreByNameRequest(BaseModel):
    name: str
    target_subject_id: str
    version: Optional[int] = None


class CreateSnapshotRequest(BaseModel):
    name: str
    source_subject_id: str
    version: int = 1
    metadata: Optional[dict] = None


@router.get("/snapshots")
async def list_snapshots_endpoint():
    """List available subject snapshots."""
    _require_snapshots()
    from server.services.snapshots import list_snapshots

    return {"snapshots": await list_snapshots()}


@router.post("/snapshots")
async def create_snapshot_endpoint(req: CreateSnapshotRequest):
    """Create a snapshot from an existing subject."""
    _require_snapshots()
    from server.services.snapshots import create_snapshot

    try:
        result = await create_snapshot(
            name=req.name,
            source_subject_id=req.source_subject_id,
            version=req.version,
            metadata=req.metadata or {},
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/snapshots/{snapshot_id}/restore")
async def restore_snapshot_endpoint(snapshot_id: uuid.UUID, req: RestoreSnapshotRequest):
    """Restore a snapshot into a new target subject.

    Creates copies of all episodes and memories with new IDs,
    remapped provenance, and shifted timestamps.
    """
    _require_snapshots()
    from server.services.snapshots import restore_snapshot

    try:
        result = await restore_snapshot(snapshot_id, req.target_subject_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/snapshots/restore-by-name")
async def restore_by_name_endpoint(req: RestoreByNameRequest):
    """Restore a snapshot by name (uses latest version if not specified).

    Convenience endpoint for demo/bootstrap flows.
    """
    _require_snapshots()
    from server.services.snapshots import get_snapshot_by_name, restore_snapshot

    snap = await get_snapshot_by_name(req.name, req.version)
    if not snap:
        raise HTTPException(status_code=404, detail=f"Snapshot '{req.name}' not found")

    try:
        result = await restore_snapshot(uuid.UUID(snap["id"]), req.target_subject_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/cleanup")
async def trigger_cleanup(
    prefix: str = Query(default="live_", description="Subject prefix to clean up"),
    max_age_hours: int = Query(default=24, description="Max age in hours"),
):
    """Manually trigger cleanup of stale ephemeral subjects."""
    _require_snapshots()
    from server.services.snapshots import cleanup_ephemeral_subjects

    count = await cleanup_ephemeral_subjects(prefix=prefix, max_age_hours=max_age_hours)
    return {"subjects_cleaned": count}


# ─── Memory portability ───
#
# Vendor-neutral memory operations: starter-pack list/import, support reseed,
# clone, export, import. All routed through `server/services/memory_packs.py`
# so the same primitives back the admin UI, the marketing widget's docs
# grounding, and any future CLI tooling.
#
# Auth: the existing `X-API-Key` middleware gates every `/admin/*` route — no
# per-route check needed. No memory content is logged from any of these
# handlers; only counts, subject ids, and pack ids appear in stdout.


class StarterPackImportRequest(BaseModel):
    pack_id: str
    target_subject_id: Optional[str] = None
    target_display_name: Optional[str] = None
    target_tenant_id: Optional[str] = None
    conflict_strategy: Literal["create_copy", "merge", "cancel"] = "create_copy"
    # Off by default — operators using the admin UI should never accidentally
    # import a pack into the marketing widget's per-visitor namespace
    # (`demo_web_*`). The marketing edge function flips this on when seeding
    # a visitor's subject from the bundled showcase pack — that's the
    # legitimate `demo_web_*` write path.
    allow_reserved_target: bool = False


class SupportReseedRequest(BaseModel):
    reason: Optional[str] = None
    # When false (default), skip work if the live subject already carries
    # the bundled pack's version — that's the no-op fast path used by
    # container-restart auto-update. When true, the drawer's manual
    # "Restore" button forces a reseed regardless of version state.
    force: bool = False


class CloneSubjectRequest(BaseModel):
    source_subject_id: str
    target_subject_id: Optional[str] = None
    target_display_name: Optional[str] = None
    target_tenant_id: Optional[str] = None
    clone_scope: Literal[
        "episodes",
        "memories",
        "episodes_and_memories",
        "episodes_memories_sources",
    ] = "episodes_memories_sources"


class ExportSubjectsRequest(BaseModel):
    subject_ids: list[str]
    tenant_id: Optional[str] = None
    export_scope: Literal[
        "episodes",
        "memories",
        "episodes_and_memories",
        "episodes_memories_sources",
    ] = "episodes_memories_sources"


class ImportPayloadRequest(BaseModel):
    payload: dict
    target_tenant_id: Optional[str] = None
    conflict_strategy: Literal["create_copy", "merge", "cancel"] = "create_copy"


@router.get("/memory/starter-packs")
async def list_starter_packs_endpoint():
    """Return metadata for the platform-bundled starter packs.

    Pack content lives in `server/starter_packs/`; this endpoint reads the
    on-disk index. No memory content is returned here — only manifest data
    so the admin UI can render selectable cards.
    """
    from server.services.memory_packs import list_starter_packs

    return {"packs": list_starter_packs()}


@router.post("/memory/starter-packs/import")
async def import_starter_pack_endpoint(req: StarterPackImportRequest):
    """Import a platform starter pack into a new tenant-owned subject.

    Default behaviour creates a fresh subject with a unique id; `merge`
    appends to an existing subject; `cancel` aborts if the target id
    already has data. Provenance metadata
    (`starter_pack_id`, `starter_pack_version`, `imported_at`) is written
    onto every resulting episode/memory so the import is traceable.
    """
    from server.services.memory_packs import (
        StarterPackError,
        import_starter_pack,
    )

    try:
        return await import_starter_pack(
            pack_id=req.pack_id,
            target_subject_id=req.target_subject_id,
            target_display_name=req.target_display_name,
            target_tenant_id=req.target_tenant_id,
            conflict_strategy=req.conflict_strategy,
            allow_reserved_target=req.allow_reserved_target,
        )
    except StarterPackError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@router.post("/memory/support/reseed")
async def support_reseed_endpoint(req: SupportReseedRequest | None = None):
    """Rebuild the shared Statewave Support docs subject (vendor-neutral).

    Imports the bundled `statewave-support-agent` starter pack into the
    `statewave-support-docs` subject. Behaviour:

      - Version-aware: when the live subject already carries the bundled
        pack's version, the call is a no-op (returns `updated=false`).
        Pass `force=true` to override and reseed unconditionally.
      - Selective purge: only rows whose metadata identifies them as
        belonging to the support pack are deleted before reimport. Rows
        an operator added alongside ours survive untouched.
      - Per-visitor `demo_web_<uuid>__statewave-support` subjects are not
        touched.
    """
    from server.services.memory_packs import (
        StarterPackError,
        reseed_support_subject,
    )

    reason = (req.reason if req and req.reason else "").strip()[:200] or None
    force = bool(req.force) if req else False
    try:
        return await reseed_support_subject(reason=reason, force=force)
    except StarterPackError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@router.get("/memory/support/state")
async def support_state_endpoint():
    """Snapshot of the support subject's reseed state.

    Returns the bundled pack version, the version currently installed in
    the live subject (read from row metadata), and counts of pack-owned vs
    operator-added rows. The drawer uses this to render the
    "installed v{x} → available v{y}" banner and gate the destructive
    Restore action.
    """
    from server.services.memory_packs import (
        StarterPackError,
        get_support_subject_state,
    )

    try:
        return await get_support_subject_state()
    except StarterPackError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@router.post("/docs-pack/reseed", deprecated=True)
async def docs_pack_reseed_alias(req: SupportReseedRequest | None = None):
    """[Deprecated] Backward-compatible alias for `/admin/memory/support/reseed`.

    Older operator scripts (and the pre-vendor-neutral admin UI) called
    `/admin/docs-pack/reseed`, which used to dispatch a GitHub Actions
    workflow that ran the reseed CLI in CI. That implementation is gone:
    the new vendor-neutral path imports the bundled
    `statewave-support-agent` starter pack directly inside the API process,
    with no GitHub token, no workflow dispatch, and no CI dependency.

    This route delegates straight to `reseed_support_subject` so old
    callers keep working. Prefer `POST /admin/memory/support/reseed` in
    new code; this alias may be removed in a future major version.
    """
    from server.services.memory_packs import (
        StarterPackError,
        reseed_support_subject,
    )

    reason = (req.reason if req and req.reason else "").strip()[:200] or None
    force = bool(req.force) if req else False
    try:
        return await reseed_support_subject(reason=reason, force=force)
    except StarterPackError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@router.post("/memory/clone")
async def clone_subject_endpoint(
    req: CloneSubjectRequest,
    x_statewave_operator_email: str | None = Header(default=None),
):
    """Clone an existing subject into a new one.

    Default scope copies episodes + compiled memories + sources (sources
    are tracked but not yet first-class cloneable records — `source_count`
    returns 0 today). Refuses to overwrite a target that already has data
    — caller must choose a different target id. Provenance metadata
    (`cloned_from_subject_id`, `cloned_at`, `cloned_by`,
    `original_episode_id` / `original_memory_id`) is written onto every
    copied record.

    Error codes:
      400 — invalid input (bad subject id, unsupported scope)
      404 — source subject not found
      409 — target subject already populated (caller picked an existing id)
      500 — unexpected failure
    """
    from server.services.memory_packs import (
        StarterPackError,
        clone_subject,
    )

    try:
        return await clone_subject(
            source_subject_id=req.source_subject_id,
            target_subject_id=req.target_subject_id,
            target_display_name=req.target_display_name,
            target_tenant_id=req.target_tenant_id,
            clone_scope=req.clone_scope,
            cloned_by=x_statewave_operator_email,
        )
    except StarterPackError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@router.post("/memory/export")
async def export_memory_endpoint(req: ExportSubjectsRequest):
    """Build a versioned export payload for one or more subjects.

    Returns plaintext JSON. The admin client encrypts this payload locally
    (passphrase never reaches the server) before saving it as a `.swmem`
    file. Refuses bodies that exceed configured size/count limits — see
    `STATEWAVE_MEMORY_IMPORT_MAX_*` settings.
    """
    from server.services.memory_packs import (
        StarterPackError,
        export_memory_payload,
    )

    try:
        return await export_memory_payload(
            subject_ids=req.subject_ids,
            tenant_id=req.tenant_id,
            export_scope=req.export_scope,
        )
    except StarterPackError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


@router.post("/memory/import")
async def import_memory_endpoint(req: ImportPayloadRequest):
    """Ingest a previously exported memory payload.

    The admin client has already decrypted the `.swmem` blob and validated
    the manifest header — the payload arrives as plaintext JSON over
    authenticated HTTPS. The server re-validates schema, enforces size and
    record-count limits, generates fresh subject ids on conflict by
    default, and refuses unknown top-level fields.
    """
    from server.services.memory_packs import (
        StarterPackError,
        import_memory_payload,
    )

    try:
        return await import_memory_payload(
            payload=req.payload,
            target_tenant_id=req.target_tenant_id,
            conflict_strategy=req.conflict_strategy,
        )
    except StarterPackError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e))


# ─── Receipts (admin operator view) ─────────────────────────────────────────
#
# Mirrors GET /v1/receipts but allows cross-tenant listing for operators
# auditing fleet-wide assembly traffic. Use the per-tenant /v1/receipts
# from inside an application; use /admin/receipts from the admin app or
# CLI when you need to see every tenant in one pane.


@router.get("/receipts")
async def admin_list_receipts(
    subject_id: str | None = Query(None, description="Filter by subject id"),
    tenant_id: str | None = Query(None, description="Filter by tenant id"),
    since: str | None = Query(None, description="Lower bound on created_at (ISO 8601)"),
    until: str | None = Query(None, description="Upper bound on created_at (ISO 8601)"),
    cursor: str | None = Query(
        None,
        description=(
            "Pagination cursor — pass the last receipt_id from the previous "
            "page. ULIDs sort lexically by time so this is stable under "
            "concurrent inserts."
        ),
    ),
    limit: int = Query(50, ge=1, le=200),
):
    """List receipts across tenants for operator audit views.

    Either `subject_id` or `tenant_id` (or both) must be supplied —
    cross-fleet unscoped listing is intentionally not exposed to avoid
    a single page fetch becoming a 200-row dump of every tenant's
    receipts at once.
    """
    if not subject_id and not tenant_id:
        raise HTTPException(
            status_code=400,
            detail="At least one of subject_id or tenant_id is required",
        )

    from datetime import datetime
    from sqlalchemy import select

    from server.db import engine as engine_module
    from server.db.tables import ReceiptRow

    def _parse(ts: str | None) -> datetime | None:
        if not ts:
            return None
        # Tolerate the trailing-Z form some clients still emit.
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))

    async with engine_module.get_session_factory()() as session:
        # Order by the SAME key the cursor predicate uses (`receipt_id < cursor`
        # below). The ULID receipt_id encodes creation time, so receipt_id-desc
        # is newest-first; ordering by created_at (DB commit time) while
        # cursoring on receipt_id silently drops/duplicates rows across pages
        # (same keyset-pagination bug fixed for repo.list_receipts in #223).
        stmt = select(ReceiptRow).order_by(ReceiptRow.receipt_id.desc()).limit(limit)
        if subject_id:
            stmt = stmt.where(ReceiptRow.subject_id == subject_id)
        if tenant_id:
            stmt = stmt.where(ReceiptRow.tenant_id == tenant_id)
        since_dt = _parse(since)
        until_dt = _parse(until)
        if since_dt is not None:
            stmt = stmt.where(ReceiptRow.created_at >= since_dt)
        if until_dt is not None:
            stmt = stmt.where(ReceiptRow.created_at <= until_dt)
        if cursor is not None:
            stmt = stmt.where(ReceiptRow.receipt_id < cursor)

        result = await session.execute(stmt)
        rows = result.scalars().all()

    next_cursor = rows[-1].receipt_id if len(rows) == limit else None
    return {
        "receipts": [row.body for row in rows],
        "next_cursor": next_cursor,
        "limit": limit,
    }


@router.get("/receipts/{receipt_id}")
async def admin_get_receipt(receipt_id: str):
    """Fetch one receipt by id across all tenants (admin view)."""
    from sqlalchemy import select

    from server.db import engine as engine_module
    from server.db.tables import ReceiptRow

    async with engine_module.get_session_factory()() as session:
        result = await session.execute(
            select(ReceiptRow).where(ReceiptRow.receipt_id == receipt_id)
        )
        row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="receipt not found")
    return row.body


@router.post("/receipts/{receipt_id}/replay")
async def admin_replay_receipt(receipt_id: str):
    """Admin-facing replay endpoint (v0.9 #160 admin app shim).

    Mirrors ``POST /v1/receipts/{id}/replay`` but is reachable through
    the admin proxy's ``/admin/*`` allowlist. The admin proxy is the
    only path the admin app talks to upstream — surfacing replay there
    keeps the proxy's allowlist tight (`/admin/*` only) instead of
    expanding it to ``/v1/`` for one operation.

    Cross-tenant by design: admin views are not tenant-scoped on the
    list endpoint either (see ``/admin/receipts`` above). The
    underlying service call passes ``tenant_id=None``, which means a
    receipt is looked up by id alone. Same response shape and same
    422 refusal codes as the public endpoint.
    """
    from server.db import engine as engine_module
    from server.services.replay import ReplayError, replay_receipt

    async with engine_module.get_session_factory()() as session:
        try:
            result = await replay_receipt(session, receipt_id=receipt_id, tenant_id=None)
        except ReplayError as exc:
            if exc.reason == "not_found":
                raise HTTPException(status_code=404, detail="receipt not found") from exc
            raise HTTPException(
                status_code=422,
                detail={
                    "code": f"unreplayable.{exc.reason}",
                    "message": exc.detail or exc.reason,
                },
            ) from exc

    return {
        "original_receipt_id": result.original_receipt_id,
        "replay_receipt_id": result.replay_receipt_id,
        "diff": result.diff,
    }


# ─── Sensitivity-label policy (issue #50) ──────────────────────────────────
#
# v1 surface: upload a YAML/JSON bundle, list bundles, set the active
# bundle (per-tenant or global), invalidate the in-process cache. The
# server consults the active bundle on every assembly call; the policy
# layer is data-driven so changes don't require a redeploy.


class UploadPolicyBundleRequest(BaseModel):
    """Body for POST /admin/policy/bundles.

    `yaml_content` is parsed at upload time so a syntactically broken
    bundle never reaches the active slot. Validation errors return
    400 with the parser message so an operator can fix the YAML
    locally before retrying.
    """

    yaml_content: str
    tenant_id: str | None = None
    activate: bool = False


@router.post("/policy/bundles")
async def upload_policy_bundle(req: UploadPolicyBundleRequest):
    """Upload a new policy bundle. Returns the content hash so the
    operator can refer to it in subsequent activate calls."""
    from sqlalchemy import select, update as sa_update

    from server.db import engine as engine_module
    from server.db.tables import PolicyBundleRow
    from server.services import policy as policy_service

    try:
        bundle = policy_service.load_bundle(req.yaml_content)
    except policy_service.PolicyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    async with engine_module.get_session_factory()() as session:
        # If a row already exists with the SAME (tenant_id, bundle_hash),
        # the bundle is a duplicate the caller already uploaded —
        # return the existing row's metadata rather than INSERT-
        # conflicting on the composite unique index. The tenant_id
        # scope on this lookup is the load-bearing fix for #79: two
        # tenants installing the same YAML must produce two rows,
        # not silently rebind the first row's tenant.
        existing_stmt = select(PolicyBundleRow).where(
            PolicyBundleRow.bundle_hash == bundle.bundle_hash
        )
        if req.tenant_id is None:
            existing_stmt = existing_stmt.where(PolicyBundleRow.tenant_id.is_(None))
        else:
            existing_stmt = existing_stmt.where(PolicyBundleRow.tenant_id == req.tenant_id)
        existing = await session.execute(existing_stmt)
        existing_row = existing.scalar_one_or_none()
        if existing_row is None:
            session.add(
                PolicyBundleRow(
                    bundle_hash=bundle.bundle_hash,
                    yaml_content=req.yaml_content,
                    active=False,
                    tenant_id=req.tenant_id,
                )
            )
            await session.commit()

        if req.activate:
            # Deactivate other bundles in the same scope so the
            # active-bundle resolver sees exactly one row.
            scope_stmt = sa_update(PolicyBundleRow).where(
                PolicyBundleRow.bundle_hash != bundle.bundle_hash
            )
            if req.tenant_id is None:
                scope_stmt = scope_stmt.where(PolicyBundleRow.tenant_id.is_(None))
            else:
                scope_stmt = scope_stmt.where(PolicyBundleRow.tenant_id == req.tenant_id)
            scope_stmt = scope_stmt.values(active=False)
            await session.execute(scope_stmt)
            # Activate ONLY the row for this (tenant, hash) — without
            # the tenant_id scope, an UPDATE on `bundle_hash=X` alone
            # would flip the active flag on another tenant's row that
            # happened to share the same content hash.
            activate_stmt = sa_update(PolicyBundleRow).where(
                PolicyBundleRow.bundle_hash == bundle.bundle_hash
            )
            if req.tenant_id is None:
                activate_stmt = activate_stmt.where(PolicyBundleRow.tenant_id.is_(None))
            else:
                activate_stmt = activate_stmt.where(PolicyBundleRow.tenant_id == req.tenant_id)
            await session.execute(activate_stmt.values(active=True))
            await session.commit()
            policy_service.invalidate_bundle_cache()

    return {
        "bundle_hash": bundle.bundle_hash,
        "version": bundle.version,
        "rule_count": bundle.rule_count,
        "tenant_id": req.tenant_id,
        "active": req.activate,
    }


@router.get("/policy/bundles")
async def list_policy_bundles(tenant_id: str | None = Query(None)):
    """List policy bundles, optionally filtered by tenant. Returns
    metadata only — bundle YAML is fetched separately via /admin/policy/bundles/{hash}."""
    from sqlalchemy import select

    from server.db import engine as engine_module
    from server.db.tables import PolicyBundleRow

    async with engine_module.get_session_factory()() as session:
        stmt = select(PolicyBundleRow).order_by(PolicyBundleRow.created_at.desc())
        if tenant_id is not None:
            stmt = stmt.where(PolicyBundleRow.tenant_id == tenant_id)
        result = await session.execute(stmt)
        rows = result.scalars().all()
    return {
        "bundles": [
            {
                "bundle_hash": r.bundle_hash,
                "tenant_id": r.tenant_id,
                "active": r.active,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
    }


@router.get("/policy/bundles/{bundle_hash}")
async def get_policy_bundle(bundle_hash: str, tenant_id: str | None = Query(None)):
    """Fetch a bundle's full YAML content + parsed rule summary.

    `tenant_id` query param disambiguates when multiple tenants have
    uploaded the same YAML (post-#79). Omit for the global-scope row
    (`tenant_id IS NULL`); pass a specific id to scope to that tenant.
    When omitted and the hash exists in multiple scopes, returns the
    global row if one exists, else 404 with a hint to specify
    `?tenant_id=` for tenant-scoped rows."""
    from sqlalchemy import select

    from server.db import engine as engine_module
    from server.db.tables import PolicyBundleRow
    from server.services import policy as policy_service

    async with engine_module.get_session_factory()() as session:
        stmt = select(PolicyBundleRow).where(PolicyBundleRow.bundle_hash == bundle_hash)
        if tenant_id is not None:
            stmt = stmt.where(PolicyBundleRow.tenant_id == tenant_id)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
        else:
            # No tenant filter — prefer the global row (tenant_id IS NULL).
            # If absent, surface that the hash exists tenant-scoped and
            # the caller needs to disambiguate.
            global_stmt = stmt.where(PolicyBundleRow.tenant_id.is_(None))
            result = await session.execute(global_stmt)
            row = result.scalar_one_or_none()
            if row is None:
                any_stmt = stmt.limit(1)
                if (await session.execute(any_stmt)).scalar_one_or_none() is not None:
                    raise HTTPException(
                        status_code=404,
                        detail=(
                            f"bundle {bundle_hash} exists in tenant-scoped form; "
                            "pass `?tenant_id=<id>` to fetch it"
                        ),
                    )
    if row is None:
        raise HTTPException(status_code=404, detail="bundle not found")
    parsed = policy_service.load_bundle(row.yaml_content)
    return {
        "bundle_hash": row.bundle_hash,
        "tenant_id": row.tenant_id,
        "active": row.active,
        "created_at": row.created_at.isoformat(),
        "yaml_content": row.yaml_content,
        "metadata": parsed.metadata,
        "rules": [
            {
                "id": r.id,
                "description": r.description,
                "when": r.when,
                "action": r.action,
            }
            for r in parsed.rules
        ],
    }


class ActivateBundleRequest(BaseModel):
    """Body for POST /admin/policy/activate.

    Identifies the bundle to activate by `(tenant_id, bundle_hash)`.
    Post-#79 the same `bundle_hash` can exist in multiple scopes
    (global + N tenants), so the request must specify which one to
    flip — otherwise the activate could silently target the wrong
    row. Use `tenant_id: null` to activate the global-scope row.
    """

    bundle_hash: str
    tenant_id: str | None = None


@router.post("/policy/activate")
async def activate_policy_bundle(req: ActivateBundleRequest):
    """Mark a bundle as the active policy for its scope. Deactivates
    any other bundle in the same scope (global or tenant-specific)
    so the active-bundle resolver sees exactly one row.

    Returns 404 if no row matches `(tenant_id, bundle_hash)`."""
    from sqlalchemy import select, update as sa_update

    from server.db import engine as engine_module
    from server.db.tables import PolicyBundleRow
    from server.services import policy as policy_service

    async with engine_module.get_session_factory()() as session:
        target_stmt = select(PolicyBundleRow).where(PolicyBundleRow.bundle_hash == req.bundle_hash)
        if req.tenant_id is None:
            target_stmt = target_stmt.where(PolicyBundleRow.tenant_id.is_(None))
        else:
            target_stmt = target_stmt.where(PolicyBundleRow.tenant_id == req.tenant_id)
        result = await session.execute(target_stmt)
        target = result.scalar_one_or_none()
        if target is None:
            raise HTTPException(status_code=404, detail="bundle not found")

        # Deactivate every other bundle in the same scope (≠ THIS
        # bundle, same tenant). Using target.id rather than
        # bundle_hash here is the load-bearing precision: we want to
        # leave the same-hash-different-tenant rows untouched.
        scope_stmt = sa_update(PolicyBundleRow).where(PolicyBundleRow.id != target.id)
        if target.tenant_id is None:
            scope_stmt = scope_stmt.where(PolicyBundleRow.tenant_id.is_(None))
        else:
            scope_stmt = scope_stmt.where(PolicyBundleRow.tenant_id == target.tenant_id)
        scope_stmt = scope_stmt.values(active=False)
        await session.execute(scope_stmt)
        await session.execute(
            sa_update(PolicyBundleRow).where(PolicyBundleRow.id == target.id).values(active=True)
        )
        await session.commit()

    policy_service.invalidate_bundle_cache()
    return {"bundle_hash": req.bundle_hash, "active": True}


@router.post("/policy/reload")
async def reload_policy_cache():
    """Force the in-process active-bundle cache to drop. Use after
    setting `policy_bundles.active` manually outside the API (e.g.
    direct DB fix-ups during incident response)."""
    from server.services import policy as policy_service

    policy_service.invalidate_bundle_cache()
    return {"reloaded": True}


@router.get("/policy/active")
async def get_active_policy(tenant_id: str | None = Query(None)):
    """Return the currently active bundle for a tenant scope, or
    the global active bundle when `tenant_id` is omitted. Returns
    JSON `null` (with HTTP 200) when no bundle is active for the
    scope — "no policy uploaded yet" is the expected default state
    on a fresh install, not an error. Returning 404 here used to
    pollute every operator's browser console on first page load.
    Admin client treats `null` and the bundle object uniformly via
    a `ActivePolicyBundle | null` return type."""
    from server.db import engine as engine_module
    from server.services import policy as policy_service

    async with engine_module.get_session_factory()() as session:
        bundle = await policy_service.resolve_active_bundle(session, tenant_id)
    if bundle is None:
        return None
    return {
        "bundle_hash": bundle.bundle_hash,
        "version": bundle.version,
        "rule_count": bundle.rule_count,
        "metadata": bundle.metadata,
        "rules": [
            {
                "id": r.id,
                "description": r.description,
                "when": r.when,
                "action": r.action,
            }
            for r in bundle.rules
        ],
    }


# ─── Memory-labels admin shim (issue #50) ──────────────────────────────────
#
# Mirrors PATCH /v1/memories/{memory_id}/labels but lives under
# /admin so the admin-app proxy (which only forwards /admin/*) can
# reach it without widening its allowlist. The endpoint accepts an
# optional `tenant_id` query param so an operator can edit labels
# across tenants; the underlying canonicalization (dedup + lowercase
# + trim) and 32-label cap are the same as the /v1 endpoint.


class AdminSetMemoryLabelsRequest(BaseModel):
    sensitivity_labels: list[str]


@router.patch("/memories/{memory_id}/labels")
async def admin_set_memory_labels(
    memory_id: uuid.UUID,
    req: AdminSetMemoryLabelsRequest,
    tenant_id: str | None = Query(None),
):
    """Operator-facing memory-labels editor. Same canonicalization
    rules as /v1/memories/{id}/labels."""
    from sqlalchemy import select, update as sa_update

    from server.db import engine as engine_module
    from server.db.tables import MemoryRow

    normalized = sorted({lbl.strip().lower() for lbl in req.sensitivity_labels if lbl.strip()})
    if len(normalized) > 32:
        raise HTTPException(status_code=400, detail="too many labels (max 32)")

    async with engine_module.get_session_factory()() as session:
        stmt = select(MemoryRow).where(MemoryRow.id == memory_id)
        if tenant_id is not None:
            stmt = stmt.where(MemoryRow.tenant_id == tenant_id)
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="memory not found")
        update_stmt = (
            sa_update(MemoryRow)
            .where(MemoryRow.id == memory_id)
            .values(sensitivity_labels=normalized)
        )
        if tenant_id is not None:
            update_stmt = update_stmt.where(MemoryRow.tenant_id == tenant_id)
        await session.execute(update_stmt)
        await session.commit()
        result = await session.execute(stmt)
        row = result.scalar_one()

    return {
        "id": str(row.id),
        "kind": row.kind,
        "content": row.content,
        "summary": row.summary,
        "confidence": row.confidence,
        "status": row.status,
        "source_episode_ids": [str(s) for s in (row.source_episode_ids or [])],
        "valid_from": row.valid_from.isoformat(),
        "valid_to": row.valid_to.isoformat() if row.valid_to else None,
        "sensitivity_labels": list(row.sensitivity_labels or []),
        "created_at": row.created_at.isoformat(),
    }


# ─── Tenant configuration (issue #49 + #50) ─────────────────────────────────
#
# Read/write the `tenant_configs.config` JSONB document. The receipt
# emission gate and the sensitivity-label policy enforcement mode both
# live in this document. Without a write endpoint, `policy_mode:
# enforce` and `require_caller_identity: true` were unreachable via
# the API — compliance customers would have had to write the values
# via direct SQL. v2 of #50 fills that gap.
#
# The write path is PATCH-shaped: only the fields you supply are
# changed; other keys in the dict are preserved verbatim. This
# matters because future per-tenant knobs (rate-limit tiers,
# webhook URLs, etc.) will land in the same document, and we don't
# want each new admin endpoint to know the full key set.
#
# Optimistic concurrency: `version` on the row is bumped on every
# write. PATCH callers may pass `expected_version` to fail-fast on
# concurrent edits.


@router.get(
    "/tenants/{tenant_id}/config",
    response_model=TenantConfigResponse,
    summary="Read a tenant's configuration document",
)
async def get_tenant_config_endpoint(tenant_id: str):
    """Returns 200 with `config: {}`, `version: 0` when the tenant has
    no row yet — that's the default state on a fresh install, not an
    error. The two known top-level keys are `receipts` (emission
    policy from #49) and `policy_mode` / `require_caller_identity`
    (from #50)."""
    from sqlalchemy import select

    from server.db import engine as engine_module
    from server.db.tables import TenantConfigRow

    async with engine_module.get_session_factory()() as session:
        result = await session.execute(
            select(TenantConfigRow).where(TenantConfigRow.tenant_id == tenant_id)
        )
        row = result.scalar_one_or_none()

    if row is None:
        return TenantConfigResponse(tenant_id=tenant_id, config={}, version=0)
    return TenantConfigResponse(
        tenant_id=row.tenant_id,
        config=row.config or {},
        version=row.version,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@router.patch(
    "/tenants/{tenant_id}/config",
    response_model=TenantConfigResponse,
    summary="Update a tenant's configuration document",
)
async def patch_tenant_config(tenant_id: str, patch: TenantConfigPatch):
    """Partial update of `tenant_configs.config`. Only supplied
    fields are changed; other keys in the existing dict are
    preserved (forward-compat for future knobs).

    Race protection: pass `expected_version` from a prior GET to
    fail-fast (409) on a concurrent write. Omit if you're the only
    writer.

    Returns the post-write document so callers can re-render
    without an extra round-trip.
    """
    from sqlalchemy import select

    from server.db import engine as engine_module
    from server.db.tables import TenantConfigRow

    # Build the partial-update dict from supplied fields only. Pydantic
    # treats `None` as "field not set" via `exclude_none`, which is
    # exactly the PATCH semantic we want — supplying a literal `null`
    # for a known field is also "don't change it" (consistent with
    # the doc strings).
    incoming = patch.model_dump(exclude_none=True)
    # `expected_version` and `force_region_pin` are request-only
    # control fields, not config keys. Pop them before they leak
    # into the JSONB blob.
    expected_version = incoming.pop("expected_version", None)
    force_region_pin = incoming.pop("force_region_pin", False)

    # Region-pin safety check (v0.9 #161). Refuse a pin that would
    # immediately lock the tenant out of this deployment.
    region_value = incoming.get("region")
    if region_value is not None:
        from server.services.residency import validate_region_pin

        refusal = validate_region_pin(
            proposed_region=region_value,
            server_region=settings.region,
        )
        if refusal is not None and not force_region_pin:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "residency.invalid_pin",
                    "message": refusal,
                },
            )

    async with engine_module.get_session_factory()() as session:
        result = await session.execute(
            select(TenantConfigRow).where(TenantConfigRow.tenant_id == tenant_id)
        )
        existing = result.scalar_one_or_none()

        if existing is None:
            if expected_version is not None and expected_version != 0:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"version mismatch: expected {expected_version}, got 0 (no row exists yet)"
                    ),
                )
            new_config = dict(incoming)
            row = TenantConfigRow(
                tenant_id=tenant_id,
                config=new_config,
                version=1,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
        else:
            if expected_version is not None and expected_version != existing.version:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"version mismatch: expected {expected_version}, "
                        f"current is {existing.version}"
                    ),
                )
            merged = {**(existing.config or {}), **incoming}
            existing.config = merged
            existing.version = existing.version + 1
            await session.commit()
            await session.refresh(existing)
            row = existing

    logger.info(
        "tenant_config_updated",
        tenant_id=tenant_id,
        version=row.version,
        updated_keys=sorted(incoming.keys()),
    )
    return TenantConfigResponse(
        tenant_id=row.tenant_id,
        config=row.config or {},
        version=row.version,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


# ─── Production-readiness check ──────────────────────────────────────────


def _compute_readiness_issues(cfg: dict) -> list[dict]:
    """Build the issues list from a config snapshot.

    Pure function on a dict so the route can run it twice: once with
    live `settings.X` (what the process is doing now), once with live
    + DB overrides merged (what the next restart would look like).
    Diffing the two lets us mark `fix_staged=True` on issues that are
    queued but not yet active — which is the only reason an operator
    who just clicked Save sees the same warning persist.
    """
    issues: list[dict] = []

    # ─── Critical ──────────────────────────────────────────────────
    api_key = cfg.get("api_key") or ""
    if not api_key:
        issues.append({
            "id": "no_backend_auth",
            "severity": "critical",
            "title": "Backend authentication is disabled",
            "summary": (
                "Any client that can reach the backend can read and write "
                "memories, run migrations, and view receipts. Enable an API "
                "key before exposing this deployment outside localhost."
            ),
            "fix": {"kind": "wizard", "id": "enable-auth"},
        })
    elif api_key.startswith("dev-") or api_key in {
        "dev-local-placeholder",
        "change-me",
        "your-api-key-here",
    }:
        issues.append({
            "id": "dev_placeholder_api_key",
            "severity": "critical",
            "title": "Backend API key looks like a development placeholder",
            "summary": (
                "Rotate to a strong random key — the current value matches "
                "a known dev/example default and is trivial to guess."
            ),
            "fix": {"kind": "wizard", "id": "enable-auth"},
        })

    # ─── High ──────────────────────────────────────────────────────
    cors = cfg.get("cors_origins") or []
    if cors == ["*"] or "*" in cors:
        issues.append({
            "id": "permissive_cors",
            "severity": "high",
            "title": "CORS allows any browser origin",
            "summary": (
                "Replace `[\"*\"]` with the explicit list of origins that "
                "actually need browser access (e.g. your admin frontend, "
                "internal apps). A wildcard is a XSS amplifier."
            ),
            "fix": {"kind": "setting", "key": "cors_origins"},
        })

    if cfg.get("debug"):
        issues.append({
            "id": "debug_logging",
            "severity": "high",
            "title": "Debug logging is enabled",
            "summary": (
                "Verbose logs can leak prompts, episode payloads, and "
                "internal IDs. Turn off `debug` outside development."
            ),
            "fix": {"kind": "setting", "key": "debug"},
        })

    if cfg.get("embedding_provider") == "stub":
        issues.append({
            "id": "stub_embeddings",
            "severity": "high",
            "title": "Embeddings are using the deterministic stub",
            "summary": (
                "Stub embeddings produce hash-based vectors — semantic "
                "search will return nonsense. Switch to a real provider "
                "(LiteLLM with an API key) for production memory recall."
            ),
            "fix": {"kind": "setting", "key": "embedding_provider"},
        })

    # ─── Medium ────────────────────────────────────────────────────
    if cfg.get("rate_limit_rpm", 0) == 0:
        issues.append({
            "id": "no_rate_limit",
            "severity": "medium",
            "title": "No rate limiting configured",
            "summary": (
                "`rate_limit_rpm = 0` disables the limiter entirely. A "
                "noisy or buggy client can exhaust LLM quota or saturate "
                "Postgres. A 60–600 RPM cap per tenant is typical."
            ),
            "fix": {"kind": "setting", "key": "rate_limit_rpm"},
        })

    if not cfg.get("strict_schema"):
        issues.append({
            "id": "lax_schema_check",
            "severity": "medium",
            "title": "Strict schema check is off",
            "summary": (
                "The server boots on a stale schema and only warns. Turn "
                "this on in production so a missed migration fails the "
                "deploy loudly instead of running with a half-applied "
                "schema."
            ),
            "fix": {"kind": "setting", "key": "strict_schema"},
        })

    # ─── Low ───────────────────────────────────────────────────────
    if cfg.get("region") is None:
        issues.append({
            "id": "no_region",
            "severity": "low",
            "title": "Server region is not pinned",
            "summary": (
                "Set `region` if you operate >1 region — the residency "
                "layer compares per-tenant region pins against it. "
                "Single-region deployments can ignore this."
            ),
            "fix": {"kind": "setting", "key": "region"},
        })

    return issues


@router.get("/readiness-check")
async def admin_readiness_check():
    """Opinionated production-readiness scan.

    Returns `{id, severity, title, summary, fix, fix_staged?}` issues.

    `fix_staged` is true when the issue is still firing against the
    LIVE process state, but a DB-stored override (queued by the
    operator) would clear it on the next backend restart. The UI uses
    it to render "Pending restart" affordances instead of the naïve
    "fix did nothing" appearance — without it, an operator who just
    saved `debug=false` keeps seeing the warning and assumes the save
    was ignored.

    Severity scale: see `_compute_readiness_issues`.
    """
    from server.core.dynamic_settings import _load_global_overrides
    from server.db import engine as engine_module

    # Live snapshot — what the process is doing RIGHT NOW. Reads
    # straight off the pydantic Settings object so it reflects what
    # the application code actually sees.
    live_cfg = {
        "api_key": settings.api_key,
        "cors_origins": list(settings.cors_origins) if settings.cors_origins else [],
        "debug": settings.debug,
        "embedding_provider": settings.embedding_provider,
        "rate_limit_rpm": settings.rate_limit_rpm,
        "strict_schema": settings.strict_schema,
        "region": settings.region,
        "webhook_url": settings.webhook_url,
    }
    live_issues = _compute_readiness_issues(live_cfg)

    # DB overrides — what the next restart would apply on top of env.
    try:
        async with engine_module.get_session_factory()() as session:
            db_overrides = await _load_global_overrides(session)
    except Exception as exc:
        # If the override layer is unreachable (DB hiccup, migrations
        # behind), the readiness check should still respond — staged
        # detection just degrades to "no staged info". Beats 500ing
        # the dashboard.
        logger.warning("readiness_db_lookup_failed", error=str(exc)[:200])
        db_overrides = {}

    next_cfg = {**live_cfg, **db_overrides}
    next_issue_ids = {i["id"] for i in _compute_readiness_issues(next_cfg)}

    # An issue is "staged" iff it currently fires AND wouldn't fire on
    # next restart. That captures both kind:'setting' and kind:'wizard'
    # remediations uniformly — both flow through the DB override layer.
    for issue in live_issues:
        if issue["id"] not in next_issue_ids:
            issue["fix_staged"] = True

    return {"issues": live_issues}


# ─── Connection info ─────────────────────────────────────────────────────


@router.get("/connection-info")
async def admin_connection_info():
    """Lightweight self-introspection for the admin dashboard's
    "Connection" card.

    Returns whatever the backend can know about itself without doing
    work — version, schema head, bind config, region, and whether auth
    is configured. The admin-server's proxy URL (what hostname clients
    use to reach this backend) is NOT here because the backend doesn't
    know it; the admin server surfaces that via
    `/api/admin/proxy-info`.

    Reads are live: `auth_enabled` reflects the current effective
    `api_key` (DB override → env), so flipping the toggle in the
    Settings page shows up here immediately after a restart.
    """
    from server.app import get_app_version
    from server.services.migrations import EXPECTED_HEAD

    return {
        "version": get_app_version(),
        "schema_head": EXPECTED_HEAD,
        "host": settings.host,
        "port": settings.port,
        "region": settings.region,
        "auth_enabled": bool(settings.api_key),
        "compiler_type": settings.compiler_type,
        "embedding_provider": settings.embedding_provider,
        "require_tenant": settings.require_tenant,
    }


# ─── Restart endpoint ───────────────────────────────────────────────────
#
# Used by the admin UI's "Restart backend" button when there are pending
# non-hot-reloadable settings overrides. The pattern is exit-then-restart:
# we schedule a delayed `os._exit(0)` (so the HTTP response completes
# first) and rely on the container orchestrator's restart policy to
# bring the process back. Works on:
#
#   - Docker / Compose with `restart: unless-stopped` / `restart: always`
#   - Kubernetes (a Pod's container restarts on exit by default)
#   - systemd with `Restart=on-success` or `Restart=always`
#
# Without a restart policy the container just stops — that's a deploy
# misconfiguration, and the UI warns about it. The endpoint is gated
# by the admin router (which is auth-protected upstream of any caller).
# We deliberately use `os._exit(0)` rather than `sys.exit` so a stuck
# background task can't keep the process alive past the requested exit.


@router.post("/restart")
async def admin_restart_endpoint(delay_seconds: float = Query(2.0, ge=0.5, le=10.0)):
    """Exit the process so the orchestrator restarts it.

    Returns 202 immediately; the actual exit happens after `delay_seconds`
    so the response can flush. In-flight requests get the standard FastAPI
    shutdown — they're allowed to finish if the worker grants the grace
    period (~2s is enough for nearly all admin / readiness probes).

    The endpoint is destructive in the "kicks every connected client off"
    sense, but not in the "loses data" sense — the DB is the source of
    truth, and pending overrides land at boot via `apply_db_overrides_to_settings`.
    """
    import asyncio
    import os

    logger.warning("admin_restart_requested", delay_seconds=delay_seconds)

    async def _exit_after_delay() -> None:
        await asyncio.sleep(delay_seconds)
        logger.warning("admin_restart_exiting")
        # `_exit` skips finalizers / atexit handlers — that's deliberate.
        # Anything that NEEDS to flush before exit (audit rows, etc.) is
        # already committed by the time the request handler returns;
        # a stuck cleanup task should not be able to block the restart.
        os._exit(0)

    asyncio.create_task(_exit_after_delay())
    return {
        "ok": True,
        "message": "Backend will exit shortly; orchestrator restart policy brings it back.",
        "exit_in_seconds": delay_seconds,
    }


# ─── System settings (DB-backed override layer) ──────────────────────────
#
# These endpoints let the admin UI list, read, edit and revert the settings
# catalogued in `server/core/settings_catalogue.py`. Values resolve at read
# time with the precedence chain:
#
#     tenant_override → global_db → env (pydantic Settings) → default
#
# Every editable PATCH / DELETE / tenant write goes through
# `server.core.dynamic_settings`, which appends an audit row and invalidates
# the in-process resolution cache. Test probes are side-effect-free and live
# at /admin/settings/test.


class SettingPatchRequest(BaseModel):
    value: object
    changed_by: str | None = None
    note: str | None = None


class TenantSettingPatchRequest(BaseModel):
    value: object
    changed_by: str | None = None
    note: str | None = None


class SettingTestRequest(BaseModel):
    key: str
    value: object


@router.get("/settings")
async def list_settings_endpoint(
    tenant_id: str | None = Query(None, description="Optional tenant scope for tenant-overridable settings"),
):
    """Return the effective value for every catalogued setting.

    Source labels indicate where each value came from:

      * ``tenant_db`` — overridden in `tenant_settings` for the given tenant
      * ``global_db`` — overridden in `system_settings`
      * ``env`` — env / `.env` (or hardcoded pydantic default)

    Secrets are returned redacted (``•••<last-3>``). Use the audit log for
    history.
    """
    from server.core.dynamic_settings import get_effective_snapshot
    from server.db import engine as engine_module

    async with engine_module.get_session_factory()() as session:
        snapshot = await get_effective_snapshot(session, tenant_id=tenant_id)
    return {"settings": snapshot, "tenant_id": tenant_id}


@router.get("/settings/audit/log")
async def settings_audit_endpoint(
    key: str | None = Query(None, description="Filter to one setting key"),
    limit: int = Query(100, ge=1, le=1000),
):
    """Recent settings changes, newest first."""
    from server.core.dynamic_settings import list_audit
    from server.db import engine as engine_module

    async with engine_module.get_session_factory()() as session:
        rows = await list_audit(session, key=key, limit=limit)
    return {"entries": rows}


@router.post("/settings/test")
async def test_setting_endpoint(req: SettingTestRequest):
    """Side-effect-free probe of a candidate setting value.

    Used by the admin UI's "Test" button before committing a change to a
    risky setting (LLM creds, webhook URL). Does NOT persist.
    """
    from server.core.dynamic_settings import test_probe
    from server.db import engine as engine_module

    async with engine_module.get_session_factory()() as session:
        result = await test_probe(req.key, req.value, session=session)
    return {"ok": result.ok, "detail": result.detail, "extra": result.extra}


@router.patch("/settings/tenants/{tenant_id}/{key}")
async def patch_tenant_setting_endpoint(
    tenant_id: str, key: str, req: TenantSettingPatchRequest
):
    """UPSERT a per-tenant setting override."""
    from server.core.dynamic_settings import (
        SettingValidationError,
        apply_tenant_override,
    )
    from server.db import engine as engine_module

    async with engine_module.get_session_factory()() as session:
        try:
            persisted = await apply_tenant_override(
                tenant_id,
                key,
                req.value,
                session,
                changed_by=req.changed_by,
                note=req.note,
            )
        except SettingValidationError as exc:
            await session.rollback()
            raise HTTPException(status_code=400, detail={"code": "settings.invalid", "message": str(exc)})
        await session.commit()
    logger.info("tenant_setting_patched", tenant_id=tenant_id, key=key)
    return {"tenant_id": tenant_id, "key": key, "value": persisted, "source": "tenant_db"}


@router.delete("/settings/tenants/{tenant_id}/{key}")
async def delete_tenant_setting_endpoint(
    tenant_id: str,
    key: str,
    changed_by: str | None = Query(None),
    note: str | None = Query(None),
):
    """Drop a per-tenant setting override → falls back to global/env."""
    from server.core.dynamic_settings import (
        SettingValidationError,
        delete_tenant_override,
    )
    from server.db import engine as engine_module

    async with engine_module.get_session_factory()() as session:
        try:
            await delete_tenant_override(
                tenant_id, key, session, changed_by=changed_by, note=note
            )
        except SettingValidationError as exc:
            await session.rollback()
            raise HTTPException(status_code=400, detail={"code": "settings.invalid", "message": str(exc)})
        await session.commit()
    logger.info("tenant_setting_reverted", tenant_id=tenant_id, key=key)
    return {"tenant_id": tenant_id, "key": key, "reverted_to": "global_or_env"}


# These two MUST come last among `/settings/*` routes — FastAPI matches in
# definition order, and `/settings/{key}` would otherwise shadow the static
# routes above (`/settings/test`, `/settings/audit/log`, `/settings/tenants/...`).
@router.get("/settings/{key}")
async def get_setting_endpoint(
    key: str,
    tenant_id: str | None = Query(None),
):
    """Return one setting's effective value + metadata."""
    from server.core.dynamic_settings import get_effective_snapshot
    from server.core.settings_catalogue import get_spec
    from server.db import engine as engine_module

    if get_spec(key) is None:
        raise HTTPException(status_code=404, detail=f"unknown setting key: {key!r}")

    async with engine_module.get_session_factory()() as session:
        snapshot = await get_effective_snapshot(session, tenant_id=tenant_id)
    return snapshot[key]


@router.patch("/settings/{key}")
async def patch_setting_endpoint(key: str, req: SettingPatchRequest):
    """UPSERT a global override + append an audit row."""
    from server.core.dynamic_settings import (
        SettingValidationError,
        apply_global_override,
    )
    from server.db import engine as engine_module

    async with engine_module.get_session_factory()() as session:
        try:
            persisted = await apply_global_override(
                key,
                req.value,
                session,
                changed_by=req.changed_by,
                note=req.note,
            )
        except SettingValidationError as exc:
            await session.rollback()
            raise HTTPException(status_code=400, detail={"code": "settings.invalid", "message": str(exc)})
        await session.commit()
    logger.info("setting_patched", key=key, changed_by=req.changed_by)
    return {"key": key, "value": persisted, "source": "global_db"}


@router.post("/settings/apply")
async def apply_settings_endpoint():
    """Apply pending hot-reloadable DB overrides to the live process in-place.

    Mutates env_settings for every hot_reloadable=True setting that has a
    global DB override, then refreshes _applied_values so the pending_restart
    flag clears on the next snapshot fetch — no orchestrator restart needed.

    Settings that are not hot_reloadable (cors_origins, api_key, embedding
    singletons, etc.) are intentionally skipped: they still require a full
    process restart, and their pending_restart flag stays true.

    Returns {applied: N, keys: [...]} confirming which settings were updated.
    """
    from server.core.dynamic_settings import apply_hot_reloadable_overrides

    applied = await apply_hot_reloadable_overrides()
    logger.info("settings_applied_hot_reload", count=len(applied), keys=sorted(applied.keys()))
    return {"applied": len(applied), "keys": sorted(applied.keys())}


@router.delete("/settings/{key}")
async def delete_setting_endpoint(
    key: str,
    changed_by: str | None = Query(None),
    note: str | None = Query(None),
):
    """Drop a global override → setting reverts to env value."""
    from server.core.dynamic_settings import (
        SettingValidationError,
        delete_global_override,
    )
    from server.db import engine as engine_module

    async with engine_module.get_session_factory()() as session:
        try:
            await delete_global_override(
                key, session, changed_by=changed_by, note=note
            )
        except SettingValidationError as exc:
            await session.rollback()
            raise HTTPException(status_code=400, detail={"code": "settings.invalid", "message": str(exc)})
        await session.commit()
    logger.info("setting_reverted", key=key, changed_by=changed_by)
    return {"key": key, "reverted_to": "env"}
