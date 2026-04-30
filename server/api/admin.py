"""Admin endpoints — operator introspection and advanced bootstrap capabilities."""

from __future__ import annotations

import uuid
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from server.core.config import settings
from server.services import webhooks

router = APIRouter(prefix="/admin", tags=["admin"])


# ─── Response Models ─────────────────────────────────────────────────────────


class SubjectListItem(BaseModel):
    subject_id: str
    tenant_id: str | None
    episode_count: int
    memory_count: int
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


class MemoryListResponse(BaseModel):
    memories: list[MemoryListItem]
    total: int
    limit: int
    offset: int


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
        async with engine_module.async_session_factory() as session:
            episodes = await session.scalar(select(func.count()).select_from(EpisodeRow)) or 0
            memories = await session.scalar(select(func.count()).select_from(MemoryRow)) or 0
            subjects = (
                await session.scalar(select(func.count(func.distinct(EpisodeRow.subject_id)))) or 0
            )
            return {"episodes": episodes, "memories": memories, "subjects": subjects}

    async def _get_job_stats():
        async with engine_module.async_session_factory() as session:
            rows = await session.execute(
                select(CompileJobRow.status, func.count()).group_by(CompileJobRow.status)
            )
            stats = {row[0]: row[1] for row in rows}
            return stats

    async def _get_health_distribution():
        """Get subject health score distribution from cache table."""
        from server.db.tables import Base

        # Check if health cache table exists in metadata
        if "subject_health_cache" not in Base.metadata.tables:
            return None
        try:
            from sqlalchemy import text

            async with engine_module.async_session_factory() as session:
                rows = await session.execute(
                    text(
                        "SELECT last_state, COUNT(*) FROM subject_health_cache GROUP BY last_state"
                    )
                )
                return {row[0]: row[1] for row in rows}
        except Exception:
            return None

    async def _get_readiness():
        from server.db.engine import engine

        async with engine.connect() as conn:
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

    async with engine_module.async_session_factory() as session:
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
                func.count().label("memory_count"),
            )
            .group_by(MemoryRow.subject_id)
            .subquery()
        )

        # Open sessions per subject
        open_sessions = (
            select(
                ResolutionRow.subject_id,
                func.count().label("open_count"),
            )
            .where(ResolutionRow.status == "open")
            .group_by(ResolutionRow.subject_id)
            .subquery()
        )

        # Health cache
        health_cache = select(SubjectHealthCacheRow).subquery()

        # Main query joining all
        stmt = select(
            ep_stats.c.subject_id,
            ep_stats.c.tenant_id,
            ep_stats.c.episode_count,
            func.coalesce(mem_stats.c.memory_count, 0).label("memory_count"),
            ep_stats.c.last_episode_at,
            health_cache.c.last_state.label("health_state"),
            health_cache.c.last_score.label("health_score"),
            func.coalesce(open_sessions.c.open_count, 0).label("open_sessions"),
        ).select_from(
            ep_stats.outerjoin(mem_stats, ep_stats.c.subject_id == mem_stats.c.subject_id)
            .outerjoin(open_sessions, ep_stats.c.subject_id == open_sessions.c.subject_id)
            .outerjoin(health_cache, ep_stats.c.subject_id == health_cache.c.subject_id)
        )

        # Exclude internal subjects
        stmt = stmt.where(ep_stats.c.subject_id.not_like("_snapshot/%"))
        stmt = stmt.where(ep_stats.c.subject_id.not_like("_bootstrap_tmp/%"))

        # Apply filters
        if search:
            stmt = stmt.where(ep_stats.c.subject_id.ilike(f"%{search}%"))

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

        subjects = [
            SubjectListItem(
                subject_id=row.subject_id,
                tenant_id=row.tenant_id,
                episode_count=row.episode_count,
                memory_count=row.memory_count,
                last_episode_at=row.last_episode_at.isoformat() if row.last_episode_at else None,
                health_state=row.health_state,
                health_score=row.health_score,
                open_sessions=row.open_sessions,
            )
            for row in rows
        ]

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

    async with engine_module.async_session_factory() as session:
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


@router.get("/subjects/{subject_id}/memories", response_model=MemoryListResponse)
async def list_subject_memories(
    subject_id: str,
    tenant_id: str | None = Query(None, description="Filter by tenant"),
    status: Literal["active", "superseded", "all"] = Query("all", description="Filter by status"),
    kind: str | None = Query(None, description="Filter by memory kind"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List memories for a subject with filtering and pagination."""
    from sqlalchemy import func, select

    from server.db import engine as engine_module
    from server.db.tables import MemoryRow

    async with engine_module.async_session_factory() as session:
        base = select(MemoryRow).where(MemoryRow.subject_id == subject_id)
        if tenant_id:
            base = base.where(MemoryRow.tenant_id == tenant_id)
        if status != "all":
            base = base.where(MemoryRow.status == status)
        if kind:
            base = base.where(MemoryRow.kind == kind)

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
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List episodes for a subject with filtering and pagination."""
    from sqlalchemy import func, select

    from server.db import engine as engine_module
    from server.db.tables import EpisodeRow

    async with engine_module.async_session_factory() as session:
        base = select(EpisodeRow).where(EpisodeRow.subject_id == subject_id)
        if tenant_id:
            base = base.where(EpisodeRow.tenant_id == tenant_id)
        if session_id:
            base = base.where(EpisodeRow.session_id == session_id)
        if type:
            base = base.where(EpisodeRow.type == type)

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

    async with engine_module.async_session_factory() as session:

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
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List compile jobs for operator debugging.

    Returns recent jobs ordered by creation time (newest first).
    """
    from server.services.compile_jobs_durable import list_jobs

    jobs = await list_jobs(
        status=status, subject_id=subject_id, tenant_id=tenant_id, limit=limit, offset=offset
    )
    return {"jobs": jobs, "limit": limit, "offset": offset}


# ─── Tenant Audit ───


@router.get("/tenant-audit")
async def tenant_audit():
    """Report rows with NULL tenant_id — helps operators backfill after enabling tenants."""
    from sqlalchemy import func, select

    from server.db.engine import async_session_factory
    from server.db.tables import CompileJobRow, EpisodeRow, MemoryRow

    async with async_session_factory() as session:
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


@router.get("/webhooks/stats")
async def webhook_stats():
    """Aggregate webhook delivery statistics."""
    return await webhooks.get_delivery_stats()


@router.get("/webhooks/{event_id}")
async def webhook_event_status(event_id: uuid.UUID):
    """Get delivery status of a specific webhook event."""
    result = await webhooks.get_event_status(event_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Webhook event not found")
    return result


# ─── Subject Snapshots (advanced bootstrap/admin, feature-flagged) ───


class RestoreSnapshotRequest(BaseModel):
    target_subject_id: str


class RestoreByNameRequest(BaseModel):
    name: str
    target_subject_id: str
    version: Optional[int] = None


@router.get("/snapshots")
async def list_snapshots_endpoint():
    """List available subject snapshots."""
    _require_snapshots()
    from server.services.snapshots import list_snapshots

    return {"snapshots": await list_snapshots()}


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
