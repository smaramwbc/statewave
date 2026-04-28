"""Durable compile job manager — Postgres-backed.

Jobs survive server restarts. Replaces the in-memory store from v0.4.
Falls back gracefully: if DB write fails, job still runs (just untracked).

Public API unchanged:
- submit_job(subject_id) → CompileJob
- get_job(job_id) → CompileJob | None
- mark_running(job_id)
- mark_completed(job_id, memories_created, memories)
- mark_failed(job_id, error)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog
from sqlalchemy import delete, select, update

from server.db.engine import async_session_factory
from server.db.tables import CompileJobRow

logger = structlog.stdlib.get_logger()


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


@dataclass
class CompileJob:
    id: str
    subject_id: str
    status: JobStatus = JobStatus.pending
    memories_created: int = 0
    memories: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    completed_at: float | None = None


def _row_to_job(row: CompileJobRow) -> CompileJob:
    return CompileJob(
        id=row.id,
        subject_id=row.subject_id,
        status=JobStatus(row.status),
        memories_created=row.memories_created,
        memories=[],  # Not stored in DB to keep table lean
        error=row.error,
        created_at=row.created_at.timestamp() if row.created_at else 0,
        completed_at=row.completed_at.timestamp() if row.completed_at else None,
    )


async def submit_job(subject_id: str, tenant_id: str | None = None) -> CompileJob:
    """Create a durable compile job and persist to Postgres."""
    job_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc)

    try:
        async with async_session_factory() as session:
            row = CompileJobRow(
                id=job_id,
                subject_id=subject_id,
                tenant_id=tenant_id,
                status="pending",
                memories_created=0,
                created_at=now,
            )
            session.add(row)
            await session.commit()
    except Exception:
        logger.warning("compile_job_persist_failed", job_id=job_id, exc_info=True)

    return CompileJob(id=job_id, subject_id=subject_id, created_at=now.timestamp())


async def get_job(job_id: str) -> CompileJob | None:
    """Retrieve a job by ID from Postgres."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(select(CompileJobRow).where(CompileJobRow.id == job_id))
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _row_to_job(row)
    except Exception:
        logger.warning("compile_job_get_failed", job_id=job_id, exc_info=True)
        return None


async def mark_running(job_id: str) -> None:
    """Mark job as running."""
    try:
        async with async_session_factory() as session:
            await session.execute(
                update(CompileJobRow)
                .where(CompileJobRow.id == job_id)
                .values(status="running", started_at=datetime.now(timezone.utc))
            )
            await session.commit()
    except Exception:
        logger.warning("compile_job_mark_running_failed", job_id=job_id, exc_info=True)


async def mark_completed(
    job_id: str, memories_created: int, memories: list[dict[str, Any]]
) -> None:
    """Mark job as completed with result count."""
    try:
        async with async_session_factory() as session:
            await session.execute(
                update(CompileJobRow)
                .where(CompileJobRow.id == job_id)
                .values(
                    status="completed",
                    memories_created=memories_created,
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
    except Exception:
        logger.warning("compile_job_mark_completed_failed", job_id=job_id, exc_info=True)


async def mark_failed(job_id: str, error: str) -> None:
    """Mark job as failed with error message."""
    try:
        async with async_session_factory() as session:
            await session.execute(
                update(CompileJobRow)
                .where(CompileJobRow.id == job_id)
                .values(
                    status="failed",
                    error=error,
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
    except Exception:
        logger.warning("compile_job_mark_failed_failed", job_id=job_id, exc_info=True)


async def list_jobs(
    status: str | None = None,
    subject_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List compile jobs for admin introspection."""
    try:
        async with async_session_factory() as session:
            stmt = select(CompileJobRow).order_by(CompileJobRow.created_at.desc())
            if status:
                stmt = stmt.where(CompileJobRow.status == status)
            if subject_id:
                stmt = stmt.where(CompileJobRow.subject_id == subject_id)
            stmt = stmt.offset(offset).limit(limit)

            result = await session.execute(stmt)
            rows = result.scalars().all()

            return [
                {
                    "job_id": row.id,
                    "subject_id": row.subject_id,
                    "tenant_id": row.tenant_id,
                    "status": row.status,
                    "memories_created": row.memories_created,
                    "error": row.error,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "started_at": row.started_at.isoformat() if row.started_at else None,
                    "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                }
                for row in rows
            ]
    except Exception:
        logger.warning("compile_jobs_list_failed", exc_info=True)
        return []


async def cleanup_old_jobs(retention_hours: int = 168) -> int:
    """Delete completed/failed jobs older than retention window.

    Default: 7 days (168 hours). Returns count of deleted rows.
    """
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                delete(CompileJobRow)
                .where(CompileJobRow.status.in_(["completed", "failed"]))
                .where(CompileJobRow.created_at < cutoff)
            )
            await session.commit()
            count = result.rowcount  # type: ignore[attr-defined]
            if count:
                logger.info("compile_jobs_cleaned", deleted=count)
            return count
    except Exception:
        logger.warning("compile_jobs_cleanup_failed", exc_info=True)
        return 0
