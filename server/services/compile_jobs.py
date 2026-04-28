"""Background compilation job manager.

Provides async compilation with job tracking:
- submit_job(): start compilation in background, return job_id
- get_job(): check job status and results
- cleanup_old_jobs(): prune stale entries

v0.5: Durable mode (Postgres-backed) via compile_jobs_durable.
Falls back to in-memory when DB is not available.
Jobs survive restarts in durable mode.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

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
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


# In-memory fallback store
_jobs: dict[str, CompileJob] = {}
_JOB_TTL_SECONDS = 300


# ---------------------------------------------------------------------------
# Durable (Postgres-backed) interface — async
# These are the primary interface used by the async compile path.
# ---------------------------------------------------------------------------


async def submit_job_durable(subject_id: str, tenant_id: str | None = None) -> CompileJob:
    """Submit job with Postgres durability."""
    try:
        from server.services.compile_jobs_durable import submit_job as _durable_submit

        job = await _durable_submit(subject_id, tenant_id)
        # Also track in memory for fast access during this process lifetime
        _jobs[job.id] = job
        return job
    except Exception:
        logger.warning("durable_submit_fallback_to_memory", exc_info=True)
        return submit_job(subject_id)


async def get_job_durable(job_id: str) -> CompileJob | None:
    """Get job from Postgres (falls back to in-memory)."""
    # Check in-memory first (fast path for same-process)
    if job_id in _jobs:
        return _jobs[job_id]
    try:
        from server.services.compile_jobs_durable import get_job as _durable_get

        return await _durable_get(job_id)
    except Exception:
        return None


async def mark_running_durable(job_id: str) -> None:
    """Mark running in both stores."""
    mark_running(job_id)
    try:
        from server.services.compile_jobs_durable import mark_running as _durable_mark

        await _durable_mark(job_id)
    except Exception:
        pass


async def mark_completed_durable(
    job_id: str, memories_created: int, memories: list[dict[str, Any]]
) -> None:
    """Mark completed in both stores."""
    mark_completed(job_id, memories_created, memories)
    try:
        from server.services.compile_jobs_durable import mark_completed as _durable_mark

        await _durable_mark(job_id, memories_created, memories)
    except Exception:
        pass


async def mark_failed_durable(job_id: str, error: str) -> None:
    """Mark failed in both stores."""
    mark_failed(job_id, error)
    try:
        from server.services.compile_jobs_durable import mark_failed as _durable_mark

        await _durable_mark(job_id, error)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# In-memory interface (legacy, still used as fast cache)
# ---------------------------------------------------------------------------


def submit_job(subject_id: str) -> CompileJob:
    """Create a new compile job and return it. Caller must start the task."""
    job_id = str(uuid.uuid4())[:8]
    job = CompileJob(id=job_id, subject_id=subject_id)
    _jobs[job_id] = job
    _cleanup_old_jobs()
    return job


def get_job(job_id: str) -> CompileJob | None:
    """Retrieve a job by ID."""
    return _jobs.get(job_id)


def mark_running(job_id: str) -> None:
    job = _jobs.get(job_id)
    if job:
        job.status = JobStatus.running


def mark_completed(job_id: str, memories_created: int, memories: list[dict[str, Any]]) -> None:
    job = _jobs.get(job_id)
    if job:
        job.status = JobStatus.completed
        job.memories_created = memories_created
        job.memories = memories
        job.completed_at = time.time()


def mark_failed(job_id: str, error: str) -> None:
    job = _jobs.get(job_id)
    if job:
        job.status = JobStatus.failed
        job.error = error
        job.completed_at = time.time()


def _cleanup_old_jobs() -> None:
    """Remove expired jobs to prevent memory leaks."""
    now = time.time()
    expired = [jid for jid, j in _jobs.items() if now - j.created_at > _JOB_TTL_SECONDS]
    for jid in expired:
        del _jobs[jid]
