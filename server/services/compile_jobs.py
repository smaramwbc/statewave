"""Background compilation job manager.

Two layers, one source of truth:

  * `compile_jobs_durable` is the source of truth — every job is a row
    in Postgres, and DB errors propagate. Jobs survive process restarts
    and are visible to other replicas.
  * The `_jobs` dict in this module is a process-local cache so that a
    `get_job_durable(id)` from the same process that submitted the job
    skips the DB round-trip. It is NOT a fallback for a missing or
    unhealthy database — if Postgres is down the operator sees a 5xx,
    not a "submitted" job that no other process can see.

Public API used by the app: `submit_job_durable`, `get_job_durable`,
`mark_running_durable`, `mark_completed_durable`, `mark_failed_durable`.

The sync, in-memory primitives at the bottom of this file (`submit_job`,
`get_job`, `mark_running`, `mark_completed`, `mark_failed`) are kept
only because unit tests use them to drive the `_jobs` cache without
mocking SQLAlchemy. They are NOT used anywhere on the request path.
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


# Process-local cache of jobs submitted from this process — speeds up
# `get_job_durable` polling without a DB round-trip per call. NOT a
# fallback store: a missing entry here is just a cache miss, never a
# substitute for the durable row.
_jobs: dict[str, CompileJob] = {}
_JOB_TTL_SECONDS = 300


# ---------------------------------------------------------------------------
# Durable (Postgres-backed) interface — async
# These are the primary interface used by the async compile path.
# A DB failure raises; the caller (memories.py) renders the error to
# the client instead of returning a job that nothing else will ever see.
# ---------------------------------------------------------------------------


async def submit_job_durable(subject_id: str, tenant_id: str | None = None) -> CompileJob:
    """Submit a Postgres-durable compile job.

    Persists the row first; only seeds the process-local cache after the
    DB write succeeds, so a half-submitted job never appears as "found"
    to a follow-up `get_job_durable` in the same process.
    """
    from server.services.compile_jobs_durable import submit_job as _durable_submit

    job = await _durable_submit(subject_id, tenant_id)
    _jobs[job.id] = job
    _cleanup_old_jobs()
    return job


async def get_job_durable(job_id: str) -> CompileJob | None:
    """Get a job by id. Process-local cache first, then Postgres.

    Cache hits skip the DB round-trip; cache misses fall through to
    Postgres. DB errors propagate so a transient outage isn't reported
    as "job not found".
    """
    if job_id in _jobs:
        return _jobs[job_id]
    from server.services.compile_jobs_durable import get_job as _durable_get

    return await _durable_get(job_id)


async def mark_running_durable(job_id: str) -> None:
    """Mark the job running in Postgres + the process-local cache.

    The DB write happens first: if it fails, we don't quietly mark the
    cache as running while the persisted row stays pending. Subsequent
    cache mutation only happens on a successful commit.
    """
    from server.services.compile_jobs_durable import mark_running as _durable_mark

    await _durable_mark(job_id)
    mark_running(job_id)


async def mark_completed_durable(
    job_id: str, memories_created: int, memories: list[dict[str, Any]]
) -> None:
    """Mark the job completed in Postgres + the process-local cache."""
    from server.services.compile_jobs_durable import mark_completed as _durable_mark

    await _durable_mark(job_id, memories_created, memories)
    mark_completed(job_id, memories_created, memories)


async def mark_failed_durable(job_id: str, error: str) -> None:
    """Mark the job failed in Postgres + the process-local cache."""
    from server.services.compile_jobs_durable import mark_failed as _durable_mark

    await _durable_mark(job_id, error)
    mark_failed(job_id, error)


async def update_progress_durable(job_id: str, memories_created: int) -> None:
    """Bump the durable `memories_created` count mid-job.

    Used by the async compile drain loop (issue #134) so operators see
    real progress while a large subject is processed batch by batch.
    Status stays `running` — only the count is updated.
    """
    from server.services.compile_jobs_durable import update_progress as _durable_update

    await _durable_update(job_id, memories_created)
    job = _jobs.get(job_id)
    if job:
        job.memories_created = memories_created


# ---------------------------------------------------------------------------
# In-memory primitives
#
# Drive the `_jobs` cache directly. Used by tests to construct fixtures
# without mocking SQLAlchemy, and by the durable wrappers above to keep
# the cache in sync after a successful DB write. Not on any request path.
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
