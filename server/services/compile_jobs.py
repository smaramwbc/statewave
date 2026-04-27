"""Background compilation job manager.

Provides async compilation with job tracking:
- submit_job(): start compilation in background, return job_id
- get_job(): check job status and results
- cleanup_old_jobs(): prune stale entries

Jobs are stored in-memory (suitable for single-instance deployments).
For multi-instance, replace with Redis or DB-backed store.
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


# In-memory store — simple and sufficient for single-instance
_jobs: dict[str, CompileJob] = {}
_JOB_TTL_SECONDS = 300  # Jobs expire after 5 minutes


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
