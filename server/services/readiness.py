"""Deep readiness checks for /readyz endpoint.

Checks:
- Database connectivity (SELECT 1)
- Job queue health (no stuck jobs older than threshold)
- LLM reachability (optional, only if configured)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from server.core.config import settings

logger = structlog.get_logger()

# A job claimed more than this long ago without completing is "stuck"
_STUCK_JOB_THRESHOLD = timedelta(minutes=30)


@dataclass
class CheckResult:
    name: str
    status: str  # "ok" | "degraded" | "fail"
    detail: str = ""
    latency_ms: float = 0.0


@dataclass
class ReadinessResult:
    status: str = "ready"  # "ready" | "degraded" | "not_ready"
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def http_status(self) -> int:
        if self.status == "not_ready":
            return 503
        return 200


async def _check_db(conn: AsyncConnection) -> CheckResult:
    """Verify database responds to a simple query."""
    import time

    start = time.perf_counter()
    try:
        await conn.execute(text("SELECT 1"))
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(name="database", status="ok", latency_ms=round(latency, 1))
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            name="database", status="fail", detail=str(exc)[:200], latency_ms=round(latency, 1)
        )


async def _check_queue(conn: AsyncConnection) -> CheckResult:
    """Check for stuck compilation jobs."""
    try:
        threshold = datetime.now(timezone.utc) - _STUCK_JOB_THRESHOLD
        result = await conn.execute(
            text(
                "SELECT COUNT(*) FROM compile_jobs "
                "WHERE status = 'running' AND started_at < :threshold"
            ),
            {"threshold": threshold},
        )
        stuck_count = result.scalar() or 0
        if stuck_count > 0:
            return CheckResult(
                name="queue",
                status="degraded",
                detail=f"{stuck_count} stuck job(s) older than {int(_STUCK_JOB_THRESHOLD.total_seconds() // 60)}m",
            )
        return CheckResult(name="queue", status="ok")
    except Exception as exc:
        # Table might not exist yet (pre-migration) — treat as degraded, not fail
        return CheckResult(name="queue", status="degraded", detail=str(exc)[:200])


async def _check_llm() -> CheckResult:
    """Verify LLM provider is reachable (lightweight completion call)."""
    if not settings.openai_api_key:
        return CheckResult(name="llm", status="ok", detail="not configured (skip)")

    import time

    start = time.perf_counter()
    try:
        import litellm

        # Use a minimal completion to verify connectivity without burning tokens
        await asyncio.wait_for(
            litellm.acompletion(
                model=settings.llm_compiler_model or "gpt-3.5-turbo",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            ),
            timeout=10.0,
        )
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(name="llm", status="ok", latency_ms=round(latency, 1))
    except asyncio.TimeoutError:
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            name="llm", status="degraded", detail="timeout (>10s)", latency_ms=round(latency, 1)
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            name="llm", status="degraded", detail=str(exc)[:200], latency_ms=round(latency, 1)
        )


async def run_readiness_checks(conn: AsyncConnection) -> ReadinessResult:
    """Run all readiness checks and return aggregated result."""
    db_check, queue_check = await asyncio.gather(
        _check_db(conn),
        _check_queue(conn),
    )

    # LLM check is independent of the DB connection
    llm_check = await _check_llm()

    checks = [db_check, queue_check, llm_check]

    # Determine overall status
    if any(c.status == "fail" for c in checks):
        status = "not_ready"
    elif any(c.status == "degraded" for c in checks):
        status = "degraded"
    else:
        status = "ready"

    result = ReadinessResult(status=status, checks=checks)
    if status != "ready":
        logger.warning("readiness_degraded", status=status, checks=[c.__dict__ for c in checks])

    return result
