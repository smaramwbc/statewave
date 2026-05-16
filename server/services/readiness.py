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
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncConnection

from server.core.config import settings
from server.services.llm import litellm_api_key_configured, llm_requires_api_key

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
    """Verify LLM provider is reachable (lightweight completion call).

    Routes through the central LLM adapter — see server.services.llm.
    No direct litellm import here; the adapter owns the SDK choice.
    """
    if llm_requires_api_key() and not litellm_api_key_configured():
        return CheckResult(
            name="llm", status="ok", detail="STATEWAVE_LITELLM_API_KEY is not set"
        )
    # Local Ollama (no key required) falls through to the real probe below —
    # if the local Ollama server is down, that *should* surface here.

    import time

    from server.services import llm as llm_adapter

    start = time.perf_counter()
    try:
        # aping issues a one-token completion through the adapter, which
        # applies the configured timeout/retry/error-mapping uniformly.
        await llm_adapter.aping(timeout=10.0)
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(name="llm", status="ok", latency_ms=round(latency, 1))
    except llm_adapter.LLMTimeoutError:
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            name="llm", status="degraded", detail="timeout (>10s)", latency_ms=round(latency, 1)
        )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        return CheckResult(
            name="llm", status="degraded", detail=str(exc)[:200], latency_ms=round(latency, 1)
        )


def database_url_status() -> tuple[str, str | None]:
    """Classify the configured DB URL *before* a connection is attempted.

    Lets `/readyz` tell a first-time deployer the difference between
    "you never configured a database" and "the database is down" — the
    most common early-setup confusion (#66). `get_engine()` builds the
    engine from `settings.database_url`, so we classify exactly that.

    Returns ``(status, detail)`` where status is ``"ok"`` (detail None),
    ``"missing"``, or ``"unparseable"``.
    """
    url = (settings.database_url or "").strip()
    if not url:
        return "missing", "DATABASE_URL is not set"
    try:
        make_url(url)
    except Exception as exc:
        # SQLAlchemy's parse error is generic and does not echo the URL,
        # so this is safe to surface (no credential leak).
        return "unparseable", f"DATABASE_URL is set but couldn't be parsed: {str(exc)[:160]}"
    return "ok", None


async def run_readiness_checks(
    conn: AsyncConnection | None,
    *,
    db_unavailable_detail: str | None = None,
) -> ReadinessResult:
    """Run all readiness checks and return aggregated result.

    `conn` is None when the DB connection could not be established at all
    (URL not set / unparseable / Postgres unreachable). In that case the
    DB check fails with `db_unavailable_detail`, the queue check is
    degraded (it needs a connection), and the LLM check still runs since
    it is independent of the database.
    """
    if conn is None:
        db_check = CheckResult(
            name="database",
            status="fail",
            detail=db_unavailable_detail or "database connection unavailable",
        )
        queue_check = CheckResult(
            name="queue", status="degraded", detail="skipped (no database connection)"
        )
    else:
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
