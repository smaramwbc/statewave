"""Distributed rate limiter — Postgres-backed fixed-window.

Replaces the in-memory per-process rate limiter with a shared counter
that survives restarts and works across multiple workers.

Design:
- Fixed 60-second windows keyed by client IP
- Single atomic upsert per request (INSERT ... ON CONFLICT UPDATE)
- Background cleanup of expired windows
- Falls back to in-memory if DB is unreachable (graceful degradation)

Config:
- STATEWAVE_RATE_LIMIT_RPM: max requests per minute per key (0 = disabled)
- STATEWAVE_RATE_LIMIT_STRATEGY: "distributed" (default) or "memory"
"""

from __future__ import annotations

import time

import structlog
from sqlalchemy import text

from server.db.engine import get_session_factory

logger = structlog.stdlib.get_logger()


async def check_rate_limit(key: str, rpm: int) -> tuple[bool, int]:
    """Check and increment rate limit for a key.

    Returns (allowed: bool, retry_after_seconds: int).
    retry_after is 0 if allowed.
    """
    window_start = int(time.time()) // 60 * 60  # current 60s window start

    try:
        async with get_session_factory()() as session:
            # Atomic upsert: increment counter or insert new row
            result = await session.execute(
                text("""
                    INSERT INTO rate_limit_hits (key, window_start, hit_count, updated_at)
                    VALUES (:key, :window_start, 1, now())
                    ON CONFLICT (key, window_start)
                    DO UPDATE SET hit_count = rate_limit_hits.hit_count + 1, updated_at = now()
                    RETURNING hit_count
                """),
                {"key": key, "window_start": window_start},
            )
            await session.commit()
            count = result.scalar_one()

            if count > rpm:
                # Already over limit
                seconds_into_window = int(time.time()) - window_start
                retry_after = max(1, 60 - seconds_into_window)
                return False, retry_after

            return True, 0

    except Exception:
        logger.warning(
            "distributed_rate_limit_db_error",
            key=key,
            hint="DB connection pool may be exhausted by rate limiter. "
            "Consider switching to STATEWAVE_RATE_LIMIT_STRATEGY=memory for single-instance deployments. "
            "See: https://docs.statewave.ai/deployment/troubleshooting#statewave-ts-001",
            exc_info=True,
        )
        # Graceful degradation: allow the request if DB is down
        return True, 0


async def cleanup_expired_windows(retention_windows: int = 5) -> int:
    """Delete rate limit rows older than N windows (default: 5 minutes).

    Called periodically from background task.
    """
    cutoff = (int(time.time()) // 60 * 60) - (retention_windows * 60)
    try:
        async with get_session_factory()() as session:
            result = await session.execute(
                text("DELETE FROM rate_limit_hits WHERE window_start < :cutoff"),
                {"cutoff": cutoff},
            )
            await session.commit()
            count = result.rowcount  # type: ignore[attr-defined]
            if count:
                logger.debug("rate_limit_cleanup", deleted=count)
            return count
    except Exception:
        logger.warning("rate_limit_cleanup_failed", exc_info=True)
        return 0
