"""Distributed rate limiter — Postgres-backed fixed-window.

Uses a **dedicated connection pool** (pool_size=2, pool_timeout=2s) that is
completely isolated from the main application pool. This prevents rate-limit
queries from ever starving real API queries — even under heavy load, the rate
limiter either responds instantly or fails open (allows the request).

Design:
- Fixed 60-second windows keyed by client IP
- Single atomic upsert per request (INSERT ... ON CONFLICT UPDATE)
- Background cleanup of expired windows
- Falls back to allowing requests if DB is unreachable (graceful degradation)
- Separate pool = zero impact on main API even if rate limiter pool is exhausted

Config:
- STATEWAVE_RATE_LIMIT_RPM: max requests per minute per key (0 = disabled)
- STATEWAVE_RATE_LIMIT_STRATEGY: "memory" (default) or "distributed"
"""

from __future__ import annotations

import time

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from server.core.config import settings

logger = structlog.stdlib.get_logger()

# Dedicated engine for rate limiting — isolated from main app pool.
# Small pool (2 connections), short timeout (2s), no pre-ping (speed over reliability).
_ratelimit_engine = None
_ratelimit_session_factory = None


def _get_ratelimit_session_factory() -> async_sessionmaker[AsyncSession]:
    """Lazy-init a dedicated session factory for rate limiting."""
    global _ratelimit_engine, _ratelimit_session_factory
    if _ratelimit_session_factory is None:
        _ratelimit_engine = create_async_engine(
            settings.database_url,
            pool_size=2,
            max_overflow=1,
            pool_timeout=2,  # fail fast — never block the request pipeline
            pool_pre_ping=False,
            echo=False,
        )
        _ratelimit_session_factory = async_sessionmaker(
            _ratelimit_engine, class_=AsyncSession, expire_on_commit=False
        )
    return _ratelimit_session_factory


async def check_rate_limit(key: str, rpm: int) -> tuple[bool, int]:
    """Check and increment rate limit for a key.

    Returns (allowed: bool, retry_after_seconds: int).
    retry_after is 0 if allowed.
    """
    window_start = int(time.time()) // 60 * 60  # current 60s window start

    try:
        factory = _get_ratelimit_session_factory()
        async with factory() as session:
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
            hint="Rate limiter DB pool exhausted or unreachable. "
            "Request allowed (fail-open). No impact on main API pool.",
            docs="https://github.com/smaramwbc/statewave-docs/blob/main/deployment/troubleshooting.md#statewave-ts-001",
        )
        # Graceful degradation: allow the request if DB is down
        return True, 0


async def cleanup_expired_windows(retention_windows: int = 5) -> int:
    """Delete rate limit rows older than N windows (default: 5 minutes).

    Called periodically from background task.
    """
    cutoff = (int(time.time()) // 60 * 60) - (retention_windows * 60)
    try:
        factory = _get_ratelimit_session_factory()
        async with factory() as session:
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
