"""FastAPI application factory."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.api import context, episodes, memories, subjects, timeline
from server.core.config import settings
from server.core.errors import register_exception_handlers
from server.core.logging import setup_logging
from server.core.middleware import RequestIDMiddleware
from server.core.auth import APIKeyMiddleware
from server.core.ratelimit import RateLimitMiddleware
from server.core.tenant import TenantMiddleware
from server.core.tracing import setup_tracing

logger = structlog.stdlib.get_logger()


async def _cleanup_loop():
    """Periodically clean up stale ephemeral demo subjects and old compile jobs."""
    from server.services.snapshots import cleanup_ephemeral_subjects
    from server.services.compile_jobs_durable import cleanup_old_jobs
    from server.services.ratelimit import cleanup_expired_windows

    while True:
        await asyncio.sleep(3600)  # every hour
        try:
            count = await cleanup_ephemeral_subjects()
            if count:
                logger.info("scheduled_cleanup_done", subjects_cleaned=count)
        except Exception as exc:
            logger.warning("scheduled_cleanup_error", error=str(exc))

        # Compile job retention
        if settings.compile_job_retention_hours > 0:
            try:
                deleted = await cleanup_old_jobs(settings.compile_job_retention_hours)
                if deleted:
                    logger.info("compile_jobs_retention_done", deleted=deleted)
            except Exception as exc:
                logger.warning("compile_jobs_retention_error", error=str(exc))

        # Rate limit window cleanup
        if settings.rate_limit_rpm > 0 and settings.rate_limit_strategy == "distributed":
            try:
                await cleanup_expired_windows()
            except Exception as exc:
                logger.warning("rate_limit_cleanup_error", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure tracing (no-op if opentelemetry not installed)
    setup_tracing()
    # Configure webhooks
    from server.services import webhooks

    webhooks.configure(url=settings.webhook_url, timeout=settings.webhook_timeout)
    await webhooks.start_worker()

    # Start background cleanup (snapshots + compile job retention + rate limit)
    cleanup_task = None
    needs_cleanup = (
        settings.enable_snapshots
        or settings.compile_job_retention_hours > 0
        or (settings.rate_limit_rpm > 0 and settings.rate_limit_strategy == "distributed")
    )
    if needs_cleanup:
        cleanup_task = asyncio.create_task(_cleanup_loop())

    # Schema compatibility check
    try:
        from server.services.migrations import check_migration_status

        migration_status = await check_migration_status()
        if migration_status.error:
            logger.error("schema_check_error", error=migration_status.error)
        elif not migration_status.is_compatible:
            msg = migration_status.summary
            logger.warning("schema_mismatch", detail=msg, pending=migration_status.pending_count)
            if settings.strict_schema:
                raise RuntimeError(f"Schema mismatch (STATEWAVE_STRICT_SCHEMA=1): {msg}")
    except RuntimeError:
        raise
    except Exception as exc:
        logger.warning("schema_check_skipped", reason=str(exc)[:200])

    logger.info("app_startup", version="0.6.1", debug=settings.debug)
    yield
    if cleanup_task:
        cleanup_task.cancel()
    await webhooks.stop_worker()
    from server.db.engine import dispose_engine

    await dispose_engine()
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    setup_logging(debug=settings.debug)

    app = FastAPI(
        title="Statewave",
        summary="Memory OS — trusted context runtime for AI agents",
        description=(
            "Statewave lets AI applications record raw interaction history, "
            "compile durable typed memories, retrieve ranked context within "
            "token budgets, and govern data by subject."
        ),
        version="0.5.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # -- Middleware -----------------------------------------------------------
    # Starlette executes add_middleware in REVERSE order (last-added = outermost).
    # Desired execution: CORS → RequestID → Auth → RateLimit → Tenant → App
    # So we register innermost first:
    app.add_middleware(
        TenantMiddleware, header=settings.tenant_header, require=settings.require_tenant
    )
    app.add_middleware(
        RateLimitMiddleware, rpm=settings.rate_limit_rpm, strategy=settings.rate_limit_strategy
    )
    app.add_middleware(APIKeyMiddleware, api_key=settings.api_key)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Exception handlers --------------------------------------------------
    register_exception_handlers(app)

    # -- Routes --------------------------------------------------------------
    app.include_router(episodes.router)
    app.include_router(memories.router)
    app.include_router(context.router)
    app.include_router(timeline.router)
    app.include_router(subjects.router)

    from server.api.resolutions import router as resolutions_router

    app.include_router(resolutions_router)

    from server.api.handoff import router as handoff_router

    app.include_router(handoff_router)

    from server.api.admin import router as admin_router

    app.include_router(admin_router)

    from server.api.health import router as health_router

    app.include_router(health_router)

    from server.api.sla import router as sla_router

    app.include_router(sla_router)

    # -- Ops endpoints -------------------------------------------------------
    @app.get("/healthz", tags=["ops"], summary="Liveness check")
    @app.get("/health", tags=["ops"], summary="Liveness check (alias)", include_in_schema=False)
    async def healthz():
        """Returns 200 if the process is alive."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"], summary="Deep readiness check")
    @app.get("/ready", tags=["ops"], summary="Readiness check (alias)", include_in_schema=False)
    async def readyz():
        """Deep readiness check: DB, queue health, LLM reachability."""
        from fastapi.responses import JSONResponse

        from server.db.engine import get_engine
        from server.services.readiness import run_readiness_checks

        async with get_engine().connect() as conn:
            result = await run_readiness_checks(conn)

        body = {
            "status": result.status,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    **({"detail": c.detail} if c.detail else {}),
                    **({"latency_ms": c.latency_ms} if c.latency_ms else {}),
                }
                for c in result.checks
            ],
        }
        return JSONResponse(content=body, status_code=result.http_status)

    @app.get("/ops/migrations", tags=["ops"], summary="Migration status")
    async def migration_status():
        """Return current schema revision, expected head, and pending migrations."""
        from server.services.migrations import check_migration_status

        status = await check_migration_status()
        return {
            "current_revision": status.current_revision,
            "expected_head": status.expected_head,
            "is_compatible": status.is_compatible,
            "pending_count": status.pending_count,
            "pending_revisions": status.pending_revisions,
            "error": status.error,
            "summary": status.summary,
        }

    return app


app = create_app()
