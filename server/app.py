"""FastAPI application factory."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

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
    """Periodically clean up stale ephemeral demo subjects."""
    from server.services.snapshots import cleanup_ephemeral_subjects

    while True:
        await asyncio.sleep(3600)  # every hour
        try:
            count = await cleanup_ephemeral_subjects()
            if count:
                logger.info("scheduled_cleanup_done", subjects_cleaned=count)
        except Exception as exc:
            logger.warning("scheduled_cleanup_error", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Propagate STATEWAVE_OPENAI_API_KEY to OPENAI_API_KEY for LiteLLM
    import os
    if settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

    # Configure tracing (no-op if opentelemetry not installed)
    setup_tracing()
    # Configure webhooks
    from server.services import webhooks
    webhooks.configure(url=settings.webhook_url, timeout=settings.webhook_timeout)
    await webhooks.start_worker()

    # Start background cleanup (only if snapshots enabled)
    cleanup_task = None
    if settings.enable_snapshots:
        cleanup_task = asyncio.create_task(_cleanup_loop())

    logger.info("app_startup", version="0.4.3", debug=settings.debug)
    yield
    if cleanup_task:
        cleanup_task.cancel()
    await webhooks.stop_worker()
    from server.db.engine import engine
    await engine.dispose()
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
        version="0.4.3",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # -- Middleware -----------------------------------------------------------
    # Starlette executes add_middleware in REVERSE order (last-added = outermost).
    # Desired execution: CORS → RequestID → Auth → RateLimit → Tenant → App
    # So we register innermost first:
    app.add_middleware(TenantMiddleware, header=settings.tenant_header, require=settings.require_tenant)
    app.add_middleware(RateLimitMiddleware, rpm=settings.rate_limit_rpm)
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

    from server.api.admin import router as admin_router
    app.include_router(admin_router)

    # -- Ops endpoints -------------------------------------------------------
    @app.get("/healthz", tags=["ops"], summary="Liveness check")
    @app.get("/health", tags=["ops"], summary="Liveness check (alias)", include_in_schema=False)
    async def healthz():
        """Returns 200 if the process is alive."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"], summary="Readiness check")
    @app.get("/ready", tags=["ops"], summary="Readiness check (alias)", include_in_schema=False)
    async def readyz():
        """Returns 200 if the app can reach the database."""
        from server.db.engine import engine

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ready"}

    return app


app = create_app()
