"""FastAPI application factory."""

from __future__ import annotations

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

logger = structlog.stdlib.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure webhooks
    from server.services import webhooks
    webhooks.configure(url=settings.webhook_url, timeout=settings.webhook_timeout)
    logger.info("app_startup", version="0.3.0", debug=settings.debug)
    yield
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
        version="0.3.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # -- Middleware (outermost first) ----------------------------------------
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(TenantMiddleware, header=settings.tenant_header, require=settings.require_tenant)
    app.add_middleware(APIKeyMiddleware, api_key=settings.api_key)
    app.add_middleware(RateLimitMiddleware, rpm=settings.rate_limit_rpm)
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

    # -- Ops endpoints -------------------------------------------------------
    @app.get("/healthz", tags=["ops"], summary="Liveness check")
    async def healthz():
        """Returns 200 if the process is alive."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"], summary="Readiness check")
    async def readyz():
        """Returns 200 if the app can reach the database."""
        from server.db.engine import engine

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "ready"}

    return app


app = create_app()
