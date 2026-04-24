"""FastAPI application factory."""

from fastapi import FastAPI

from server.api import context, episodes, memories, subjects, timeline
from server.core.config import settings
from server.core.logging import setup_logging


def create_app() -> FastAPI:
    setup_logging(debug=settings.debug)

    app = FastAPI(
        title="Statewave",
        description="Memory OS — trusted context runtime for AI agents",
        version="0.1.0",
    )

    app.include_router(episodes.router)
    app.include_router(memories.router)
    app.include_router(context.router)
    app.include_router(timeline.router)
    app.include_router(subjects.router)

    @app.get("/healthz", tags=["ops"])
    async def healthz():
        return {"status": "ok"}

    return app


app = create_app()
