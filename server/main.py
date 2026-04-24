"""Entrypoint for running the server directly."""

import uvicorn

from server.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "server.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
