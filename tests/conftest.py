"""Shared test fixtures."""

from __future__ import annotations

# Pin the test environment to match CI BEFORE any `server.*` import constructs
# the Settings singleton. CI runs with no `.env`; a developer's local `.env`
# (which may carry a live LiteLLM key) would otherwise leak into tests and
# diverge them from CI — e.g. the settings-redaction tests would see the real
# key instead of the value they monkeypatch. `setdefault` lets a developer
# still point at a dedicated test env file via `STATEWAVE_ENV_FILE=.env.test`.
import os

os.environ.setdefault("STATEWAVE_ENV_FILE", "")

from typing import Generator  # noqa: E402

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from starlette.routing import BaseRoute  # noqa: E402

from server.app import create_app  # noqa: E402


def iter_routes(app_or_router) -> Generator[BaseRoute, None, None]:
    """Recursively yield all leaf routes, handling _IncludedRouter wrappers.

    Starlette ≥1.3 stores included routers as _IncludedRouter objects with an
    `original_router` attribute rather than flattening APIRoute entries into
    app.routes directly.  Older versions flatten them.  This walker handles
    both shapes so route-registration tests stay version-agnostic.
    """
    for route in getattr(app_or_router, "routes", []):
        if hasattr(route, "path") and isinstance(route.path, str):
            yield route
        # Starlette <1.3: sub-routers expose their routes via .routes
        if hasattr(route, "routes"):
            yield from iter_routes(route)
        # Starlette ≥1.3: _IncludedRouter wraps the original APIRouter
        if hasattr(route, "original_router"):
            yield from iter_routes(route.original_router)

@pytest.fixture(autouse=True)
def _no_api_key_in_tests(monkeypatch):
    """Unit tests assume open access unless they explicitly set settings.api_key.
    A developer .env with STATEWAVE_API_KEY would otherwise make every HTTP
    test hit the auth middleware and return 401.
    """
    from server.core.config import settings
    monkeypatch.setattr(settings, "api_key", None)


@pytest.fixture
async def client():
    """Async test client that talks to the app without needing a real DB.

    For integration tests that need Postgres, skip or use a test-scoped DB.
    """
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    
    # Dispose engine after test to ensure clean state for next test
    from server.db.engine import dispose_engine
    await dispose_engine()
