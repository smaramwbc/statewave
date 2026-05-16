"""Shared test fixtures."""

import pytest
from httpx import ASGITransport, AsyncClient

from server.app import create_app

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
