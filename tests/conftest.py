"""Shared test fixtures."""

import pytest
from httpx import ASGITransport, AsyncClient

from server.app import app


@pytest.fixture
async def client():
    """Async test client that talks to the app without needing a real DB.

    For integration tests that need Postgres, skip or use a test-scoped DB.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
