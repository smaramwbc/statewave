"""Integration test fixtures — real FastAPI app against a real test database.

Requires a running Postgres with a `statewave_test` database:
    createdb statewave_test

The test suite creates/drops all tables per session and uses unique subject IDs
per test for isolation, so tests can run in parallel without conflicts.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from server.db.tables import Base

# ---------------------------------------------------------------------------
# Test database URL — mirrors production but targets `statewave_test`
# ---------------------------------------------------------------------------
TEST_DATABASE_URL = "postgresql+asyncpg://statewave:statewave@localhost:5432/statewave_test"


# ---------------------------------------------------------------------------
# Session-scoped event loop for session-scoped async fixtures
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Session-scoped: create tables once, drop after all tests
# ---------------------------------------------------------------------------

# Engine and session factory are created inside the session fixture to ensure
# they're bound to the test event loop (pytest-asyncio 1.x creates loops lazily).
_engine = None
_session_factory = None


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session", autouse=True)
async def _setup_database():
    """Create pgvector extension + all tables, then tear down after the session."""
    global _engine, _session_factory
    # NullPool ensures each request gets a fresh connection — avoids asyncpg
    # "another operation is in progress" errors from connection reuse.
    _engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)
    _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)

    async with _engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await _engine.dispose()


# ---------------------------------------------------------------------------
# Per-test: override the app's DB session dependency
# ---------------------------------------------------------------------------


async def _override_get_session():
    async with _session_factory() as session:
        yield session


@pytest.fixture
async def client():
    """Async HTTP client wired to the real app with test DB."""
    from server.app import create_app
    from server.db.engine import get_session, set_engine_for_testing

    app = create_app()
    app.dependency_overrides[get_session] = _override_get_session

    # Patch the module-level engine and session factory so that readyz
    # and background tasks use the test engine (same event loop).
    prev_engine, prev_factory = set_engine_for_testing(_engine, _session_factory)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    set_engine_for_testing(prev_engine, prev_factory)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def subject_id() -> str:
    """Unique subject ID per test — guarantees isolation."""
    return f"test-{uuid.uuid4().hex[:12]}"
