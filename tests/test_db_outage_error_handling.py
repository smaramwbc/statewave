"""Regression test for issue #354.

A database outage (e.g. Postgres unreachable) raises `sqlalchemy.exc.OperationalError`
from inside a route handler. `register_exception_handlers`'s catch-all `Exception`
handler caught it and returned 500/`internal_error`, indistinguishable from a genuine
endpoint bug and not a signal most retry policies back off on. `/readyz` already
reports 503 for the identical condition (`ReadinessResult.http_status`), so the two
surfaces disagreed about what a DB outage means.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import OperationalError

from server.app import create_app
from server.db.engine import dispose_engine, get_engine, get_session

pytestmark = pytest.mark.asyncio


async def _raise_operational_error():
    raise OperationalError("SELECT 1", {}, Exception("connection refused"))
    yield  # pragma: no cover - unreachable, keeps this an async generator


async def test_db_outage_returns_503_not_500():
    app = create_app()
    app.dependency_overrides[get_session] = _raise_operational_error

    # No raise_app_exceptions=False here: the OperationalError handler lives in
    # ExceptionMiddleware, which sends its response WITHOUT re-raising — so this
    # also pins that a DB outage produces no propagated exception at all.
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/v1/subjects/does-not-matter/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "service_unavailable"
    assert "Retry-After" in resp.headers


async def test_non_db_exceptions_still_return_500():
    """The catch-all handler must still cover unrelated bugs: only
    `OperationalError` gets the 503 treatment."""
    app = create_app()

    async def _raise_value_error():
        raise ValueError("not a database problem")
        yield  # pragma: no cover

    app.dependency_overrides[get_session] = _raise_value_error

    # The Exception catch-all lives in ServerErrorMiddleware, which sends its
    # 500 response and then re-raises the original exception (so real servers
    # log it); raise_app_exceptions=False lets the test client see the response
    # instead of the propagated ValueError.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/v1/subjects/does-not-matter/health")

    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "internal_error"


# ---------------------------------------------------------------------------
# The asyncpg dialect does not raise OperationalError for real outages on its
# own: connect-time failures escape as raw OSError subclasses and a mid-query
# disconnect surfaces as a bare DBAPIError. server/db/engine.py normalizes
# both; these tests pin that normalization end to end (issue #354's actual
# repro is the app-level test below — no dependency override, a real engine
# pointed at a dead port).
# ---------------------------------------------------------------------------


async def test_engine_wraps_connect_failure_as_operational_error(monkeypatch):
    from sqlalchemy import text
    from sqlalchemy.exc import OperationalError

    from server.core.config import settings

    monkeypatch.setattr(
        settings, "database_url", "postgresql+asyncpg://u:p@127.0.0.1:1/statewave"
    )
    await dispose_engine()
    try:
        engine = get_engine()
        with pytest.raises(OperationalError) as excinfo:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        assert isinstance(excinfo.value.orig, OSError)
    finally:
        await dispose_engine()


async def test_db_outage_returns_503_without_dependency_override(monkeypatch):
    """Issue #354's reproduction: the database is unreachable, a normal request
    comes in, and the real get_session/engine path (no override) must yield the
    503 payload — this fails if OperationalError never surfaces from the engine."""
    from server.core.config import settings

    monkeypatch.setattr(
        settings, "database_url", "postgresql+asyncpg://u:p@127.0.0.1:1/statewave"
    )
    await dispose_engine()
    try:
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/v1/subjects/does-not-matter/health")

        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "service_unavailable"
        assert "Retry-After" in resp.headers
    finally:
        await dispose_engine()
