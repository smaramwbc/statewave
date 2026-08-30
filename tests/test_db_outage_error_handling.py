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
from server.db.engine import get_session

pytestmark = pytest.mark.asyncio


async def _raise_operational_error():
    raise OperationalError("SELECT 1", {}, Exception("connection refused"))
    yield  # pragma: no cover - unreachable, keeps this an async generator


async def test_db_outage_returns_503_not_500():
    app = create_app()
    app.dependency_overrides[get_session] = _raise_operational_error

    # Starlette's ServerErrorMiddleware sends the handler's response and then
    # always re-raises the original exception (so a real server can log it):
    # raise_app_exceptions=False is what lets the test client see the response
    # this handler wrote instead of the propagated OperationalError.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
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

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/v1/subjects/does-not-matter/health")

    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "internal_error"
