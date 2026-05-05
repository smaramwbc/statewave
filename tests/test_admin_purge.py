"""Unit + middleware-level tests for the admin purge endpoints.

Covers:
- Service-layer validation (empty filter, non-terminal status) without a DB.
- Route registration on the real app (DELETE methods present).
- Auth gate: when STATEWAVE_API_KEY is set, the purge endpoints are 401/403
  unprotected, mirroring the rest of /admin/*.

Real-DB happy-path / tenant scoping / no-cascade behavior lives in
tests/integration/test_admin_purge.py — those need Postgres, while the
checks here run on every CI invocation regardless of DB availability.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from server.services import webhooks as webhooks_service
from server.services import compile_jobs_durable as jobs_service


# ─── Service-layer validation (mocked DB) ────────────────────────────────────


class TestPurgeJobsValidation:
    @pytest.mark.anyio
    async def test_empty_filter_raises(self):
        with pytest.raises(ValueError, match="at least one filter is required"):
            await jobs_service.purge_jobs()

    @pytest.mark.anyio
    async def test_non_terminal_status_rejected(self):
        for bad in ("pending", "running", "anything"):
            with pytest.raises(ValueError, match="status must be one of"):
                await jobs_service.purge_jobs(status=bad)

    @pytest.mark.anyio
    async def test_terminal_statuses_constant_matches_implementation(self):
        # The frontend's TERMINAL_STATUSES (jobs page) hardcodes the same
        # pair. If this constant ever drifts, the UI either over-restricts
        # the operator or sends a status the backend rejects.
        assert jobs_service.TERMINAL_JOB_STATUSES == ("completed", "failed")

    @pytest.mark.anyio
    async def test_subject_filter_alone_is_accepted(self):
        # Subject-only filters skip the status check but still must hit the
        # DB; a mocked session lets us prove the validation didn't reject.
        mock_result = AsyncMock()
        mock_result.rowcount = 0
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            "server.services.compile_jobs_durable.get_session_factory",
            return_value=lambda: mock_session,
        ):
            count = await jobs_service.purge_jobs(subject_id="sub-1")
        assert count == 0


class TestPurgeWebhooksValidation:
    @pytest.mark.anyio
    async def test_empty_filter_raises(self):
        with pytest.raises(ValueError, match="at least one filter is required"):
            await webhooks_service.purge_events()

    @pytest.mark.anyio
    async def test_non_terminal_status_rejected(self):
        for bad in ("pending", "anything"):
            with pytest.raises(ValueError, match="status must be one of"):
                await webhooks_service.purge_events(status=bad)

    @pytest.mark.anyio
    async def test_terminal_statuses_constant_matches_implementation(self):
        assert webhooks_service.TERMINAL_WEBHOOK_STATUSES == (
            "delivered",
            "dead_letter",
        )

    @pytest.mark.anyio
    async def test_event_type_filter_alone_is_accepted(self):
        mock_result = AsyncMock()
        mock_result.rowcount = 0
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            "server.services.webhooks.get_session_factory",
            return_value=lambda: mock_session,
        ):
            count = await webhooks_service.purge_events(event_type="episode.created")
        assert count == 0


# ─── Route registration (no DB) ──────────────────────────────────────────────


def _methods_for_path(path: str) -> set[str]:
    """Aggregate all HTTP methods registered at `path` across the app.

    FastAPI registers `@router.get` and `@router.delete` on the same path
    as two separate APIRoute entries, so we have to walk the full list
    rather than stop at the first hit.
    """
    from server.app import app

    methods: set[str] = set()
    for route in app.routes:
        if hasattr(route, "path") and route.path == path:
            methods.update(route.methods)
    return methods


class TestRouteRegistration:
    def test_delete_jobs_registered(self):
        assert "DELETE" in _methods_for_path("/admin/jobs")

    def test_delete_webhooks_registered(self):
        assert "DELETE" in _methods_for_path("/admin/webhooks")


# ─── Endpoint validation (router → service) ──────────────────────────────────


class TestEndpointValidation:
    """Hit the real router with an in-process client. The service is
    mocked, so we're verifying the FastAPI route correctly translates the
    `ValueError` raised by the service into a 400 with the same message.
    """

    @pytest.fixture
    async def client_no_auth(self):
        from server.app import create_app

        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
        from server.db.engine import dispose_engine

        await dispose_engine()

    @pytest.mark.anyio
    async def test_delete_jobs_empty_filter_returns_400(self, client_no_auth):
        with patch(
            "server.services.compile_jobs_durable.purge_jobs",
            new=AsyncMock(side_effect=ValueError("at least one filter is required")),
        ):
            r = await client_no_auth.request("DELETE", "/admin/jobs")
        assert r.status_code == 400
        assert "at least one filter" in r.json()["error"]["message"]

    @pytest.mark.anyio
    async def test_delete_jobs_non_terminal_returns_400(self, client_no_auth):
        with patch(
            "server.services.compile_jobs_durable.purge_jobs",
            new=AsyncMock(side_effect=ValueError("status must be one of (...)")),
        ):
            r = await client_no_auth.request(
                "DELETE", "/admin/jobs", params={"status": "running"}
            )
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_delete_webhooks_empty_filter_returns_400(self, client_no_auth):
        with patch(
            "server.services.webhooks.purge_events",
            new=AsyncMock(side_effect=ValueError("at least one filter is required")),
        ):
            r = await client_no_auth.request("DELETE", "/admin/webhooks")
        assert r.status_code == 400
        assert "at least one filter" in r.json()["error"]["message"]

    @pytest.mark.anyio
    async def test_delete_webhooks_non_terminal_returns_400(self, client_no_auth):
        with patch(
            "server.services.webhooks.purge_events",
            new=AsyncMock(side_effect=ValueError("status must be one of (...)")),
        ):
            r = await client_no_auth.request(
                "DELETE", "/admin/webhooks", params={"status": "pending"}
            )
        assert r.status_code == 400

    @pytest.mark.anyio
    async def test_delete_jobs_happy_path_passes_filter(self, client_no_auth):
        spy = AsyncMock(return_value=3)
        with patch("server.services.compile_jobs_durable.purge_jobs", new=spy):
            r = await client_no_auth.request(
                "DELETE",
                "/admin/jobs",
                params={"status": "completed", "tenant_id": "acme"},
            )
        assert r.status_code == 200
        assert r.json() == {"deleted": 3}
        spy.assert_awaited_once_with(
            status="completed", subject_id=None, tenant_id="acme"
        )

    @pytest.mark.anyio
    async def test_delete_webhooks_happy_path_passes_filter(self, client_no_auth):
        spy = AsyncMock(return_value=7)
        with patch("server.services.webhooks.purge_events", new=spy):
            r = await client_no_auth.request(
                "DELETE",
                "/admin/webhooks",
                params={"status": "dead_letter", "event_type": "episode.created"},
            )
        assert r.status_code == 200
        assert r.json() == {"deleted": 7}
        spy.assert_awaited_once_with(
            status="dead_letter", event_type="episode.created", tenant_id=None
        )


# ─── Auth gate ────────────────────────────────────────────────────────────────


class TestAuthGate:
    """When STATEWAVE_API_KEY is set, every /admin/* route — including the
    new DELETE endpoints — must require it. We construct an app with the
    real middleware enabled and verify the gate rejects the request before
    the service is ever called.
    """

    @pytest.fixture
    async def client_with_key(self, monkeypatch):
        from server.core.config import settings
        from server.app import create_app

        monkeypatch.setattr(settings, "api_key", "test-secret-key")
        app = create_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
        from server.db.engine import dispose_engine

        await dispose_engine()

    @pytest.mark.anyio
    async def test_delete_jobs_requires_api_key(self, client_with_key):
        r = await client_with_key.request(
            "DELETE", "/admin/jobs", params={"status": "completed"}
        )
        assert r.status_code == 401
        assert "missing_api_key" in r.json()["error"]["code"]

    @pytest.mark.anyio
    async def test_delete_jobs_rejects_wrong_key(self, client_with_key):
        r = await client_with_key.request(
            "DELETE",
            "/admin/jobs",
            params={"status": "completed"},
            headers={"X-API-Key": "wrong"},
        )
        assert r.status_code == 403

    @pytest.mark.anyio
    async def test_delete_webhooks_requires_api_key(self, client_with_key):
        r = await client_with_key.request(
            "DELETE", "/admin/webhooks", params={"status": "delivered"}
        )
        assert r.status_code == 401

    @pytest.mark.anyio
    async def test_delete_webhooks_rejects_wrong_key(self, client_with_key):
        r = await client_with_key.request(
            "DELETE",
            "/admin/webhooks",
            params={"status": "delivered"},
            headers={"X-API-Key": "wrong"},
        )
        assert r.status_code == 403
