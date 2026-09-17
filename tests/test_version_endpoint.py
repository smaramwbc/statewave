"""Tests for the public /v1/version discovery endpoint (#178)."""

from importlib.metadata import PackageNotFoundError

from httpx import ASGITransport, AsyncClient

from server.app import create_app


async def test_version_endpoint_shape(client):
    resp = await client.get("/v1/version")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("version"), str) and body["version"]
    assert body["api_contract"] == "v1"


async def test_version_endpoint_is_public_when_api_key_set(monkeypatch):
    """With an API key configured, /v1/version needs no auth header — while a
    guarded /v1 path still 401s. Confirms the public-path exemption works."""
    from server.core.config import settings

    monkeypatch.setattr(settings, "api_key", "secret-key")
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        public = await ac.get("/v1/version")
    assert public.status_code == 200
    assert public.json()["api_contract"] == "v1"

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        guarded = await ac.get("/v1/subjects")  # no X-API-Key
    assert guarded.status_code == 401


async def test_version_endpoint_matches_openapi_info_version(client):
    """The discovery endpoint and OpenAPI `info.version` describe the same
    running server, so they must never answer differently (#396)."""
    endpoint = await client.get("/v1/version")
    openapi = await client.get("/openapi.json")
    assert endpoint.json()["version"] == openapi.json()["info"]["version"]


async def test_not_installed_sentinel_is_shared_by_both_surfaces(monkeypatch):
    """Not-installed is one condition, so it gets one sentinel.

    The two surfaces used to disagree — `0.0.0-dev` in `info.version`,
    `unknown` from the endpoint — and a client comparing the server against
    a required floor warned on the first and ignored the second.
    """
    import server.app as app_module

    def _not_installed(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(app_module, "version", _not_installed)
    assert app_module.get_app_version() == "0.0.0-dev"

    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/v1/version")
    assert resp.json()["version"] == app.version == "0.0.0-dev"
