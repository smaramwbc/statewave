"""Tests for the public /v1/version discovery endpoint (#178)."""

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
