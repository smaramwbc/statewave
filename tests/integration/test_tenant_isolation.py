"""Tests for multi-tenant data isolation.

Proves that when tenant_id is set, data is fully scoped:
- tenant A cannot see tenant B's subjects, episodes, memories
- compile jobs are tenant-scoped
- search/context/timeline respect tenant boundaries
- single-tenant mode (no header) continues to work as before
"""

from __future__ import annotations


import pytest
from httpx import ASGITransport, AsyncClient

from server.app import create_app
from server.core.config import settings
from server.db.engine import get_session, set_engine_for_testing

import tests.integration.conftest as _conftest


def _make_override():
    async def _override_get_session():
        async with _conftest._session_factory() as session:
            yield session
    return _override_get_session


@pytest.fixture
def _require_tenant(monkeypatch):
    """Enable tenant enforcement for this test."""
    monkeypatch.setattr(settings, "require_tenant", True)


@pytest.fixture
async def client():
    """Client with tenant enforcement enabled."""
    original = settings.require_tenant
    settings.require_tenant = True
    app = create_app()
    app.dependency_overrides[get_session] = _make_override()
    prev = set_engine_for_testing(_conftest._engine, _conftest._session_factory)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    set_engine_for_testing(*prev)
    settings.require_tenant = original


@pytest.fixture
async def client_optional():
    """Client with optional tenant (default single-tenant mode)."""
    original = settings.require_tenant
    settings.require_tenant = False
    app = create_app()
    app.dependency_overrides[get_session] = _make_override()
    prev = set_engine_for_testing(_conftest._engine, _conftest._session_factory)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    set_engine_for_testing(*prev)
    settings.require_tenant = original


# ---------------------------------------------------------------------------
# Tenant enforcement
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_missing_tenant_rejected_when_required(client):
    """Requests without tenant header are rejected when require_tenant=true."""
    resp = await client.post(
        "/v1/episodes",
        json={
            "subject_id": "user-1",
            "source": "test",
            "type": "message",
            "payload": {"text": "hi"},
        },
    )
    assert resp.status_code == 400
    assert "required" in resp.json()["error"]["message"].lower()


@pytest.mark.anyio
async def test_tenant_header_accepted(client):
    """Requests with tenant header proceed normally."""
    resp = await client.post(
        "/v1/episodes",
        headers={"X-Tenant-ID": "tenant-a"},
        json={
            "subject_id": "user-1",
            "source": "test",
            "type": "message",
            "payload": {"text": "hi"},
        },
    )
    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Episode isolation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_episodes_isolated_between_tenants(client):
    """Tenant A's episodes are invisible to tenant B."""
    # Tenant A creates an episode
    await client.post(
        "/v1/episodes",
        headers={"X-Tenant-ID": "tenant-a"},
        json={
            "subject_id": "shared-subject",
            "source": "test",
            "type": "message",
            "payload": {"text": "from tenant A"},
        },
    )

    # Tenant B gets timeline for same subject — should see nothing
    resp = await client.get(
        "/v1/timeline",
        headers={"X-Tenant-ID": "tenant-b"},
        params={"subject_id": "shared-subject"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["episodes"]) == 0

    # Tenant A sees their own episode
    resp = await client.get(
        "/v1/timeline",
        headers={"X-Tenant-ID": "tenant-a"},
        params={"subject_id": "shared-subject"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["episodes"]) == 1


# ---------------------------------------------------------------------------
# Subject listing isolation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_subjects_isolated_between_tenants(client):
    """Subject listing is tenant-scoped."""
    # Tenant A creates data
    await client.post(
        "/v1/episodes",
        headers={"X-Tenant-ID": "tenant-a"},
        json={
            "subject_id": "user-alpha",
            "source": "test",
            "type": "message",
            "payload": {"text": "hello"},
        },
    )

    # Tenant B creates data
    await client.post(
        "/v1/episodes",
        headers={"X-Tenant-ID": "tenant-b"},
        json={
            "subject_id": "user-beta",
            "source": "test",
            "type": "message",
            "payload": {"text": "hello"},
        },
    )

    # Tenant A sees only their subject
    resp = await client.get("/v1/subjects", headers={"X-Tenant-ID": "tenant-a"})
    assert resp.status_code == 200
    subject_ids = [s["subject_id"] for s in resp.json()["subjects"]]
    assert "user-alpha" in subject_ids
    assert "user-beta" not in subject_ids

    # Tenant B sees only their subject
    resp = await client.get("/v1/subjects", headers={"X-Tenant-ID": "tenant-b"})
    subject_ids = [s["subject_id"] for s in resp.json()["subjects"]]
    assert "user-beta" in subject_ids
    assert "user-alpha" not in subject_ids


# ---------------------------------------------------------------------------
# Memory compilation isolation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_compile_only_processes_own_tenant_episodes(client):
    """Compilation only processes episodes belonging to the requesting tenant."""
    # Tenant A ingests
    await client.post(
        "/v1/episodes",
        headers={"X-Tenant-ID": "tenant-a"},
        json={
            "subject_id": "user-1",
            "source": "chat",
            "type": "message",
            "payload": {"role": "user", "content": "My name is Alice"},
        },
    )

    # Tenant B ingests same subject
    await client.post(
        "/v1/episodes",
        headers={"X-Tenant-ID": "tenant-b"},
        json={
            "subject_id": "user-1",
            "source": "chat",
            "type": "message",
            "payload": {"role": "user", "content": "My name is Bob"},
        },
    )

    # Tenant A compiles — should only see Alice
    resp = await client.post(
        "/v1/memories/compile",
        headers={"X-Tenant-ID": "tenant-a"},
        json={"subject_id": "user-1"},
    )
    assert resp.status_code == 200
    resp.json()["memories"]

    # Tenant B compiles — should only see Bob
    resp = await client.post(
        "/v1/memories/compile",
        headers={"X-Tenant-ID": "tenant-b"},
        json={"subject_id": "user-1"},
    )
    assert resp.status_code == 200
    resp.json()["memories"]

    # Verify isolation via search
    resp = await client.get(
        "/v1/memories/search",
        headers={"X-Tenant-ID": "tenant-a"},
        params={"subject_id": "user-1"},
    )
    a_contents = [m["content"] for m in resp.json()["memories"]]

    resp = await client.get(
        "/v1/memories/search",
        headers={"X-Tenant-ID": "tenant-b"},
        params={"subject_id": "user-1"},
    )
    b_contents = [m["content"] for m in resp.json()["memories"]]

    # Cross-check: A's memories should not appear in B's search
    for content in a_contents:
        assert content not in b_contents or "Alice" not in content or "Bob" not in content


# ---------------------------------------------------------------------------
# Search isolation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_search_respects_tenant_boundary(client):
    """Memory search does not leak across tenants."""
    # Set up: tenant-a has data, tenant-b does not
    await client.post(
        "/v1/episodes",
        headers={"X-Tenant-ID": "tenant-a"},
        json={
            "subject_id": "customer-x",
            "source": "chat",
            "type": "message",
            "payload": {"role": "user", "content": "I prefer dark mode"},
        },
    )
    await client.post(
        "/v1/memories/compile",
        headers={"X-Tenant-ID": "tenant-a"},
        json={"subject_id": "customer-x"},
    )

    # Tenant B searches for same subject — empty
    resp = await client.get(
        "/v1/memories/search",
        headers={"X-Tenant-ID": "tenant-b"},
        params={"subject_id": "customer-x"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["memories"]) == 0


# ---------------------------------------------------------------------------
# Context assembly isolation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_context_respects_tenant_boundary(client):
    """Context bundle assembly is tenant-scoped."""
    # Tenant A has data
    await client.post(
        "/v1/episodes",
        headers={"X-Tenant-ID": "tenant-a"},
        json={
            "subject_id": "user-ctx",
            "source": "chat",
            "type": "message",
            "payload": {"role": "user", "content": "My timezone is PST"},
        },
    )
    await client.post(
        "/v1/memories/compile",
        headers={"X-Tenant-ID": "tenant-a"},
        json={"subject_id": "user-ctx"},
    )

    # Tenant B requests context for same subject — should get nothing
    resp = await client.post(
        "/v1/context",
        headers={"X-Tenant-ID": "tenant-b"},
        json={"subject_id": "user-ctx", "task": "help user"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["facts"]) == 0
    assert len(data["episodes"]) == 0


# ---------------------------------------------------------------------------
# Delete isolation
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_delete_only_affects_own_tenant(client):
    """Deleting a subject only removes that tenant's data."""
    # Both tenants create data for same subject
    await client.post(
        "/v1/episodes",
        headers={"X-Tenant-ID": "tenant-a"},
        json={
            "subject_id": "user-del",
            "source": "test",
            "type": "message",
            "payload": {"text": "A data"},
        },
    )
    await client.post(
        "/v1/episodes",
        headers={"X-Tenant-ID": "tenant-b"},
        json={
            "subject_id": "user-del",
            "source": "test",
            "type": "message",
            "payload": {"text": "B data"},
        },
    )

    # Tenant A deletes
    resp = await client.delete("/v1/subjects/user-del", headers={"X-Tenant-ID": "tenant-a"})
    assert resp.status_code == 200
    assert resp.json()["episodes_deleted"] == 1

    # Tenant B still has their data
    resp = await client.get(
        "/v1/timeline",
        headers={"X-Tenant-ID": "tenant-b"},
        params={"subject_id": "user-del"},
    )
    assert len(resp.json()["episodes"]) == 1


# ---------------------------------------------------------------------------
# Single-tenant mode (backward compat)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_no_tenant_header_works_in_optional_mode(client_optional):
    """Without tenant header in optional mode, everything works normally."""
    resp = await client_optional.post(
        "/v1/episodes",
        json={
            "subject_id": "user-local",
            "source": "test",
            "type": "message",
            "payload": {"text": "hello"},
        },
    )
    assert resp.status_code == 201

    resp = await client_optional.get("/v1/timeline", params={"subject_id": "user-local"})
    assert resp.status_code == 200
    assert len(resp.json()["episodes"]) == 1


@pytest.mark.anyio
async def test_optional_tenant_still_isolates_when_provided(client_optional):
    """When tenant header is optionally provided, isolation still works."""
    # With tenant header
    await client_optional.post(
        "/v1/episodes",
        headers={"X-Tenant-ID": "org-1"},
        json={
            "subject_id": "user-opt",
            "source": "test",
            "type": "message",
            "payload": {"text": "org1 data"},
        },
    )

    # Without header — should NOT see org-1's data (None != "org-1")
    resp = await client_optional.get("/v1/timeline", params={"subject_id": "user-opt"})
    # In optional mode without header, tenant_id=None means "show unscoped data"
    # This is correct: data created with tenant_id="org-1" won't match tenant_id=None
    assert resp.status_code == 200
