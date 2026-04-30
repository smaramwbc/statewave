"""Integration tests for backup/restore (requires Postgres)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from server.app import create_app


@pytest.fixture
async def client():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_export_returns_valid_document(client):
    """Export endpoint returns a properly structured document."""
    # Create some data first
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": "backup-test-user",
            "source": "chat",
            "type": "message",
            "payload": {"role": "user", "content": "I like dark mode"},
        },
    )
    await client.post("/v1/memories/compile", json={"subject_id": "backup-test-user"})

    # Export
    resp = await client.get("/admin/export/backup-test-user")
    assert resp.status_code == 200
    doc = resp.json()
    assert doc["format_version"] == "1.0"
    assert doc["subject_id"] == "backup-test-user"
    assert doc["counts"]["episodes"] >= 1
    assert "checksum" in doc


@pytest.mark.anyio
async def test_export_empty_subject_returns_404(client):
    """Exporting a subject with no data returns 404."""
    resp = await client.get("/admin/export/nonexistent-subject-xyz")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_roundtrip_export_import(client):
    """Export then import to a new subject preserves data."""
    # Create data
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": "roundtrip-src",
            "source": "chat",
            "type": "message",
            "payload": {"role": "user", "content": "My timezone is UTC+1"},
        },
    )
    await client.post("/v1/memories/compile", json={"subject_id": "roundtrip-src"})

    # Export
    resp = await client.get("/admin/export/roundtrip-src")
    assert resp.status_code == 200
    doc = resp.json()

    # Delete original
    await client.delete("/v1/subjects/roundtrip-src")

    # Import to new subject
    resp = await client.post(
        "/admin/import",
        json={
            "document": doc,
            "target_subject_id": "roundtrip-dst",
            "preserve_ids": False,
        },
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["subject_id"] == "roundtrip-dst"
    assert result["episodes_imported"] >= 1

    # Verify data is accessible
    resp = await client.get("/v1/timeline", params={"subject_id": "roundtrip-dst"})
    assert resp.status_code == 200
    assert len(resp.json()["episodes"]) >= 1


@pytest.mark.anyio
async def test_import_rejects_bad_checksum(client):
    """Import with tampered data is rejected."""
    resp = await client.post(
        "/admin/import",
        json={
            "document": {
                "format_version": "1.0",
                "subject_id": "test",
                "tenant_id": None,
                "episodes": [],
                "memories": [],
                "checksum": "bad",
                "counts": {"episodes": 0, "memories": 0},
                "exported_at": "2026-01-01T00:00:00+00:00",
            },
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    # Check for checksum error in either 'detail' or 'message' field
    error_text = body.get("detail") or body.get("message") or str(body)
    assert "checksum" in error_text.lower() or "Checksum" in error_text


@pytest.mark.anyio
async def test_import_with_tenant_override(client):
    """Import can override tenant_id."""
    # Create and export
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": "tenant-move-src",
            "source": "test",
            "type": "event",
            "payload": {"data": "value"},
        },
    )
    resp = await client.get("/admin/export/tenant-move-src")
    doc = resp.json()

    # Delete original
    await client.delete("/v1/subjects/tenant-move-src")

    # Import with tenant override
    resp = await client.post(
        "/admin/import",
        json={
            "document": doc,
            "target_subject_id": "tenant-move-dst",
            "target_tenant_id": "org-new",
            "preserve_ids": False,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["tenant_id"] == "org-new"
