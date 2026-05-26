"""End-to-end residency enforcement tests (v0.9 #161).

Covers:
  * Middleware: refuses tenant-scoped requests when the tenant is
    pinned to a different region than ``settings.region``.
  * Middleware: passes through when residency is disabled
    (`settings.region` is None) — the v0.8 upgrade path.
  * Middleware: passes through when the tenant is not pinned.
  * Admin patch: refuses pinning a tenant to a region this server
    doesn't serve unless `force_region_pin: true` is set.
  * Admin patch: accepts pinning to the local region.
  * Region is stamped on receipts emitted in multi-region mode.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from server.core.config import settings
from server.db.tables import ReceiptRow, TenantConfigRow


# ---------------------------------------------------------------------------
# Middleware enforcement
# ---------------------------------------------------------------------------


async def _set_tenant_region(session_factory, *, tenant_id: str, region: str | None):
    """Direct DB write so the test controls the pinned region precisely."""
    async with session_factory() as session:
        result = await session.execute(
            select(TenantConfigRow).where(TenantConfigRow.tenant_id == tenant_id)
        )
        row = result.scalar_one_or_none()
        new_config: dict = {}
        if row is not None:
            new_config = dict(row.config or {})
        if region is None:
            new_config.pop("region", None)
        else:
            new_config["region"] = region

        if row is None:
            session.add(
                TenantConfigRow(
                    tenant_id=tenant_id,
                    config=new_config,
                    version=1,
                )
            )
        else:
            row.config = new_config
            row.version = row.version + 1
        await session.commit()


@pytest.mark.anyio
async def test_residency_disabled_is_a_passthrough(
    client: AsyncClient, session_factory, monkeypatch
):
    """When `settings.region` is None (single-region mode, the
    default), the middleware must not refuse any request — even if
    the tenant has a region pinned in its config."""
    monkeypatch.setattr(settings, "region", None)
    tenant_id = f"t-{uuid.uuid4().hex[:8]}"
    await _set_tenant_region(session_factory, tenant_id=tenant_id, region="us")

    r = await client.get(
        "/admin/tenants",
        headers={"X-Tenant-ID": tenant_id},
    )
    # Any tenant-scoped route must succeed when residency is off.
    assert r.status_code == 200, r.text


@pytest.mark.anyio
async def test_unpinned_tenant_passes_through_in_multi_region(
    client: AsyncClient, session_factory, monkeypatch
):
    """A tenant without `region` in its config is unpinned: legacy
    pre-v0.9 tenants and globally-mobile tenants both take this path
    and must not be refused."""
    monkeypatch.setattr(settings, "region", "eu")
    tenant_id = f"t-{uuid.uuid4().hex[:8]}"
    # Deliberately no region pin.
    await _set_tenant_region(session_factory, tenant_id=tenant_id, region=None)

    r = await client.get(
        f"/admin/tenants/{tenant_id}/config",
        headers={"X-Tenant-ID": tenant_id},
    )
    assert r.status_code == 200, r.text


@pytest.mark.anyio
async def test_matching_region_allowed(client: AsyncClient, session_factory, monkeypatch):
    monkeypatch.setattr(settings, "region", "eu")
    tenant_id = f"t-{uuid.uuid4().hex[:8]}"
    await _set_tenant_region(session_factory, tenant_id=tenant_id, region="eu")

    r = await client.get(
        f"/admin/tenants/{tenant_id}/config",
        headers={"X-Tenant-ID": tenant_id},
    )
    assert r.status_code == 200, r.text


@pytest.mark.anyio
async def test_mismatched_region_refused_with_403(
    client: AsyncClient, session_factory, monkeypatch
):
    """The load-bearing test: a request for a tenant pinned to `us`
    arriving at an `eu` process must be refused with 403 +
    `residency.mismatch`, no exceptions."""
    monkeypatch.setattr(settings, "region", "eu")
    tenant_id = f"t-{uuid.uuid4().hex[:8]}"
    await _set_tenant_region(session_factory, tenant_id=tenant_id, region="us")

    r = await client.post(
        "/v1/episodes",
        headers={"X-Tenant-ID": tenant_id},
        json={
            "subject_id": "anyone",
            "source": "test",
            "type": "chat.session",
            "payload": {"messages": []},
        },
    )
    assert r.status_code == 403, r.text
    err = r.json()["error"]
    assert err["code"] == "residency.mismatch"
    # The wire response must NOT echo the server's region — only
    # the tenant's pinned region is safe to surface.
    assert "us" in err["message"]
    assert "eu" not in err["message"]


@pytest.mark.anyio
async def test_mismatched_region_refused_on_admin_path_too(
    client: AsyncClient, session_factory, monkeypatch
):
    """v0.9 #161 design decision: total isolation. Cross-region
    admin reads are forbidden. The same 403 applies to /admin/."""
    monkeypatch.setattr(settings, "region", "eu")
    tenant_id = f"t-{uuid.uuid4().hex[:8]}"
    await _set_tenant_region(session_factory, tenant_id=tenant_id, region="us")

    r = await client.get(
        f"/admin/tenants/{tenant_id}/config",
        headers={"X-Tenant-ID": tenant_id},
    )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "residency.mismatch"


@pytest.mark.anyio
async def test_anonymous_request_skipped(client: AsyncClient, session_factory, monkeypatch):
    """Requests without a tenant header have nothing to enforce
    against. The middleware must pass them through; `require_tenant`
    is the separate knob that gates anonymous traffic."""
    monkeypatch.setattr(settings, "region", "eu")

    r = await client.get("/healthz")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Admin patch — pinning a tenant
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_admin_patch_accepts_pin_to_local_region(client: AsyncClient, monkeypatch):
    """Pinning a tenant to the local region from this server is the
    happy path — it doesn't lock the tenant out."""
    monkeypatch.setattr(settings, "region", "eu")
    tenant_id = f"t-{uuid.uuid4().hex[:8]}"

    r = await client.patch(
        f"/admin/tenants/{tenant_id}/config",
        json={"region": "eu"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["config"]["region"] == "eu"


@pytest.mark.anyio
async def test_admin_patch_refuses_pin_to_foreign_region_without_force(
    client: AsyncClient, monkeypatch
):
    """Pinning a tenant to a region this server doesn't serve would
    immediately lock the tenant out. Refuse with 422
    `residency.invalid_pin` unless `force_region_pin: true`."""
    monkeypatch.setattr(settings, "region", "eu")
    tenant_id = f"t-{uuid.uuid4().hex[:8]}"

    r = await client.patch(
        f"/admin/tenants/{tenant_id}/config",
        json={"region": "us"},
    )
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "residency.invalid_pin"


@pytest.mark.anyio
async def test_admin_patch_accepts_force_region_pin(client: AsyncClient, monkeypatch):
    """`force_region_pin: true` bypasses the safety check — used by
    scripted bulk-config migrations."""
    monkeypatch.setattr(settings, "region", "eu")
    tenant_id = f"t-{uuid.uuid4().hex[:8]}"

    r = await client.patch(
        f"/admin/tenants/{tenant_id}/config",
        json={"region": "us", "force_region_pin": True},
    )
    # Note: the tenant is now locked out of THIS server, but the
    # admin-patch endpoint is reachable because the tenant didn't
    # have a residency pin BEFORE this call. (Middleware reads the
    # pre-call config.)
    assert r.status_code == 200, r.text
    assert r.json()["config"]["region"] == "us"
    # `force_region_pin` is a request-only flag, not a config key —
    # it must not be persisted on the JSONB.
    assert "force_region_pin" not in r.json()["config"]


@pytest.mark.anyio
async def test_admin_patch_allows_any_pin_in_single_region_mode(client: AsyncClient, monkeypatch):
    """Dev / single-region servers (`settings.region` is None) must
    allow operators to author multi-region configs that will be
    deployed elsewhere."""
    monkeypatch.setattr(settings, "region", None)
    tenant_id = f"t-{uuid.uuid4().hex[:8]}"

    r = await client.patch(
        f"/admin/tenants/{tenant_id}/config",
        json={"region": "ap-south-1"},
    )
    assert r.status_code == 200
    assert r.json()["config"]["region"] == "ap-south-1"


# ---------------------------------------------------------------------------
# Region stamped on receipts
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_receipts_stamp_local_region(client: AsyncClient, session_factory, monkeypatch):
    """Receipts emitted in multi-region mode must carry the local
    region so the audit trail records *where* the decision was made,
    not just *what* it was."""
    monkeypatch.setattr(settings, "region", "eu")
    tenant_id = f"t-{uuid.uuid4().hex[:8]}"
    # Pin to EU so the middleware doesn't refuse the request.
    await _set_tenant_region(session_factory, tenant_id=tenant_id, region="eu")
    subject_id = f"s-{uuid.uuid4().hex[:8]}"

    r = await client.post(
        "/v1/episodes",
        headers={"X-Tenant-ID": tenant_id},
        json={
            "subject_id": subject_id,
            "source": "test",
            "type": "chat.session",
            "payload": {
                "messages": [{"role": "user", "content": "hi"}],
            },
        },
    )
    assert r.status_code == 201
    r = await client.post(
        "/v1/memories/compile",
        headers={"X-Tenant-ID": tenant_id},
        json={"subject_id": subject_id},
    )
    assert r.status_code == 200
    r = await client.post(
        "/v1/context",
        headers={"X-Tenant-ID": tenant_id},
        json={"subject_id": subject_id, "task": "hi", "emit_receipt": True},
    )
    assert r.status_code == 200, r.text
    receipt_id = r.json().get("receipt_id")
    assert receipt_id

    async with session_factory() as session:
        row = (
            await session.execute(select(ReceiptRow).where(ReceiptRow.receipt_id == receipt_id))
        ).scalar_one()
    assert row.region == "eu"
    assert row.body.get("region") == "eu"


@pytest.mark.anyio
async def test_receipts_have_null_region_in_single_region_mode(
    client: AsyncClient, session_factory, monkeypatch
):
    """When `settings.region` is None, receipts carry `region=None`
    — the v0.8-backwards-compatible default."""
    monkeypatch.setattr(settings, "region", None)
    subject_id = f"s-{uuid.uuid4().hex[:8]}"

    r = await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "source": "test",
            "type": "chat.session",
            "payload": {"messages": [{"role": "user", "content": "hi"}]},
        },
    )
    assert r.status_code == 201
    r = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert r.status_code == 200
    r = await client.post(
        "/v1/context",
        json={"subject_id": subject_id, "task": "hi", "emit_receipt": True},
    )
    assert r.status_code == 200
    receipt_id = r.json().get("receipt_id")
    assert receipt_id

    async with session_factory() as session:
        row = (
            await session.execute(select(ReceiptRow).where(ReceiptRow.receipt_id == receipt_id))
        ).scalar_one()
    assert row.region is None
