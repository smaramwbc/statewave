"""Integration tests for the v0.9 receipt retention worker (issue #156).

`cleanup_expired_receipts` reads from `tenant_configs` and writes to
`receipts` — both DB tables — so the function isn't pure and lives here
rather than in the unit-test file.

Pattern follows `tests/integration/test_memory_ttl.py`: insert rows
directly through the session factory so the test controls `created_at`
precisely (compiling through the API would stamp `created_at` from
``now()`` and force monkey-patching).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from server.db.tables import ReceiptRow, TenantConfigRow
from server.services.receipts import cleanup_expired_receipts, new_ulid


# ---------------------------------------------------------------------------
# Helpers — direct DB writes so created_at is fully controlled
# ---------------------------------------------------------------------------


async def _insert_receipt(
    session_factory,
    *,
    tenant_id: str | None,
    subject_id: str,
    created_at: datetime,
    status: str = "active",
    receipt_id: str | None = None,
) -> str:
    rid = receipt_id or new_ulid()
    async with session_factory() as session:
        session.add(
            ReceiptRow(
                receipt_id=rid,
                parent_receipt_id=None,
                mode="retrieval",
                tenant_id=tenant_id,
                subject_id=subject_id,
                query_id=None,
                task_id=None,
                context_hash="0" * 64,
                context_size_bytes=0,
                policy_bundle_hash=None,
                region=None,
                receipt_signature=None,
                body={
                    "receipt_id": rid,
                    "mode": "retrieval",
                    "subject_id": subject_id,
                    "tenant_id": tenant_id,
                    "stub": True,
                },
                as_of=created_at,
                created_at=created_at,
                status=status,
            )
        )
        await session.commit()
    return rid


async def _set_tenant_retention(
    session_factory,
    tenant_id: str,
    days: int | None | object,
) -> None:
    """Set (or unset) `tenant_configs.config.receipt_retention_days`.

    ``days=None`` clears the key; non-int values are accepted (the
    worker's input-validation path is exercised by passing through here)."""
    async with session_factory() as session:
        existing = await session.get(TenantConfigRow, tenant_id)
        if existing is None:
            cfg: dict = {} if days is None else {"receipt_retention_days": days}
            session.add(TenantConfigRow(tenant_id=tenant_id, config=cfg, version=1))
        else:
            new_cfg = dict(existing.config or {})
            if days is None:
                new_cfg.pop("receipt_retention_days", None)
            else:
                new_cfg["receipt_retention_days"] = days
            existing.config = new_cfg
            existing.version = (existing.version or 0) + 1
        await session.commit()


def _t(label: str) -> str:
    return f"tenant-{label}-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Worker — happy path
# ---------------------------------------------------------------------------


async def test_tombstones_expired_active_receipt(session_factory, subject_id):
    tenant = _t("tomb")
    await _set_tenant_retention(session_factory, tenant, 30)
    old = datetime.now(timezone.utc) - timedelta(days=40)
    rid = await _insert_receipt(
        session_factory, tenant_id=tenant, subject_id=subject_id, created_at=old
    )

    async with session_factory() as session:
        count = await cleanup_expired_receipts(session)
        await session.commit()
    assert count == 1

    async with session_factory() as session:
        row = await session.get(ReceiptRow, rid)
        assert row.status == "tombstoned"
        assert row.tombstoned_at is not None


async def test_leaves_active_receipt_within_window(session_factory, subject_id):
    tenant = _t("within")
    await _set_tenant_retention(session_factory, tenant, 30)
    recent = datetime.now(timezone.utc) - timedelta(days=5)
    rid = await _insert_receipt(
        session_factory, tenant_id=tenant, subject_id=subject_id, created_at=recent
    )

    async with session_factory() as session:
        count = await cleanup_expired_receipts(session)
        await session.commit()
    assert count == 0

    async with session_factory() as session:
        row = await session.get(ReceiptRow, rid)
        assert row.status == "active"
        assert row.tombstoned_at is None


async def test_tenant_without_retention_is_untouched(session_factory, subject_id):
    """A tenant that hasn't configured retention sees no transitions, even
    on ancient receipts — retention is an explicit opt-in per tenant."""
    tenant = _t("noret")
    # No tenant_configs row for this tenant at all
    old = datetime.now(timezone.utc) - timedelta(days=400)
    rid = await _insert_receipt(
        session_factory, tenant_id=tenant, subject_id=subject_id, created_at=old
    )

    async with session_factory() as session:
        count = await cleanup_expired_receipts(session)
        await session.commit()
    assert count == 0

    async with session_factory() as session:
        row = await session.get(ReceiptRow, rid)
        assert row.status == "active"


# ---------------------------------------------------------------------------
# Worker — idempotency + replay safety
# ---------------------------------------------------------------------------


async def test_idempotent_second_pass_is_noop(session_factory, subject_id):
    tenant = _t("idem")
    await _set_tenant_retention(session_factory, tenant, 30)
    old = datetime.now(timezone.utc) - timedelta(days=40)
    await _insert_receipt(
        session_factory, tenant_id=tenant, subject_id=subject_id, created_at=old
    )

    async with session_factory() as session:
        first = await cleanup_expired_receipts(session)
        await session.commit()
    async with session_factory() as session:
        second = await cleanup_expired_receipts(session)
        await session.commit()
    assert first == 1
    assert second == 0


async def test_already_tombstoned_row_is_untouched(session_factory, subject_id):
    tenant = _t("alreadytomb")
    await _set_tenant_retention(session_factory, tenant, 30)
    old = datetime.now(timezone.utc) - timedelta(days=40)
    rid = await _insert_receipt(
        session_factory,
        tenant_id=tenant,
        subject_id=subject_id,
        created_at=old,
        status="tombstoned",
    )

    async with session_factory() as session:
        count = await cleanup_expired_receipts(session)
        await session.commit()
    assert count == 0

    # `tombstoned_at` is still None because the row was inserted as
    # tombstoned without one — the worker shouldn't backfill it.
    async with session_factory() as session:
        row = await session.get(ReceiptRow, rid)
        assert row.tombstoned_at is None


# ---------------------------------------------------------------------------
# Worker — tenant isolation
# ---------------------------------------------------------------------------


async def test_tenant_isolation(session_factory, subject_id):
    """Tenant A's retention setting must not affect tenant B's receipts."""
    tenant_a = _t("iso-a")
    tenant_b = _t("iso-b")
    await _set_tenant_retention(session_factory, tenant_a, 30)
    # tenant_b has no retention — its 40-day-old receipt must survive.
    old = datetime.now(timezone.utc) - timedelta(days=40)
    rid_a = await _insert_receipt(
        session_factory, tenant_id=tenant_a, subject_id=subject_id, created_at=old
    )
    rid_b = await _insert_receipt(
        session_factory, tenant_id=tenant_b, subject_id=subject_id, created_at=old
    )

    async with session_factory() as session:
        count = await cleanup_expired_receipts(session)
        await session.commit()
    assert count == 1

    async with session_factory() as session:
        a = await session.get(ReceiptRow, rid_a)
        b = await session.get(ReceiptRow, rid_b)
        assert a.status == "tombstoned"
        assert b.status == "active"


# ---------------------------------------------------------------------------
# Worker — defensive parsing of malformed config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", ["thirty", -5, 0, True, [30], {"days": 30}])
async def test_malformed_retention_value_is_skipped_not_fatal(
    session_factory, subject_id, bad_value
):
    """A SQL-shell-poked or pre-v0.9 garbage value in tenant_configs.config
    must not halt the purge tick — the worker skips that tenant and
    keeps processing the rest. Per-tenant config endpoint validates at
    write time but the worker can't trust historical writes."""
    bad_tenant = _t("bad")
    good_tenant = _t("good")
    await _set_tenant_retention(session_factory, bad_tenant, bad_value)
    await _set_tenant_retention(session_factory, good_tenant, 30)

    old = datetime.now(timezone.utc) - timedelta(days=40)
    bad_rid = await _insert_receipt(
        session_factory, tenant_id=bad_tenant, subject_id=subject_id, created_at=old
    )
    good_rid = await _insert_receipt(
        session_factory, tenant_id=good_tenant, subject_id=subject_id, created_at=old
    )

    async with session_factory() as session:
        count = await cleanup_expired_receipts(session)
        await session.commit()

    # Bad tenant's receipt untouched; good tenant's receipt tombstoned.
    assert count == 1
    async with session_factory() as session:
        assert (await session.get(ReceiptRow, bad_rid)).status == "active"
        assert (await session.get(ReceiptRow, good_rid)).status == "tombstoned"


# ---------------------------------------------------------------------------
# API — GET /v1/receipts/{id} surfaces `status` and `tombstoned_at`
# ---------------------------------------------------------------------------


async def test_get_receipt_shows_active_status(client, session_factory, subject_id):
    tenant = _t("api-active")
    rid = await _insert_receipt(
        session_factory,
        tenant_id=tenant,
        subject_id=subject_id,
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
    )

    resp = await client.get(f"/v1/receipts/{rid}", headers={"X-Tenant-ID": tenant})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "active"
    assert "tombstoned_at" not in body
    assert body["receipt_id"] == rid


async def test_get_receipt_shows_tombstoned_status_with_timestamp(
    client, session_factory, subject_id
):
    """Tombstoned receipts remain individually addressable for forensic
    lookup, with `status` + `tombstoned_at` surfaced on the wire."""
    tenant = _t("api-tomb")
    await _set_tenant_retention(session_factory, tenant, 30)
    rid = await _insert_receipt(
        session_factory,
        tenant_id=tenant,
        subject_id=subject_id,
        created_at=datetime.now(timezone.utc) - timedelta(days=40),
    )

    # Run the worker to transition the row.
    async with session_factory() as session:
        await cleanup_expired_receipts(session)
        await session.commit()

    resp = await client.get(f"/v1/receipts/{rid}", headers={"X-Tenant-ID": tenant})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "tombstoned"
    assert "tombstoned_at" in body
    assert body["receipt_id"] == rid


# ---------------------------------------------------------------------------
# API — GET /v1/receipts list filters by status
# ---------------------------------------------------------------------------


async def test_list_excludes_tombstoned_by_default(client, session_factory, subject_id):
    tenant = _t("api-list-default")
    await _set_tenant_retention(session_factory, tenant, 30)
    active_rid = await _insert_receipt(
        session_factory,
        tenant_id=tenant,
        subject_id=subject_id,
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    tomb_rid = await _insert_receipt(
        session_factory,
        tenant_id=tenant,
        subject_id=subject_id,
        created_at=datetime.now(timezone.utc) - timedelta(days=40),
    )
    async with session_factory() as session:
        await cleanup_expired_receipts(session)
        await session.commit()

    resp = await client.get(
        f"/v1/receipts?subject_id={subject_id}",
        headers={"X-Tenant-ID": tenant},
    )
    assert resp.status_code == 200
    ids = [r["receipt_id"] for r in resp.json()["receipts"]]
    assert active_rid in ids
    assert tomb_rid not in ids


async def test_list_includes_tombstoned_when_requested(
    client, session_factory, subject_id
):
    tenant = _t("api-list-include")
    await _set_tenant_retention(session_factory, tenant, 30)
    active_rid = await _insert_receipt(
        session_factory,
        tenant_id=tenant,
        subject_id=subject_id,
        created_at=datetime.now(timezone.utc) - timedelta(days=1),
    )
    tomb_rid = await _insert_receipt(
        session_factory,
        tenant_id=tenant,
        subject_id=subject_id,
        created_at=datetime.now(timezone.utc) - timedelta(days=40),
    )
    async with session_factory() as session:
        await cleanup_expired_receipts(session)
        await session.commit()

    resp = await client.get(
        f"/v1/receipts?subject_id={subject_id}&include_tombstoned=true",
        headers={"X-Tenant-ID": tenant},
    )
    assert resp.status_code == 200
    ids = [r["receipt_id"] for r in resp.json()["receipts"]]
    statuses = {r["receipt_id"]: r["status"] for r in resp.json()["receipts"]}
    assert active_rid in ids
    assert tomb_rid in ids
    assert statuses[active_rid] == "active"
    assert statuses[tomb_rid] == "tombstoned"
