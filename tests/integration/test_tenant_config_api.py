"""Integration tests for the tenant-config admin endpoints (#50 follow-up).

Covers the lifecycle that was missing before this surface shipped:
  * GET on an unconfigured tenant returns the default-empty document.
  * PATCH creates a row when none exists, with version=1.
  * PATCH merges into existing config (preserves unknown keys —
    the forward-compat property future per-tenant knobs depend on).
  * PATCH validates enum and bound values at the API boundary,
    not silently in JSONB.
  * Optimistic concurrency via `expected_version` returns 409 on
    mismatch.
  * End-to-end: PATCH `policy_mode: enforce` actually flips the
    assembly path to filtering. Before this endpoint existed, this
    flip required a direct SQL write.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select, update

from server.db.tables import MemoryRow, PolicyBundleRow
from server.services import policy as policy_service


PII_BUNDLE_YAML = """
version: 1
rules:
  - id: deny-pii-marketing
    when:
      memory_has_any_label: [pii]
      caller_type: marketing_tool
    action: deny
"""


async def _seed(client: AsyncClient, subject_id: str) -> None:
    """Ingest + compile a small subject with PII content."""
    r = await client.post(
        "/v1/episodes",
        json={
            "subject_id": subject_id,
            "source": "x",
            "type": "y",
            "payload": {
                "messages": [
                    {"role": "user", "content": "I'm Alice, email alice@globex.example"},
                    {"role": "assistant", "content": "Noted."},
                ]
            },
        },
    )
    assert r.status_code == 201
    r = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert r.status_code == 200


@pytest.mark.anyio
async def test_get_returns_empty_default_for_unconfigured_tenant(client: AsyncClient):
    """A fresh tenant has no row in tenant_configs. The endpoint must
    return 200 + `{config: {}, version: 0}` rather than 404 — the
    same hygiene principle that motivated the /admin/policy/active
    fix."""
    r = await client.get("/admin/tenants/never-configured/config")
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "tenant_id": "never-configured",
        "config": {},
        "version": 0,
        "created_at": None,
        "updated_at": None,
    }


@pytest.mark.anyio
async def test_patch_creates_row_when_none_exists(client: AsyncClient):
    """First PATCH should INSERT and return version=1."""
    tenant = "new-tenant-1"
    r = await client.patch(
        f"/admin/tenants/{tenant}/config",
        json={"policy_mode": "enforce", "receipts": "always"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tenant_id"] == tenant
    assert body["config"] == {"policy_mode": "enforce", "receipts": "always"}
    assert body["version"] == 1


@pytest.mark.anyio
async def test_patch_merges_and_preserves_unknown_keys(client: AsyncClient):
    """The merge semantic is load-bearing — future admin endpoints
    will add keys to the same JSONB, and a PATCH that only touches
    one knob must not clobber the others."""
    tenant = "merge-test"
    # First write: set policy_mode.
    await client.patch(
        f"/admin/tenants/{tenant}/config", json={"policy_mode": "enforce"}
    )
    # Second write: ONLY change receipts. policy_mode must survive.
    r = await client.patch(
        f"/admin/tenants/{tenant}/config", json={"receipts": "always"}
    )
    body = r.json()
    assert body["config"] == {"policy_mode": "enforce", "receipts": "always"}
    assert body["version"] == 2

    # Third write: an unknown forward-compat key. PATCH must accept
    # the documented keys but the existing config must still be
    # preserved. We test the preservation property by writing a
    # known key and confirming nothing else was lost.
    r = await client.patch(
        f"/admin/tenants/{tenant}/config", json={"require_caller_identity": True}
    )
    body = r.json()
    assert body["config"] == {
        "policy_mode": "enforce",
        "receipts": "always",
        "require_caller_identity": True,
    }
    assert body["version"] == 3


@pytest.mark.anyio
async def test_patch_validates_enum_at_api_boundary(client: AsyncClient):
    """A typo like `policy_mode: "enforced"` would silently leave
    enforcement off if we validated inside the JSONB. Validate at
    the request layer — fail fast with 422."""
    r = await client.patch(
        "/admin/tenants/typo/config", json={"policy_mode": "enforced"}
    )
    assert r.status_code == 422


@pytest.mark.anyio
async def test_patch_validates_bounds(client: AsyncClient):
    """receipt_retention_days has a ge=0 / le=36500 bound."""
    r = await client.patch(
        "/admin/tenants/bad/config", json={"receipt_retention_days": -1}
    )
    assert r.status_code == 422


@pytest.mark.anyio
async def test_patch_optimistic_concurrency_returns_409_on_mismatch(client: AsyncClient):
    """expected_version must reflect the current row; mismatch = 409."""
    tenant = "concurrency-test"
    # Bring the row into existence.
    r = await client.patch(
        f"/admin/tenants/{tenant}/config", json={"policy_mode": "log_only"}
    )
    assert r.json()["version"] == 1

    # PATCH with stale version should 409.
    r = await client.patch(
        f"/admin/tenants/{tenant}/config",
        json={"policy_mode": "enforce", "expected_version": 0},
    )
    assert r.status_code == 409, r.text
    assert "version mismatch" in r.json()["error"]["message"]

    # PATCH with correct version should succeed.
    r = await client.patch(
        f"/admin/tenants/{tenant}/config",
        json={"policy_mode": "enforce", "expected_version": 1},
    )
    assert r.status_code == 200
    assert r.json()["version"] == 2


@pytest.mark.anyio
async def test_expected_version_zero_for_new_tenant(client: AsyncClient):
    """expected_version=0 should succeed for a tenant that has never
    been configured — that's the natural "I'm creating the row"
    semantic."""
    tenant = "v0-create"
    r = await client.patch(
        f"/admin/tenants/{tenant}/config",
        json={"policy_mode": "enforce", "expected_version": 0},
    )
    assert r.status_code == 200
    assert r.json()["version"] == 1


@pytest.mark.anyio
async def test_expected_version_one_for_new_tenant_returns_409(client: AsyncClient):
    """Symmetric to the above: claiming version=1 for a row that
    doesn't exist yet is a lost-update misread, return 409."""
    r = await client.patch(
        "/admin/tenants/v1-on-empty/config",
        json={"policy_mode": "enforce", "expected_version": 1},
    )
    assert r.status_code == 409


# ──────────────────────────────────────────────────────────────────────────
# The end-to-end test that justifies this whole endpoint existing.
# ──────────────────────────────────────────────────────────────────────────


async def _install_active_policy(session_factory, yaml_content: str, tenant_id: str) -> str:
    bundle = policy_service.load_bundle(yaml_content)
    async with session_factory() as session:
        # Idempotent: insert-if-missing + flip active.
        result = await session.execute(
            select(PolicyBundleRow).where(
                PolicyBundleRow.bundle_hash == bundle.bundle_hash
            )
        )
        if result.scalar_one_or_none() is None:
            session.add(
                PolicyBundleRow(
                    bundle_hash=bundle.bundle_hash,
                    yaml_content=yaml_content,
                    active=True,
                    tenant_id=tenant_id,
                )
            )
        else:
            await session.execute(
                update(PolicyBundleRow)
                .where(PolicyBundleRow.bundle_hash == bundle.bundle_hash)
                .values(active=True, tenant_id=tenant_id)
            )
        await session.commit()
    return bundle.bundle_hash


@pytest.mark.anyio
async def test_patch_policy_mode_enforce_actually_filters_memories(
    client: AsyncClient, subject_id: str, session_factory
):
    """The load-bearing end-to-end test: before this endpoint existed,
    flipping policy_mode=enforce required a direct SQL write — there
    was no API path. This verifies a tenant operator can flip enforce
    via the admin API and that the assembly path immediately starts
    filtering.

    Sequence: configure + label + install policy → PATCH enforce →
    /v1/context as the disallowed caller_type → assert facts/provenance
    are empty (the policy actually fired)."""
    tenant = "e2e-enforce"
    client.headers["X-Tenant-ID"] = tenant

    await _seed(client, subject_id)
    # Label every memory pii so the deny rule covers any retrieval winner.
    async with session_factory() as session:
        ids = (await session.execute(
            select(MemoryRow.id).where(MemoryRow.subject_id == subject_id)
        )).scalars().all()
        await session.execute(
            update(MemoryRow)
            .where(MemoryRow.id.in_(ids))
            .values(sensitivity_labels=["pii"])
        )
        await session.commit()
    await _install_active_policy(session_factory, PII_BUNDLE_YAML, tenant)

    # Baseline: under log_only (the default), the marketing call gets
    # memories AND the receipt records what would be denied. This
    # confirms the test setup is real before we touch enforce.
    r = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "x",
            "caller_type": "marketing_tool",
            "caller_id": "agent",
            "emit_receipt": True,
        },
    )
    body = r.json()
    assert body["facts"], "log_only must still deliver the memories"
    r_receipt = await client.get(f"/v1/receipts/{body['receipt_id']}")
    assert r_receipt.json()["policy"]["filters_applied"], (
        "log_only must record what would be filtered"
    )

    # The flip. This is the thing the endpoint exists to make
    # possible without a SQL shell.
    r = await client.patch(
        f"/admin/tenants/{tenant}/config", json={"policy_mode": "enforce"}
    )
    assert r.status_code == 200
    assert r.json()["config"]["policy_mode"] == "enforce"

    # Same call, but now under enforce → memories are denied.
    r = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "x",
            "caller_type": "marketing_tool",
            "caller_id": "agent",
            "emit_receipt": True,
        },
    )
    body = r.json()
    assert body["facts"] == [], (
        f"enforce mode must filter denied memories, got facts={body['facts']}"
    )
    assert body["provenance"].get("fact_ids", []) == []


@pytest.mark.anyio
async def test_patch_require_caller_identity_blocks_anonymous_after_flip(
    client: AsyncClient, subject_id: str
):
    """Same shape as the policy_mode flip — verify
    require_caller_identity actually takes effect after the PATCH."""
    tenant = "e2e-caller-identity"
    client.headers["X-Tenant-ID"] = tenant

    await _seed(client, subject_id)

    # Before the flip — anonymous call succeeds.
    r = await client.post("/v1/context", json={"subject_id": subject_id, "task": "x"})
    assert r.status_code == 200

    # The flip.
    r = await client.patch(
        f"/admin/tenants/{tenant}/config", json={"require_caller_identity": True}
    )
    assert r.status_code == 200

    # Same anonymous call → now 401.
    r = await client.post("/v1/context", json={"subject_id": subject_id, "task": "x"})
    assert r.status_code == 401

    # With identity it works again.
    r = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "x",
            "caller_id": "agent",
            "caller_type": "support",
        },
    )
    assert r.status_code == 200
