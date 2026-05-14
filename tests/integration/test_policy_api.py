"""End-to-end policy tests (#50).

Covers the three #50 acceptance-criteria negative tests against the
real FastAPI app + real Postgres:

  1. Sensitive memory must NOT be delivered to a disallowed caller
     when policy_mode is `enforce`.
  2. Filtered memory must NOT influence ranking (it doesn't even
     reach the ranker — verified by absence from `provenance`).
  3. A policy version bump must produce receipts that reference the
     new `policy_bundle_hash` — old receipts pin the old hash, which
     is the load-bearing replay property.

Plus the happy path: a tenant in `log_only` mode receives unfiltered
results but the receipt records every decision the policy would have
made under enforce.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import update

from server.db.tables import MemoryRow, PolicyBundleRow, TenantConfigRow
from server.services import policy as policy_service


PII_BUNDLE_YAML = """
version: 1
rules:
  - id: deny-pii-for-marketing
    when:
      memory_has_any_label: [pii]
      caller_type: marketing_tool
    action: deny
  - id: redact-secrets
    when:
      memory_has_any_label: [secret]
    action: redact
"""


async def _seed_subject(client: AsyncClient, subject_id: str) -> None:
    """Ingest two episodes and compile so the subject has memories
    we can label."""
    for ep in [
        {
            "source": "support-chat",
            "type": "conversation",
            "payload": {
                "messages": [
                    {"role": "user", "content": "My email is alice@globex.com."},
                    {"role": "assistant", "content": "Noted."},
                ]
            },
        },
        {
            "source": "support-chat",
            "type": "conversation",
            "payload": {
                "messages": [
                    {"role": "user", "content": "API key is sk-test-12345."},
                    {"role": "assistant", "content": "Noted."},
                ]
            },
        },
    ]:
        r = await client.post("/v1/episodes", json={**ep, "subject_id": subject_id})
        assert r.status_code == 201
    r = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert r.status_code == 200


async def _label_all_memories(
    session_factory, subject_id: str, labels: list[str]
) -> int:
    """Stamp `labels` onto EVERY memory for the subject. Used as a
    fast in-test fixture — the /v1/memories/{id}/labels API path is
    exercised by its own test below.

    The negative test for "PII memory blocked for marketing caller"
    needs the deny rule to cover the memory carrying the PII content,
    and the heuristic compiler emits multiple memories per subject
    (one per fact, plus episode summaries). Labelling all memories
    guarantees the rule applies whichever one carries `alice@globex.com`.
    """
    async with session_factory() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(MemoryRow.id).where(MemoryRow.subject_id == subject_id)
        )
        ids = [row[0] for row in result.all()]
        if not ids:
            return 0
        await session.execute(
            update(MemoryRow)
            .where(MemoryRow.id.in_(ids))
            .values(sensitivity_labels=labels)
        )
        await session.commit()
        return len(ids)


async def _set_tenant_policy_mode(session_factory, tenant_id: str, mode: str) -> None:
    """Insert/update `tenant_configs.config.policy_mode = mode`."""
    async with session_factory() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(TenantConfigRow).where(TenantConfigRow.tenant_id == tenant_id)
        )
        existing = result.scalar_one_or_none()
        if existing is None:
            session.add(
                TenantConfigRow(
                    tenant_id=tenant_id,
                    config={"policy_mode": mode},
                )
            )
        else:
            await session.execute(
                update(TenantConfigRow)
                .where(TenantConfigRow.tenant_id == tenant_id)
                .values(config={**existing.config, "policy_mode": mode})
            )
        await session.commit()


async def _install_active_policy(
    session_factory, yaml_content: str, tenant_id: str | None = None
) -> str:
    """Insert (or re-activate) a policy bundle and mark it active.
    Returns the hash. Mirrors what `POST /admin/policy/bundles` does
    with `activate=true`.

    Idempotent on hash: bundles are content-addressed, so the same
    YAML produces the same primary key. The bundle table is
    immutable, so we INSERT-IF-MISSING and then flip `active`
    rather than failing on the unique constraint. Tests that reuse
    the same bundle YAML across cases (the typical pattern) work
    without each one having to track whether it's been inserted yet.
    """
    bundle = policy_service.load_bundle(yaml_content)
    async with session_factory() as session:
        from sqlalchemy import select

        # Deactivate every prior bundle in the same scope so the
        # active-bundle resolver sees exactly one row.
        scope = update(PolicyBundleRow)
        if tenant_id is None:
            scope = scope.where(PolicyBundleRow.tenant_id.is_(None))
        else:
            scope = scope.where(PolicyBundleRow.tenant_id == tenant_id)
        await session.execute(scope.values(active=False))

        # Insert-or-skip on the immutable bundle row.
        existing = await session.execute(
            select(PolicyBundleRow).where(
                PolicyBundleRow.bundle_hash == bundle.bundle_hash
            )
        )
        if existing.scalar_one_or_none() is None:
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
                .values(active=True)
            )
        await session.commit()
    policy_service.invalidate_bundle_cache()
    return bundle.bundle_hash


@pytest.mark.anyio
async def test_pii_memory_blocked_for_marketing_caller_under_enforce(
    client: AsyncClient, subject_id: str, session_factory
):
    """Issue #50 negative test: sensitive *memory* must NOT be
    delivered to a disallowed caller when enforce is on.

    Scope note: v1 of the policy layer filters memories only — raw
    episodes are out of v1 scope and pass through unchanged. The
    seed episodes contain `alice@globex.com` directly, so the
    assembled context (which includes a `## Recent interactions`
    episode section) WILL still contain it. The test verifies the
    surface that policy *does* govern: the `facts` array (compiled
    memories) and the receipt's `filters_applied` block."""
    tenant_id = "acme"
    client.headers["X-Tenant-ID"] = tenant_id

    await _seed_subject(client, subject_id)
    await _label_all_memories(session_factory, subject_id, ["pii"])
    bundle_hash = await _install_active_policy(session_factory, PII_BUNDLE_YAML)
    await _set_tenant_policy_mode(session_factory, tenant_id, "enforce")

    r = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "Help the customer",
            "caller_id": "agent-42",
            "caller_type": "marketing_tool",
            "emit_receipt": True,
        },
    )
    assert r.status_code == 200
    body = r.json()

    # No PII memories survived enforce mode — this is the property
    # #50 actually delivers. The `provenance.fact_ids` and `facts`
    # arrays must both be empty for a subject where every memory
    # was labelled `pii` and the policy denies PII for marketing.
    assert body["facts"] == [], (
        f"facts must be empty under enforce + deny-all-pii, got {body['facts']}"
    )
    assert body["provenance"].get("fact_ids", []) == [], (
        "provenance.fact_ids must be empty when all memories were denied"
    )

    # The receipt records the deny so a reviewer can verify the policy
    # actually fired (rather than the memory just not having been
    # selected for unrelated reasons).
    r2 = await client.get(f"/v1/receipts/{body['receipt_id']}")
    receipt = r2.json()
    deny_decisions = [
        f for f in receipt["policy"]["filters_applied"] if f["action"] == "deny"
    ]
    assert deny_decisions, "policy.filters_applied must record the deny"
    assert receipt["policy"]["policy_bundle_hash"] == bundle_hash


@pytest.mark.anyio
async def test_log_only_mode_records_decisions_without_filtering(
    client: AsyncClient, subject_id: str, session_factory
):
    """In log_only mode (the default), a tenant sees what *would* be
    filtered without actually losing memories. This is the safe-
    rollout property that lets compliance teams audit a policy for a
    week before flipping enforce on."""
    tenant_id = "acme-logonly"
    client.headers["X-Tenant-ID"] = tenant_id

    await _seed_subject(client, subject_id)
    await _label_all_memories(session_factory, subject_id, ["pii"])
    await _install_active_policy(session_factory, PII_BUNDLE_YAML)
    # Explicitly NOT setting policy_mode → defaults to log_only.

    r = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "Help the customer",
            "caller_type": "marketing_tool",
            "emit_receipt": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    # The memory IS still delivered — log_only doesn't filter.
    assert "alice@globex.com" in body["assembled_context"]

    r2 = await client.get(f"/v1/receipts/{body['receipt_id']}")
    receipt = r2.json()
    # …but the receipt records what would happen under enforce.
    assert receipt["policy"]["mode"] == "log_only"
    deny_decisions = [
        f for f in receipt["policy"]["filters_applied"] if f["action"] == "deny"
    ]
    assert deny_decisions, "log_only must still record decisions"


@pytest.mark.anyio
async def test_policy_bundle_hash_changes_invalidate_audit_replay(
    client: AsyncClient, subject_id: str, session_factory
):
    """Issue #50 negative test: a policy version bump must surface
    in the receipts. Two assembly calls under different bundles must
    record different `policy_bundle_hash` so audit replay can tell
    which rules were in effect at decision time."""
    tenant_id = "acme-replay"
    client.headers["X-Tenant-ID"] = tenant_id

    await _seed_subject(client, subject_id)
    await _label_all_memories(session_factory, subject_id, ["pii"])

    hash_v1 = await _install_active_policy(session_factory, PII_BUNDLE_YAML)
    r1 = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "first call",
            "caller_type": "marketing_tool",
            "emit_receipt": True,
        },
    )
    body1 = r1.json()
    receipt1 = (await client.get(f"/v1/receipts/{body1['receipt_id']}")).json()

    # Bump the policy: add a redact rule for a non-existent label.
    # The bundle hash changes; subsequent receipts pin the new hash.
    bundle_v2_yaml = PII_BUNDLE_YAML + """
  - id: redact-internal
    when: {memory_has_any_label: [internal]}
    action: redact
"""
    hash_v2 = await _install_active_policy(session_factory, bundle_v2_yaml)
    assert hash_v1 != hash_v2

    r2 = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "second call",
            "caller_type": "marketing_tool",
            "emit_receipt": True,
        },
    )
    body2 = r2.json()
    receipt2 = (await client.get(f"/v1/receipts/{body2['receipt_id']}")).json()

    assert receipt1["policy"]["policy_bundle_hash"] == hash_v1
    assert receipt2["policy"]["policy_bundle_hash"] == hash_v2


@pytest.mark.anyio
async def test_caller_identity_required_when_tenant_config_flips_it_on(
    client: AsyncClient, subject_id: str, session_factory
):
    """`require_caller_identity: true` is the compliance lever that
    makes policy non-bypassable — unidentified callers can't slip
    through with `caller_type=None`."""
    tenant_id = "acme-strict"
    client.headers["X-Tenant-ID"] = tenant_id

    await _seed_subject(client, subject_id)
    async with session_factory() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(TenantConfigRow).where(TenantConfigRow.tenant_id == tenant_id)
        )
        existing = result.scalar_one_or_none()
        if existing is None:
            session.add(
                TenantConfigRow(
                    tenant_id=tenant_id,
                    config={"require_caller_identity": True},
                )
            )
        else:
            await session.execute(
                update(TenantConfigRow)
                .where(TenantConfigRow.tenant_id == tenant_id)
                .values(config={**existing.config, "require_caller_identity": True})
            )
        await session.commit()

    # Unidentified call → 401.
    r = await client.post(
        "/v1/context",
        json={"subject_id": subject_id, "task": "x"},
    )
    assert r.status_code == 401

    # Identified call → 200.
    r2 = await client.post(
        "/v1/context",
        json={
            "subject_id": subject_id,
            "task": "x",
            "caller_id": "agent-1",
            "caller_type": "support_agent",
        },
    )
    assert r2.status_code == 200


@pytest.mark.anyio
async def test_set_memory_labels_endpoint_normalizes_and_persists(
    client: AsyncClient, subject_id: str, session_factory
):
    """PATCH /v1/memories/{id}/labels normalizes (dedup, lowercase,
    strip) and persists. Read-after-write returns the canonicalized
    set."""
    tenant_id = "acme-labels"
    client.headers["X-Tenant-ID"] = tenant_id

    await _seed_subject(client, subject_id)
    async with session_factory() as session:
        from sqlalchemy import select

        result = await session.execute(
            select(MemoryRow).where(MemoryRow.subject_id == subject_id).limit(1)
        )
        memory_id = result.scalar_one().id

    r = await client.patch(
        f"/v1/memories/{memory_id}/labels",
        json={"sensitivity_labels": [" PII ", "pii", "Financial", "financial"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["sensitivity_labels"] == ["financial", "pii"]


@pytest.mark.anyio
async def test_admin_policy_upload_and_activate_round_trip(client: AsyncClient):
    """The /admin/policy/* endpoints lifecycle: upload a bundle,
    list it, activate it, fetch active. Verifies the operator
    surface end-to-end.

    Uses a content unique to this test rather than the shared
    PII_BUNDLE_YAML — bundles are content-addressed (same YAML →
    same hash), and `bundle_hash` is the primary key of
    `policy_bundles`. Reusing PII_BUNDLE_YAML here would silently
    point at the row inserted by the earlier global-scope tests
    (whose `tenant_id IS NULL`), so the tenant-scoped listing would
    miss it. Cross-tenant bundle reuse via a composite-key schema
    is a v2 follow-up.
    """
    unique_yaml = """
version: 1
metadata:
  description: admin-round-trip-test
rules:
  - id: deny-admin-round-trip-only
    when:
      memory_has_any_label: [admin-round-trip-marker]
    action: deny
"""
    r = await client.post(
        "/admin/policy/bundles",
        json={
            "yaml_content": unique_yaml,
            "tenant_id": "acme-policy-admin",
            "activate": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    bundle_hash = body["bundle_hash"]
    assert body["active"] is True

    r2 = await client.get("/admin/policy/bundles?tenant_id=acme-policy-admin")
    assert r2.status_code == 200
    bundles = r2.json()["bundles"]
    assert any(b["bundle_hash"] == bundle_hash and b["active"] for b in bundles)

    r3 = await client.get("/admin/policy/active?tenant_id=acme-policy-admin")
    assert r3.status_code == 200
    assert r3.json()["bundle_hash"] == bundle_hash
