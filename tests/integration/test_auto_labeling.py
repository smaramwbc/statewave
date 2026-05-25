"""End-to-end integration tests for auto-labeling (v0.9 #158).

Covers:
  * Compile path stamps ``suggested_labels`` on real MemoryRows when
    the feature flag is on, and stamps nothing when it is off.
  * Authoritative ``sensitivity_labels`` is NEVER mutated by the
    pipeline — this is the load-bearing isolation property (the
    policy evaluator reads sensitivity_labels; a regression that
    leaked suggestions there would silently tighten policy).
  * ``GET /admin/memories/with-suggested-labels`` returns only rows
    with suggestions, supports filtering, and surfaces the catalogue.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from server.core.config import settings
from server.db.tables import MemoryRow


_EMAIL_PAYLOAD = {
    "source": "test",
    "type": "chat.session",
    "payload": {
        "messages": [
            {"role": "user", "content": "Reach me at alice@example.com any time."},
            {"role": "assistant", "content": "Got it, will email there."},
        ]
    },
}

_CARD_PAYLOAD = {
    "source": "test",
    "type": "chat.session",
    "payload": {
        "messages": [
            {"role": "user", "content": "My card is 4111-1111-1111-1111."},
        ]
    },
}


# ---------------------------------------------------------------------------
# Compile path — feature flag gates the pipeline
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_compile_stamps_suggested_labels_when_enabled(
    client: AsyncClient, subject_id: str, monkeypatch
):
    """With the flag ON, the compile path must stamp pii.email on the
    memories derived from an email-containing episode."""
    monkeypatch.setattr(settings, "auto_labeling_enabled", True)

    resp = await client.post("/v1/episodes", json={**_EMAIL_PAYLOAD, "subject_id": subject_id})
    assert resp.status_code == 201

    resp = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert resp.status_code == 200
    assert resp.json()["memories_created"] > 0

    # Read back from DB and verify the column.
    from server.db.engine import get_session_factory

    async with get_session_factory()() as session:
        rows = (
            (await session.execute(select(MemoryRow).where(MemoryRow.subject_id == subject_id)))
            .scalars()
            .all()
        )

    assert rows, "compile should have produced memories"
    assert any("pii.email" in (m.suggested_labels or []) for m in rows), (
        "at least one memory should carry pii.email"
    )


@pytest.mark.anyio
async def test_compile_skips_suggested_labels_when_disabled(
    client: AsyncClient, subject_id: str, monkeypatch
):
    """With the flag OFF (the default), no memory should carry any
    suggested label even when the episode obviously contains PII.
    This is the backwards-compatibility guarantee."""
    monkeypatch.setattr(settings, "auto_labeling_enabled", False)

    resp = await client.post("/v1/episodes", json={**_EMAIL_PAYLOAD, "subject_id": subject_id})
    assert resp.status_code == 201

    resp = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert resp.status_code == 200

    from server.db.engine import get_session_factory

    async with get_session_factory()() as session:
        rows = (
            (await session.execute(select(MemoryRow).where(MemoryRow.subject_id == subject_id)))
            .scalars()
            .all()
        )

    assert rows
    for m in rows:
        assert (m.suggested_labels or []) == [], (
            f"memory {m.id} carries suggestions while flag is OFF"
        )


# ---------------------------------------------------------------------------
# Policy isolation — load-bearing
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_auto_labeling_never_writes_sensitivity_labels(
    client: AsyncClient, subject_id: str, monkeypatch
):
    """The pipeline writes to ``suggested_labels`` only; the
    authoritative ``sensitivity_labels`` column the policy evaluator
    reads must remain empty (or whatever the tenant set explicitly).
    A regression that swapped these columns would silently tighten
    policy on real traffic — keep the assertion explicit."""
    monkeypatch.setattr(settings, "auto_labeling_enabled", True)

    resp = await client.post("/v1/episodes", json={**_CARD_PAYLOAD, "subject_id": subject_id})
    assert resp.status_code == 201

    resp = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert resp.status_code == 200

    from server.db.engine import get_session_factory

    async with get_session_factory()() as session:
        rows = (
            (await session.execute(select(MemoryRow).where(MemoryRow.subject_id == subject_id)))
            .scalars()
            .all()
        )

    assert rows
    # At least one row must carry the suggested label; otherwise the
    # rest of the assertion is meaningless.
    assert any("financial.card" in (m.suggested_labels or []) for m in rows)
    # NOT ONE row may carry it under sensitivity_labels.
    for m in rows:
        assert "financial.card" not in (m.sensitivity_labels or [])
        assert "pii.email" not in (m.sensitivity_labels or [])
        assert "secret.token" not in (m.sensitivity_labels or [])


# ---------------------------------------------------------------------------
# Admin review endpoint
# ---------------------------------------------------------------------------


async def _insert_memory_with_suggested(
    session_factory,
    *,
    subject_id: str,
    tenant_id: str | None,
    suggested: list[str],
) -> uuid.UUID:
    mid = uuid.uuid4()
    async with session_factory() as session:
        session.add(
            MemoryRow(
                id=mid,
                subject_id=subject_id,
                tenant_id=tenant_id,
                kind="profile_fact",
                content="alice@example.com",
                summary="alice email",
                confidence=1.0,
                valid_from=datetime.now(timezone.utc),
                valid_to=None,
                source_episode_ids=[],
                metadata_={},
                status="active",
                sensitivity_labels=[],
                suggested_labels=suggested,
            )
        )
        await session.commit()
    return mid


@pytest.mark.anyio
async def test_admin_list_only_returns_rows_with_suggestions(client: AsyncClient, session_factory):
    """The endpoint filters out memories with empty suggested_labels."""
    s_with = f"with-{uuid.uuid4().hex[:8]}"
    s_without = f"without-{uuid.uuid4().hex[:8]}"

    await _insert_memory_with_suggested(
        session_factory, subject_id=s_with, tenant_id=None, suggested=["pii.email"]
    )
    await _insert_memory_with_suggested(
        session_factory, subject_id=s_without, tenant_id=None, suggested=[]
    )

    resp = await client.get(
        "/admin/memories/with-suggested-labels",
        params={"subject_id": s_with},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["memories"][0]["suggested_labels"] == ["pii.email"]

    # Same call against the empty-suggestions subject returns nothing.
    resp = await client.get(
        "/admin/memories/with-suggested-labels",
        params={"subject_id": s_without},
    )
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


@pytest.mark.anyio
async def test_admin_list_label_filter(client: AsyncClient, session_factory):
    """The `label` query param narrows to memories carrying that
    specific suggestion. Uses the GIN-indexed overlap path."""
    s = f"filter-{uuid.uuid4().hex[:8]}"
    await _insert_memory_with_suggested(
        session_factory, subject_id=s, tenant_id=None, suggested=["pii.email"]
    )
    await _insert_memory_with_suggested(
        session_factory, subject_id=s, tenant_id=None, suggested=["financial.card"]
    )

    resp = await client.get(
        "/admin/memories/with-suggested-labels",
        params={"subject_id": s, "label": "pii.email"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["memories"][0]["suggested_labels"] == ["pii.email"]


@pytest.mark.anyio
async def test_admin_list_includes_catalogue(client: AsyncClient, session_factory):
    """Response carries the detector catalogue so the admin UI can
    populate filter dropdowns without a second round-trip."""
    resp = await client.get("/admin/memories/with-suggested-labels")
    assert resp.status_code == 200
    body = resp.json()
    labels = {entry["label"] for entry in body["catalogue"]}
    assert {"pii.email", "pii.phone", "financial.card", "secret.token"}.issubset(labels)


@pytest.mark.anyio
async def test_admin_list_pagination(client: AsyncClient, session_factory):
    """limit + offset behave as advertised."""
    s = f"page-{uuid.uuid4().hex[:8]}"
    for _ in range(3):
        await _insert_memory_with_suggested(
            session_factory, subject_id=s, tenant_id=None, suggested=["pii.email"]
        )

    resp = await client.get(
        "/admin/memories/with-suggested-labels",
        params={"subject_id": s, "limit": 2, "offset": 0},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert len(body["memories"]) == 2
    assert body["limit"] == 2
    assert body["offset"] == 0


# ---------------------------------------------------------------------------
# Promote suggested → sensitivity labels (v0.9 #160)
# ---------------------------------------------------------------------------


async def _read_memory(session_factory, memory_id: uuid.UUID) -> MemoryRow:
    from sqlalchemy import select

    async with session_factory() as session:
        return (
            await session.execute(select(MemoryRow).where(MemoryRow.id == memory_id))
        ).scalar_one()


@pytest.mark.anyio
async def test_promote_labels_moves_suggested_to_sensitivity(client: AsyncClient, session_factory):
    """Happy path: a label currently in suggested_labels lands in
    sensitivity_labels and is removed from suggested_labels."""
    s = f"promote-{uuid.uuid4().hex[:8]}"
    mid = await _insert_memory_with_suggested(
        session_factory,
        subject_id=s,
        tenant_id=None,
        suggested=["pii.email", "pii.phone"],
    )

    resp = await client.post(
        f"/admin/memories/{mid}/promote-labels",
        json={"labels": ["pii.email"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["memory_id"] == str(mid)
    assert body["promoted"] == ["pii.email"]
    assert body["sensitivity_labels"] == ["pii.email"]
    # `pii.phone` stays in suggested_labels — only `pii.email` was promoted.
    assert body["suggested_labels"] == ["pii.phone"]

    # DB matches the response and an audit entry was stamped.
    row = await _read_memory(session_factory, mid)
    assert "pii.email" in row.sensitivity_labels
    assert "pii.email" not in row.suggested_labels
    promotions = row.metadata_.get("label_promotions") or []
    assert len(promotions) == 1
    assert promotions[0]["labels"] == ["pii.email"]
    assert promotions[0]["promoted_at"]
    # promoted_by is null in v0.9 — admin identity TODO lands later
    assert promotions[0]["promoted_by"] is None


@pytest.mark.anyio
async def test_promote_labels_preserves_existing_sensitivity_labels(
    client: AsyncClient, session_factory
):
    """Pre-existing sensitivity_labels (from explicit tenant SDK writes)
    must survive a promotion — the endpoint merges, never overwrites."""
    s = f"merge-{uuid.uuid4().hex[:8]}"
    mid = uuid.uuid4()
    async with session_factory() as session:
        session.add(
            MemoryRow(
                id=mid,
                subject_id=s,
                tenant_id=None,
                kind="profile_fact",
                content="alice@example.com",
                summary="x",
                confidence=1.0,
                valid_from=datetime.now(timezone.utc),
                source_episode_ids=[],
                metadata_={},
                status="active",
                sensitivity_labels=["legal.contract"],  # tenant-set
                suggested_labels=["pii.email"],
            )
        )
        await session.commit()

    resp = await client.post(
        f"/admin/memories/{mid}/promote-labels",
        json={"labels": ["pii.email"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Result is sorted + deduped, tenant's existing label preserved.
    assert body["sensitivity_labels"] == ["legal.contract", "pii.email"]


@pytest.mark.anyio
async def test_promote_labels_rejects_unsuggested_label(client: AsyncClient, session_factory):
    """Ad-hoc promotion is rejected — every label in the request must
    currently be in suggested_labels. This is the load-bearing
    constraint that keeps the endpoint review-only and prevents it
    from becoming a backdoor write surface."""
    s = f"reject-{uuid.uuid4().hex[:8]}"
    mid = await _insert_memory_with_suggested(
        session_factory, subject_id=s, tenant_id=None, suggested=["pii.email"]
    )

    resp = await client.post(
        f"/admin/memories/{mid}/promote-labels",
        json={"labels": ["financial.card"]},  # never suggested on this memory
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "promote_labels.not_suggested"


@pytest.mark.anyio
async def test_promote_labels_empty_request_rejected(client: AsyncClient, session_factory):
    s = f"empty-{uuid.uuid4().hex[:8]}"
    mid = await _insert_memory_with_suggested(
        session_factory, subject_id=s, tenant_id=None, suggested=["pii.email"]
    )

    resp = await client.post(f"/admin/memories/{mid}/promote-labels", json={"labels": []})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "promote_labels.empty"


@pytest.mark.anyio
async def test_promote_labels_duplicate_in_request_rejected(client: AsyncClient, session_factory):
    """Duplicates inside the request body are rejected — keeps the
    audit entry honest (a single API call promotes a unique set)."""
    s = f"dupe-{uuid.uuid4().hex[:8]}"
    mid = await _insert_memory_with_suggested(
        session_factory, subject_id=s, tenant_id=None, suggested=["pii.email"]
    )

    resp = await client.post(
        f"/admin/memories/{mid}/promote-labels",
        json={"labels": ["pii.email", "pii.email"]},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "promote_labels.duplicate_labels"


@pytest.mark.anyio
async def test_promote_labels_404_for_unknown_memory(client: AsyncClient):
    bogus = uuid.uuid4()
    resp = await client.post(
        f"/admin/memories/{bogus}/promote-labels", json={"labels": ["pii.email"]}
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_promote_labels_400_for_invalid_uuid(client: AsyncClient):
    resp = await client.post(
        "/admin/memories/not-a-uuid/promote-labels", json={"labels": ["pii.email"]}
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_promote_labels_idempotent_rejects_second_call(client: AsyncClient, session_factory):
    """After the first promotion, the label is no longer in
    suggested_labels, so a second call with the same label hits the
    not_suggested guard. This is the desired idempotency contract —
    re-running converges silently from the caller's side: the row is
    in the right state, and the audit entry from the first call is
    preserved."""
    s = f"idem-{uuid.uuid4().hex[:8]}"
    mid = await _insert_memory_with_suggested(
        session_factory, subject_id=s, tenant_id=None, suggested=["pii.email"]
    )

    resp = await client.post(
        f"/admin/memories/{mid}/promote-labels", json={"labels": ["pii.email"]}
    )
    assert resp.status_code == 200

    resp = await client.post(
        f"/admin/memories/{mid}/promote-labels", json={"labels": ["pii.email"]}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "promote_labels.not_suggested"

    # Single audit entry survived.
    row = await _read_memory(session_factory, mid)
    promotions = row.metadata_.get("label_promotions") or []
    assert len(promotions) == 1


@pytest.mark.anyio
async def test_promote_labels_tenant_scoped(client: AsyncClient, session_factory):
    """Passing a tenant_id that doesn't own the memory returns 404 —
    defence in depth against a misconfigured admin caller."""
    s = f"tenant-{uuid.uuid4().hex[:8]}"
    mid = await _insert_memory_with_suggested(
        session_factory, subject_id=s, tenant_id="tenant-a", suggested=["pii.email"]
    )

    resp = await client.post(
        f"/admin/memories/{mid}/promote-labels",
        params={"tenant_id": "tenant-b"},
        json={"labels": ["pii.email"]},
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_promote_labels_appends_subsequent_audit_entries(
    client: AsyncClient, session_factory
):
    """Two promotions of different labels on the same memory produce
    two distinct audit entries, in chronological order."""
    s = f"audit-{uuid.uuid4().hex[:8]}"
    mid = await _insert_memory_with_suggested(
        session_factory,
        subject_id=s,
        tenant_id=None,
        suggested=["pii.email", "pii.phone"],
    )

    r1 = await client.post(f"/admin/memories/{mid}/promote-labels", json={"labels": ["pii.email"]})
    assert r1.status_code == 200
    r2 = await client.post(f"/admin/memories/{mid}/promote-labels", json={"labels": ["pii.phone"]})
    assert r2.status_code == 200

    row = await _read_memory(session_factory, mid)
    promotions = row.metadata_.get("label_promotions") or []
    assert len(promotions) == 2
    assert promotions[0]["labels"] == ["pii.email"]
    assert promotions[1]["labels"] == ["pii.phone"]
    assert promotions[0]["promoted_at"] <= promotions[1]["promoted_at"]
    assert set(row.sensitivity_labels) == {"pii.email", "pii.phone"}
    assert row.suggested_labels == []
