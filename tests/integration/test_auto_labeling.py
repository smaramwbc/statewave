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
