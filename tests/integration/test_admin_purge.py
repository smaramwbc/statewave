"""Real-DB tests for the admin purge endpoints.

Lives under tests/integration/ because these exercise the actual SQL
DELETE against a Postgres test database. The unit-level tests in
tests/test_admin_purge.py already cover validation and route
registration without needing the DB.

What we verify here:
- Happy path: only matching rows are deleted, the count is correct.
- Tenant scoping: a tenant filter cannot reach into another tenant's rows.
- No cascade: purging jobs/events does not touch unrelated tables (memories
  are not in scope, but we keep a sentinel row to prove the DELETE is
  narrow).
- Status restriction: a `pending`/`running` row is never touched even
  when its other selectors match the request — the service must enforce
  the terminal-status floor regardless of caller intent.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from server.db.tables import CompileJobRow, WebhookEventRow

pytestmark = pytest.mark.anyio


# ─── Helpers ─────────────────────────────────────────────────────────────────


async def _seed_jobs(session_factory, *rows: dict) -> None:
    """Insert CompileJobRow rows directly. Each dict overrides defaults."""
    async with session_factory() as session:
        for r in rows:
            session.add(
                CompileJobRow(
                    id=r.get("id", str(uuid.uuid4())),
                    subject_id=r["subject_id"],
                    tenant_id=r.get("tenant_id"),
                    status=r["status"],
                    memories_created=r.get("memories_created", 0),
                    error=r.get("error"),
                    created_at=r.get("created_at", datetime.now(timezone.utc)),
                )
            )
        await session.commit()


async def _seed_events(session_factory, *rows: dict) -> None:
    async with session_factory() as session:
        for r in rows:
            session.add(
                WebhookEventRow(
                    id=r.get("id", uuid.uuid4()),
                    tenant_id=r.get("tenant_id"),
                    event=r["event"],
                    payload=r.get("payload", {}),
                    status=r["status"],
                    attempts=r.get("attempts", 0),
                    max_attempts=r.get("max_attempts", 5),
                )
            )
        await session.commit()


async def _count_jobs(session_factory, **filters) -> int:
    async with session_factory() as session:
        stmt = select(CompileJobRow)
        for k, v in filters.items():
            stmt = stmt.where(getattr(CompileJobRow, k) == v)
        result = await session.execute(stmt)
        return len(result.scalars().all())


async def _count_events(session_factory, **filters) -> int:
    async with session_factory() as session:
        stmt = select(WebhookEventRow)
        for k, v in filters.items():
            stmt = stmt.where(getattr(WebhookEventRow, k) == v)
        result = await session.execute(stmt)
        return len(result.scalars().all())


# ─── Compile Jobs ────────────────────────────────────────────────────────────


async def test_purge_jobs_status_only(client: AsyncClient, session_factory):
    """`status=failed` deletes the failed row and leaves the others."""
    tag = uuid.uuid4().hex[:8]
    await _seed_jobs(
        session_factory,
        {"subject_id": f"keep_{tag}_a", "status": "completed"},
        {"subject_id": f"go_{tag}_b", "status": "failed"},
        {"subject_id": f"go_{tag}_c", "status": "failed"},
        {"subject_id": f"keep_{tag}_d", "status": "running"},
    )

    r = await client.request("DELETE", "/admin/jobs", params={"status": "failed"})
    assert r.status_code == 200
    assert r.json()["deleted"] >= 2  # the two failed rows we seeded

    # The non-failed rows we seeded are still there.
    assert await _count_jobs(session_factory, subject_id=f"keep_{tag}_a") == 1
    assert await _count_jobs(session_factory, subject_id=f"keep_{tag}_d") == 1
    # The failed rows are gone.
    assert await _count_jobs(session_factory, subject_id=f"go_{tag}_b") == 0
    assert await _count_jobs(session_factory, subject_id=f"go_{tag}_c") == 0


async def test_purge_jobs_rejects_running(client: AsyncClient, session_factory):
    """A `running` job can never be deleted via this endpoint, even when
    other selectors match. The 4xx surfaces the constraint to the caller."""
    tag = uuid.uuid4().hex[:8]
    await _seed_jobs(
        session_factory, {"subject_id": f"running_{tag}", "status": "running"}
    )

    r = await client.request(
        "DELETE",
        "/admin/jobs",
        params={"status": "running", "subject_id": f"running_{tag}"},
    )
    assert r.status_code == 400
    # Sentinel survives.
    assert await _count_jobs(session_factory, subject_id=f"running_{tag}") == 1


async def test_purge_jobs_empty_filter_rejected(client: AsyncClient, session_factory):
    """No filter → no delete. Belt-and-suspenders with the unit test, but
    this one runs against the real DB so we catch any case where the
    validation doesn't actually short-circuit before the SQL fires."""
    tag = uuid.uuid4().hex[:8]
    await _seed_jobs(
        session_factory, {"subject_id": f"sentinel_{tag}", "status": "completed"}
    )
    before = await _count_jobs(session_factory)

    r = await client.request("DELETE", "/admin/jobs")
    assert r.status_code == 400

    after = await _count_jobs(session_factory)
    assert before == after
    assert await _count_jobs(session_factory, subject_id=f"sentinel_{tag}") == 1


async def test_purge_jobs_tenant_scoping(client: AsyncClient, session_factory):
    """A tenant filter cannot reach into another tenant's rows."""
    tag = uuid.uuid4().hex[:8]
    await _seed_jobs(
        session_factory,
        {
            "subject_id": f"a_{tag}",
            "tenant_id": f"tenant-a-{tag}",
            "status": "failed",
        },
        {
            "subject_id": f"b_{tag}",
            "tenant_id": f"tenant-b-{tag}",
            "status": "failed",
        },
    )

    r = await client.request(
        "DELETE",
        "/admin/jobs",
        params={"status": "failed", "tenant_id": f"tenant-a-{tag}"},
    )
    assert r.status_code == 200

    # Tenant A's row is gone, tenant B's survives untouched.
    assert await _count_jobs(session_factory, tenant_id=f"tenant-a-{tag}") == 0
    assert await _count_jobs(session_factory, tenant_id=f"tenant-b-{tag}") == 1


async def test_purge_jobs_subject_filter(client: AsyncClient, session_factory):
    """A subject filter narrows the delete to a single subject's terminal
    rows, leaving every other subject's data alone."""
    tag = uuid.uuid4().hex[:8]
    target = f"target_{tag}"
    other = f"other_{tag}"
    await _seed_jobs(
        session_factory,
        {"subject_id": target, "status": "completed"},
        {"subject_id": target, "status": "failed"},
        {"subject_id": other, "status": "completed"},
    )

    r = await client.request(
        "DELETE", "/admin/jobs", params={"subject_id": target}
    )
    assert r.status_code == 200

    assert await _count_jobs(session_factory, subject_id=target) == 0
    assert await _count_jobs(session_factory, subject_id=other) == 1


# ─── Webhook Events ──────────────────────────────────────────────────────────


async def test_purge_webhooks_status_only(client: AsyncClient, session_factory):
    """`status=dead_letter` deletes the dead-letter rows; pending survives."""
    tag = uuid.uuid4().hex[:8]
    await _seed_events(
        session_factory,
        {"event": f"e_{tag}_a", "status": "dead_letter"},
        {"event": f"e_{tag}_b", "status": "dead_letter"},
        {"event": f"e_{tag}_c", "status": "pending"},
        {"event": f"e_{tag}_d", "status": "delivered"},
    )

    r = await client.request(
        "DELETE", "/admin/webhooks", params={"status": "dead_letter"}
    )
    assert r.status_code == 200

    # Pending and delivered survive; dead-letter rows are gone.
    assert await _count_events(session_factory, event=f"e_{tag}_a") == 0
    assert await _count_events(session_factory, event=f"e_{tag}_b") == 0
    assert await _count_events(session_factory, event=f"e_{tag}_c") == 1
    assert await _count_events(session_factory, event=f"e_{tag}_d") == 1


async def test_purge_webhooks_rejects_pending(client: AsyncClient, session_factory):
    """`pending` events may still be in flight in the delivery worker —
    a caller asking to delete them must be rejected before any SQL runs."""
    tag = uuid.uuid4().hex[:8]
    await _seed_events(
        session_factory, {"event": f"pending_{tag}", "status": "pending"}
    )

    r = await client.request(
        "DELETE", "/admin/webhooks", params={"status": "pending"}
    )
    assert r.status_code == 400
    assert await _count_events(session_factory, event=f"pending_{tag}") == 1


async def test_purge_webhooks_event_type(client: AsyncClient, session_factory):
    """`event_type` alone (no status) wipes both terminal statuses for
    that type while leaving in-flight rows of the same type alone."""
    tag = uuid.uuid4().hex[:8]
    target_event = f"target.{tag}"
    other_event = f"other.{tag}"
    await _seed_events(
        session_factory,
        {"event": target_event, "status": "delivered"},
        {"event": target_event, "status": "dead_letter"},
        {"event": target_event, "status": "pending"},
        {"event": other_event, "status": "delivered"},
    )

    r = await client.request(
        "DELETE", "/admin/webhooks", params={"event_type": target_event}
    )
    assert r.status_code == 200

    # Only the two terminal target rows are gone.
    assert (
        await _count_events(
            session_factory, event=target_event, status="delivered"
        )
        == 0
    )
    assert (
        await _count_events(
            session_factory, event=target_event, status="dead_letter"
        )
        == 0
    )
    # Pending of the same type survives — the worker may still pick it up.
    assert (
        await _count_events(session_factory, event=target_event, status="pending")
        == 1
    )
    # Unrelated event-type row is untouched.
    assert (
        await _count_events(
            session_factory, event=other_event, status="delivered"
        )
        == 1
    )


async def test_purge_webhooks_tenant_scoping(client: AsyncClient, session_factory):
    """Tenant filter cannot delete cross-tenant events."""
    tag = uuid.uuid4().hex[:8]
    await _seed_events(
        session_factory,
        {
            "event": f"a_{tag}",
            "tenant_id": f"tenant-a-{tag}",
            "status": "dead_letter",
        },
        {
            "event": f"b_{tag}",
            "tenant_id": f"tenant-b-{tag}",
            "status": "dead_letter",
        },
    )

    r = await client.request(
        "DELETE",
        "/admin/webhooks",
        params={"status": "dead_letter", "tenant_id": f"tenant-a-{tag}"},
    )
    assert r.status_code == 200

    assert await _count_events(session_factory, tenant_id=f"tenant-a-{tag}") == 0
    assert await _count_events(session_factory, tenant_id=f"tenant-b-{tag}") == 1


async def test_purge_webhooks_empty_filter_rejected(
    client: AsyncClient, session_factory
):
    tag = uuid.uuid4().hex[:8]
    await _seed_events(
        session_factory, {"event": f"sentinel_{tag}", "status": "delivered"}
    )

    r = await client.request("DELETE", "/admin/webhooks")
    assert r.status_code == 400
    assert await _count_events(session_factory, event=f"sentinel_{tag}") == 1
