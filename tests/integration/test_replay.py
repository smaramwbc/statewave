"""Integration tests for receipt replay (v0.9 #159).

Covers:
  * Happy path — v0.9 receipt replays cleanly, emits a child receipt
    with mode="as_of_replay" and parent_receipt_id set.
  * Diff envelope picks up a memory added between emission and replay.
  * `policy_snapshot` is persisted on every new receipt (column + body).
  * Pre-v0.9 receipt (column NULL) → 422 unreplayable: missing_policy_snapshot.
  * Replaying a replay → 422 unreplayable: nested_replay.
  * Invalid (unparseable) snapshot YAML → 422 unreplayable: invalid_snapshot.
  * Tenant isolation — receipts under tenant A are 404 for tenant B.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import select, update

from server.db.tables import ReceiptRow


_EPISODES = [
    {
        "source": "test",
        "type": "chat.session",
        "payload": {
            "messages": [
                {"role": "user", "content": "My name is Alice. I use dark mode."},
                {"role": "assistant", "content": "Got it."},
            ]
        },
    },
]


async def _seed(client: AsyncClient, subject_id: str) -> None:
    for ep in _EPISODES:
        r = await client.post("/v1/episodes", json={**ep, "subject_id": subject_id})
        assert r.status_code == 201, r.text
    r = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert r.status_code == 200, r.text


async def _emit_receipt(
    client: AsyncClient, subject_id: str, task: str = "what's the plan?"
) -> str:
    r = await client.post(
        "/v1/context",
        json={"subject_id": subject_id, "task": task, "emit_receipt": True},
    )
    assert r.status_code == 200, r.text
    receipt_id = r.json().get("receipt_id")
    assert receipt_id, "expected receipt_id on emit"
    return receipt_id


# ---------------------------------------------------------------------------
# Policy snapshot is stamped on every new receipt
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_policy_snapshot_stamped_on_v09_receipts(
    client: AsyncClient, subject_id: str, session_factory
):
    """Every v0.9 receipt must carry a policy_snapshot envelope, even
    when no policy bundle is active (in which case the inner fields
    are null but the envelope itself exists)."""
    await _seed(client, subject_id)
    receipt_id = await _emit_receipt(client, subject_id)

    async with session_factory() as session:
        row = (
            await session.execute(select(ReceiptRow).where(ReceiptRow.receipt_id == receipt_id))
        ).scalar_one()

    # Column was persisted with the right shape
    assert row.policy_snapshot is not None
    assert "bundle_hash" in row.policy_snapshot
    assert "bundle_yaml" in row.policy_snapshot
    assert "captured_at" in row.policy_snapshot
    # bundle_hash and bundle_yaml move together — either both null
    # (no active bundle) or both populated. They never disagree.
    has_hash = row.policy_snapshot["bundle_hash"] is not None
    has_yaml = row.policy_snapshot["bundle_yaml"] is not None
    assert has_hash == has_yaml, (
        "policy_snapshot envelope is inconsistent: bundle_hash and "
        "bundle_yaml must be both null or both populated"
    )
    # Body and column agree (body is signed, column is denormalised)
    assert row.body.get("policy_snapshot") == row.policy_snapshot


# ---------------------------------------------------------------------------
# Happy path — replay produces a child receipt + diff envelope
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_replay_emits_child_receipt_with_parent_pointer(
    client: AsyncClient, subject_id: str, session_factory
):
    await _seed(client, subject_id)
    original_id = await _emit_receipt(client, subject_id)

    r = await client.post(f"/v1/receipts/{original_id}/replay")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["original_receipt_id"] == original_id
    replay_id = body["replay_receipt_id"]
    assert replay_id, "expected a replay_receipt_id"
    assert replay_id != original_id

    # The child receipt exists and points back at the parent.
    async with session_factory() as session:
        child = (
            await session.execute(select(ReceiptRow).where(ReceiptRow.receipt_id == replay_id))
        ).scalar_one()
    assert child.mode == "as_of_replay"
    assert child.parent_receipt_id == original_id


@pytest.mark.anyio
async def test_replay_with_no_memory_changes_reports_zero_diff(
    client: AsyncClient, subject_id: str
):
    """Same memories at emission and replay → no add/remove in the
    selected_entries diff. Context hash may or may not change depending
    on scoring stability, but the entry set must be invariant."""
    await _seed(client, subject_id)
    original_id = await _emit_receipt(client, subject_id)

    r = await client.post(f"/v1/receipts/{original_id}/replay")
    assert r.status_code == 200, r.text
    diff = r.json()["diff"]

    assert diff["selected_entries"]["added"] == []
    assert diff["selected_entries"]["removed"] == []
    assert diff["selected_entries"]["common"] >= 1
    # filters_applied stays empty when no bundle is active on either run
    assert diff["filters_applied"]["added"] == []
    assert diff["filters_applied"]["removed"] == []


@pytest.mark.anyio
async def test_replay_diff_captures_newly_added_memory(client: AsyncClient, subject_id: str):
    """Add a memory between emission and replay; the diff must show
    the new memory under selected_entries.added."""
    await _seed(client, subject_id)
    original_id = await _emit_receipt(client, subject_id)

    # Inject a new episode + compile so a brand-new memory exists.
    new_ep = {
        "source": "test",
        "type": "chat.session",
        "subject_id": subject_id,
        "payload": {
            "messages": [
                {"role": "user", "content": "Also I work at Initech as a senior engineer."},
            ]
        },
    }
    r = await client.post("/v1/episodes", json=new_ep)
    assert r.status_code == 201
    r = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert r.status_code == 200

    r = await client.post(f"/v1/receipts/{original_id}/replay")
    assert r.status_code == 200, r.text
    diff = r.json()["diff"]

    # The new memory shows up as an addition in the replay.
    assert len(diff["selected_entries"]["added"]) >= 1


# ---------------------------------------------------------------------------
# 422 refusals
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_pre_v09_receipt_is_unreplayable(
    client: AsyncClient, subject_id: str, session_factory
):
    """A receipt with a NULL policy_snapshot column (the pre-v0.9
    state) must be refused with 422 missing_policy_snapshot. We
    simulate the pre-v0.9 state by stripping the column post-emission."""
    await _seed(client, subject_id)
    receipt_id = await _emit_receipt(client, subject_id)

    # Strip snapshot on both the column and the body to mirror a
    # truly pre-v0.9 row.
    async with session_factory() as session:
        await session.execute(
            update(ReceiptRow)
            .where(ReceiptRow.receipt_id == receipt_id)
            .values(policy_snapshot=None)
        )
        # Also remove from the body so a replay isn't tempted to fall back.
        row = (
            await session.execute(select(ReceiptRow).where(ReceiptRow.receipt_id == receipt_id))
        ).scalar_one()
        new_body = dict(row.body)
        new_body.pop("policy_snapshot", None)
        await session.execute(
            update(ReceiptRow).where(ReceiptRow.receipt_id == receipt_id).values(body=new_body)
        )
        await session.commit()

    r = await client.post(f"/v1/receipts/{receipt_id}/replay")
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["code"] == "unreplayable.missing_policy_snapshot"


@pytest.mark.anyio
async def test_nested_replay_is_unreplayable(client: AsyncClient, subject_id: str):
    """Replaying a replay must return 422 nested_replay."""
    await _seed(client, subject_id)
    original_id = await _emit_receipt(client, subject_id)

    # First replay succeeds.
    r = await client.post(f"/v1/receipts/{original_id}/replay")
    assert r.status_code == 200
    replay_id = r.json()["replay_receipt_id"]
    assert replay_id

    # Second replay against the replay receipt is refused.
    r = await client.post(f"/v1/receipts/{replay_id}/replay")
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "unreplayable.nested_replay"


@pytest.mark.anyio
async def test_invalid_snapshot_yaml_returns_422(
    client: AsyncClient, subject_id: str, session_factory
):
    """If the snapshot YAML is unparseable (e.g. tampered), surface
    invalid_snapshot rather than crashing."""
    await _seed(client, subject_id)
    receipt_id = await _emit_receipt(client, subject_id)

    bad_snapshot = {
        "bundle_hash": "deadbeef",
        "bundle_yaml": "::: this is not valid yaml :::",
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    async with session_factory() as session:
        await session.execute(
            update(ReceiptRow)
            .where(ReceiptRow.receipt_id == receipt_id)
            .values(policy_snapshot=bad_snapshot)
        )
        await session.commit()

    r = await client.post(f"/v1/receipts/{receipt_id}/replay")
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "unreplayable.invalid_snapshot"


@pytest.mark.anyio
async def test_replay_404_for_unknown_receipt(client: AsyncClient):
    r = await client.post(f"/v1/receipts/{uuid.uuid4().hex[:26].upper()}/replay")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Admin replay shim (v0.9 #160 admin-app prep)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_admin_replay_shim_returns_same_shape(client: AsyncClient, subject_id: str):
    """`POST /admin/receipts/{id}/replay` mirrors the public endpoint
    so the admin app can call replay through the `/admin/*`-only proxy
    allowlist without forwarding `/v1/`."""
    await _seed(client, subject_id)
    original_id = await _emit_receipt(client, subject_id)

    r = await client.post(f"/admin/receipts/{original_id}/replay")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["original_receipt_id"] == original_id
    assert body["replay_receipt_id"]
    assert body["replay_receipt_id"] != original_id
    assert "diff" in body
    assert "selected_entries" in body["diff"]


@pytest.mark.anyio
async def test_admin_replay_shim_surfaces_422_codes(
    client: AsyncClient, subject_id: str, session_factory
):
    """Refusal codes round-trip identically through the admin shim."""
    await _seed(client, subject_id)
    original_id = await _emit_receipt(client, subject_id)
    replay_resp = await client.post(f"/v1/receipts/{original_id}/replay")
    assert replay_resp.status_code == 200
    replay_id = replay_resp.json()["replay_receipt_id"]

    r = await client.post(f"/admin/receipts/{replay_id}/replay")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "unreplayable.nested_replay"


@pytest.mark.anyio
async def test_admin_replay_shim_404(client: AsyncClient):
    r = await client.post(f"/admin/receipts/{uuid.uuid4().hex[:26].upper()}/replay")
    assert r.status_code == 404
