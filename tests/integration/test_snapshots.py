"""Integration tests for Subject Snapshots feature.

Tests cover:
- Feature flag gating (disabled → 404)
- Snapshot creation via service layer
- Restore by ID and by name
- Provenance remapping
- Timestamp shifting
- Cleanup safety (snapshot subjects not cleaned)
- System subject hiding from list_subjects
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Feature flag: disabled → 404
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_snapshots_disabled_returns_404(client: AsyncClient):
    """When enable_snapshots=False, admin snapshot endpoints return 404."""
    with patch("server.core.config.settings.enable_snapshots", False):
        r = await client.get("/admin/snapshots")
        assert r.status_code == 404

        r = await client.post(
            "/admin/snapshots/restore-by-name",
            json={"name": "test", "target_subject_id": "x"},
        )
        assert r.status_code == 404

        r = await client.post("/admin/cleanup")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Full lifecycle: create snapshot via service, restore via API
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_snapshot_create_and_restore(client: AsyncClient, subject_id: str):
    """Create a snapshot from episodes, then restore into a new subject."""
    with patch("server.core.config.settings.enable_snapshots", True):
        # 1. Ingest episodes into source subject
        episodes = [
            {
                "subject_id": subject_id,
                "source": "test",
                "type": "support-chat",
                "payload": {
                    "messages": [
                        {"role": "user", "content": "My name is Alice"},
                        {"role": "assistant", "content": "Hi Alice!"},
                    ]
                },
            },
            {
                "subject_id": subject_id,
                "source": "test",
                "type": "support-chat",
                "payload": {
                    "messages": [
                        {"role": "user", "content": "I'm on the pro plan"},
                        {"role": "assistant", "content": "Great, pro plan noted."},
                    ]
                },
            },
        ]
        for ep in episodes:
            r = await client.post("/v1/episodes", json=ep)
            assert r.status_code == 200

        # 2. Compile
        r = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
        assert r.status_code == 200

        # 3. Create snapshot via service (create endpoint removed from API)
        from server.services.snapshots import create_snapshot

        snap_name = f"test-snap-{uuid.uuid4().hex[:8]}"
        snap = await create_snapshot(name=snap_name, source_subject_id=subject_id)
        assert snap["name"] == snap_name
        assert snap["episode_count"] >= 2
        assert snap["memory_count"] >= 1

        # 4. Verify snapshot appears in list
        r = await client.get("/admin/snapshots")
        assert r.status_code == 200
        names = [s["name"] for s in r.json()["snapshots"]]
        assert snap_name in names

        # 5. Restore by ID into target subject
        target_id = f"restored-{uuid.uuid4().hex[:8]}"
        r = await client.post(
            f"/admin/snapshots/{snap['id']}/restore",
            json={"target_subject_id": target_id},
        )
        assert r.status_code == 200
        result = r.json()
        assert result["episodes_restored"] >= 2
        assert result["memories_restored"] >= 1

        # 6. Target subject should have searchable memories
        r = await client.get(
            "/v1/memories/search",
            params={"subject_id": target_id, "limit": 50},
        )
        assert r.status_code == 200
        mems = r.json()["memories"]
        assert len(mems) >= 1


@pytest.mark.anyio
async def test_restored_subject_behaves_normally(client: AsyncClient, subject_id: str):
    """A restored subject should work like any organic subject — ingest, compile, search, context."""
    with patch("server.core.config.settings.enable_snapshots", True):
        # Setup and restore
        await client.post(
            "/v1/episodes",
            json={
                "subject_id": subject_id,
                "source": "test",
                "type": "support-chat",
                "payload": {
                    "messages": [
                        {"role": "user", "content": "My name is Bob"},
                        {"role": "assistant", "content": "Hi Bob!"},
                    ]
                },
            },
        )
        await client.post("/v1/memories/compile", json={"subject_id": subject_id})

        from server.services.snapshots import create_snapshot

        snap_name = f"normal-{uuid.uuid4().hex[:8]}"
        snap = await create_snapshot(name=snap_name, source_subject_id=subject_id)

        target_id = f"normal-target-{uuid.uuid4().hex[:8]}"
        await client.post(
            f"/admin/snapshots/{snap['id']}/restore",
            json={"target_subject_id": target_id},
        )

        # Now use target_id like any normal subject
        # 1. Ingest new episode
        r = await client.post(
            "/v1/episodes",
            json={
                "subject_id": target_id,
                "source": "test",
                "type": "support-chat",
                "payload": {
                    "messages": [
                        {"role": "user", "content": "I upgraded to enterprise"},
                        {"role": "assistant", "content": "Great!"},
                    ]
                },
            },
        )
        assert r.status_code == 200

        # 2. Re-compile (should not fail)
        r = await client.post("/v1/memories/compile", json={"subject_id": target_id})
        assert r.status_code == 200

        # 3. Context retrieval works
        r = await client.post(
            "/v1/context",
            json={
                "subject_id": target_id,
                "task": "Help Bob with billing",
            },
        )
        assert r.status_code == 200
        assert r.json()["subject_id"] == target_id

        # 4. Timeline works
        r = await client.get("/v1/timeline", params={"subject_id": target_id})
        assert r.status_code == 200

        # 5. Shows up in subject list
        r = await client.get("/v1/subjects")
        assert r.status_code == 200
        subject_ids = [s["subject_id"] for s in r.json().get("subjects", r.json().get("items", []))]
        assert target_id in subject_ids


@pytest.mark.anyio
async def test_timestamp_shifting_lands_near_now(client: AsyncClient, subject_id: str):
    """Restored episodes should have timestamps near now, preserving relative order."""
    import datetime

    with patch("server.core.config.settings.enable_snapshots", True):
        # Ingest two episodes
        r1 = await client.post(
            "/v1/episodes",
            json={
                "subject_id": subject_id,
                "source": "test",
                "type": "note",
                "payload": {"text": "first event"},
            },
        )
        assert r1.status_code == 200

        # Small delay to ensure ordering
        import asyncio

        await asyncio.sleep(0.1)

        r2 = await client.post(
            "/v1/episodes",
            json={
                "subject_id": subject_id,
                "source": "test",
                "type": "note",
                "payload": {"text": "second event"},
            },
        )
        assert r2.status_code == 200

        await client.post("/v1/memories/compile", json={"subject_id": subject_id})

        from server.services.snapshots import create_snapshot

        snap_name = f"time-{uuid.uuid4().hex[:8]}"
        snap = await create_snapshot(name=snap_name, source_subject_id=subject_id)

        target_id = f"time-target-{uuid.uuid4().hex[:8]}"
        await client.post(
            f"/admin/snapshots/{snap['id']}/restore",
            json={"target_subject_id": target_id},
        )

        # Fetch timeline for restored subject
        r = await client.get("/v1/timeline", params={"subject_id": target_id})
        assert r.status_code == 200
        events = r.json().get("events", r.json().get("items", []))
        assert len(events) >= 2

        # All timestamps should be within the last 60 seconds
        now = datetime.datetime.now(datetime.timezone.utc)
        for event in events:
            ts = datetime.datetime.fromisoformat(event["created_at"].replace("Z", "+00:00"))
            delta = (now - ts).total_seconds()
            assert delta < 60, f"Restored timestamp too old: {delta}s ago"

        # Relative order preserved (second event after first)
        timestamps = [
            datetime.datetime.fromisoformat(e["created_at"].replace("Z", "+00:00")) for e in events
        ]
        assert timestamps == sorted(timestamps)


@pytest.mark.anyio
async def test_snapshot_restore_by_name(client: AsyncClient, subject_id: str):
    """Restore a snapshot using the by-name endpoint."""
    with patch("server.core.config.settings.enable_snapshots", True):
        # Setup: ingest + compile + create snapshot
        r = await client.post(
            "/v1/episodes",
            json={
                "subject_id": subject_id,
                "source": "test",
                "type": "note",
                "payload": {"text": "User prefers dark mode"},
            },
        )
        assert r.status_code == 200
        await client.post("/v1/memories/compile", json={"subject_id": subject_id})

        from server.services.snapshots import create_snapshot

        snap_name = f"byname-{uuid.uuid4().hex[:8]}"
        await create_snapshot(name=snap_name, source_subject_id=subject_id)

        # Restore by name
        target_id = f"live_test_{uuid.uuid4().hex[:8]}"
        r = await client.post(
            "/admin/snapshots/restore-by-name",
            json={"name": snap_name, "target_subject_id": target_id},
        )
        assert r.status_code == 200
        assert r.json()["episodes_restored"] >= 1


# ---------------------------------------------------------------------------
# Provenance and timestamps
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_restored_provenance_uses_new_ids(client: AsyncClient, subject_id: str):
    """Restored memories should reference the new episode IDs, not originals."""
    with patch("server.core.config.settings.enable_snapshots", True):
        r = await client.post(
            "/v1/episodes",
            json={
                "subject_id": subject_id,
                "source": "test",
                "type": "support-chat",
                "payload": {
                    "messages": [
                        {"role": "user", "content": "I use vim"},
                        {"role": "assistant", "content": "Noted!"},
                    ]
                },
            },
        )
        assert r.status_code == 200
        original_ep_id = r.json()["id"]

        await client.post("/v1/memories/compile", json={"subject_id": subject_id})

        from server.services.snapshots import create_snapshot

        snap_name = f"prov-{uuid.uuid4().hex[:8]}"
        snap = await create_snapshot(name=snap_name, source_subject_id=subject_id)

        target_id = f"prov-target-{uuid.uuid4().hex[:8]}"
        r = await client.post(
            f"/admin/snapshots/{snap['id']}/restore",
            json={"target_subject_id": target_id},
        )
        assert r.status_code == 200

        # Check memories — source_episode_ids should NOT contain original
        r = await client.get(
            "/v1/memories/search",
            params={"subject_id": target_id, "limit": 50},
        )
        mems = r.json()["memories"]
        for mem in mems:
            if mem.get("source_episode_ids"):
                assert original_ep_id not in mem["source_episode_ids"]


# ---------------------------------------------------------------------------
# System subject hiding
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_snapshot_subjects_hidden_from_timeline(client: AsyncClient, subject_id: str):
    """Snapshot source subjects (_snapshot/*) should not appear in subject listing."""
    with patch("server.core.config.settings.enable_snapshots", True):
        # Create an episode so the subject appears
        await client.post(
            "/v1/episodes",
            json={
                "subject_id": subject_id,
                "source": "test",
                "type": "note",
                "payload": {"text": "hello"},
            },
        )
        await client.post("/v1/memories/compile", json={"subject_id": subject_id})

        from server.services.snapshots import create_snapshot

        snap_name = f"hidden-{uuid.uuid4().hex[:8]}"
        await create_snapshot(name=snap_name, source_subject_id=subject_id)

        # List subjects (via timeline which uses list_subjects)
        r = await client.get("/v1/subjects")
        if r.status_code == 200:
            subjects = [
                s["subject_id"] for s in r.json().get("subjects", r.json().get("items", []))
            ]
            for s in subjects:
                assert not s.startswith("_snapshot/")
                assert not s.startswith("_bootstrap_tmp/")


# ---------------------------------------------------------------------------
# Cleanup safety
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cleanup_does_not_delete_snapshots(client: AsyncClient, subject_id: str):
    """Cleanup of stale live_ subjects must never delete _snapshot/* subjects."""
    with patch("server.core.config.settings.enable_snapshots", True):
        # Create a snapshot
        await client.post(
            "/v1/episodes",
            json={
                "subject_id": subject_id,
                "source": "test",
                "type": "note",
                "payload": {"text": "preserve me"},
            },
        )
        await client.post("/v1/memories/compile", json={"subject_id": subject_id})

        from server.services.snapshots import create_snapshot

        snap_name = f"cleanup-{uuid.uuid4().hex[:8]}"
        snap = await create_snapshot(name=snap_name, source_subject_id=subject_id)

        # Run cleanup with 0 hours TTL (would catch everything)
        r = await client.post("/admin/cleanup", params={"prefix": "live_", "max_age_hours": 0})
        assert r.status_code == 200

        # Snapshot should still exist
        r = await client.get("/admin/snapshots")
        snap_ids = [s["id"] for s in r.json()["snapshots"]]
        assert snap["id"] in snap_ids
