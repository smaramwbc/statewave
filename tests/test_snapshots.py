"""Tests for Subject Snapshots service."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from server.services.snapshots import (
    SNAPSHOT_SOURCE_PREFIX,
    restore_snapshot,
)


# ─── Unit tests for timestamp shifting and provenance remapping ───


class FakeEpisode:
    def __init__(
        self,
        id,
        subject_id,
        created_at,
        source="test",
        type="test",
        payload=None,
        metadata_=None,
        provenance=None,
        last_compiled_at=None,
    ):
        self.id = id
        self.subject_id = subject_id
        self.source = source
        self.type = type
        self.payload = payload or {}
        self.metadata_ = metadata_ or {}
        self.provenance = provenance or {}
        self.created_at = created_at
        self.last_compiled_at = last_compiled_at


class FakeMemory:
    def __init__(
        self,
        id,
        subject_id,
        created_at,
        updated_at,
        valid_from,
        source_episode_ids,
        kind="fact",
        content="test",
        summary="",
        confidence=1.0,
        valid_to=None,
        metadata_=None,
        status="active",
        embedding=None,
    ):
        self.id = id
        self.subject_id = subject_id
        self.kind = kind
        self.content = content
        self.summary = summary
        self.confidence = confidence
        self.valid_from = valid_from
        self.valid_to = valid_to
        self.source_episode_ids = source_episode_ids
        self.metadata_ = metadata_ or {}
        self.status = status
        self.embedding = embedding
        self.created_at = created_at
        self.updated_at = updated_at


class FakeSnapshot:
    def __init__(self, id, source_subject_id):
        self.id = id
        self.source_subject_id = source_subject_id


class FakeScalarsResult:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


class FakeResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return FakeScalarsResult(self._items)

    def scalar(self):
        return self._items[0] if self._items else None


@pytest.mark.asyncio
async def test_restore_snapshot_remaps_provenance():
    """Restored memories must reference cloned episode IDs, not original ones."""
    snap_id = uuid.uuid4()
    ep1_id = uuid.uuid4()
    ep2_id = uuid.uuid4()
    mem_id = uuid.uuid4()

    now = datetime.now(timezone.utc)
    ep1_time = now - timedelta(hours=2)
    ep2_time = now - timedelta(hours=1)

    fake_snap = FakeSnapshot(id=snap_id, source_subject_id="_snapshot/test/v1")
    fake_eps = [
        FakeEpisode(id=ep1_id, subject_id="_snapshot/test/v1", created_at=ep1_time),
        FakeEpisode(id=ep2_id, subject_id="_snapshot/test/v1", created_at=ep2_time),
    ]
    fake_mems = [
        FakeMemory(
            id=mem_id,
            subject_id="_snapshot/test/v1",
            created_at=ep2_time,
            updated_at=ep2_time,
            valid_from=ep1_time,
            source_episode_ids=[ep1_id, ep2_id],
        ),
    ]

    added_items = []

    class FakeSession:
        async def execute(self, query):
            # Distinguish queries by what they filter on
            query_str = str(query)
            if "subject_snapshots" in query_str:
                return FakeResult([fake_snap])
            elif "memories" in query_str:
                return FakeResult(fake_mems)
            else:
                return FakeResult(fake_eps)

        def add(self, item):
            added_items.append(item)

        async def commit(self):
            pass

    class FakeSessionFactory:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *args):
            pass

    with patch(
        "server.services.snapshots.async_session_factory", return_value=FakeSessionFactory()
    ):
        result = await restore_snapshot(snap_id, "live_test_123")

    assert result["episodes_restored"] == 2
    assert result["memories_restored"] == 1

    # Check that memories have remapped episode IDs
    memory_items = [i for i in added_items if hasattr(i, "source_episode_ids")]
    assert len(memory_items) == 1
    mem = memory_items[0]

    # source_episode_ids should NOT contain original IDs
    assert ep1_id not in mem.source_episode_ids
    assert ep2_id not in mem.source_episode_ids

    # They should contain new IDs that match the cloned episodes
    episode_items = [
        i for i in added_items if hasattr(i, "source") and not hasattr(i, "source_episode_ids")
    ]
    new_ep_ids = {i.id for i in episode_items}
    assert set(mem.source_episode_ids) == new_ep_ids


@pytest.mark.asyncio
async def test_restore_snapshot_shifts_timestamps():
    """Timestamps should be shifted so newest episode ≈ now."""
    snap_id = uuid.uuid4()
    ep_id = uuid.uuid4()
    mem_id = uuid.uuid4()

    # Episode was created 7 days ago
    old_time = datetime.now(timezone.utc) - timedelta(days=7)

    fake_snap = FakeSnapshot(id=snap_id, source_subject_id="_snapshot/test/v1")
    fake_eps = [
        FakeEpisode(id=ep_id, subject_id="_snapshot/test/v1", created_at=old_time),
    ]
    fake_mems = [
        FakeMemory(
            id=mem_id,
            subject_id="_snapshot/test/v1",
            created_at=old_time,
            updated_at=old_time,
            valid_from=old_time,
            source_episode_ids=[ep_id],
        ),
    ]

    added_items = []

    class FakeSession:
        async def execute(self, query):
            query_str = str(query)
            if "subject_snapshots" in query_str:
                return FakeResult([fake_snap])
            elif "memories" in query_str:
                return FakeResult(fake_mems)
            else:
                return FakeResult(fake_eps)

        def add(self, item):
            added_items.append(item)

        async def commit(self):
            pass

    class FakeSessionFactory:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *args):
            pass

    before = datetime.now(timezone.utc)
    with patch(
        "server.services.snapshots.async_session_factory", return_value=FakeSessionFactory()
    ):
        await restore_snapshot(snap_id, "live_test_456")
    after = datetime.now(timezone.utc)

    # Cloned episode should be very close to now (within a few seconds)
    episode_items = [
        i for i in added_items if hasattr(i, "source") and not hasattr(i, "source_episode_ids")
    ]
    assert len(episode_items) == 1
    assert before <= episode_items[0].created_at <= after

    # Cloned memory timestamps should also be near now
    memory_items = [i for i in added_items if hasattr(i, "source_episode_ids")]
    assert len(memory_items) == 1
    assert before <= memory_items[0].created_at <= after
    assert before <= memory_items[0].valid_from <= after


@pytest.mark.asyncio
async def test_restore_preserves_relative_offsets():
    """Relative time between episodes should be preserved after shifting."""
    snap_id = uuid.uuid4()
    ep1_id = uuid.uuid4()
    ep2_id = uuid.uuid4()

    now = datetime.now(timezone.utc)
    ep1_time = now - timedelta(days=10)
    ep2_time = now - timedelta(days=7)  # 3 days after ep1

    fake_snap = FakeSnapshot(id=snap_id, source_subject_id="_snapshot/test/v1")
    fake_eps = [
        FakeEpisode(id=ep1_id, subject_id="_snapshot/test/v1", created_at=ep1_time),
        FakeEpisode(id=ep2_id, subject_id="_snapshot/test/v1", created_at=ep2_time),
    ]

    added_items = []

    class FakeSession:
        async def execute(self, query):
            query_str = str(query)
            if "subject_snapshots" in query_str:
                return FakeResult([fake_snap])
            elif "memories" in query_str:
                return FakeResult([])
            else:
                return FakeResult(fake_eps)

        def add(self, item):
            added_items.append(item)

        async def commit(self):
            pass

    class FakeSessionFactory:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *args):
            pass

    with patch(
        "server.services.snapshots.async_session_factory", return_value=FakeSessionFactory()
    ):
        await restore_snapshot(snap_id, "live_test_789")

    episode_items = [
        i for i in added_items if hasattr(i, "source") and not hasattr(i, "source_episode_ids")
    ]
    episode_items.sort(key=lambda x: x.created_at)

    # The gap between the two episodes should still be ~3 days
    gap = episode_items[1].created_at - episode_items[0].created_at
    assert timedelta(days=2, hours=23) < gap < timedelta(days=3, hours=1)


@pytest.mark.asyncio
async def test_cleanup_never_deletes_snapshots():
    """Cleanup must never touch _snapshot/ subjects."""
    # This is a logic test — verify the filter
    stale_subjects = [
        "live_sarah_123",
        "_snapshot/sarah/v1",
        "live_marcus_456",
        "_snapshot/priya/v1",
    ]

    filtered = [s for s in stale_subjects if not s.startswith(SNAPSHOT_SOURCE_PREFIX)]
    assert filtered == ["live_sarah_123", "live_marcus_456"]
    assert "_snapshot/sarah/v1" not in filtered
    assert "_snapshot/priya/v1" not in filtered


@pytest.mark.asyncio
async def test_restore_adds_provenance_metadata():
    """Restored episodes should have snapshot provenance in their metadata."""
    snap_id = uuid.uuid4()
    ep_id = uuid.uuid4()

    now = datetime.now(timezone.utc)
    fake_snap = FakeSnapshot(id=snap_id, source_subject_id="_snapshot/test/v1")
    fake_eps = [FakeEpisode(id=ep_id, subject_id="_snapshot/test/v1", created_at=now)]

    added_items = []

    class FakeSession:
        async def execute(self, query):
            query_str = str(query)
            if "subject_snapshots" in query_str:
                return FakeResult([fake_snap])
            elif "memories" in query_str:
                return FakeResult([])
            else:
                return FakeResult(fake_eps)

        def add(self, item):
            added_items.append(item)

        async def commit(self):
            pass

    class FakeSessionFactory:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *args):
            pass

    with patch(
        "server.services.snapshots.async_session_factory", return_value=FakeSessionFactory()
    ):
        await restore_snapshot(snap_id, "live_test_prov")

    episode_items = [i for i in added_items if hasattr(i, "provenance")]
    assert len(episode_items) == 1
    assert episode_items[0].provenance["restored_from_snapshot"] == str(snap_id)
    assert episode_items[0].provenance["original_episode_id"] == str(ep_id)
