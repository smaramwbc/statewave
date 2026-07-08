"""DELETE /v1/subjects fires subject.deleted only when something was actually
deleted (issue #282).

The delete is idempotent — deleting a missing subject (or the same subject
twice) still returns 200 with honest zero counts — but it must NOT emit a
subject.deleted webhook for a no-op, which would be a spurious deletion signal
to consumers (cache invalidation, audit logging, compliance records).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from server.api import subjects as subjects_api


class _FakeSession:
    """Minimal async session — delete_subject only calls .commit() on it; the
    repo delete helpers are mocked."""

    async def commit(self):
        return None


def _stub_repo_deletes(monkeypatch, *, episodes: int, memories: int) -> None:
    monkeypatch.setattr(
        subjects_api.repo, "delete_episodes_by_subject", AsyncMock(return_value=episodes)
    )
    monkeypatch.setattr(
        subjects_api.repo, "delete_memories_by_subject", AsyncMock(return_value=memories)
    )
    for name in (
        "delete_resolutions_by_subject",
        "delete_health_cache_by_subject",
        "delete_entities_by_subject",
    ):
        monkeypatch.setattr(subjects_api.repo, name, AsyncMock(return_value=0))


async def test_no_webhook_when_nothing_deleted(monkeypatch):
    _stub_repo_deletes(monkeypatch, episodes=0, memories=0)
    fire = AsyncMock()
    monkeypatch.setattr(subjects_api.webhooks, "fire", fire)

    resp = await subjects_api.delete_subject(
        "ghost-subject", session=_FakeSession(), tenant_id=None
    )

    # Idempotent: still 200 with honest zero counts...
    assert resp.episodes_deleted == 0
    assert resp.memories_deleted == 0
    # ...but no spurious subject.deleted event.
    fire.assert_not_called()


async def test_webhook_fires_when_something_deleted(monkeypatch):
    _stub_repo_deletes(monkeypatch, episodes=3, memories=2)
    fire = AsyncMock()
    monkeypatch.setattr(subjects_api.webhooks, "fire", fire)

    resp = await subjects_api.delete_subject(
        "real-subject", session=_FakeSession(), tenant_id=None
    )

    assert resp.episodes_deleted == 3
    assert resp.memories_deleted == 2
    fire.assert_called_once()
    event_name = fire.call_args.args[0]
    payload = fire.call_args.args[1]
    assert event_name == "subject.deleted"
    assert payload["episodes_deleted"] == 3
    assert payload["memories_deleted"] == 2


async def test_webhook_fires_when_only_memories_deleted(monkeypatch):
    # ep_count=0 but mem_count>0 is still a real deletion → fire.
    _stub_repo_deletes(monkeypatch, episodes=0, memories=1)
    fire = AsyncMock()
    monkeypatch.setattr(subjects_api.webhooks, "fire", fire)

    await subjects_api.delete_subject("m-only", session=_FakeSession(), tenant_id=None)

    fire.assert_called_once()
