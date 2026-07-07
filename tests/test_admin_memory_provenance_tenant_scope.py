from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from server.api.admin import memory_provenance


class _FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def scalar_one_or_none(self):
        return self.rows[0] if self.rows else None

    def scalars(self):
        return self

    def all(self):
        return self.rows


class _FakeSession:
    def __init__(self, results):
        self.results = list(results)
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return self.results.pop(0)


class _FakeSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _compiled(statement):
    return statement.compile(dialect=postgresql.dialect())


def _memory_row(*, memory_id, subject_id, tenant_id, source_episode_ids):
    return SimpleNamespace(
        id=memory_id,
        subject_id=subject_id,
        tenant_id=tenant_id,
        kind="profile_fact",
        content=f"{tenant_id} memory",
        summary=f"{tenant_id} memory",
        confidence=1.0,
        status="active",
        created_at=datetime.now(timezone.utc),
        source_episode_ids=source_episode_ids,
    )


def _episode_row(*, episode_id, subject_id, tenant_id):
    return SimpleNamespace(
        id=episode_id,
        subject_id=subject_id,
        tenant_id=tenant_id,
        source="test",
        type="message",
        payload={"text": f"{tenant_id} episode"},
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.anyio
async def test_memory_provenance_scopes_memory_episodes_and_siblings_by_tenant(
    monkeypatch,
):
    subject_id = "shared-subject"
    tenant_id = "tenant-a"
    source_episode_id = uuid.uuid4()
    memory_id = uuid.uuid4()
    sibling_id = uuid.uuid4()

    memory = _memory_row(
        memory_id=memory_id,
        subject_id=subject_id,
        tenant_id=tenant_id,
        source_episode_ids=[source_episode_id],
    )
    source_episode = _episode_row(
        episode_id=source_episode_id,
        subject_id=subject_id,
        tenant_id=tenant_id,
    )
    sibling = _memory_row(
        memory_id=sibling_id,
        subject_id=subject_id,
        tenant_id=tenant_id,
        source_episode_ids=[source_episode_id],
    )
    session = _FakeSession(
        [
            _FakeResult([memory]),
            _FakeResult([source_episode]),
            _FakeResult([sibling]),
        ]
    )

    monkeypatch.setattr(
        "server.db.engine.get_session_factory",
        lambda: lambda: _FakeSessionContext(session),
    )

    response = await memory_provenance(
        subject_id,
        str(memory_id),
        tenant_id=tenant_id,
    )

    assert response.memory.id == str(memory_id)
    assert [ep.id for ep in response.source_episodes] == [str(source_episode_id)]
    assert [mem.id for mem in response.sibling_memories] == [str(sibling_id)]

    memory_stmt, episode_stmt, sibling_stmt = session.statements
    for statement in (memory_stmt, episode_stmt, sibling_stmt):
        compiled = _compiled(statement)
        assert "tenant_id =" in str(compiled)
        assert tenant_id in compiled.params.values()

