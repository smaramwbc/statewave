from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql

from server.api.admin import memory_compiler_trace


class _FakeSession:
    def __init__(self, row):
        self.row = row
        self.scalar_statement = None

    async def scalar(self, statement):
        self.scalar_statement = statement
        return self.row


class _FakeSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _compiled(statement):
    return statement.compile(dialect=postgresql.dialect())


@pytest.mark.anyio
async def test_memory_compiler_trace_scopes_memory_and_source_episodes_by_tenant(
    monkeypatch,
):
    source_episode_id = uuid.uuid4()
    row = SimpleNamespace(
        id=uuid.uuid4(),
        subject_id="shared-subject",
        tenant_id="tenant-a",
        kind="profile_fact",
        content="tenant-scoped memory",
        summary="tenant-scoped memory",
        confidence=1.0,
        status="active",
        created_at=datetime.now(timezone.utc),
        metadata_={},
        source_episode_ids=[source_episode_id],
    )
    session = _FakeSession(row)
    get_episodes_by_ids = AsyncMock(return_value=[])

    monkeypatch.setattr(
        "server.db.engine.get_session_factory",
        lambda: lambda: _FakeSessionContext(session),
    )
    monkeypatch.setattr(
        "server.db.repositories.get_episodes_by_ids",
        get_episodes_by_ids,
    )

    response = await memory_compiler_trace(
        "shared-subject",
        str(row.id),
        tenant_id="tenant-a",
    )

    compiled = _compiled(session.scalar_statement)
    assert "memories.tenant_id =" in str(compiled)
    assert "tenant-a" in compiled.params.values()
    get_episodes_by_ids.assert_awaited_once_with(
        session,
        [source_episode_id],
        tenant_id="tenant-a",
    )
    assert response.memory_id == str(row.id)
    assert response.reconstructed_input == []
