from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from server.api.admin import (
    BulkDeleteCommitRequest,
    BulkDeleteFilter,
    BulkDeleteSample,
    _delete_subject_key,
    _matching_subjects,
    commit_bulk_delete,
)


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Rowcount:
    def __init__(self, rowcount):
        self.rowcount = rowcount


class _FakeSession:
    def __init__(self, *, episode_rows=None, memory_rows=None):
        self._results = [
            list(episode_rows or []),
            list(memory_rows or []),
        ]
        self.statements = []
        self.commits = 0
        self.rollbacks = 0

    async def execute(self, statement):
        self.statements.append(statement)
        return _Rows(self._results.pop(0))

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        self.rollbacks += 1


class _DeleteSession:
    def __init__(self, rowcounts):
        self._rowcounts = list(rowcounts)
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return _Rowcount(self._rowcounts.pop(0))


class _FakeSessionContext:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _install_session(monkeypatch, session):
    monkeypatch.setattr(
        "server.db.engine.get_session_factory",
        lambda: lambda: _FakeSessionContext(session),
    )


def _row(**kwargs):
    return SimpleNamespace(**kwargs)


def _compiled(statement):
    return statement.compile(dialect=postgresql.dialect())


@pytest.mark.anyio
async def test_matching_subjects_includes_memory_only_subjects(monkeypatch):
    session = _FakeSession(
        episode_rows=[],
        memory_rows=[
            _row(subject_id="memory-only", tenant_id="tenant-a", mem_count=2),
        ],
    )
    _install_session(monkeypatch, session)

    matches, total_eps, total_mems = await _matching_subjects(
        BulkDeleteFilter(tenant_id="tenant-a")
    )

    assert total_eps == 0
    assert total_mems == 2
    assert [m.model_dump() for m in matches] == [
        {
            "subject_id": "memory-only",
            "tenant_id": "tenant-a",
            "episode_count": 0,
            "memory_count": 2,
            "last_episode_at": None,
        }
    ]


@pytest.mark.anyio
async def test_matching_subjects_sums_memory_counts_only_for_matched_tenant(monkeypatch):
    session = _FakeSession(
        episode_rows=[
            _row(
                subject_id="shared",
                tenant_id="tenant-a",
                ep_count=1,
                last_episode_at=None,
            ),
        ],
        memory_rows=[
            _row(subject_id="shared", tenant_id="tenant-a", mem_count=2),
            # Defensive: even if a future query regression fetched this row,
            # totals must only include the concrete matched key.
            _row(subject_id="shared", tenant_id="tenant-b", mem_count=7),
        ],
    )
    _install_session(monkeypatch, session)

    matches, total_eps, total_mems = await _matching_subjects(
        BulkDeleteFilter(tenant_id="tenant-a")
    )

    assert total_eps == 1
    assert total_mems == 2
    assert matches[0].memory_count == 2


@pytest.mark.anyio
async def test_matching_subjects_escapes_prefix_like_metacharacters(monkeypatch):
    session = _FakeSession()
    _install_session(monkeypatch, session)

    await _matching_subjects(BulkDeleteFilter(subject_id_prefix="demo_web_"))

    for statement in session.statements:
        compiled = _compiled(statement)
        assert " ESCAPE '\\\\'" in str(compiled)
        assert "demo\\_web\\_%" in set(compiled.params.values())


@pytest.mark.anyio
async def test_delete_subject_key_treats_none_tenant_as_global_only():
    session = _DeleteSession(rowcounts=[1, 2, 3, 4])

    ep_count, mem_count = await _delete_subject_key(session, "shared", tenant_id=None)

    assert (ep_count, mem_count) == (1, 2)
    assert len(session.statements) == 4
    for statement in session.statements:
        sql = str(_compiled(statement))
        assert "tenant_id IS NULL" in sql
        assert "tenant_id =" not in sql


@pytest.mark.anyio
async def test_commit_bulk_delete_uses_concrete_none_tenant_key(monkeypatch):
    session = _FakeSession()
    _install_session(monkeypatch, session)
    calls = []

    async def fake_matching(_req):
        return (
            [
                BulkDeleteSample(
                    subject_id="global-subject",
                    tenant_id=None,
                    episode_count=0,
                    memory_count=1,
                    last_episode_at=None,
                )
            ],
            0,
            1,
        )

    async def fake_delete_subject_key(session_arg, subject_id, tenant_id):
        calls.append((session_arg, subject_id, tenant_id))
        return 0, 1

    async def fake_fire(*_args, **_kwargs):
        return None

    monkeypatch.setattr("server.api.admin._matching_subjects", fake_matching)
    monkeypatch.setattr("server.api.admin._delete_subject_key", fake_delete_subject_key)
    monkeypatch.setattr("server.api.admin.webhooks.fire", fake_fire)

    result = await commit_bulk_delete(
        BulkDeleteCommitRequest(match_all=True, expected_count=1, confirm=True)
    )

    assert calls == [(session, "global-subject", None)]
    assert result.deleted_subjects == 1
    assert result.deleted_episodes == 0
    assert result.deleted_memories == 1
    assert session.commits == 1
    assert session.rollbacks == 0


@pytest.mark.parametrize("older_than_days", [-1, 0])
def test_bulk_delete_filter_rejects_non_positive_age(older_than_days):
    with pytest.raises(ValidationError):
        BulkDeleteFilter(older_than_days=older_than_days)
