"""Shared DB-session test doubles.

Several admin endpoint tests need to stand in for a real async SQLAlchemy
session without hitting Postgres: they monkeypatch
``server.db.engine.get_session_factory`` with a factory that returns one of
these fakes, then inspect the compiled statement(s) the endpoint executed.

This module holds only the generic scaffolding shared by callers with the
same lightweight execute/scalar shape used by ``test_admin_like_escape.py``,
plus ``make_async_session()`` for tests that pass a session straight into a
service. Tests that need a differently-shaped fake session (e.g. one that
tracks commits/rollbacks, or pops results off a queue) should keep their own
local fake rather than force-fitting it into this one.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


class FakeScalarRows:
    def all(self):
        return []


class FakeResult:
    def all(self):
        return []

    def scalars(self):
        return FakeScalarRows()


class FakeSession:
    def __init__(self):
        self.statements = []

    async def scalar(self, statement):
        self.statements.append(statement)
        return 0

    async def execute(self, statement):
        self.statements.append(statement)
        return FakeResult()


class FakeSessionContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return False


def make_empty_result() -> MagicMock:
    """A `Result` double whose accessors are sync and find nothing.

    `Result` is returned by an awaited `execute()`, but its own accessors are
    plain methods. Left to an `AsyncMock` they hand back coroutines, so
    `row = result.scalar_one_or_none()` is truthy and the caller walks its
    "row found" branch holding a coroutine instead of a row.
    """
    result = MagicMock()
    result.scalar.return_value = None
    result.scalar_one_or_none.return_value = None
    result.first.return_value = None
    result.all.return_value = []
    result.scalars.return_value.all.return_value = []
    result.scalars.return_value.first.return_value = None
    return result


def make_async_session() -> AsyncMock:
    """An `AsyncSession` double that keeps the session's sync API sync.

    `AsyncSession.add()` is a plain method: an `AsyncMock` returns a coroutine
    nobody awaits, which both leaks a RuntimeWarning pointing at the
    production `session.add(row)` and lets a test pass against code that
    awaits `add()` — a TypeError against a real session. `add_all()` is the
    same kind of method and gets the same treatment.
    """
    session = AsyncMock()
    session.add = MagicMock(return_value=None)  # as the real add() returns
    session.add_all = MagicMock(return_value=None)
    session.execute.return_value = make_empty_result()
    return session


def install_fake_session_factory(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(
        "server.db.engine.get_session_factory",
        lambda: lambda: FakeSessionContext(session),
    )
    return session
