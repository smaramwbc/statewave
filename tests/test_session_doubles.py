"""Guard the shape of the shared AsyncSession double.

A double that answers everything with a coroutine accepts code a real
``AsyncSession`` would reject, so the tests built on it stop proving what
they claim. These check the two halves the suite actually relies on: the
session's sync API and the sync accessors on an awaited ``execute()``.
"""

from __future__ import annotations

import inspect

from server.services import policy
from tests._fakes import make_async_session, make_empty_result


def test_add_stays_synchronous_like_the_real_session():
    session = make_async_session()

    returned = session.add(object())

    assert not inspect.iscoroutine(returned)
    assert returned is None


def test_result_accessors_stay_synchronous():
    result = make_empty_result()

    assert result.scalar_one_or_none() is None
    assert result.scalars().all() == []
    assert not inspect.iscoroutine(result.first())


async def test_empty_result_reaches_the_no_bundle_branch():
    # scalar_one_or_none() answering with a coroutine is truthy, so the loader
    # takes its row-found branch and raises; the caller's fail-open except
    # returns None as well, and the test reads as if the empty-table path ran.
    assert await policy._load_active_from_db(make_async_session(), None) is None
    assert await policy._load_active_from_db(make_async_session(), "tenant-a") is None
