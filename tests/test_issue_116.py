"""Regression test for issue #116.

/v1/context must flag when nothing stored is relevant to the task instead
of silently presenting unrelated memory as "About this user".
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from server.services.context import _NO_TASK_CONTEXT_NOTE, assemble_context


def _fact(content: str) -> SimpleNamespace:
    now = datetime(2026, 5, 16, tzinfo=timezone.utc)
    return SimpleNamespace(
        id=uuid.uuid4(),
        subject_id="alice_bob_conversation",
        kind="profile_fact",
        content=content,
        summary=content[:80],
        confidence=0.9,
        valid_from=now,
        valid_to=None,
        source_episode_ids=[],
        metadata_={},
        status="active",
        sensitivity_labels=[],
        created_at=now,
        updated_at=now,
    )


_WEATHER_FACTS = [
    _fact("Bob checked the weather on the morning of 2026-05-16."),
    _fact("The temperature was around 60 degrees Fahrenheit and windy on 2026-05-16."),
    _fact("Alice brought her jacket on 2026-05-16 because it was cold outside."),
]


@contextmanager
def _mock_repos(facts):
    async def _search_memories(session, subject_id, *, tenant_id=None, kind=None, limit=None):
        return list(facts) if kind == "profile_fact" else []

    with (
        patch(
            "server.services.context.repo.search_memories",
            new=AsyncMock(side_effect=_search_memories),
        ),
        patch(
            "server.services.context.repo.list_episodes_by_subject",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "server.services.context.repo.search_memories_by_embedding",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch("server.services.context.get_embedding_provider", return_value=None),
        patch(
            "server.services.context.repo.get_resolved_session_ids",
            new_callable=AsyncMock,
            return_value=set(),
        ),
        patch(
            "server.services.context.repo.get_open_session_ids",
            new_callable=AsyncMock,
            return_value=set(),
        ),
        patch(
            "server.services.context.repo.list_resolutions",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "server.services.context.policy_service.resolve_active_bundle",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        yield


@pytest.mark.asyncio
async def test_116_offtopic_task_gets_caveat_but_keeps_facts():
    """An unrelated task surfaces the caveat — and still returns the facts."""
    with _mock_repos(_WEATHER_FACTS):
        result = await assemble_context(
            AsyncMock(), "alice_bob_conversation",
            "What is the airline ticket price?", max_tokens=4000,
        )
    assert _NO_TASK_CONTEXT_NOTE in result.assembled_context
    # Recall preserved: the weather facts are still in the bundle.
    assert "60 degrees Fahrenheit" in result.assembled_context
    assert len(result.facts) == len(_WEATHER_FACTS)


@pytest.mark.asyncio
async def test_116_relevant_task_has_no_caveat():
    """A task that overlaps stored content must NOT trigger the caveat."""
    with _mock_repos(_WEATHER_FACTS):
        result = await assemble_context(
            AsyncMock(), "alice_bob_conversation",
            "What was the weather like that morning?", max_tokens=4000,
        )
    assert _NO_TASK_CONTEXT_NOTE not in result.assembled_context
    assert "60 degrees Fahrenheit" in result.assembled_context


@pytest.mark.asyncio
async def test_116_empty_task_never_caveats():
    """No task → relevance is undefined; never emit the caveat."""
    with _mock_repos(_WEATHER_FACTS):
        result = await assemble_context(
            AsyncMock(), "alice_bob_conversation", "", max_tokens=4000,
        )
    assert _NO_TASK_CONTEXT_NOTE not in result.assembled_context
