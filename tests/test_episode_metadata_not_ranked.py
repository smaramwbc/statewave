"""Episode `metadata` is inert for retrieval and ranking.

The ingest API documents `metadata` as stored and returned unchanged. A caller
who labels an episode `{"topic": "..."}` is told those labels will not help it
surface and will not reach the model either. These tests pin that promise
against the assembly path, so the documented sentence cannot quietly become
false.

Context assembly has exactly one exception: the reserved `outcome` envelope
from #415, whose behaviour is pinned in test_episode_outcome_in_bundle.py. The
documented rule has to name that exception, which is what the last test here
checks, because it went a while without naming it (#442).
"""

from __future__ import annotations

import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from server.api.episodes import create_episode, create_episodes_batch
from server.schemas.requests import CreateEpisodeRequest
from server.services.context import OUTCOME_METADATA_KEY, assemble_context


TASK = "deploy pipeline staging"


def _make_episode_row(
    *,
    row_id: uuid.UUID,
    text: str,
    minutes_ago: int,
    metadata: dict | None = None,
):
    """Create a fake episode row matching the ORM shape."""
    when = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return SimpleNamespace(
        id=row_id,
        subject_id="user-1",
        source="chat",
        type="message",
        payload={"messages": [{"role": "user", "content": text}]},
        metadata_=metadata or {},
        provenance={},
        session_id=None,
        occurred_at=when,
        created_at=when,
    )


@contextmanager
def _mock_repos(episodes):
    """Patch the repo calls assemble_context makes, memories left empty."""
    with (
        patch(
            "server.services.context.repo.search_memories", new_callable=AsyncMock, return_value=[]
        ),
        patch(
            "server.services.context.repo.list_episodes_by_subject",
            new_callable=AsyncMock,
            return_value=episodes,
        ),
        patch(
            "server.services.context.repo.superseded_only_episode_ids",
            new_callable=AsyncMock,
            return_value=set(),
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
    ):
        yield


async def _assemble(episodes):
    with _mock_repos(episodes):
        return await assemble_context(AsyncMock(), "user-1", TASK, max_tokens=4000)


def _episode_pair(*, labelled: bool):
    """Two episodes whose only possible difference is metadata.

    Ids and timestamps are fixed across both variants so the two bundles are
    comparable verbatim. The RELEVANT episode matches the task in its text;
    the recent one does not, and — in the labelled variant — carries the
    task's own words in metadata instead. If metadata were scored, that would
    be enough to move it.
    """
    relevant_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    recent_id = uuid.UUID("22222222-2222-2222-2222-222222222222")
    return [
        _make_episode_row(
            row_id=relevant_id,
            text="the deploy pipeline failed on staging again",
            minutes_ago=600,
        ),
        _make_episode_row(
            row_id=recent_id,
            text="the invoice pdf renders upside down",
            minutes_ago=1,
            metadata={"topic": TASK, "priority": "critical"} if labelled else None,
        ),
    ]


@pytest.mark.asyncio
async def test_metadata_changes_neither_selection_nor_order():
    plain = await _assemble(_episode_pair(labelled=False))
    labelled = await _assemble(_episode_pair(labelled=True))

    assert labelled.provenance["episode_ids"] == plain.provenance["episode_ids"]


@pytest.mark.asyncio
async def test_metadata_never_reaches_the_assembled_context():
    labelled = await _assemble(_episode_pair(labelled=True))
    plain = await _assemble(_episode_pair(labelled=False))

    assert labelled.assembled_context == plain.assembled_context
    assert labelled.token_estimate == plain.token_estimate
    assert "critical" not in labelled.assembled_context


@pytest.mark.asyncio
async def test_metadata_is_returned_unchanged():
    # The other half of the promise: inert for assembly, but not dropped.
    labelled = await _assemble(_episode_pair(labelled=True))

    returned = {str(ep.id): ep.metadata for ep in labelled.episodes}
    assert returned["22222222-2222-2222-2222-222222222222"] == {
        "topic": TASK,
        "priority": "critical",
    }


# The sentence this catches, in the two phrasings it has actually been written
# in: "takes no part in retrieval, ranking or context assembly" and "retrieval,
# ranking and context assembly never read it". Both were true until the
# `outcome` envelope landed, and both survived it.
_INERT_FOR_ASSEMBLY_CLAIM = re.compile(r"(no part in|never read)[^.]*context assembl", re.I)


def test_the_documented_rule_names_the_outcome_exception():
    """The rule callers read has to match the rule the code follows.

    The `metadata` field description is published in the OpenAPI schema and
    the two ingest routes carry the same rule in their docstrings, so all
    three are places a caller learns what `metadata` does.
    """
    documented = {
        "metadata field description": CreateEpisodeRequest.model_fields["metadata"].description,
        "create_episode docstring": create_episode.__doc__,
        "create_episodes_batch docstring": create_episodes_batch.__doc__,
    }
    for where, text in documented.items():
        assert text, f"{where} is empty"
        # Docstrings are hard-wrapped, so the claim is matched against one
        # long line: otherwise a line break in the middle of the phrase would
        # be enough to slip it past this test.
        flat = " ".join(text.split())
        assert OUTCOME_METADATA_KEY in flat, f"{where} does not name the `outcome` exception"
        assert not _INERT_FOR_ASSEMBLY_CLAIM.search(flat), (
            f"{where} still says context assembly never reads metadata"
        )
