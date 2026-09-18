"""A later, broader restatement must not delete the exception it dropped (#414).

Jaccard overlap is symmetric, so the lexical pass could not tell "later and
different" from "later and emptier": "Refunds are approved up to 500 EUR for
orders under 30 days" followed by "Refunds are approved up to 500 EUR for
orders" scores 0.75 against the 0.6 profile_fact threshold, superseded the
narrow row, and took "under 30 days" out of every retrieval path.

The guard is not about specificity in the abstract — it is about whether the
newer statement ADDS anything. The three shapes that must keep superseding are
pinned here beside the one that must not, because a guard that blocks a real
supersession is worse than the bug it fixes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from server.db.tables import MemoryRow
from server.services.conflicts import _are_conflicting, _tokenize, resolve_conflicts
from server.services.supersession import (
    RULE_LEXICAL,
    RULE_WIDENING_SKIPPED,
    SUPERSEDING_RULES,
    SupersessionDecision,
)
from tests._fakes import make_async_session

# The reported pair, verbatim from the issue.
NARROW = "Refunds are approved up to 500 EUR for orders under 30 days"
WIDE = "Refunds are approved up to 500 EUR for orders"


def _mem(content: str, *, kind: str = "profile_fact", days_ago: int = 0) -> MemoryRow:
    when = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return MemoryRow(
        id=uuid.uuid4(),
        subject_id="user-1",
        kind=kind,
        content=content,
        summary=content[:200],
        confidence=0.8,
        created_at=when,
        valid_from=when,
        source_episode_ids=[uuid.uuid4()],
        metadata_={},
        status="active",
    )


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    return len(ta & tb) / len(ta | tb)


def _records(session) -> list:
    rows = []
    for call in session.add_all.call_args_list:
        rows.extend(call.args[0])
    return rows


async def _resolve(memories, **kwargs):
    session = make_async_session()
    with patch("server.services.conflicts.repo") as mock_repo:
        mock_repo.list_active_memories_by_subject = AsyncMock(return_value=memories)
        mock_repo.mark_memories_superseded = AsyncMock()
        superseded = await resolve_conflicts(session, "user-1", **kwargs)
    return session, mock_repo, superseded


# --------------------------------------------------------------------------- #
# The reported defect
# --------------------------------------------------------------------------- #


def test_the_reported_pair_really_does_clear_the_threshold():
    """The fixture has to fail the OLD way, or every test below passes for the
    wrong reason. 0.6 is the profile_fact threshold."""
    assert _jaccard(NARROW, WIDE) == pytest.approx(0.75)


async def test_a_widening_restatement_leaves_the_narrow_memory_retrievable():
    narrow = _mem(NARROW, days_ago=5)
    wide = _mem(WIDE, days_ago=0)

    _session, mock_repo, superseded = await _resolve([narrow, wide])

    assert superseded == []
    mock_repo.mark_memories_superseded.assert_not_called()
    # `_unexpired` reads both of these; either one set would drop the row out
    # of search_memories, the embedding path and the hybrid path alike.
    assert narrow.status == "active"
    assert narrow.valid_to is None


def test_the_detector_and_the_resolver_agree_about_widening():
    assert _are_conflicting(_mem(NARROW, days_ago=5), _mem(WIDE)) is False


# --------------------------------------------------------------------------- #
# What must still supersede
# --------------------------------------------------------------------------- #


async def test_value_replacement_still_supersedes():
    """Munich -> Berlin. Each side has a content word the other lacks, so the
    newer is not a subset: the older is stale and must go."""
    older = _mem("Alice lives in Munich", days_ago=5)
    newer = _mem("Alice lives in Berlin", days_ago=0)

    _session, _repo, superseded = await _resolve([older, newer])

    assert superseded == [older.id]


async def test_narrowing_still_supersedes():
    """Alice -> Alice Chen. The newer is a strict SUPERSET; it carries
    everything the older said, so nothing is lost by retiring it."""
    older = _mem("my name is Alice", days_ago=5)
    newer = _mem("my name is Alice Chen", days_ago=0)

    _session, _repo, superseded = await _resolve([older, newer])

    assert superseded == [older.id]


async def test_a_reworded_duplicate_still_supersedes():
    """Equal sets: nothing is strict, so nothing is dropped and the older row
    is redundant, not an exception."""
    older = _mem("I use Stripe", days_ago=5)
    newer = _mem("I use Stripe.", days_ago=0)

    _session, _repo, superseded = await _resolve([older, newer])

    assert superseded == [older.id]


# --------------------------------------------------------------------------- #
# The two views the guard compares on
# --------------------------------------------------------------------------- #


async def test_a_dropped_content_word_is_enough_to_widen():
    """No number anywhere in the pair — the content-word view alone must see
    that the scope condition is gone."""
    older = _mem("Refunds require a manager signature for enterprise accounts", days_ago=5)
    newer = _mem("Refunds require a manager signature", days_ago=0)
    assert _jaccard(older.content, newer.content) >= 0.6

    _session, _repo, superseded = await _resolve([older, newer])

    assert superseded == []


async def test_a_dropped_short_number_is_enough_to_widen():
    """Content words are IDENTICAL here: "2" is one character, so it never
    reaches the content-word view. Only the number view can see that "Tier 2"
    became "Tier" — which is why the two are checked separately."""
    older = _mem("Tier 2 escalation applies to enterprise accounts", days_ago=5)
    newer = _mem("Tier escalation applies to enterprise accounts", days_ago=0)
    assert _jaccard(older.content, newer.content) >= 0.6

    _session, _repo, superseded = await _resolve([older, newer])

    assert superseded == []


# --------------------------------------------------------------------------- #
# The skip is a recorded decision, not a silent one
# --------------------------------------------------------------------------- #


async def test_the_skip_is_recorded_with_the_score_it_declined_to_act_on():
    """"How often does this fire, and on what" has to be a query. The rule
    name is what keeps the row out of every supersession reader."""
    narrow = _mem(NARROW, days_ago=5)
    wide = _mem(WIDE, days_ago=0)

    session, _repo, _superseded = await _resolve([narrow, wide], compile_job_id="job-9")

    (record,) = _records(session)
    assert record.rule == RULE_WIDENING_SKIPPED
    assert record.superseded_memory_id == narrow.id
    assert record.superseding_memory_id == wide.id
    assert record.score == pytest.approx(0.75)
    assert record.threshold == pytest.approx(0.6)
    assert record.compile_job_id == "job-9"


async def test_the_skip_record_describes_the_drop_without_quoting_it():
    """Counts, not tokens: the dropped words are memory text, and this table
    holds none (#419/#423)."""
    narrow = _mem(NARROW, days_ago=5)
    wide = _mem(WIDE, days_ago=0)

    session, _repo, _superseded = await _resolve([narrow, wide])

    (record,) = _records(session)
    # "days" left the content words; "30" left the numbers.
    assert record.details == {"dropped_sig_tokens": 1, "dropped_numbers": 1}
    stored = " ".join(str(v) for v in record.details.values())
    assert "days" not in stored and "30" not in stored


async def test_a_skip_alone_still_performs_no_status_write():
    """The call contract: at most one `mark_memories_superseded`, and none at
    all when nothing was superseded. A skip must not smuggle an id into it."""
    session, mock_repo, superseded = await _resolve([_mem(NARROW, days_ago=5), _mem(WIDE)])

    assert superseded == []
    mock_repo.mark_memories_superseded.assert_not_called()
    assert len(_records(session)) == 1


# --------------------------------------------------------------------------- #
# A skip is not immunity
# --------------------------------------------------------------------------- #


async def test_a_skipped_pair_does_not_shield_the_older_row_from_a_real_change():
    """The guard moves on to the next candidate instead of abandoning the row.
    Stopping at the first skip would leave a genuinely stale memory active for
    as long as some later restatement happened to widen it."""
    narrow = _mem(NARROW, days_ago=10)
    wide = _mem(WIDE, days_ago=5)
    changed = _mem("Refunds are approved up to 500 EUR for orders under 60 days", days_ago=1)

    session, _repo, superseded = await _resolve([narrow, wide, changed])

    # 30 days -> 60 days is a value replacement: the narrow row IS stale now.
    assert narrow.id in superseded
    rules = sorted({r.rule for r in _records(session)})
    assert rules == [RULE_LEXICAL, RULE_WIDENING_SKIPPED]


# --------------------------------------------------------------------------- #
# The rule set every reader filters on
# --------------------------------------------------------------------------- #


def test_the_skip_rule_is_not_a_superseding_rule():
    """The whole safety of writing skips into `supersession_records` rests on
    this: readers ask the rule, and an admin panel reading an unfiltered row
    would report an ACTIVE memory as superseded."""
    assert RULE_WIDENING_SKIPPED not in SUPERSEDING_RULES
    assert RULE_LEXICAL in SUPERSEDING_RULES


def test_a_decision_knows_whether_it_retired_anything():
    ids = {"superseded_memory_id": uuid.uuid4(), "superseding_memory_id": uuid.uuid4()}
    assert SupersessionDecision(rule=RULE_LEXICAL, **ids).supersedes() is True
    assert SupersessionDecision(rule=RULE_WIDENING_SKIPPED, **ids).supersedes() is False
