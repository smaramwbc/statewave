"""The decision behind a supersession is recorded, not discarded (#419).

Both compile-time producers are covered, because both retire memories:
`conflicts.resolve_conflicts` (claim + lexical) and `reconcile` (LLM
UPDATE/DELETE). Reconcile is enabled by default and retires more memories than
the resolver, so a table fed by only one of them would leave the admin view
with nothing to read for most superseded memories.

What a record may NOT carry is pinned here too: ids, a rule, a claim KEY and
numbers — never memory text or claim values. `memories` is joined for anything
readable, so deleting a subject still takes its text with it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.db.tables import MemoryRow, SupersessionRecordRow
from server.services import reconcile
from server.services.conflicts import resolve_conflicts
from tests._fakes import make_async_session


def _claim(key: str, value: str) -> dict:
    return {"claim": {"schema_version": 1, "key": key, "value": value}}


def _mem(
    content: str,
    *,
    kind: str = "profile_fact",
    days_ago: int = 0,
    metadata: dict | None = None,
) -> MemoryRow:
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
        metadata_=metadata or {},
        status="active",
    )


def _records(session) -> list[SupersessionRecordRow]:
    """Every row the code under test staged on the session."""
    rows: list[SupersessionRecordRow] = []
    for call in session.add_all.call_args_list:
        rows.extend(call.args[0])
    return rows


async def _resolve(memories, session=None, **kwargs):
    session = session or make_async_session()
    with patch("server.services.conflicts.repo") as mock_repo:
        mock_repo.list_active_memories_by_subject = AsyncMock(return_value=memories)
        mock_repo.mark_memories_superseded = AsyncMock()
        superseded = await resolve_conflicts(session, "user-1", **kwargs)
    return session, superseded


# --------------------------------------------------------------------------- #
# conflicts.py — the deterministic resolver
# --------------------------------------------------------------------------- #


async def test_lexical_supersession_records_rule_score_and_successor():
    older = _mem("I use Stripe", days_ago=5)
    newer = _mem("I use Stripe.", days_ago=0)

    session, superseded = await _resolve([older, newer])

    assert superseded == [older.id]
    (record,) = _records(session)
    assert record.superseded_memory_id == older.id
    assert record.superseding_memory_id == newer.id
    assert record.rule == "lexical"
    # The number the decision was actually made on: identical token sets, so a
    # Jaccard of 1.0 against the profile_fact threshold.
    assert record.score == pytest.approx(1.0)
    assert record.threshold == pytest.approx(0.6)
    assert record.claim_key is None
    assert record.subject_id == "user-1"


async def test_lexical_record_carries_the_kind_specific_threshold():
    """A non-fact kind is judged at 0.8, and the record says so — a score
    filed under the wrong threshold cannot be argued about later."""
    older = _mem("The user asked about billing", kind="episode_summary", days_ago=5)
    newer = _mem("The user asked about billing issues", kind="episode_summary", days_ago=0)

    session, superseded = await _resolve([older, newer])

    assert superseded == [older.id]
    (record,) = _records(session)
    assert record.threshold == pytest.approx(0.8)
    assert record.score >= record.threshold


async def test_claim_contradiction_records_the_key_and_no_values():
    older = _mem(
        "Lives in Munich",
        days_ago=5,
        metadata=_claim("location.current_home", "Munich"),
    )
    newer = _mem(
        "Lives in Berlin",
        days_ago=0,
        metadata=_claim("location.current_home", "Berlin"),
    )

    session, superseded = await _resolve([older, newer])

    assert superseded == [older.id]
    (record,) = _records(session)
    assert record.rule == "claim_contradiction"
    assert record.claim_key == "location.current_home"
    # A registry key, not a value and not the v2 bucket (which carries the
    # claim's entity). Neither value, nor either memory's text, is anywhere in
    # the row.
    stored = " ".join(
        str(getattr(record, col.name)) for col in SupersessionRecordRow.__table__.columns
    )
    for secret in ("Munich", "Berlin", older.content, newer.content):
        assert secret not in stored
    # Lexical numbers belong to the lexical rule only.
    assert record.score is None
    assert record.threshold is None


async def test_claim_duplicate_is_recorded_under_its_own_rule():
    """Same key, same value, overlapping windows: a repeated observation, not
    a contradiction. The two are different decisions and are recorded as such."""
    older = _mem(
        "Works at Globex",
        days_ago=5,
        metadata=_claim("employment.current_employer", "Globex"),
    )
    newer = _mem(
        "Just joined the Globex team",
        days_ago=0,
        metadata=_claim("employment.current_employer", "Globex"),
    )

    session, superseded = await _resolve([older, newer])

    assert superseded == [older.id]
    (record,) = _records(session)
    assert record.rule == "claim_duplicate"
    assert record.claim_key == "employment.current_employer"


async def test_nothing_superseded_records_nothing():
    a = _mem("my name is Alice")
    b = _mem("I work at Globex Corporation")

    session, superseded = await _resolve([a, b])

    assert superseded == []
    assert _records(session) == []


async def test_record_carries_the_compile_job_that_made_the_decision():
    older = _mem("I use Stripe", days_ago=5)
    newer = _mem("I use Stripe.", days_ago=0)

    session, _ = await _resolve([older, newer], compile_job_id="job-42")

    (record,) = _records(session)
    assert record.compile_job_id == "job-42"


async def test_record_is_staged_not_committed():
    """The record rides the compile batch's transaction. A resolver that
    committed on its own would leave records behind for a batch that later
    rolled back."""
    older = _mem("I use Stripe", days_ago=5)
    newer = _mem("I use Stripe.", days_ago=0)

    session, _ = await _resolve([older, newer])

    # not_called, not just not_awaited: a sync `session.commit()` would leak a
    # never-awaited coroutine rather than commit, and still has no business here.
    session.commit.assert_not_called()
    session.flush.assert_not_called()


# --------------------------------------------------------------------------- #
# reconcile.py — the LLM producer, on by default
# --------------------------------------------------------------------------- #


def _patch_reconcile(existing, decisions=None, llm_exc=None):
    list_mock = AsyncMock(return_value=existing)
    if llm_exc is not None:
        llm_mock = AsyncMock(side_effect=llm_exc)
    else:
        llm_mock = AsyncMock(return_value={"decisions": decisions or []})
    return (
        patch.object(reconcile.repo, "list_active_memories_by_subject", list_mock),
        patch.object(reconcile.llm_adapter, "acomplete_json", llm_mock),
    )


@pytest.mark.parametrize(
    ("action", "expected_rule"),
    [("UPDATE", "reconcile_update"), ("DELETE", "reconcile_delete")],
)
async def test_reconcile_records_which_action_retired_the_memory(action, expected_rule):
    session = MagicMock()
    e0 = _mem("Alice lives in Munich", days_ago=30)
    c0 = _mem("Alice moved to Berlin in 2024", days_ago=1)
    p_list, p_llm = _patch_reconcile([e0], decisions=[{"i": 0, "action": action, "target": "E0"}])

    with p_list, p_llm:
        kept, superseded = await reconcile.reconcile_compile_batch(session, "user-1", [c0])

    assert superseded == {e0.id}
    (record,) = _records(session)
    assert record.superseded_memory_id == e0.id
    assert record.superseding_memory_id == c0.id
    assert record.rule == expected_rule
    # The model returns an action, not a score, and its `reason` text is never
    # recorded — it is written from the memories.
    assert record.score is None
    assert record.threshold is None


async def test_reconcile_records_no_successor_when_the_winner_is_dropped():
    """A candidate can retire an existing memory and then be retired itself by
    a later candidate, so it is never inserted. The existing memory stays
    superseded, so the decision is recorded with no successor rather than
    pointing at a row that will not exist (and rather than not at all, which
    would send the admin view back to guessing)."""
    session = MagicMock()
    e0 = _mem("API averages 250ms", days_ago=30)
    c_old = _mem("API now averages 150ms", days_ago=10)
    c_new = _mem("API now averages 90ms", days_ago=1)
    p_list, p_llm = _patch_reconcile(
        [e0],
        decisions=[
            {"i": 0, "action": "UPDATE", "target": "E0"},  # c_old retires e0
            {"i": 1, "action": "UPDATE", "target": "N0"},  # c_new retires c_old
        ],
    )

    with p_list, p_llm:
        kept, superseded = await reconcile.reconcile_compile_batch(
            session, "user-1", [c_new, c_old]
        )

    assert [m.id for m in kept] == [c_new.id]
    assert superseded == {e0.id}
    (record,) = _records(session)
    assert record.superseded_memory_id == e0.id
    assert record.superseding_memory_id is None


async def test_reconcile_fail_open_records_nothing():
    """The caller treats a failed reconcile as "superseded nothing". A record
    written on that path would claim a supersession that never happened."""
    session = MagicMock()
    e0 = _mem("Alice lives in Munich", days_ago=30)
    c0 = _mem("Alice moved to Berlin", days_ago=1)
    p_list, p_llm = _patch_reconcile([e0], llm_exc=RuntimeError("provider down"))

    with p_list, p_llm:
        kept, superseded = await reconcile.reconcile_compile_batch(session, "user-1", [c0])

    assert [m.id for m in kept] == [c0.id]
    assert superseded == set()
    assert _records(session) == []


async def test_reconcile_records_carry_tenant_and_job():
    session = MagicMock()
    e0 = _mem("Alice lives in Munich", days_ago=30)
    c0 = _mem("Alice moved to Berlin in 2024", days_ago=1)
    p_list, p_llm = _patch_reconcile([e0], decisions=[{"i": 0, "action": "UPDATE", "target": "E0"}])

    with p_list, p_llm:
        await reconcile.reconcile_compile_batch(
            session, "user-1", [c0], tenant_id="tenant-a", compile_job_id="job-7"
        )

    (record,) = _records(session)
    assert record.tenant_id == "tenant-a"
    assert record.compile_job_id == "job-7"


# --------------------------------------------------------------------------- #
# The table's own contract
# --------------------------------------------------------------------------- #


def test_the_table_has_no_column_that_could_hold_memory_text():
    """#423 is open on subject text outliving subject deletion. This table was
    built not to widen that: a reader joins `memories`. A new column here that
    could carry content, a claim value or a model rationale breaks this test on
    purpose — it is a decision to reopen, not a detail to add."""
    assert {c.name for c in SupersessionRecordRow.__table__.columns} == {
        "id",
        "subject_id",
        "tenant_id",
        "superseded_memory_id",
        "superseding_memory_id",
        "rule",
        "claim_key",
        "score",
        "threshold",
        "compile_job_id",
        "details",
        "created_at",
    }


def test_the_table_has_no_foreign_keys():
    """Reconcile's successor may be a candidate that is never inserted, so an
    FK would abort the compile; an FK cascade would also erase the audit row
    with the memory it explains."""
    assert list(SupersessionRecordRow.__table__.foreign_keys) == []
