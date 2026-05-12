"""Unit tests for state-assembly receipts (#49).

Covers three layers:
  1. Pure helpers: emission decision matrix, ULID properties,
     canonicalization, provenance_hash.
  2. Receipt body construction: every field present, correct shape,
     deterministic.
  3. The six acceptance-criteria negative tests from issue #49 —
     each is a deterministic assertion against the receipt body, no
     access to assembly internals required.

These tests are unit-level (no Postgres). The integration-level
test that round-trips through the API lives in
tests/integration/test_receipts_api.py and requires the test DB.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone

from server.services.receipts import (
    CANONICALIZATION_VERSION,
    EmissionDecision,
    SelectedEpisode,
    SelectedMemory,
    build_receipt_body,
    canonicalize_context,
    decide_emission,
    new_ulid,
    provenance_hash,
    supersession_status_from_row,
)


# ---------------------------------------------------------------------------
# decide_emission — full input matrix
# ---------------------------------------------------------------------------


def test_emit_default_off_when_nothing_requests_it():
    d = decide_emission(request_flag=None, tenant_config=None)
    assert d == EmissionDecision(emit=False, reason="default_off")


def test_emit_on_per_request_flag():
    d = decide_emission(request_flag=True, tenant_config=None)
    assert d.emit and d.reason == "request_flag"


def test_emit_off_when_request_flag_false_and_no_tenant_override():
    d = decide_emission(request_flag=False, tenant_config={"receipts": "on_request"})
    assert not d.emit and d.reason == "default_off"


def test_emit_tenant_always_overrides_request_false():
    d = decide_emission(request_flag=False, tenant_config={"receipts": "always"})
    assert d.emit and d.reason == "tenant_always"


def test_emit_tenant_never_overrides_request_true():
    d = decide_emission(request_flag=True, tenant_config={"receipts": "never"})
    assert not d.emit and d.reason == "tenant_never"


def test_emit_policy_force_on_overrides_everything():
    # Policy decision is reserved for #50; this test pins the contract.
    d = decide_emission(
        request_flag=False,
        tenant_config={"receipts": "never"},
        policy_decision="must_emit",
    )
    assert d.emit and d.reason == "policy_force_on"


def test_emit_policy_force_off_overrides_request_true():
    d = decide_emission(
        request_flag=True,
        tenant_config={"receipts": "always"},
        policy_decision="must_skip",
    )
    assert not d.emit and d.reason == "policy_force_off"


def test_emit_env_kill_switch_wins_over_all(monkeypatch):
    monkeypatch.setenv("STATEWAVE_RECEIPTS_DISABLED", "true")
    d = decide_emission(
        request_flag=True,
        tenant_config={"receipts": "always"},
        policy_decision="must_emit",
    )
    assert not d.emit and d.reason == "kill_switch"


# ---------------------------------------------------------------------------
# ULID
# ---------------------------------------------------------------------------


def test_ulid_length_and_alphabet():
    ulid = new_ulid()
    assert len(ulid) == 26
    # Crockford Base32 alphabet
    assert all(c in "0123456789ABCDEFGHJKMNPQRSTVWXYZ" for c in ulid)


def test_ulid_sorts_by_time(monkeypatch):
    # Two ULIDs minted in sequence — the later one should sort >= the earlier.
    a = new_ulid()
    b = new_ulid()
    # The 10-char time prefix is monotonic; full string ordering matches the
    # time ordering as long as the time prefix differs. Within the same
    # millisecond ordering is random, so allow >= rather than strict >.
    assert a[:10] <= b[:10]


def test_ulid_uniqueness_at_scale():
    # 10k receipts in the same process should never collide.
    ids = {new_ulid() for _ in range(10_000)}
    assert len(ids) == 10_000


# ---------------------------------------------------------------------------
# canonicalize_context + provenance_hash
# ---------------------------------------------------------------------------


def test_canonicalize_matches_raw_utf8_sha256():
    body = "## Task\nHello\n\n## About this user\n- some fact\n"
    digest, size = canonicalize_context(body)
    expected = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert digest == expected
    assert size == len(body.encode("utf-8"))


def test_canonicalize_deterministic_across_calls():
    body = "the same bytes"
    assert canonicalize_context(body) == canonicalize_context(body)


def test_canonicalize_changes_with_one_byte_flip():
    a, _ = canonicalize_context("hello")
    b, _ = canonicalize_context("Hello")
    assert a != b


def test_provenance_hash_stable_across_order():
    ids = [uuid.uuid4() for _ in range(3)]
    assert provenance_hash(ids) == provenance_hash(list(reversed(ids)))


def test_provenance_hash_empty_is_stable():
    assert provenance_hash([]) == provenance_hash([])


def test_provenance_hash_changes_with_content():
    a = provenance_hash([uuid.uuid4()])
    b = provenance_hash([uuid.uuid4()])
    assert a != b  # cryptographically negligible chance of collision


# ---------------------------------------------------------------------------
# supersession_status_from_row
# ---------------------------------------------------------------------------


def _fake_row(**kwargs):
    """Build a minimal duck-typed row object for the helpers under test."""
    class _R:
        pass

    r = _R()
    for k, v in kwargs.items():
        setattr(r, k, v)
    return r


def test_supersession_status_passes_through_active():
    assert supersession_status_from_row(_fake_row(status="active")) == "active"


def test_supersession_status_passes_through_superseded():
    assert supersession_status_from_row(_fake_row(status="superseded")) == "superseded"


def test_supersession_status_passes_through_tombstoned():
    # The receipt records the stored status as-is. A stale memory whose
    # valid_to is in the past but whose `status` is still "active" must
    # NOT be relabeled here — the negative-test for "stale fact" depends
    # on the receipt showing exactly what the assembly path saw.
    assert supersession_status_from_row(_fake_row(status="tombstoned")) == "tombstoned"


# ---------------------------------------------------------------------------
# build_receipt_body — shape, completeness, determinism
# ---------------------------------------------------------------------------


def _sample_body(**overrides):
    """Build a representative receipt body for the negative tests."""
    now = datetime.now(timezone.utc)
    selected_memories = overrides.pop(
        "selected_memories",
        [
            SelectedMemory(
                memory_id=uuid.uuid4(),
                kind="profile_fact",
                valid_from=now - timedelta(days=10),
                valid_to=None,
                supersession_status="active",
                source_episode_ids=[uuid.uuid4()],
                rank=1,
            )
        ],
    )
    selected_episodes = overrides.pop("selected_episodes", [])
    base = dict(
        receipt_id=new_ulid(),
        mode="retrieval",
        tenant_id="acme",
        subject_id="user:42",
        task="What's their plan tier?",
        as_of=now,
        selected_memories=selected_memories,
        selected_episodes=selected_episodes,
        context_hash="deadbeef" * 8,
        context_size_bytes=128,
        token_estimate=32,
    )
    base.update(overrides)
    return build_receipt_body(**base)


def test_receipt_body_has_all_top_level_keys():
    body = _sample_body()
    for key in (
        "receipt_id",
        "parent_receipt_id",
        "mode",
        "query_id",
        "task_id",
        "tenant_id",
        "subject_id",
        "task",
        "as_of",
        "created_at",
        "selected_entries",
        "policy",
        "output",
        "region",
        "receipt_signature",
    ):
        assert key in body, f"missing top-level field: {key}"


def test_receipt_body_policy_block_v1_is_empty_no_opinion():
    # v1 must surface the policy keys with empty lists — that's the
    # contract the SDK / admin UI write against today, before #50 lands.
    body = _sample_body()
    assert body["policy"]["filters_applied"] == []
    assert body["policy"]["filters_skipped"] == []
    assert body["policy"]["mode"] == "log_only"
    assert body["policy"]["policy_bundle_hash"] is None


def test_receipt_body_output_block_carries_canonicalization_version():
    body = _sample_body()
    assert body["output"]["canonicalization_version"] == CANONICALIZATION_VERSION


def test_receipt_body_selected_memories_include_provenance_hash():
    src_ids = [uuid.uuid4(), uuid.uuid4()]
    body = _sample_body(
        selected_memories=[
            SelectedMemory(
                memory_id=uuid.uuid4(),
                kind="profile_fact",
                valid_from=datetime.now(timezone.utc),
                valid_to=None,
                supersession_status="active",
                source_episode_ids=src_ids,
                rank=1,
            )
        ]
    )
    entry = body["selected_entries"][0]
    assert entry["provenance_hash"] == provenance_hash(src_ids)


# ---------------------------------------------------------------------------
# Issue #49 — six negative-test acceptance criteria
#
# Each test below asserts a failure mode is DETECTABLE from the receipt
# alone, with no access to assembly internals. The assertions are the
# load-bearing part — they're what a compliance reviewer / eval harness
# would actually run.
# ---------------------------------------------------------------------------


def test_negative_1_stale_fact_detectable():
    """A memory with valid_to in the past appears in selected_entries —
    a reviewer can detect it by checking each entry's valid_to against
    the receipt's as_of."""
    now = datetime.now(timezone.utc)
    stale = SelectedMemory(
        memory_id=uuid.uuid4(),
        kind="profile_fact",
        valid_from=now - timedelta(days=30),
        valid_to=now - timedelta(days=5),  # past
        supersession_status="active",  # status hasn't been swept yet
        source_episode_ids=[uuid.uuid4()],
        rank=1,
    )
    body = _sample_body(as_of=now, selected_memories=[stale])

    stale_entries = [
        e
        for e in body["selected_entries"]
        if e.get("valid_to") and e["valid_to"] < body["as_of"]
    ]
    assert stale_entries, "stale fact must be detectable from receipt alone"


def test_negative_2_superseded_memory_detectable():
    """A memory with supersession_status=superseded that influenced
    ranking is recorded as such."""
    superseded = SelectedMemory(
        memory_id=uuid.uuid4(),
        kind="profile_fact",
        valid_from=datetime.now(timezone.utc) - timedelta(days=10),
        valid_to=None,
        supersession_status="superseded",
        source_episode_ids=[],
        rank=2,
    )
    body = _sample_body(selected_memories=[superseded])
    bad = [e for e in body["selected_entries"] if e["supersession_status"] == "superseded"]
    assert bad, "superseded entries must be visible in receipt"


def test_negative_3_tombstoned_memory_resurrected_detectable():
    """A tombstoned memory that was nonetheless included shows up with
    supersession_status=tombstoned."""
    tomb = SelectedMemory(
        memory_id=uuid.uuid4(),
        kind="profile_fact",
        valid_from=datetime.now(timezone.utc) - timedelta(days=30),
        valid_to=None,
        supersession_status="tombstoned",
        source_episode_ids=[],
        rank=1,
    )
    body = _sample_body(selected_memories=[tomb])
    resurrected = [
        e for e in body["selected_entries"] if e["supersession_status"] == "tombstoned"
    ]
    assert resurrected


def test_negative_4_conflict_status_field_exists_for_conflict_grouping():
    """v1 of the conflict-resolution path doesn't write fact_key /
    conflict_status yet, but the receipt schema reserves both fields so
    a future test can assert "two entries share a fact_key and neither
    has conflict_status=merged"."""
    body = _sample_body(
        selected_memories=[
            SelectedMemory(
                memory_id=uuid.uuid4(),
                kind="profile_fact",
                valid_from=datetime.now(timezone.utc),
                valid_to=None,
                supersession_status="active",
                source_episode_ids=[],
                rank=1,
                fact_key="email",
                conflict_status="merged",
            ),
            SelectedMemory(
                memory_id=uuid.uuid4(),
                kind="profile_fact",
                valid_from=datetime.now(timezone.utc),
                valid_to=None,
                supersession_status="active",
                source_episode_ids=[],
                rank=2,
                fact_key="email",
                conflict_status="none",  # the failure mode
            ),
        ]
    )
    by_key: dict[str, list[dict]] = {}
    for e in body["selected_entries"]:
        if e.get("fact_key"):
            by_key.setdefault(e["fact_key"], []).append(e)
    unresolved = [
        k
        for k, es in by_key.items()
        if len(es) >= 2 and not all(e["conflict_status"] == "merged" for e in es)
    ]
    assert unresolved == ["email"]


def test_negative_5_as_of_drift_detectable():
    """A caller comparing the requested as_of vs the receipt's as_of can
    detect silent fall-back to latest-state recall."""
    requested_as_of = datetime(2024, 1, 1, tzinfo=timezone.utc)
    actual_as_of = datetime.now(timezone.utc)
    body = _sample_body(as_of=actual_as_of)
    # The caller knows what they asked for; the receipt records what
    # resolved. They differ → silent fall-back.
    from datetime import datetime as _dt
    receipt_as_of = _dt.fromisoformat(body["as_of"])
    assert receipt_as_of != requested_as_of


def test_negative_6_context_hash_tamper_detectable():
    """A consumer recomputing sha256 over the bytes they received and
    comparing to receipt.output.context_hash detects any byte-level
    drift between what the receipt says was returned and what actually
    was."""
    delivered = "## Task\nhello\n"
    body = _sample_body(
        context_hash=canonicalize_context(delivered)[0],
        context_size_bytes=len(delivered.encode("utf-8")),
    )
    # Honest case
    digest, _ = canonicalize_context(delivered)
    assert digest == body["output"]["context_hash"]
    # Tampered case
    digest_tampered, _ = canonicalize_context(delivered + " ")
    assert digest_tampered != body["output"]["context_hash"]


# ---------------------------------------------------------------------------
# Receipt construction with episodes
# ---------------------------------------------------------------------------


def test_receipt_records_episodes_with_rank_and_type():
    ep = SelectedEpisode(
        episode_id=uuid.uuid4(),
        source="slack",
        type="message",
        occurred_at=datetime.now(timezone.utc),
        rank=7,
    )
    body = _sample_body(selected_episodes=[ep])
    ep_entries = [e for e in body["selected_entries"] if e["type"] == "episode"]
    assert ep_entries[0]["rank"] == 7
    assert ep_entries[0]["source"] == "slack"
    assert ep_entries[0]["event_type"] == "message"
