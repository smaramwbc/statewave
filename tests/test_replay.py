"""Unit tests for the receipt replay engine (v0.9 #159).

Pure tests — exercise `_compute_diff` and `build_policy_snapshot`
without the DB. End-to-end behaviour (load receipt, re-run assembly,
emit replay receipt) is covered by
`tests/integration/test_replay.py`.
"""

from __future__ import annotations

from server.services.receipts import build_policy_snapshot
from server.services.replay import _compute_diff


# ---------------------------------------------------------------------------
# build_policy_snapshot
# ---------------------------------------------------------------------------


def test_policy_snapshot_with_bundle():
    snap = build_policy_snapshot(
        bundle_hash="abc123",
        bundle_yaml="version: 1\nrules: []\n",
    )
    assert snap["bundle_hash"] == "abc123"
    assert snap["bundle_yaml"] == "version: 1\nrules: []\n"
    assert snap["captured_at"], "captured_at must be set"


def test_policy_snapshot_no_bundle_active():
    """A null pair records 'no policy was active' — a valid, replayable
    state. The replay path treats this as the no-policy fallback."""
    snap = build_policy_snapshot(bundle_hash=None, bundle_yaml=None)
    assert snap["bundle_hash"] is None
    assert snap["bundle_yaml"] is None
    assert snap["captured_at"]


# ---------------------------------------------------------------------------
# Diff envelope shape + matching
# ---------------------------------------------------------------------------


def _entry(memory_id: str, rank: int = 1) -> dict:
    return {"type": "memory", "memory_id": memory_id, "rank": rank}


def _ep(episode_id: str, rank: int = 1) -> dict:
    return {"type": "episode", "episode_id": episode_id, "rank": rank}


def _body(entries: list[dict], filters: list[dict] | None = None, ctx: str = "h") -> dict:
    return {
        "selected_entries": entries,
        "policy": {"filters_applied": filters or []},
        "output": {"context_hash": ctx},
    }


def test_diff_identical_bodies_have_no_changes():
    body = _body([_entry("m1"), _entry("m2")])
    d = _compute_diff(original_body=body, replay_body=body)
    assert d["context_hash"]["changed"] is False
    assert d["selected_entries"]["added"] == []
    assert d["selected_entries"]["removed"] == []
    assert d["selected_entries"]["common"] == 2
    assert d["filters_applied"]["added"] == []
    assert d["filters_applied"]["removed"] == []


def test_diff_added_and_removed_entries():
    original = _body([_entry("m1"), _entry("m2")], ctx="a")
    replay = _body([_entry("m2"), _entry("m3")], ctx="b")
    d = _compute_diff(original_body=original, replay_body=replay)
    assert d["context_hash"]["changed"] is True
    assert d["context_hash"]["original"] == "a"
    assert d["context_hash"]["replay"] == "b"
    added = {e["memory_id"] for e in d["selected_entries"]["added"]}
    removed = {e["memory_id"] for e in d["selected_entries"]["removed"]}
    assert added == {"m3"}
    assert removed == {"m1"}
    assert d["selected_entries"]["common"] == 1


def test_diff_rank_change_alone_is_not_an_add_remove():
    """Re-ordering an entry must NOT register as add+remove — the diff
    is keyed off entry id, not rank/score, so an operator reviewing the
    envelope can tell churn (added/removed) apart from re-ranking."""
    original = _body([_entry("m1", rank=1), _entry("m2", rank=2)])
    replay = _body([_entry("m1", rank=2), _entry("m2", rank=1)])
    d = _compute_diff(original_body=original, replay_body=replay)
    assert d["selected_entries"]["added"] == []
    assert d["selected_entries"]["removed"] == []
    assert d["selected_entries"]["common"] == 2


def test_diff_episode_entries_use_episode_id():
    """Episodes are matched by `episode_id`, memories by `memory_id` —
    the two never collide even if the underlying UUIDs are identical
    strings."""
    original = _body([_ep("e1"), _entry("m1")])
    replay = _body([_ep("e1"), _entry("m1")])
    d = _compute_diff(original_body=original, replay_body=replay)
    assert d["selected_entries"]["common"] == 2


def test_diff_filters_keyed_by_rule_memory_action():
    """Two filters with the same rule_id but different memory_ids are
    distinct rows in the diff."""
    original = _body(
        [_entry("m1")],
        filters=[
            {"rule_id": "r1", "memory_id": "m1", "action": "redact"},
            {"rule_id": "r1", "memory_id": "m2", "action": "redact"},
        ],
    )
    replay = _body(
        [_entry("m1")], filters=[{"rule_id": "r1", "memory_id": "m1", "action": "redact"}]
    )
    d = _compute_diff(original_body=original, replay_body=replay)
    removed = d["filters_applied"]["removed"]
    assert len(removed) == 1
    assert removed[0]["memory_id"] == "m2"


def test_diff_replay_body_none_reports_full_removal():
    """When the replay receipt failed to write, the diff is everything-
    removed: callers still see what was in the original."""
    original = _body([_entry("m1"), _entry("m2")], ctx="a")
    d = _compute_diff(original_body=original, replay_body=None)
    assert d["context_hash"]["original"] == "a"
    assert d["context_hash"]["replay"] is None
    assert d["context_hash"]["changed"] is True
    assert len(d["selected_entries"]["removed"]) == 2
    assert d["selected_entries"]["added"] == []
    assert d["selected_entries"]["common"] == 0


def test_diff_skips_malformed_entries():
    """An entry without a `type` or id field is malformed — skipped
    rather than crashing the diff."""
    original = _body([{"type": "memory", "memory_id": "m1"}, {"random": "junk"}])
    replay = _body([{"type": "memory", "memory_id": "m1"}])
    d = _compute_diff(original_body=original, replay_body=replay)
    # m1 common; junk entry has no key and gets dropped from both sides.
    assert d["selected_entries"]["common"] == 1
    assert d["selected_entries"]["added"] == []
    assert d["selected_entries"]["removed"] == []


def test_diff_missing_policy_section_treated_as_no_filters():
    """A body missing the `policy` section entirely (e.g. a malformed
    legacy receipt) must not crash the diff."""
    original = {
        "selected_entries": [_entry("m1")],
        "output": {"context_hash": "a"},
    }
    replay = {
        "selected_entries": [_entry("m1")],
        "output": {"context_hash": "a"},
    }
    d = _compute_diff(original_body=original, replay_body=replay)
    assert d["filters_applied"]["added"] == []
    assert d["filters_applied"]["removed"] == []
