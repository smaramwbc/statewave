"""Unit tests for the sensitivity-label policy layer (#50).

Three layers covered:
  1. Bundle loading + validation — every documented schema error
     surfaces a clear `PolicyError` rather than silently mis-parsing.
  2. Evaluator semantics — the per-rule predicates each behave as
     documented, AND-semantics inside `when:`, first-match-wins
     across rules, default-allow when nothing matches.
  3. Receipt projection — `build_filters_applied` and
     `build_filters_skipped` produce the exact shape the receipt
     consumes, including the bounded `filters_skipped` size guarantee.

The integration-level tests for policy log_only vs enforce behaviour
against a real assembly call live in tests/integration/test_policy_api.py.
"""

from __future__ import annotations

import uuid

import pytest

from server.services.policy import (
    PolicyContext,
    PolicyError,
    apply_decisions,
    build_filters_applied,
    build_filters_skipped,
    evaluate_memory,
    load_bundle,
    REDACTED_MARKER,
)


# ---------------------------------------------------------------------------
# load_bundle — schema validation
# ---------------------------------------------------------------------------


def test_load_minimal_bundle_succeeds():
    b = load_bundle(
        """
        version: 1
        rules:
          - id: r1
            when: {memory_has_any_label: [pii]}
            action: deny
        """
    )
    assert b.rule_count == 1
    assert b.rules[0].id == "r1"
    assert b.rules[0].action == "deny"


def test_load_json_form_also_works():
    """JSON is a subset of YAML 1.2 — both should load cleanly."""
    b = load_bundle(
        '{"version": 1, "rules": [{"id": "r1", "when": {"memory_has_any_label": ["pii"]}, "action": "deny"}]}'
    )
    assert b.rule_count == 1


def test_load_rejects_wrong_version():
    with pytest.raises(PolicyError, match="version 1"):
        load_bundle("version: 2\nrules: []")


def test_load_rejects_top_level_list():
    with pytest.raises(PolicyError, match="mapping at the top level"):
        load_bundle("- not\n- a\n- bundle")


def test_load_rejects_empty_when_block():
    # A rule that matches every memory is almost always a bug — the
    # parser refuses rather than silently accepting it.
    with pytest.raises(PolicyError, match="non-empty mapping"):
        load_bundle(
            """
            version: 1
            rules:
              - id: r1
                when: {}
                action: deny
            """
        )


def test_load_rejects_unknown_predicate():
    with pytest.raises(PolicyError, match="unknown predicate"):
        load_bundle(
            """
            version: 1
            rules:
              - id: r1
                when: {memory_has_any_label: [pii], future_predicate: x}
                action: deny
            """
        )


def test_load_rejects_unknown_action():
    with pytest.raises(PolicyError, match="action must be"):
        load_bundle(
            """
            version: 1
            rules:
              - id: r1
                when: {memory_has_any_label: [pii]}
                action: log
            """
        )


def test_load_rejects_duplicate_rule_id():
    with pytest.raises(PolicyError, match="duplicate rule id"):
        load_bundle(
            """
            version: 1
            rules:
              - id: r1
                when: {memory_has_any_label: [pii]}
                action: deny
              - id: r1
                when: {memory_has_any_label: [financial]}
                action: deny
            """
        )


def test_load_rejects_predicate_type_mismatch():
    """`caller_type: [x]` instead of `caller_type: "x"` would
    silently never match. The loader refuses."""
    with pytest.raises(PolicyError, match="must be a non-empty string"):
        load_bundle(
            """
            version: 1
            rules:
              - id: r1
                when: {memory_has_any_label: [pii], caller_type: [mt]}
                action: deny
            """
        )


def test_bundle_hash_is_stable_under_reformatting():
    """Two bundles with identical logical rules should hash to the
    same value regardless of YAML indentation, key order, comments."""
    a = load_bundle(
        """
        # this comment must not change the hash
        version: 1
        rules:
          - id: r1
            when:
              memory_has_any_label: [pii, financial]
            action: deny
        """
    )
    b = load_bundle(
        """
        version: 1
        rules:
          - action: deny
            when: {memory_has_any_label: [pii, financial]}
            id: r1
        """
    )
    assert a.bundle_hash == b.bundle_hash


def test_bundle_hash_changes_when_a_rule_changes():
    a = load_bundle(
        """
        version: 1
        rules:
          - id: r1
            when: {memory_has_any_label: [pii]}
            action: deny
        """
    )
    b = load_bundle(
        """
        version: 1
        rules:
          - id: r1
            when: {memory_has_any_label: [financial]}
            action: deny
        """
    )
    assert a.bundle_hash != b.bundle_hash


# ---------------------------------------------------------------------------
# evaluate_memory — predicate semantics
# ---------------------------------------------------------------------------


def _ctx(caller_type=None, caller_id=None, tenant_id=None):
    return PolicyContext(
        caller_id=caller_id, caller_type=caller_type, tenant_id=tenant_id
    )


def test_eval_default_allow_when_no_bundle():
    d = evaluate_memory(
        memory_labels=["pii"], bundle=None, context=_ctx(caller_type="x")
    )
    assert d.action == "allow"
    assert d.rule_id is None


def test_eval_default_allow_when_no_rules_match():
    b = load_bundle(
        """
        version: 1
        rules:
          - id: r1
            when: {memory_has_any_label: [pii]}
            action: deny
        """
    )
    d = evaluate_memory(
        memory_labels=["general"], bundle=b, context=_ctx(caller_type="x")
    )
    assert d.action == "allow"


def test_eval_memory_has_any_label_disjunctive():
    b = load_bundle(
        """
        version: 1
        rules:
          - id: r1
            when: {memory_has_any_label: [pii, financial]}
            action: deny
        """
    )
    # Either label matches → deny.
    assert evaluate_memory(memory_labels=["pii"], bundle=b, context=_ctx()).action == "deny"
    assert evaluate_memory(memory_labels=["financial"], bundle=b, context=_ctx()).action == "deny"
    # Neither label → allow.
    assert evaluate_memory(memory_labels=["general"], bundle=b, context=_ctx()).action == "allow"


def test_eval_memory_has_all_labels_conjunctive():
    b = load_bundle(
        """
        version: 1
        rules:
          - id: r1
            when: {memory_has_all_labels: [pii, internal]}
            action: deny
        """
    )
    # Only fires when BOTH labels present.
    assert (
        evaluate_memory(memory_labels=["pii", "internal"], bundle=b, context=_ctx()).action == "deny"
    )
    assert (
        evaluate_memory(memory_labels=["pii"], bundle=b, context=_ctx()).action == "allow"
    )


def test_eval_predicates_AND_inside_when():
    b = load_bundle(
        """
        version: 1
        rules:
          - id: r1
            when:
              memory_has_any_label: [pii]
              caller_type: marketing_tool
            action: deny
        """
    )
    # Both must match.
    assert (
        evaluate_memory(
            memory_labels=["pii"], bundle=b, context=_ctx(caller_type="marketing_tool")
        ).action
        == "deny"
    )
    assert (
        evaluate_memory(
            memory_labels=["pii"], bundle=b, context=_ctx(caller_type="admin")
        ).action
        == "allow"
    )
    assert (
        evaluate_memory(
            memory_labels=["general"], bundle=b, context=_ctx(caller_type="marketing_tool")
        ).action
        == "allow"
    )


def test_eval_caller_type_not_in_negation():
    b = load_bundle(
        """
        version: 1
        rules:
          - id: redact-for-non-admin
            when:
              memory_has_any_label: [secret]
              caller_type_not_in: [admin, security]
            action: redact
        """
    )
    assert (
        evaluate_memory(
            memory_labels=["secret"], bundle=b, context=_ctx(caller_type="marketing")
        ).action
        == "redact"
    )
    assert (
        evaluate_memory(
            memory_labels=["secret"], bundle=b, context=_ctx(caller_type="admin")
        ).action
        == "allow"
    )


def test_eval_first_match_wins():
    """Two rules could both match — only the first one in declaration
    order takes effect, so operators can author override rules near
    the top of the bundle."""
    b = load_bundle(
        """
        version: 1
        rules:
          - id: redact-first
            when: {memory_has_any_label: [pii]}
            action: redact
          - id: deny-second
            when: {memory_has_any_label: [pii]}
            action: deny
        """
    )
    d = evaluate_memory(memory_labels=["pii"], bundle=b, context=_ctx())
    assert d.action == "redact"
    assert d.rule_id == "redact-first"


def test_eval_matched_labels_records_intersection():
    b = load_bundle(
        """
        version: 1
        rules:
          - id: r1
            when: {memory_has_any_label: [pii, financial, secret]}
            action: deny
        """
    )
    d = evaluate_memory(memory_labels=["pii", "financial", "marketing"], bundle=b, context=_ctx())
    # Only the labels in the intersection — `marketing` was on the
    # memory but not on the rule, so it doesn't appear.
    assert d.matched_labels == ["financial", "pii"]


# ---------------------------------------------------------------------------
# apply_decisions — log_only vs enforce
# ---------------------------------------------------------------------------


class _FakeRow:
    """Minimal duck-typed row for apply_decisions."""

    def __init__(self, row_id, content):
        self.id = row_id
        self.content = content


def test_apply_decisions_log_only_keeps_everything():
    a = _FakeRow(uuid.uuid4(), "alpha")
    b = _FakeRow(uuid.uuid4(), "beta")
    from server.services.policy import MemoryPolicyDecision
    decisions = {a.id: MemoryPolicyDecision(action="deny", rule_id="r1")}
    kept, denied = apply_decisions([a, b], decisions, enforce=False)
    assert len(kept) == 2
    assert denied == []
    # `a` was decided "deny" but still has its original content under
    # log_only — the response is untouched. The receipt records what
    # *would* happen under enforce.
    assert a.content == "alpha"


def test_apply_decisions_enforce_removes_denied():
    a = _FakeRow(uuid.uuid4(), "alpha")
    b = _FakeRow(uuid.uuid4(), "beta")
    from server.services.policy import MemoryPolicyDecision
    decisions = {a.id: MemoryPolicyDecision(action="deny", rule_id="r1")}
    kept, denied = apply_decisions([a, b], decisions, enforce=True)
    assert kept == [b]
    assert denied == [a]


def test_apply_decisions_enforce_redacts_content():
    a = _FakeRow(uuid.uuid4(), "leaked secret value")
    from server.services.policy import MemoryPolicyDecision
    decisions = {a.id: MemoryPolicyDecision(action="redact", rule_id="r1")}
    kept, denied = apply_decisions([a], decisions, enforce=True)
    assert kept == [a]
    assert denied == []
    assert a.content == REDACTED_MARKER


# ---------------------------------------------------------------------------
# build_filters_applied / build_filters_skipped — receipt projection
# ---------------------------------------------------------------------------


def test_filters_applied_omits_default_allow():
    from server.services.policy import MemoryPolicyDecision
    decisions = {
        uuid.uuid4(): MemoryPolicyDecision(action="deny", rule_id="r1"),
        uuid.uuid4(): MemoryPolicyDecision(action="allow", rule_id=None),
    }
    applied = build_filters_applied(decisions)
    # The "allow" decision is implicit; recording every default-allow
    # would blow up receipt size at no forensic benefit.
    assert len(applied) == 1
    assert applied[0]["action"] == "deny"


def test_filters_skipped_is_bounded_by_rule_count():
    """`filters_skipped` lists rules that fired against nothing — its
    cardinality is `len(rules)`, NOT `len(memories)`. This is the
    guarantee that keeps receipt size predictable even on assembly
    calls that touched a thousand memories."""
    b = load_bundle(
        """
        version: 1
        rules:
          - id: r1
            when: {memory_has_any_label: [pii]}
            action: deny
          - id: r2
            when: {memory_has_any_label: [financial]}
            action: deny
          - id: r3
            when: {memory_has_any_label: [secret]}
            action: redact
        """
    )
    from server.services.policy import MemoryPolicyDecision
    decisions = {
        uuid.uuid4(): MemoryPolicyDecision(action="deny", rule_id="r1")
    }
    skipped = build_filters_skipped(decisions, b)
    # Two rules (r2, r3) did not fire; their ids appear in skipped.
    assert {s["rule_id"] for s in skipped} == {"r2", "r3"}


def test_filters_skipped_empty_when_no_bundle():
    skipped = build_filters_skipped({}, None)
    assert skipped == []
