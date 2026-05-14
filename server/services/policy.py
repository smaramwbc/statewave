"""Sensitivity-label policy layer (issue #50).

The policy layer reads per-memory `sensitivity_labels` (added in
migration 0018) and decides, for each memory in a context assembly,
whether the caller is allowed to receive it. Decisions are always
recorded into the state-assembly receipt's `policy.filters_applied`
field; whether they're *enforced* (memory dropped or redacted) is
governed by the tenant's `policy_mode` config:

  * `log_only` (default) — decisions go into the receipt; nothing is
    filtered from the response. This is the safe-rollout mode: a
    tenant can observe what *would* be filtered for a week or two
    before flipping to enforce.
  * `enforce` — decisions are applied. `deny` drops the memory from
    `selected_entries`; `redact` replaces its content with a stable
    redaction marker (the memory still appears so the receipt records
    that the policy fired).

## Policy format

Policy bundles are YAML (or JSON — both parse with `yaml.safe_load`).
Each bundle is content-hashed at load time; receipts reference the
hash, which makes "what did policy abc123 say on date Y?" answerable
forever — the load-bearing replay feature.

    version: 1
    metadata:
      description: "Production policy bundle v3"
      authored_by: "security-team@example.com"
    rules:
      - id: deny-pii-for-marketing-tools
        description: PII memories cannot be read by marketing tools
        when:
          memory_has_any_label: [pii, sensitive_personal]
          caller_type: marketing_tool
        action: deny
      - id: redact-secrets-for-non-admin
        when:
          memory_has_any_label: [secret, api_key]
          caller_type_not_in: [admin, security]
        action: redact

## v1 predicates and actions

Predicates (inside `when:`):
  * `memory_has_any_label: [tag, ...]` — fires when the memory's
    `sensitivity_labels` overlaps the list (disjunctive).
  * `memory_has_all_labels: [tag, ...]` — fires only when every
    listed label is present (conjunctive).
  * `caller_type: "..."` — exact-match on the request's caller_type.
  * `caller_type_in: [...]` — list-membership.
  * `caller_type_not_in: [...]` — negated list-membership.
  * `caller_id: "..."` — exact-match on caller_id.

All listed predicates within a single `when:` are AND-ed. A rule with
no predicates is a load-time validation error (it would match every
memory, which is almost always a bug).

Actions:
  * `deny` — exclude the memory under enforce mode.
  * `redact` — keep the memory but replace `content` with the
    redaction marker under enforce mode.

Rules are evaluated in declaration order; the first matching rule
wins. A memory that matches no rule falls through to default-allow.
Regex predicates and explicit `allow` rules are deferred to v2 — the
schema reserves the space for them.

## Bundle resolution

For an assembly call running under tenant `T`:
  1. Look up the `active=true` row in `policy_bundles` where
     `tenant_id = T`. If one exists, use it.
  2. Otherwise, look up the global active row (`tenant_id = NULL`).
     If one exists, use it.
  3. Otherwise, no policy is loaded — every memory is default-allow,
     `filters_applied` and `filters_skipped` on the receipt stay
     empty, and `policy_bundle_hash` stays null.

Caching: the active bundle for a tenant is cached in-process for
60 seconds. `POST /admin/policy/reload` busts the cache immediately.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import yaml
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.tables import PolicyBundleRow

# Single redaction marker — stable so consumers can dedupe / detect
# redaction without re-parsing the rule id.
REDACTED_MARKER = "[REDACTED by policy]"

# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyRule:
    """One rule inside a bundle. Predicates are stored as a dict so a
    rule's serialization round-trips through YAML/JSON unchanged for
    the content-hash to remain stable across releases."""

    id: str
    description: str
    when: dict[str, Any]
    action: str  # "deny" | "redact"


@dataclass(frozen=True)
class PolicyBundle:
    """Immutable loaded policy. Identified by its content hash."""

    bundle_hash: str
    version: int
    metadata: dict[str, Any]
    rules: tuple[PolicyRule, ...]
    yaml_content: str

    @property
    def rule_count(self) -> int:
        return len(self.rules)


@dataclass(frozen=True)
class PolicyContext:
    """The per-request inputs the evaluator consults."""

    caller_id: str | None
    caller_type: str | None
    tenant_id: str | None


@dataclass(frozen=True)
class MemoryPolicyDecision:
    """Decision for a single (memory, request) pair."""

    action: str  # "allow" | "deny" | "redact"
    rule_id: str | None
    matched_labels: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bundle loading + validation
# ---------------------------------------------------------------------------


# Predicate keys the v1 evaluator understands. Anything else in a
# rule's `when:` block is a load-time validation error — unknown
# predicates can't silently pass through as no-ops because a tenant
# would think their rule is firing when it isn't.
_KNOWN_PREDICATES = frozenset(
    [
        "memory_has_any_label",
        "memory_has_all_labels",
        "caller_type",
        "caller_type_in",
        "caller_type_not_in",
        "caller_id",
    ]
)

_KNOWN_ACTIONS = frozenset(["deny", "redact"])


class PolicyError(ValueError):
    """Raised on policy load-time validation failures."""


def load_bundle(yaml_or_json: str) -> PolicyBundle:
    """Parse + validate a policy bundle. Computes the content hash
    over a canonical JSON form so the same logical rules hash to the
    same value regardless of YAML formatting / comments / key order.
    """
    try:
        doc = yaml.safe_load(yaml_or_json)
    except yaml.YAMLError as e:
        raise PolicyError(f"invalid YAML/JSON: {e}") from e
    if not isinstance(doc, dict):
        raise PolicyError("policy bundle must be a mapping at the top level")

    version = doc.get("version")
    if version != 1:
        raise PolicyError("only policy schema version 1 is supported in v1")

    metadata = doc.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise PolicyError("metadata must be a mapping")

    rules_raw = doc.get("rules") or []
    if not isinstance(rules_raw, list):
        raise PolicyError("rules must be a list")

    rules: list[PolicyRule] = []
    seen_ids: set[str] = set()
    for idx, r in enumerate(rules_raw):
        if not isinstance(r, dict):
            raise PolicyError(f"rule #{idx} must be a mapping")
        rid = r.get("id")
        if not isinstance(rid, str) or not rid:
            raise PolicyError(f"rule #{idx} is missing a non-empty `id`")
        if rid in seen_ids:
            raise PolicyError(f"duplicate rule id: {rid!r}")
        seen_ids.add(rid)
        when = r.get("when") or {}
        if not isinstance(when, dict) or not when:
            raise PolicyError(
                f"rule {rid!r}: `when` must be a non-empty mapping — a rule "
                "with no predicates would match every memory"
            )
        unknown = set(when.keys()) - _KNOWN_PREDICATES
        if unknown:
            raise PolicyError(
                f"rule {rid!r}: unknown predicate(s): {sorted(unknown)}"
            )
        _validate_predicate_shapes(rid, when)
        action = r.get("action")
        if action not in _KNOWN_ACTIONS:
            raise PolicyError(
                f"rule {rid!r}: action must be one of {sorted(_KNOWN_ACTIONS)}, "
                f"got {action!r}"
            )
        description = r.get("description") or ""
        if not isinstance(description, str):
            raise PolicyError(f"rule {rid!r}: description must be a string")
        rules.append(
            PolicyRule(
                id=rid,
                description=description,
                when=dict(when),
                action=action,
            )
        )

    # Content hash: canonical JSON with sorted keys so YAML reflow,
    # comments, or key reordering don't change the hash.
    canonical = json.dumps(
        {
            "version": version,
            "metadata": metadata,
            "rules": [
                {
                    "id": r.id,
                    "description": r.description,
                    "when": r.when,
                    "action": r.action,
                }
                for r in rules
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    bundle_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return PolicyBundle(
        bundle_hash=bundle_hash,
        version=version,
        metadata=metadata,
        rules=tuple(rules),
        yaml_content=yaml_or_json,
    )


def _validate_predicate_shapes(rule_id: str, when: dict[str, Any]) -> None:
    """Each predicate's value must be the right Python type — a typo
    that swaps `caller_type: x` for `caller_type: [x]` would silently
    never match, which is exactly the failure mode we don't want."""
    for key, value in when.items():
        if key in {"memory_has_any_label", "memory_has_all_labels"}:
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise PolicyError(
                    f"rule {rule_id!r}: {key} must be a list of strings"
                )
            if not value:
                raise PolicyError(
                    f"rule {rule_id!r}: {key} cannot be empty — an empty "
                    "label-set match is ambiguous"
                )
        elif key in {"caller_type_in", "caller_type_not_in"}:
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise PolicyError(
                    f"rule {rule_id!r}: {key} must be a list of strings"
                )
            if not value:
                raise PolicyError(
                    f"rule {rule_id!r}: {key} cannot be empty"
                )
        elif key in {"caller_type", "caller_id"}:
            if not isinstance(value, str) or not value:
                raise PolicyError(
                    f"rule {rule_id!r}: {key} must be a non-empty string"
                )


# ---------------------------------------------------------------------------
# Bundle resolution
# ---------------------------------------------------------------------------
#
# The active bundle is resolved from `policy_bundles` on every
# assembly call. There is intentionally **no in-process cache**:
#
#   * The query is `SELECT * FROM policy_bundles WHERE tenant_id=$1
#     AND active=true LIMIT 1`, served from the existing
#     `(tenant_id, active)` index. Sub-millisecond on warm
#     connections, negligible against the rest of the assembly path
#     (semantic search, scoring, token-budget packing).
#
#   * A module-level cache USED to live here with a 60s TTL and a
#     local `invalidate_bundle_cache()` call after admin writes.
#     That works in single-process tests and in single-machine
#     deployments. It DOES NOT work on multi-replica Fly (or any
#     horizontally-scaled deploy): an `/admin/policy/bundles` call
#     lands on one machine and invalidates its cache; a subsequent
#     `/v1/context` call may route to a different machine whose
#     cache still serves the pre-upload value (None, typically) for
#     the remainder of the TTL. Under enforce mode that's a real
#     security regression — sensitive memories pass through
#     unfiltered for up to 60 seconds after a tenant activates a
#     policy. Caught in prod smoke testing of #50.
#
# Cross-replica cache busting would need either Postgres
# LISTEN/NOTIFY (each replica subscribes; bust on NOTIFY) or a
# version-row pattern with an extra SELECT-per-call (which is the
# same cost as just dropping the cache). Both add complexity for no
# practical benefit. Drop the cache.


def invalidate_bundle_cache() -> None:
    """No-op kept for API stability — older admin endpoints still call
    this after policy uploads. The active bundle is now resolved from
    DB on every assembly call, so there is nothing to invalidate."""
    return None


async def resolve_active_bundle(
    session: AsyncSession,
    tenant_id: str | None,
) -> PolicyBundle | None:
    """Return the active bundle for a tenant, falling back to the
    global active bundle. None if neither exists.

    Always hits the database. See the module-level comment above for
    why there is no cache.

    Fail-open: if the DB call raises (table missing pre-migration,
    transient connection failure), this returns None and the caller
    defaults to "no bundle loaded → everything allowed." That matches
    pre-#50 behaviour exactly, so a degraded policy table never
    breaks agent serving. The failure is logged.
    """
    try:
        return await _load_active_from_db(session, tenant_id)
    except Exception:
        import structlog
        structlog.stdlib.get_logger().warning(
            "policy_bundle_resolution_failed",
            tenant_id=tenant_id,
            exc_info=True,
        )
        return None


async def _load_active_from_db(
    session: AsyncSession,
    tenant_id: str | None,
) -> PolicyBundle | None:
    """Try tenant-specific first, then global. Returns the parsed
    bundle or None when neither slot has an active row."""
    if tenant_id is not None:
        stmt = (
            select(PolicyBundleRow)
            .where(PolicyBundleRow.tenant_id == tenant_id)
            .where(PolicyBundleRow.active.is_(True))
            .order_by(PolicyBundleRow.created_at.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is not None:
            return load_bundle(row.yaml_content)

    stmt = (
        select(PolicyBundleRow)
        .where(PolicyBundleRow.tenant_id.is_(None))
        .where(PolicyBundleRow.active.is_(True))
        .order_by(PolicyBundleRow.created_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    if row is not None:
        return load_bundle(row.yaml_content)
    return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_memory(
    *,
    memory_labels: Sequence[str],
    bundle: PolicyBundle | None,
    context: PolicyContext,
) -> MemoryPolicyDecision:
    """Apply the bundle's rules to one memory in declaration order.
    First matching rule wins; no match → default-allow.

    Pure function — no DB, no IO. Tested directly in
    tests/test_policy.py with synthetic rules and memories.
    """
    if bundle is None or not bundle.rules:
        return MemoryPolicyDecision(action="allow", rule_id=None)

    for rule in bundle.rules:
        matched, matched_labels = _rule_matches(rule, memory_labels, context)
        if matched:
            return MemoryPolicyDecision(
                action=rule.action,
                rule_id=rule.id,
                matched_labels=matched_labels,
            )
    return MemoryPolicyDecision(action="allow", rule_id=None)


def _rule_matches(
    rule: PolicyRule,
    memory_labels: Sequence[str],
    context: PolicyContext,
) -> tuple[bool, list[str]]:
    """Returns `(matches, labels_that_matched)`. All predicates inside
    `when` are AND-ed; if any returns False the whole rule does not
    match. `labels_that_matched` is the intersection of memory labels
    with the rule's label set, when label predicates participated."""
    labels_set = set(memory_labels or [])
    matched_labels: list[str] = []
    for key, value in rule.when.items():
        if key == "memory_has_any_label":
            overlap = labels_set & set(value)
            if not overlap:
                return False, []
            matched_labels.extend(sorted(overlap))
        elif key == "memory_has_all_labels":
            required = set(value)
            if not required.issubset(labels_set):
                return False, []
            matched_labels.extend(sorted(required))
        elif key == "caller_type":
            if context.caller_type != value:
                return False, []
        elif key == "caller_type_in":
            if context.caller_type not in value:
                return False, []
        elif key == "caller_type_not_in":
            if context.caller_type in value:
                return False, []
        elif key == "caller_id":
            if context.caller_id != value:
                return False, []
    return True, sorted(set(matched_labels))


# ---------------------------------------------------------------------------
# Helpers for the assembly path
# ---------------------------------------------------------------------------


def apply_decisions(
    rows: Iterable[Any],
    decisions: dict[Any, MemoryPolicyDecision],
    *,
    enforce: bool,
) -> tuple[list[Any], list[Any]]:
    """Given a list of memory rows and per-id decisions, return
    `(kept_rows, denied_rows)`. When `enforce=False` (log-only mode),
    every row is kept and `denied_rows` is empty — but the decisions
    dict is still consumed downstream to populate the receipt.

    `redact` is applied by mutating the in-memory row's `content`
    field in place (the row object is a transient SQLAlchemy ORM
    instance; the underlying DB row is not touched). The receipt
    still records the redaction so consumers can detect it.
    """
    if not enforce:
        return list(rows), []
    kept: list[Any] = []
    denied: list[Any] = []
    for row in rows:
        decision = decisions.get(row.id)
        if decision is None or decision.action == "allow":
            kept.append(row)
        elif decision.action == "deny":
            denied.append(row)
        elif decision.action == "redact":
            # Mutate the transient row — the DB-side memory is
            # untouched. Downstream rendering sees the marker;
            # the receipt's selected_entries still records the row.
            row.content = REDACTED_MARKER
            kept.append(row)
        else:  # defensive — load-time validation already gates this
            kept.append(row)
    return kept, denied


def build_filters_applied(
    decisions: dict[Any, MemoryPolicyDecision],
) -> list[dict[str, Any]]:
    """Build the receipt's `policy.filters_applied` block — one entry
    per memory where a rule fired (deny or redact). Memories that
    fell through to default-allow are NOT recorded; doing so would
    add an entry per memory per call and blow up receipt size at no
    forensic benefit."""
    out: list[dict[str, Any]] = []
    for memory_id, decision in decisions.items():
        if decision.action == "allow":
            continue
        out.append(
            {
                "memory_id": str(memory_id),
                "rule_id": decision.rule_id,
                "action": decision.action,
                "matched_labels": list(decision.matched_labels),
            }
        )
    return out


def build_filters_skipped(
    decisions: dict[Any, MemoryPolicyDecision],
    bundle: PolicyBundle | None,
) -> list[dict[str, Any]]:
    """One entry per *rule* that was evaluated but matched zero
    memories. Bounded by `len(bundle.rules)` — does not scale with
    memory count.

    This is the answer to issue #49's `filters_skipped`: "evaluated
    but did not match." Per-memory miss-by-miss would be unbounded
    and uninteresting; per-rule miss totals are exactly the signal a
    compliance reviewer wants ("did our PII rule actually fire on
    anything this hour?")."""
    if bundle is None:
        return []
    fired_rule_ids = {
        d.rule_id for d in decisions.values() if d.rule_id is not None
    }
    return [
        {"rule_id": rule.id, "action": rule.action, "matched": False}
        for rule in bundle.rules
        if rule.id not in fired_rule_ids
    ]
