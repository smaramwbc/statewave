"""State-assembly receipt replay (v0.9, issue #159).

Re-runs the original assembly against **current memories** but the
**original policy bundle** captured in the receipt's ``policy_snapshot``
envelope. Emits a fresh receipt with ``mode="as_of_replay"`` and
``parent_receipt_id`` pointing at the original, plus a diff envelope
describing what changed between the two.

## Semantic

The user-approved semantic for v0.9: *current code + original policy*.
Replay is **not** byte-for-byte reproduction:

  * Memories may have been added, tombstoned, or supersession-resolved
    between original emission and replay. Those are real-world changes
    and appear in the diff as added/removed selected_entries.
  * Episodes may have been ingested. New episodes appearing in scope
    show up as added entries.
  * The compiler / scoring algorithms run against whatever code is
    deployed at replay time. If a heuristic changed, ranks may differ
    — the diff records that as a context_hash change.
  * The policy bundle, by contrast, is frozen on the original receipt
    via ``policy_snapshot.bundle_yaml`` and replayed verbatim — so a
    rule that the operator later deleted from the live bundle still
    fires during replay. This is the load-bearing property of the
    snapshot.

## Refusals (422 ``unreplayable``)

  * ``missing_policy_snapshot`` — pre-v0.9 receipt, never captured a
    snapshot. Operator must wait for fresh v0.9 traffic, or accept
    that the audit trail for that emission stops at "what was stored."
  * ``nested_replay`` — the receipt is itself a replay
    (``mode == "as_of_replay"``). v0.9 ships one level only; replaying
    a replay is not supported.
  * ``invalid_snapshot`` — the snapshot's YAML fails to parse (it
    parsed at emission, so corruption / tampering is the likely
    cause). Surface explicitly rather than crashing.

Failure to emit the replay receipt itself does NOT raise — same
contract as `services.receipts.write_receipt`. The endpoint returns
the diff envelope unconditionally; ``replay_receipt_id`` will be null
on write failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.db.tables import ReceiptRow
from server.services import context as context_service
from server.services import policy as policy_service

logger = structlog.stdlib.get_logger()


class ReplayError(Exception):
    """Raised when a receipt cannot be replayed. ``reason`` is one of
    the documented ``unreplayable`` codes; the API layer maps it to
    HTTP 422 with the same code in the response body."""

    def __init__(self, reason: str, detail: str | None = None):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of a successful replay. Returned by `replay_receipt`."""

    original_receipt_id: str
    replay_receipt_id: str | None
    """ULID of the new replay receipt, or None if the write failed
    (the diff is still authoritative — failure is logged)."""
    diff: dict[str, Any]
    """Structural diff between original and replay receipt bodies.
    See `_compute_diff` for the shape."""


async def replay_receipt(
    session: AsyncSession,
    *,
    receipt_id: str,
    tenant_id: str | None = None,
) -> ReplayResult:
    """Load the original receipt, replay against current memories, and
    return the diff envelope plus the new receipt id.

    Tenant-scoped: callers passing ``tenant_id`` cannot replay another
    tenant's receipts even if they guess the ULID.

    Raises:
        ReplayError: when the receipt cannot be replayed. ``reason``
            is one of: ``not_found``, ``missing_policy_snapshot``,
            ``nested_replay``, ``invalid_snapshot``.
    """
    original = await repo.get_receipt_by_id(session, receipt_id, tenant_id=tenant_id)
    if original is None:
        raise ReplayError("not_found", f"receipt {receipt_id} not found")

    # Refuse v0.9 nested replay — replaying a replay would compound
    # diffs and obscure the audit trail.
    if original.mode == "as_of_replay":
        raise ReplayError(
            "nested_replay",
            "this receipt is itself a replay; replay the original parent instead",
        )

    snapshot = original.policy_snapshot
    if snapshot is None:
        raise ReplayError(
            "missing_policy_snapshot",
            "pre-v0.9 receipt — no policy snapshot was captured at emission",
        )

    # Parse the snapshot bundle. A null inner pair is a valid
    # "no policy was active" snapshot; we replay that as "no bundle."
    snapshot_bundle: policy_service.PolicyBundle | None = None
    bundle_yaml = snapshot.get("bundle_yaml")
    if bundle_yaml:
        try:
            snapshot_bundle = policy_service.load_bundle(bundle_yaml)
        except policy_service.PolicyError as exc:
            raise ReplayError(
                "invalid_snapshot",
                f"snapshot YAML failed to parse: {exc}",
            ) from exc

    # Re-extract the request fields the assembly path needs. The body
    # is authoritative — the columns are denormalised for indexing.
    body = original.body or {}
    subject_id = original.subject_id
    task = body.get("task")
    if not task:
        # A receipt without a `task` cannot be replayed: the assembly
        # path's scoring is keyed off it. v0.8+ always populates this,
        # so a missing value is structural corruption — surface
        # rather than emit an empty replay.
        raise ReplayError(
            "invalid_snapshot",
            "original receipt body is missing `task` — cannot replay",
        )

    # Replay assembly. `_use_policy_bundle_override` flag is the only
    # signal that distinguishes "skip the snapshot bundle (None)"
    # from "use the live bundle"; the value itself can be None when
    # the snapshot recorded no active policy.
    bundle = await context_service.assemble_context(
        session,
        subject_id=subject_id,
        task=task,
        tenant_id=original.tenant_id,
        emit_receipt=True,
        parent_receipt_id=original.receipt_id,
        caller_id=body.get("caller_id"),
        caller_type=body.get("caller_type"),
        _policy_bundle_override=snapshot_bundle,
        _use_policy_bundle_override=True,
        _mode_override="as_of_replay",
    )

    # Load the freshly-emitted replay receipt so we can diff its body
    # against the original. The assembler returns the receipt_id only;
    # the persisted body is the authoritative source of truth (the
    # in-memory `bundle` carries only the assembled-context surface,
    # not the receipt body itself).
    replay_row: ReceiptRow | None = None
    if bundle.receipt_id:
        replay_row = await repo.get_receipt_by_id(
            session, bundle.receipt_id, tenant_id=original.tenant_id
        )
    diff = _compute_diff(
        original_body=body,
        replay_body=replay_row.body if replay_row else None,
    )

    return ReplayResult(
        original_receipt_id=original.receipt_id,
        replay_receipt_id=bundle.receipt_id,
        diff=diff,
    )


# ---------------------------------------------------------------------------
# Diff envelope
# ---------------------------------------------------------------------------


def _compute_diff(
    *,
    original_body: dict[str, Any],
    replay_body: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare two receipt bodies and return the structural diff envelope.

    Envelope shape (stable contract; see docs/replay.md):

        {
          "context_hash": {
              "original": "...",
              "replay":   "...",
              "changed":  True | False
          },
          "selected_entries": {
              "added":    [<entry>, ...],   # in replay, not in original
              "removed":  [<entry>, ...],   # in original, not in replay
              "common":   <int>             # entries present in both, by id
          },
          "filters_applied": {
              "added":    [<filter>, ...],
              "removed":  [<filter>, ...]
          }
        }

    Entries are matched by their ``memory_id`` / ``episode_id`` —
    re-ranking the same entry does not show up as add+remove. The
    ``common`` count counts entries present in both sets regardless
    of rank or score.

    If ``replay_body`` is None (the replay receipt write failed),
    the diff is reported as everything-removed: the original entries
    are listed as ``removed`` and ``replay`` fields are null, so the
    caller can still see what was on the original.
    """
    original_entries = list(original_body.get("selected_entries", []) or [])
    original_filters = list(
        (original_body.get("policy", {}) or {}).get("filters_applied", []) or []
    )
    original_hash = (original_body.get("output", {}) or {}).get("context_hash")

    if replay_body is None:
        return {
            "context_hash": {
                "original": original_hash,
                "replay": None,
                "changed": True,
            },
            "selected_entries": {
                "added": [],
                "removed": original_entries,
                "common": 0,
            },
            "filters_applied": {
                "added": [],
                "removed": original_filters,
            },
        }

    replay_entries = list(replay_body.get("selected_entries", []) or [])
    replay_filters = list((replay_body.get("policy", {}) or {}).get("filters_applied", []) or [])
    replay_hash = (replay_body.get("output", {}) or {}).get("context_hash")

    # Entries keyed by their object id (memory_id or episode_id). An
    # entry with neither is malformed; skip rather than crash.
    def _key(entry: dict[str, Any]) -> str | None:
        if entry.get("type") == "memory":
            return entry.get("memory_id")
        if entry.get("type") == "episode":
            return entry.get("episode_id")
        return None

    orig_by_id = {k: e for e in original_entries if (k := _key(e))}
    repl_by_id = {k: e for e in replay_entries if (k := _key(e))}
    added = [repl_by_id[k] for k in repl_by_id.keys() - orig_by_id.keys()]
    removed = [orig_by_id[k] for k in orig_by_id.keys() - repl_by_id.keys()]
    common = len(orig_by_id.keys() & repl_by_id.keys())

    # Filters keyed by `(rule_id, memory_id, action)` so a tenant
    # tweaking one rule but firing the same memory shows up as a
    # remove+add rather than a silent overwrite.
    def _filter_key(f: dict[str, Any]) -> tuple:
        return (f.get("rule_id"), f.get("memory_id"), f.get("action"))

    orig_filter_set = {_filter_key(f): f for f in original_filters}
    repl_filter_set = {_filter_key(f): f for f in replay_filters}
    filters_added = [repl_filter_set[k] for k in repl_filter_set.keys() - orig_filter_set.keys()]
    filters_removed = [orig_filter_set[k] for k in orig_filter_set.keys() - repl_filter_set.keys()]

    return {
        "context_hash": {
            "original": original_hash,
            "replay": replay_hash,
            "changed": original_hash != replay_hash,
        },
        "selected_entries": {
            "added": added,
            "removed": removed,
            "common": common,
        },
        "filters_applied": {
            "added": filters_added,
            "removed": filters_removed,
        },
    }
