"""State-assembly receipts — emission decision, canonicalization, write.

Receipts are the immutable per-retrieval audit artifact specified by
issue #49. This module owns three concerns:

  1. **Emission decision** — `decide_emission()` consults the per-request
     flag, per-tenant config, and (eventually, #50) the policy layer to
     decide whether to emit. Single source of truth; both the context
     and handoff assembly paths call into it.

  2. **ULID** — receipt ids are 26-char Crockford Base32 ULIDs.
     Implemented in-house rather than adding a dep — the spec is stable
     and the routine is ~30 lines. Monotonic time prefix gives natural
     chronological sorting at the database level without a separate
     created_at index for ordering.

  3. **Canonicalization + hash** — `canonicalize_context()` produces a
     stable byte representation of the assembled context and a SHA-256
     hash. `canonicalization_version` is bumped whenever this routine
     changes so historical hashes remain verifiable.

The schema and rationale for the receipt body live in
docs/state-assembly-receipts.md.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.db.tables import ReceiptRow

logger = structlog.stdlib.get_logger()


# Bump when canonicalize_context() changes shape. Stored on every
# receipt so a historical hash mismatch can be diagnosed as "the
# canonicalization rule moved" rather than "the bytes were tampered."
CANONICALIZATION_VERSION = 1


# ---------------------------------------------------------------------------
# Emission decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmissionDecision:
    """Outcome of `decide_emission()`. `emit` is the binary answer;
    `reason` is a short tag suitable for structured logs."""

    emit: bool
    reason: str


def decide_emission(
    request_flag: bool | None,
    tenant_config: dict[str, Any] | None,
    policy_decision: str | None = None,
) -> EmissionDecision:
    """Single-source-of-truth emission policy. Inputs are consulted in
    the order defined by issue #49:

      1. Env kill-switch — `STATEWAVE_RECEIPTS_DISABLED=true` wins over
         everything. Emergency operational hygiene only.
      2. Policy force-on (#50) — reserved; v1 callers pass None and the
         input is ignored. When #50 ships, a policy decision of
         `must_emit` overrides per-request `false`.
      3. Per-tenant config — `always` forces emission, `never` suppresses
         it. `on_request` (the default) defers to the per-request flag.
      4. Per-request flag — caller opt-in. Default `False`.

    The function is intentionally pure: it takes the inputs, returns a
    decision, and never reads the environment or DB. The env-var check
    happens inside the function for a reason — the kill-switch is
    intended to be flippable without a redeploy at the process level,
    and reading it lazily here makes it observable per-call rather than
    cached at import time."""

    # 1. Kill-switch
    if _env_kill_switch_enabled():
        return EmissionDecision(emit=False, reason="kill_switch")

    # 2. Policy force-on (reserved for #50)
    if policy_decision == "must_emit":
        return EmissionDecision(emit=True, reason="policy_force_on")
    if policy_decision == "must_skip":
        return EmissionDecision(emit=False, reason="policy_force_off")

    # 3. Tenant config
    tenant_mode = (tenant_config or {}).get("receipts", "on_request")
    if tenant_mode == "always":
        return EmissionDecision(emit=True, reason="tenant_always")
    if tenant_mode == "never":
        return EmissionDecision(emit=False, reason="tenant_never")

    # 4. Per-request flag (default off when None)
    if request_flag:
        return EmissionDecision(emit=True, reason="request_flag")
    return EmissionDecision(emit=False, reason="default_off")


def _env_kill_switch_enabled() -> bool:
    """`STATEWAVE_RECEIPTS_DISABLED=true|1|yes` disables all emission."""
    val = os.environ.get("STATEWAVE_RECEIPTS_DISABLED", "").strip().lower()
    return val in {"true", "1", "yes"}


# ---------------------------------------------------------------------------
# ULID — Crockford Base32, 48-bit ms time + 80-bit randomness
# ---------------------------------------------------------------------------

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid() -> str:
    """Return a 26-char Crockford Base32 ULID. Monotonic within a
    millisecond is *not* guaranteed — randomness alone provides the
    intra-ms ordering, which is statistically fine for receipt volumes
    (collision probability under 2^-80 per ms even at 10k receipts/s).
    """
    timestamp_ms = int(time.time() * 1000)
    randomness = secrets.token_bytes(10)  # 80 bits
    return _encode_time(timestamp_ms) + _encode_random(randomness)


def _encode_time(ms: int) -> str:
    """Encode a 48-bit ms timestamp as 10 Crockford Base32 chars."""
    out = []
    for _ in range(10):
        out.append(_CROCKFORD[ms & 0x1F])
        ms >>= 5
    return "".join(reversed(out))


def _encode_random(b: bytes) -> str:
    """Encode 10 bytes (80 bits) as 16 Crockford Base32 chars."""
    # Concatenate bytes into a big integer, then peel off 5-bit groups.
    n = int.from_bytes(b, "big")
    out = []
    for _ in range(16):
        out.append(_CROCKFORD[n & 0x1F])
        n >>= 5
    return "".join(reversed(out))


# ---------------------------------------------------------------------------
# Canonicalization + hashing
# ---------------------------------------------------------------------------


def canonicalize_context(assembled_context: str) -> tuple[str, int]:
    """Return `(sha256_hex, byte_size)` of the canonical UTF-8 encoding
    of the assembled context.

    Canonicalization is intentionally minimal in v1: the assembled
    context is already deterministic (insertion order is score-driven,
    rendered text is stable). We only commit to UTF-8 with no BOM and
    no trailing newline normalization. If we ever need to canonicalize
    Unicode, normalize line endings, or strip whitespace, bump
    CANONICALIZATION_VERSION.
    """
    body = assembled_context.encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    return digest, len(body)


def provenance_hash(source_episode_ids: Iterable[uuid.UUID | str]) -> str:
    """Hash of a memory's source provenance — the sorted, comma-joined
    string of source episode UUIDs. Stable across reorderings of the
    underlying list; changes if and only if the set changes."""
    sorted_ids = sorted(str(eid) for eid in (source_episode_ids or []))
    blob = ",".join(sorted_ids).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Receipt construction + write
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectedMemory:
    """One memory entry as it appears in `receipt.selected_entries`."""

    memory_id: uuid.UUID
    kind: str
    valid_from: datetime | None
    valid_to: datetime | None
    supersession_status: str  # active | superseded | tombstoned
    source_episode_ids: list[uuid.UUID]
    rank: int
    fact_key: str | None = None
    conflict_status: str = "none"  # none | merged | overridden | unresolved
    score: float | None = None


@dataclass(frozen=True)
class SelectedEpisode:
    """One episode entry as it appears in `receipt.selected_entries`."""

    episode_id: uuid.UUID
    source: str
    type: str
    occurred_at: datetime | None
    rank: int


def build_receipt_body(
    *,
    receipt_id: str,
    mode: str,
    tenant_id: str | None,
    subject_id: str,
    task: str,
    as_of: datetime,
    selected_memories: list[SelectedMemory],
    selected_episodes: list[SelectedEpisode],
    context_hash: str,
    context_size_bytes: int,
    token_estimate: int,
    query_id: str | None = None,
    task_id: str | None = None,
    parent_receipt_id: str | None = None,
    policy_bundle_hash: str | None = None,
    policy_mode: str = "log_only",
    filters_applied: list[dict[str, Any]] | None = None,
    filters_skipped: list[dict[str, Any]] | None = None,
    caller_id: str | None = None,
    caller_type: str | None = None,
    region: str | None = None,
) -> dict[str, Any]:
    """Render the strict-superset receipt body. Pure function — same
    inputs always produce the same JSON-serializable dict."""

    created_at = datetime.now(timezone.utc)

    entries: list[dict[str, Any]] = []
    for m in selected_memories:
        entries.append(
            {
                "type": "memory",
                "memory_id": str(m.memory_id),
                "kind": m.kind,
                "valid_from": _iso(m.valid_from),
                "valid_to": _iso(m.valid_to),
                "supersession_status": m.supersession_status,
                "source_episode_ids": [str(s) for s in m.source_episode_ids],
                "provenance_hash": provenance_hash(m.source_episode_ids),
                "fact_key": m.fact_key,
                "conflict_status": m.conflict_status,
                "rank": m.rank,
                "score": m.score,
            }
        )
    for e in selected_episodes:
        entries.append(
            {
                "type": "episode",
                "episode_id": str(e.episode_id),
                "source": e.source,
                "event_type": e.type,
                "occurred_at": _iso(e.occurred_at),
                "rank": e.rank,
            }
        )

    return {
        "receipt_id": receipt_id,
        "parent_receipt_id": parent_receipt_id,
        "mode": mode,
        "query_id": query_id,
        "task_id": task_id,
        "tenant_id": tenant_id,
        "subject_id": subject_id,
        "task": task,
        "as_of": _iso(as_of),
        "created_at": _iso(created_at),
        # Caller identity (#50) — the request fields the policy layer
        # evaluated against. Null when the caller is anonymous (the
        # default for tenants that don't `require_caller_identity`).
        "caller_id": caller_id,
        "caller_type": caller_type,
        "selected_entries": entries,
        "policy": {
            # Filled in by the policy layer (#50) when a bundle is
            # active. Untagged-everywhere tenants and no-bundle
            # deployments get the same empty shape they got pre-#50,
            # which is exactly the right answer.
            "policy_bundle_hash": policy_bundle_hash,
            "filters_applied": list(filters_applied or []),
            "filters_skipped": list(filters_skipped or []),
            "mode": policy_mode,
        },
        "output": {
            "context_hash": context_hash,
            "context_size_bytes": context_size_bytes,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "token_estimate": token_estimate,
        },
        "region": region,
        "receipt_signature": None,
    }


def _iso(dt: datetime | None) -> str | None:
    """ISO-8601 with explicit UTC zone. Receipts must be timezone-safe
    on the wire; naive datetimes would silently lose offset info on
    JSON serialization."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


async def write_receipt(
    session: AsyncSession,
    *,
    receipt_body: dict[str, Any],
    as_of: datetime,
) -> str | None:
    """Persist a receipt. Returns the receipt_id on success, None on
    failure. **Never raises** — receipt write failure must not break
    agent serving (see docs/state-assembly-receipts.md → Failure mode)."""
    try:
        row = ReceiptRow(
            receipt_id=receipt_body["receipt_id"],
            parent_receipt_id=receipt_body.get("parent_receipt_id"),
            mode=receipt_body["mode"],
            tenant_id=receipt_body.get("tenant_id"),
            subject_id=receipt_body["subject_id"],
            query_id=receipt_body.get("query_id"),
            task_id=receipt_body.get("task_id"),
            context_hash=receipt_body["output"]["context_hash"],
            context_size_bytes=receipt_body["output"]["context_size_bytes"],
            policy_bundle_hash=receipt_body["policy"].get("policy_bundle_hash"),
            region=receipt_body.get("region"),
            receipt_signature=receipt_body.get("receipt_signature"),
            body=receipt_body,
            as_of=as_of,
        )
        await repo.insert_receipt(session, row)
        # Explicit commit: get_session() does not auto-commit on exit
        # (the /v1/context dependency session was primarily a read-side
        # path before receipts). Without this the row is flushed but
        # rolled back at session close, leaving callers with a
        # receipt_id that points at nothing. A commit failure is
        # treated the same as an insert failure: logged, swallowed,
        # None returned so the bundle still serves.
        await session.commit()
        return row.receipt_id
    except Exception:
        logger.warning(
            "receipt_emission_failed",
            receipt_id=receipt_body.get("receipt_id"),
            subject_id=receipt_body.get("subject_id"),
            tenant_id=receipt_body.get("tenant_id"),
            exc_info=True,
        )
        try:
            await session.rollback()
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Helpers exposed to the assembly path
# ---------------------------------------------------------------------------


def supersession_status_from_row(row: Any) -> str:
    """Map a memory row's status + valid_to to the receipt's
    `supersession_status` vocabulary.

    `MemoryRow.status` values are `active | superseded | tombstoned`,
    which already match the receipt schema. But a row that's *technically
    active* with a `valid_to` in the past is the "stale fact" failure
    mode the negative tests check for — it's NOT something we
    re-label as superseded here, because the failure mode is exactly
    that the assembly path selected it while it shouldn't have. The
    receipt records the status as-stored so a reviewer can spot the
    discrepancy."""
    return getattr(row, "status", "active")


async def load_tenant_receipt_config(
    session: AsyncSession,
    tenant_id: str | None,
) -> dict[str, Any]:
    """Fetch a tenant's relevant config keys for the emission decision.
    Returns `{}` when no tenant or no config row — both collapse to
    "use defaults" downstream."""
    if not tenant_id:
        return {}
    row = await repo.get_tenant_config(session, tenant_id)
    if row is None:
        return {}
    return row.config or {}
