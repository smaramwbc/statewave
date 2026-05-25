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
import hmac
import json
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.db.tables import ReceiptRow, TenantConfigRow

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
    policy_snapshot: dict[str, Any] | None = None,
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
        # v0.9 (#159): self-contained bundle YAML + hash for replay.
        # See `build_policy_snapshot()`. Pre-v0.9 receipts have this
        # column NULL — the replay endpoint refuses those.
        "policy_snapshot": policy_snapshot,
        "output": {
            "context_hash": context_hash,
            "context_size_bytes": context_size_bytes,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "token_estimate": token_estimate,
        },
        "region": region,
        "receipt_signature": None,
    }


def build_policy_snapshot(
    *,
    bundle_hash: str | None,
    bundle_yaml: str | None,
) -> dict[str, Any]:
    """Compose the snapshot envelope embedded into every v0.9 receipt.

    Shape:
        {"bundle_hash": "<sha256>" | null,
         "bundle_yaml": "<verbatim YAML>" | null,
         "captured_at": "<ISO-8601 UTC>"}

    A null inner pair (``bundle_hash`` AND ``bundle_yaml`` both null)
    records "no policy bundle was active at emission" — which is a
    valid, replayable state. The replay path treats it as the
    no-policy fallback (all memories allowed). The column being
    NULL on the row, by contrast, marks "pre-v0.9 receipt, no
    snapshot was ever captured" — the replay endpoint refuses
    those with 422 ``missing_policy_snapshot``.

    ``captured_at`` is recorded so an operator inspecting a receipt
    weeks later can see when the YAML was frozen, which may differ
    from the receipt's own ``created_at`` if the same bundle had been
    active long before the receipt fired.
    """
    return {
        "bundle_hash": bundle_hash,
        "bundle_yaml": bundle_yaml,
        "captured_at": _iso(datetime.now(timezone.utc)),
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
    tenant_config: dict[str, Any] | None = None,
) -> str | None:
    """Persist a receipt. Returns the receipt_id on success, None on
    failure. **Never raises** — receipt write failure must not break
    agent serving (see docs/state-assembly-receipts.md → Failure mode).

    When ``tenant_config`` carries ``receipt_signing_key_id`` AND the
    referenced key is configured in ``settings.receipt_signing_keys``,
    the body is signed via HMAC-SHA256 over its canonical v1 form
    before persistence (v0.9 / issue #157). If signing is configured
    but the key is unavailable, the receipt emits unsigned and a
    `receipt_signing_key_unavailable` warning is logged — agent
    serving is never blocked by audit infra."""
    try:
        # In-place stamp `receipt_signature`, `receipt_signature_key_id`,
        # and `receipt_signature_algorithm` on the body if signing
        # applies. No-op for tenants without signing configured.
        _apply_signature(receipt_body, tenant_config or {})

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
            receipt_signature_key_id=receipt_body.get("receipt_signature_key_id"),
            receipt_signature_algorithm=receipt_body.get("receipt_signature_algorithm"),
            body=receipt_body,
            # Mirror the snapshot onto the dedicated column too — the
            # body field is authoritative (signed) but the column lets
            # the replay engine + admin list-views filter without
            # rummaging through JSONB. Both reads should agree.
            policy_snapshot=receipt_body.get("policy_snapshot"),
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


# ---------------------------------------------------------------------------
# Retention worker — v0.9 (issue #156)
# ---------------------------------------------------------------------------


async def cleanup_expired_receipts(session: AsyncSession) -> int:
    """Tombstone receipts past their tenant's retention window.

    v0.8 shipped the configurable surface (``tenant_configs.config ->>
    'receipt_retention_days'``) but no worker. This function is that
    worker — called once an hour from `_cleanup_loop` in `server.app`.

    Walks every tenant with a positive integer ``receipt_retention_days``
    in its config and issues one ``UPDATE`` per tenant transitioning
    ``status='active'`` receipts older than ``now() - retention_days``
    into ``status='tombstoned'``. Soft-delete only — the rows persist so
    a forensic lookup ("a receipt with id X was emitted and later
    retired") still works.

    Idempotent — re-running against the same DB state is a no-op (no
    active rows past the cutoff remain to transition).

    Tenant isolation is structural: each ``UPDATE`` is scoped to a
    single ``tenant_id``, so misconfiguring tenant A cannot tombstone
    tenant B's receipts.

    Caller is responsible for committing the session.
    """
    # Find tenants with retention configured. JSONB `->>` returns NULL
    # for missing keys; the `IS NOT NULL` lets non-configured tenants
    # short-circuit without paying for a full ReceiptRow scan.
    stmt = select(TenantConfigRow.tenant_id, TenantConfigRow.config).where(
        TenantConfigRow.config.op("->>")("receipt_retention_days").isnot(None)
    )
    result = await session.execute(stmt)
    tenants = result.all()
    if not tenants:
        return 0

    now = datetime.now(timezone.utc)
    total = 0
    for tenant_id, config in tenants:
        days = (config or {}).get("receipt_retention_days")
        # Defensive: the per-tenant config endpoint validates the type
        # at write time, but a SQL-shell write or pre-v0.9 garbage
        # entry could land non-int values. Skip silently — we don't
        # want one bad tenant config to halt the whole purge tick.
        if not isinstance(days, int) or isinstance(days, bool) or days <= 0:
            continue
        cutoff = now - timedelta(days=days)
        update_stmt = (
            update(ReceiptRow)
            .where(ReceiptRow.tenant_id == tenant_id)
            .where(ReceiptRow.status == "active")
            .where(ReceiptRow.created_at < cutoff)
            .values(status="tombstoned", tombstoned_at=now)
        )
        r = await session.execute(update_stmt)
        total += r.rowcount or 0
    if total:
        logger.info("receipts_retention_tombstoned", count=total)
    return total


# ---------------------------------------------------------------------------
# HMAC signing & verification — v0.9 (issue #157)
#
# Receipts are immutable per-retrieval audit artifacts. v0.8 stored a
# SHA-256 of the assembled context bytes inside the body, but the body
# itself wasn't tamper-evident. v0.9 signs the body with HMAC-SHA256
# under a tenant-scoped operator-provided key.
#
# Algorithm string is `hmac-sha256-canonical-v1` — algorithm + canonical
# form version baked into one slot so a future migration (RFC 8785 JCS,
# or asymmetric signing) lands as a new string without a schema break.
#
# The `receipt_signature` field is excluded from canonicalization
# (signing a body that contains its own signature is circular). All
# other fields — including `receipt_signature_key_id` and
# `receipt_signature_algorithm` — are inside the signature's coverage,
# so an attacker can't tell the verifier "this was signed by a
# different key" without invalidating the signature.
#
# Keys never persist in the database. The dict lives in
# `settings.receipt_signing_keys`, sourced from operator env / secret
# manager at process startup, with the field marked `repr=False` so a
# stray `print(settings)` cannot leak them.
# ---------------------------------------------------------------------------


#: The only signature algorithm + canonicalization variant v0.9 emits.
#: Verify accepts the same. Future bumps (`-v2`, JCS, ed25519) extend
#: this constant and the verify dispatch without a schema migration.
SUPPORTED_SIGNATURE_ALGORITHM = "hmac-sha256-canonical-v1"

#: Body fields excluded from canonicalization (the signature signs
#: everything else). Wrapped in a frozenset for fast lookup.
_UNSIGNED_BODY_FIELDS = frozenset({"receipt_signature"})


def canonicalize_receipt_body_v1(body: dict[str, Any]) -> bytes:
    """Stable byte representation of a receipt body for HMAC signing.

    v1 = ``json.dumps(body_minus_signature, sort_keys=True,
    separators=(",", ":"), ensure_ascii=False)`` UTF-8 encoded. Sorted
    keys keep the bytes deterministic across Python dict-ordering
    quirks; the compact separators eliminate whitespace ambiguity;
    ``ensure_ascii=False`` keeps multi-byte content as its UTF-8
    representation rather than \\uXXXX escapes (the body already
    contains user content that may include non-ASCII).

    The version string is baked into the algorithm tag stored on each
    receipt, so a future move to RFC 8785 JCS or another canonical form
    lands as ``canonicalize_receipt_body_v2`` + a new algorithm string,
    with v1 still verifying historical receipts.
    """
    sanitized = {k: v for k, v in body.items() if k not in _UNSIGNED_BODY_FIELDS}
    return json.dumps(
        sanitized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sign_receipt_body(body: dict[str, Any], key: bytes) -> str:
    """Compute HMAC-SHA256 over the canonical v1 form of ``body``.

    Returns the signature as a lowercase hex string (64 chars). The
    raw key bytes are passed directly to ``hmac.new`` — never logged,
    never serialised, never returned by any caller-visible surface.
    """
    canonical = canonicalize_receipt_body_v1(body)
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()


def _apply_signature(
    receipt_body: dict[str, Any],
    tenant_config: dict[str, Any],
) -> None:
    """Sign ``receipt_body`` in place if the tenant has a key id configured
    AND the key is resolvable from ``settings.receipt_signing_keys``.

    Three terminal states:

    - **Signed** — body gains ``receipt_signature``,
      ``receipt_signature_key_id``, ``receipt_signature_algorithm``.
    - **Configured but key unavailable** — body left unsigned; a
      ``receipt_signing_key_unavailable`` warning is emitted with the
      key_id only (never the key bytes). The verifier later reports
      ``valid: null, reason: "key_unavailable"`` for this receipt.
    - **Unconfigured** — body untouched. The verifier reports
      ``valid: null, reason: "no_signature"``.

    Signing failures past key resolution (an unlikely Python-level
    error inside ``sign_receipt_body``) are swallowed with a warning
    so the receipt still emits unsigned — agent serving must not be
    blocked by audit infra.
    """
    key_id = tenant_config.get("receipt_signing_key_id") if tenant_config else None
    if not key_id:
        return  # tenant hasn't opted in

    # Lazy import keeps the receipts module testable without booting
    # the full Settings object.
    from server.core.config import settings

    keys_map = settings.receipt_signing_keys or {}
    key = keys_map.get(key_id)
    if key is None:
        logger.warning(
            "receipt_signing_key_unavailable",
            tenant_id=receipt_body.get("tenant_id"),
            key_id=key_id,
        )
        return

    # Stamp the metadata BEFORE computing the signature so the key_id
    # and algorithm are covered by the HMAC — an attacker who swaps
    # them on a stored receipt then has to also rewrite the signature.
    # On signing failure we roll back the stamps so the body persists
    # in a consistent unsigned state.
    receipt_body["receipt_signature_key_id"] = key_id
    receipt_body["receipt_signature_algorithm"] = SUPPORTED_SIGNATURE_ALGORITHM
    try:
        signature = sign_receipt_body(receipt_body, key)
    except Exception:
        # Defensive: a hash computation should never fail at runtime,
        # but if it does we emit unsigned + warn rather than fail the
        # whole assembly.
        receipt_body.pop("receipt_signature_key_id", None)
        receipt_body.pop("receipt_signature_algorithm", None)
        logger.warning(
            "receipt_signing_failed",
            tenant_id=receipt_body.get("tenant_id"),
            key_id=key_id,
            exc_info=True,
        )
        return

    receipt_body["receipt_signature"] = signature


def verify_receipt(
    row: ReceiptRow,
    *,
    keys_map: dict[str, bytes] | None = None,
) -> dict[str, Any]:
    """Verify the HMAC signature on a stored receipt row.

    Returns a dict with the shape documented in ``api/v1-contract.md``:

        { "valid": True | False | None,
          "key_id": "..." | None,
          "algorithm": "..." | None,
          "reason": "ok" | "signature_mismatch" | "key_unavailable"
                    | "no_signature" | "unsupported_algorithm" }

    ``valid: null`` covers the "cannot determine" cases (key was
    removed from operator config, no signature present, or the
    algorithm string names a variant this binary doesn't implement).
    ``valid: false`` is reserved for "we checked the math and the
    signature doesn't cover the body."

    Comparison uses ``hmac.compare_digest`` for constant-time
    behaviour against timing attacks. The signing key bytes never
    appear in the response.

    ``keys_map`` defaults to ``settings.receipt_signing_keys``;
    overridable for tests.
    """
    # No-signature path — pre-v0.9 receipts or tenants that didn't
    # opt in to signing. Distinct from `key_unavailable`: the receipt
    # was never signed in the first place.
    if not row.receipt_signature or not row.receipt_signature_key_id:
        return {
            "valid": None,
            "key_id": None,
            "algorithm": None,
            "reason": "no_signature",
        }

    algorithm = row.receipt_signature_algorithm or SUPPORTED_SIGNATURE_ALGORITHM
    if algorithm != SUPPORTED_SIGNATURE_ALGORITHM:
        # Future-proofing: a receipt signed under `canonical-v2` or
        # `ed25519-canonical-v1` on a newer server would land here on
        # a v0.9 binary. Report explicitly rather than misleading the
        # caller with `signature_mismatch`.
        return {
            "valid": None,
            "key_id": row.receipt_signature_key_id,
            "algorithm": algorithm,
            "reason": "unsupported_algorithm",
        }

    if keys_map is None:
        from server.core.config import settings

        keys_map = settings.receipt_signing_keys or {}

    key = keys_map.get(row.receipt_signature_key_id)
    if key is None:
        # Key rotated out of operator config; historical receipt is
        # no longer verifiable on this binary. Never a 500.
        return {
            "valid": None,
            "key_id": row.receipt_signature_key_id,
            "algorithm": algorithm,
            "reason": "key_unavailable",
        }

    expected = sign_receipt_body(row.body, key)
    if hmac.compare_digest(expected, row.receipt_signature):
        return {
            "valid": True,
            "key_id": row.receipt_signature_key_id,
            "algorithm": algorithm,
            "reason": "ok",
        }
    return {
        "valid": False,
        "key_id": row.receipt_signature_key_id,
        "algorithm": algorithm,
        "reason": "signature_mismatch",
    }
