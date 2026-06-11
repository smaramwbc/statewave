"""Memory conflict resolution service.

Two paths, hybrid and strictly additive:

* **Claim path** (new, issue #236) — for memories carrying a valid,
  registry-recognized *single-valued* claim (see ``server.services.claims``).
  Within one canonical key, a newer claim whose normalized value DIFFERS and
  whose validity interval OVERLAPS the older one supersedes it — wording-
  independent contradiction resolution. Non-overlapping intervals coexist
  (history preserved). This path is authoritative for single-valued, same-key,
  different-value pairs and explicitly removes them from the lexical pass, so
  lexical overlap can never undo its cardinality/temporal decisions.

* **Legacy path** (unchanged) — token-overlap (Jaccard) supersession for
  EVERYTHING else: unkeyed, malformed, unknown-key, unsupported-version,
  multi-valued, and mixed keyed/unkeyed pairings, plus single-valued *same
  value* duplicates (existing duplicate handling). Byte-identical to before for
  any memory without a usable single-valued claim.

No opt-in flag, no schema change, no migration. The resolver runs only at
compile time (``server.api.memories``); reads and server startup never invoke
it, so an upgrade alone never changes stored supersession state.

Strategy:
- Group active memories by (subject_id, kind) for the legacy path and by
  (subject_id, claim_key) for the claim path.
- When two memories conflict, the older one is marked as "superseded" with
  valid_to set to the newer memory's valid_from.
"""

from __future__ import annotations

import functools
import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import repositories as repo
from server.db.tables import MemoryRow
from server.services.claims import (
    SCOPE_SINGLE,
    ResolvedClaim,
    intervals_overlap,
    resolve_claim,
)
from server.services.tokenization import EDGE_PUNCT, tokenize

logger = structlog.stdlib.get_logger()

# Similarity threshold for word-overlap conflict detection (0–1)
_WORD_OVERLAP_THRESHOLD = 0.6

# Tokenization is centralized in server.services.tokenization so this detector
# and the relevance scorer can never drift apart. Drift silently welds a
# trailing "." onto a word ("Stripe." != "Stripe"), dragging Jaccard similarity
# below threshold and skipping a valid supersession (#198/#199).
_TOKEN_EDGE_PUNCT = EDGE_PUNCT
_tokenize = tokenize

_MIN_DT = datetime.min.replace(tzinfo=timezone.utc)


def _aware(dt: datetime | None) -> datetime:
    """Coerce to a comparable, tz-aware datetime (None -> -infinity, naive -> UTC)."""
    if dt is None:
        return _MIN_DT
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


async def resolve_conflicts(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
) -> list[uuid.UUID]:
    """Detect and resolve conflicting memories. Returns IDs of superseded memories.

    Call contract unchanged: exactly one ``list_active_memories_by_subject``
    read and at most one ``mark_memories_superseded`` write.
    """
    memories = await repo.list_active_memories_by_subject(session, subject_id, tenant_id=tenant_id)
    if len(memories) < 2:
        return []

    # Validate claims once. ONLY registry-recognized single-valued claims are
    # authoritative for the claim path; anything else (no claim, malformed,
    # unknown key, unsupported version, multi-valued) resolves to None here and
    # is handled exactly as today by the legacy lexical path.
    claims: dict[uuid.UUID, ResolvedClaim] = {}
    for m in memories:
        rc = resolve_claim(m.metadata_)
        if rc is not None and rc.scope == SCOPE_SINGLE:
            claims[m.id] = rc

    superseded_ids: list[uuid.UUID] = []
    # Claim path first; it marks losers status="superseded" in place, so the
    # legacy pass skips them via its existing status guard. The legacy pass
    # additionally skips single-valued same-key *different-value* pairs so it
    # can never undo a temporal coexistence the claim path established.
    superseded_ids.extend(_resolve_single_valued_claims(memories, claims))
    superseded_ids.extend(_legacy_resolve(memories, claims))

    if superseded_ids:
        await repo.mark_memories_superseded(session, superseded_ids)

    return superseded_ids


# --------------------------------------------------------------------------- #
# Claim path — cardinality- and temporally-gated supersession
# --------------------------------------------------------------------------- #


def _resolve_single_valued_claims(
    memories: list[MemoryRow],
    claims: dict[uuid.UUID, ResolvedClaim],
) -> list[uuid.UUID]:
    keyed = [m for m in memories if m.id in claims]
    if len(keyed) < 2:
        return []

    # Bucket by the GENERIC contradiction identity. v1 buckets by key alone; v2
    # by (key, entity, canonical qualifiers). The resolver never inspects
    # entity/qualifiers itself — it only requires that two claims share a bucket.
    by_bucket: dict[tuple, list[MemoryRow]] = {}
    for m in keyed:
        by_bucket.setdefault(claims[m.id].bucket, []).append(m)

    superseded_ids: list[uuid.UUID] = []
    for _bucket, group in by_bucket.items():
        if len(group) < 2:
            continue
        group.sort(key=functools.cmp_to_key(lambda a, b: _claim_cmp(a, b, claims)))
        for i in range(len(group)):
            if group[i].status != "active":
                continue
            ci = claims[group[i].id]
            for j in range(i + 1, len(group)):
                if group[j].status != "active":
                    continue
                cj = claims[group[j].id]
                # Authoritative ONLY for a genuine contradiction: same bucket (by
                # grouping) + DIFFERENT canonical value + OVERLAPPING validity.
                # Same value is a duplicate (left to legacy dedup); non-
                # overlapping windows coexist as history.
                if ci.value == cj.value:
                    continue
                if not _claim_overlap(group[i], ci, group[j], cj):
                    continue
                superseded_ids.append(group[i].id)
                group[i].status = "superseded"
                group[i].valid_to = group[j].valid_from or datetime.now(timezone.utc)
                logger.info(
                    "memory_superseded",
                    old_id=str(group[i].id),
                    new_id=str(group[j].id),
                    claim_key=ci.canonical_key,
                    strategy="claim_contradiction",
                )
                break
    return superseded_ids


def _claim_cmp(a: MemoryRow, b: MemoryRow, claims: dict[uuid.UUID, ResolvedClaim]) -> int:
    """Deterministic total order for the claim path: semantic valid_from (when
    BOTH claims supply it) → persisted created_at → stable memory id. Ascending,
    so the last (newest) element is the winner and earlier ones are superseded.
    Input and DB return order never affect the result."""
    ca, cb = claims[a.id], claims[b.id]
    if ca.valid_from is not None and cb.valid_from is not None:
        av, bv = _aware(ca.valid_from), _aware(cb.valid_from)
        if av != bv:
            return -1 if av < bv else 1
    ac, bc = _aware(a.created_at), _aware(b.created_at)
    if ac != bc:
        return -1 if ac < bc else 1
    sa, sb = str(a.id), str(b.id)
    return -1 if sa < sb else (1 if sa > sb else 0)


def _claim_overlap(mi: MemoryRow, ci: ResolvedClaim, mj: MemoryRow, cj: ResolvedClaim) -> bool:
    """Temporal overlap of two claims, using each claim's own validity window
    when supplied and otherwise the memory row's valid_from/valid_to."""
    i_from = ci.valid_from if ci.valid_from is not None else mi.valid_from
    i_to = ci.valid_to if ci.valid_to is not None else mi.valid_to
    j_from = cj.valid_from if cj.valid_from is not None else mj.valid_from
    j_to = cj.valid_to if cj.valid_to is not None else mj.valid_to
    return intervals_overlap(i_from, i_to, j_from, j_to)


# --------------------------------------------------------------------------- #
# Legacy path — unchanged lexical (Jaccard) supersession
# --------------------------------------------------------------------------- #


def _legacy_resolve(
    memories: list[MemoryRow],
    claims: dict[uuid.UUID, ResolvedClaim],
) -> list[uuid.UUID]:
    if len(memories) < 2:
        return []

    by_kind: dict[str, list[MemoryRow]] = {}
    for m in memories:
        by_kind.setdefault(m.kind, []).append(m)

    superseded_ids: list[uuid.UUID] = []
    for kind, group in by_kind.items():
        if len(group) < 2:
            continue
        group.sort(key=_legacy_sort_key)
        for i in range(len(group)):
            if group[i].status != "active":
                continue
            for j in range(i + 1, len(group)):
                if group[j].status != "active":
                    continue
                # The claim path owns single-valued same-key DIFFERENT-value
                # pairs; never let lexical overlap undo its temporal/cardinality
                # call. (Same-value duplicates are intentionally NOT skipped, so
                # existing duplicate handling still dedups them.)
                if _claim_owned_pair(group[i], group[j], claims):
                    continue
                if _are_conflicting(group[i], group[j]):
                    superseded_ids.append(group[i].id)
                    group[i].status = "superseded"
                    group[i].valid_to = group[j].valid_from or datetime.now(timezone.utc)
                    logger.info(
                        "memory_superseded",
                        old_id=str(group[i].id),
                        new_id=str(group[j].id),
                        kind=kind,
                        strategy="lexical",
                    )
                    break
    return superseded_ids


def _claim_owned_pair(a: MemoryRow, b: MemoryRow, claims: dict[uuid.UUID, ResolvedClaim]) -> bool:
    ca, cb = claims.get(a.id), claims.get(b.id)
    if ca is None or cb is None:
        return False  # a mixed keyed/unkeyed pair keeps existing legacy behavior
    # The claim path owns EVERY pair of single-valued claims except an exact
    # duplicate (same bucket + same value), which it defers to legacy dedup. In
    # particular two DIFFERENT claim identities must coexist — lexical overlap
    # between their texts must never supersede across distinct buckets (e.g.
    # Stripe-card vs Stripe-ACH rates).
    return not (ca.bucket == cb.bucket and ca.value == cb.value)


def _legacy_sort_key(m: MemoryRow) -> tuple[datetime, datetime, str]:
    """Preserve current created_at-primary ordering; add valid_from then id as
    deterministic tie-breaks so previously input/DB-order-dependent ties (equal
    or missing created_at) now resolve identically every run."""
    return (_aware(m.created_at), _aware(m.valid_from), str(m.id))


def _are_conflicting(older: MemoryRow, newer: MemoryRow) -> bool:
    """Determine if two memories of the same kind conflict.

    For profile_facts: high word overlap means the newer one replaces the older.
    For other kinds: require very high overlap.
    """
    threshold = _WORD_OVERLAP_THRESHOLD
    if older.kind != "profile_fact":
        threshold = 0.8  # stricter for non-facts

    older_tokens = _tokenize(older.content)
    newer_tokens = _tokenize(newer.content)

    if not older_tokens or not newer_tokens:
        return False

    # Jaccard similarity
    intersection = len(older_tokens & newer_tokens)
    union = len(older_tokens | newer_tokens)
    similarity = intersection / union if union else 0.0

    return similarity >= threshold
