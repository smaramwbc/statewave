"""Optional, additive **claim envelope** for memories.

A claim names *what* a memory asserts — a canonical key (entity + attribute)
plus a value — so the resolver can detect contradictions independent of
wording ("employer: Acme" vs "employer: Globex") instead of relying on lexical
overlap, which conflates duplicate / distinct / contradiction.

Design invariants (see issue smaramwbc/statewave#236):

* **Additive only.** The envelope lives under ``metadata_["claim"]`` (JSONB).
  No new column, no required field, no migration. Memories without it behave
  exactly as before.
* **Registry is authoritative.** Cardinality (single- vs multi-valued) comes
  from the controlled :data:`CLAIM_REGISTRY`, never from caller/LLM input. An
  unregistered key is non-authoritative and never drives supersession.
* **Fail safe to legacy.** Anything missing, malformed, unknown-key, or of an
  unsupported ``schema_version`` resolves to *no claim* (``None``) — the
  resolver then uses the legacy lexical path. The preferred failure mode is
  under-resolution, never wrongful deletion or ingestion failure.

Nothing here ever raises on bad input: :func:`resolve_claim` swallows
validation errors and returns ``None`` so a malformed claim can never break
ingestion, compilation, retrieval, or serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

logger = structlog.stdlib.get_logger()

# Sub-key under MemoryRow.metadata_ that carries the envelope. Chosen to not
# collide with existing metadata producers (memory_packs uses flat top-level
# keys like ``starter_pack_id``; the llm compiler uses ``compiler``/``model``).
CLAIM_METADATA_KEY = "claim"

# Bumped only on a breaking envelope-shape change. Unknown/newer versions are
# ignored (degrade to legacy), so an older reader never trips on future data.
CLAIM_SCHEMA_VERSION = 1

# Cardinality of a canonical claim key.
SCOPE_SINGLE = "single"  # one current value wins (name, current employer, …)
SCOPE_MULTI = "multi"  # values coexist (tools used, skills, preferences, …)


@dataclass(frozen=True)
class ClaimKeySpec:
    """Authoritative spec for a registered canonical claim key."""

    key: str
    scope: str  # SCOPE_SINGLE | SCOPE_MULTI


# --------------------------------------------------------------------------- #
# Canonical registry — deliberately small and high-confidence.
#
# ONLY keys listed here are authoritative. Aliases normalize *to* a canonical
# key and are accepted ONLY when explicitly registered (deterministic mapping).
# Everything else is non-authoritative: parsed but never allowed to supersede.
# Unknown keys default to coexistence-safe behavior by being absent here.
# --------------------------------------------------------------------------- #
_SINGLE_VALUED_KEYS = (
    "identity.name",
    "employment.current_employer",
    "location.current_home",
    "billing.primary_payment_processor",
)

_MULTI_VALUED_KEYS = (
    "tools.used",
    "skills",
    "preferences",
    "payment.processors_used",
)

CLAIM_REGISTRY: dict[str, ClaimKeySpec] = {
    **{k: ClaimKeySpec(k, SCOPE_SINGLE) for k in _SINGLE_VALUED_KEYS},
    **{k: ClaimKeySpec(k, SCOPE_MULTI) for k in _MULTI_VALUED_KEYS},
}

# Approved deterministic aliases → canonical key. Unknown aliases do NOT create
# new authoritative buckets; they resolve to ``None`` (legacy fallback).
CLAIM_ALIASES: dict[str, str] = {
    "name": "identity.name",
    "full_name": "identity.name",
    "employer": "employment.current_employer",
    "works_at": "employment.current_employer",
    "company": "employment.current_employer",
    "current_employer": "employment.current_employer",
    "home": "location.current_home",
    "home_location": "location.current_home",
    "current_home": "location.current_home",
    "primary_payment_processor": "billing.primary_payment_processor",
    "tools": "tools.used",
    "payment_processors": "payment.processors_used",
}


def canonicalize_key(raw_key: str | None) -> str | None:
    """Map an envelope key (or approved alias) to a registered canonical key.

    Returns ``None`` for anything not explicitly registered — so unknown or
    drifted keys (employer vs works_at vs company are handled; arbitrary
    strings are not) never become authoritative contradiction buckets.
    """
    if not raw_key or not isinstance(raw_key, str):
        return None
    key = raw_key.strip()
    if key in CLAIM_REGISTRY:
        return key
    return CLAIM_ALIASES.get(key)


def normalize_value(value: Any) -> str | None:
    """Normalize a claim value for equality comparison.

    Lowercased, stripped, internal whitespace collapsed. Non-string or empty
    values yield ``None`` (claim not usable). Intentionally conservative: it
    does NOT attempt entity resolution ("Acme" vs "Acme Corp" stay distinct).
    """
    if not isinstance(value, str):
        return None
    norm = " ".join(value.strip().split()).lower()
    return norm or None


class ClaimEnvelope(BaseModel):
    """Strict-but-tolerant validation of the stored envelope shape.

    ``extra='ignore'`` keeps the envelope forward-compatible (a newer producer
    may add fields an older reader ignores). Validation is non-destructive:
    callers never mutate the stored dict based on this model.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: int
    key: str
    value: str
    scope: str | None = None  # advisory only — registry is authoritative
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float | None = None
    source: str | None = None


@dataclass(frozen=True)
class ResolvedClaim:
    """A validated, registry-authoritative claim ready for the resolver."""

    canonical_key: str
    value: str  # normalized
    scope: str  # authoritative, from registry
    valid_from: datetime | None
    valid_to: datetime | None


def resolve_claim(metadata: dict | None) -> ResolvedClaim | None:
    """Parse + validate ``metadata['claim']`` into a :class:`ResolvedClaim`.

    Returns ``None`` (→ legacy behavior) for every unsafe case: no envelope,
    malformed envelope, unsupported ``schema_version``, unknown/unregistered
    key, or unusable value. Never raises.
    """
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get(CLAIM_METADATA_KEY)
    if not isinstance(raw, dict):
        return None

    try:
        env = ClaimEnvelope.model_validate(raw)
    except ValidationError:
        # Observability without leaking the value itself.
        logger.info("claim_envelope_invalid", reason="validation_error",
                    raw_key=raw.get("key") if isinstance(raw.get("key"), str) else None)
        return None

    if env.schema_version != CLAIM_SCHEMA_VERSION:
        logger.info("claim_envelope_unsupported_version", version=env.schema_version)
        return None

    canonical = canonicalize_key(env.key)
    if canonical is None:
        # Unknown / unregistered key → non-authoritative, never supersedes.
        return None

    value = normalize_value(env.value)
    if value is None:
        return None

    spec = CLAIM_REGISTRY[canonical]
    return ResolvedClaim(
        canonical_key=canonical,
        value=value,
        scope=spec.scope,
        valid_from=env.valid_from,
        valid_to=env.valid_to,
    )


def build_claim_envelope(
    key: str,
    value: str,
    *,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    confidence: float | None = None,
    source: str = "heuristic",
) -> dict | None:
    """Build a storable envelope for a compiler — or ``None`` to omit it.

    Returns ``None`` (attach nothing, keep legacy behavior) unless ``key``
    resolves to a registered canonical key and ``value`` normalizes cleanly.
    Compilers must treat a ``None`` return as "uncertain — emit the memory
    without a claim".
    """
    canonical = canonicalize_key(key)
    if canonical is None or normalize_value(value) is None:
        return None
    env: dict[str, Any] = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "key": canonical,
        "value": value,
        "scope": CLAIM_REGISTRY[canonical].scope,
        "source": source,
    }
    if valid_from is not None:
        env["valid_from"] = valid_from.isoformat()
    if valid_to is not None:
        env["valid_to"] = valid_to.isoformat()
    if confidence is not None:
        env["confidence"] = confidence
    return {CLAIM_METADATA_KEY: env}


def intervals_overlap(
    a_from: datetime | None,
    a_to: datetime | None,
    b_from: datetime | None,
    b_to: datetime | None,
) -> bool:
    """Half-open temporal overlap test, ``None`` = open-ended (±infinity).

    Two validity intervals ``[from, to)`` overlap iff ``a_from < b_to`` and
    ``b_from < a_to``. A missing ``from`` is treated as the beginning of time
    and a missing ``to`` as open-ended, so two "current" facts (no ``valid_to``)
    always overlap — the case that must resolve newest-wins. Touching-only
    intervals (``a_to == b_from``) do NOT overlap, so a past fact whose
    ``valid_to`` equals the successor's ``valid_from`` coexists as history.
    """
    a_from_eff = a_from or datetime.min.replace(tzinfo=None)
    b_from_eff = b_from or datetime.min.replace(tzinfo=None)

    def _lt(lo: datetime | None, hi: datetime | None) -> bool:
        # lo < hi where a None hi means +inf (always greater).
        if hi is None:
            return True
        return _aware(lo) < _aware(hi)

    return _lt(a_from_eff, b_to) and _lt(b_from_eff, a_to)


def _aware(dt: datetime) -> datetime:
    """Make a datetime safely comparable (assume UTC-naive == UTC)."""
    from datetime import timezone

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
