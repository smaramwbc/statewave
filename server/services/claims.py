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

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

logger = structlog.stdlib.get_logger()

# Sub-key under MemoryRow.metadata_ that carries the envelope. Chosen to not
# collide with existing metadata producers (memory_packs uses flat top-level
# keys like ``starter_pack_id``; the llm compiler uses ``compiler``/``model``).
CLAIM_METADATA_KEY = "claim"

# Supported envelope schema versions.
#   v1 — subject-relative claim: a single canonical key + scalar value (legacy).
#   v2 — external-entity claim:  key + entity_key + canonical qualifier map +
#        structured canonical value (issue #236 follow-up).
# An UNSUPPORTED version degrades to legacy (never authoritative), so an older
# reader never trips on future data. ``CLAIM_SCHEMA_VERSION`` stays 1 for the
# v1 builder's stamp; v2 producers stamp 2 explicitly.
CLAIM_SCHEMA_VERSION = 1
_SUPPORTED_SCHEMA_VERSIONS = (1, 2)

# Cardinality of a canonical claim key.
SCOPE_SINGLE = "single"  # one current value wins (name, current employer, …)
SCOPE_MULTI = "multi"  # values coexist (tools used, skills, preferences, …)

# How a claim's value is canonicalized for equality (registry-declared).
VALUE_STRING = "string"  # v1 scalar: lowercase/strip/collapse-ws
VALUE_CANONICAL_JSON = "canonical_json"  # v2 structured: deterministic JSON


@dataclass(frozen=True)
class ClaimDefinition:
    """Declarative, authoritative policy for one canonical claim key.

    The resolver consumes definitions *generically* — it never special-cases a
    domain. Cardinality, entity/qualifier requirements, and value normalization
    all live here, so adding a new domain (pricing, ratings, …) is a registry
    entry, not a resolver branch.
    """

    key: str
    scope: str  # SCOPE_SINGLE | SCOPE_MULTI — authoritative cardinality
    entity_required: bool = False
    required_qualifiers: frozenset[str] = field(default_factory=frozenset)
    optional_qualifiers: frozenset[str] = field(default_factory=frozenset)
    value_normalization: str = VALUE_STRING

    @property
    def supports_v2(self) -> bool:
        return self.value_normalization == VALUE_CANONICAL_JSON or self.entity_required


# Back-compat alias: older imports referenced ClaimKeySpec.
ClaimKeySpec = ClaimDefinition


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

CLAIM_REGISTRY: dict[str, ClaimDefinition] = {
    **{k: ClaimDefinition(k, SCOPE_SINGLE) for k in _SINGLE_VALUED_KEYS},
    **{k: ClaimDefinition(k, SCOPE_MULTI) for k in _MULTI_VALUED_KEYS},
    # v2 external-entity claim. Generic and reusable for ANY organization — the
    # entity is identified by `entity_key`, never hard-coded. Currency and
    # charge unit are part of IDENTITY (different currency = different fact), so
    # they are required qualifiers, not value fields.
    "pricing.processing_rate": ClaimDefinition(
        key="pricing.processing_rate",
        scope=SCOPE_SINGLE,
        entity_required=True,
        required_qualifiers=frozenset({"payment_method", "rate_type", "currency", "charge_unit"}),
        optional_qualifiers=frozenset({"region", "customer_segment", "channel", "plan"}),
        value_normalization=VALUE_CANONICAL_JSON,
    ),
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


# --------------------------------------------------------------------------- #
# v2 canonicalization — entity, qualifiers, and structured value.
# All deterministic and domain-agnostic. Any failure returns None so the claim
# is simply non-authoritative (under-resolution), never raising.
# --------------------------------------------------------------------------- #


def canonical_entity_key(entity_key: Any) -> str | None:
    """Normalize an entity key (e.g. ``organization:stripe``). Lowercased,
    stripped, internal whitespace collapsed. Non-string/empty → None."""
    return normalize_value(entity_key)


def normalize_qualifiers(quals: Any, definition: ClaimDefinition) -> dict[str, str] | None:
    """Validate + normalize the qualifier map into a clean dict.

    Rules: keys/values normalized deterministically; object ordering irrelevant;
    ALL required qualifiers must be present (missing → non-authoritative, NOT a
    wildcard); unapproved qualifier keys are REJECTED (the documented safe
    policy); only string scalar values are accepted."""
    if not isinstance(quals, dict):
        return None
    allowed = definition.required_qualifiers | definition.optional_qualifiers
    norm: dict[str, str] = {}
    for k, v in quals.items():
        if not isinstance(k, str):
            return None
        nk = normalize_value(k)
        nv = normalize_value(v)
        if nk is None or nv is None:
            return None
        if nk not in allowed:
            return None  # unapproved qualifier → reject (safe policy)
        if nk in norm:
            return None  # duplicate after normalization → ambiguous
        norm[nk] = nv
    if not definition.required_qualifiers <= set(norm):
        return None  # a required qualifier is missing → non-authoritative
    return norm


def canonical_qualifiers(quals: Any, definition: ClaimDefinition) -> str | None:
    """Stable string identity for a qualifier map (sorted keys). Two different
    qualifier maps canonicalize differently and never share a bucket."""
    norm = normalize_qualifiers(quals, definition)
    if norm is None:
        return None
    return json.dumps(norm, sort_keys=True, separators=(",", ":"))


def _json_safe(value: Any) -> bool:
    """True iff value is a finite JSON scalar / nested dict(str-keys)/list."""
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_safe(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_json_safe(v) for v in value)
    return False  # no arbitrary Python objects


def canonical_value(value: Any, definition: ClaimDefinition) -> str | None:
    """Canonicalize a claim value per its definition, deterministically.

    ``string``: v1 scalar normalization. ``canonical_json``: sorted-key JSON of
    a JSON-safe structured value (reject non-finite numbers and arbitrary
    objects), so equivalent object orderings compare equal and avoid raw
    floating-point surprises (percentages as basis points, money as minor units)."""
    if definition.value_normalization == VALUE_STRING:
        return normalize_value(value)
    if definition.value_normalization == VALUE_CANONICAL_JSON:
        if not _json_safe(value):
            return None
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            return None
    return None


def _parse_dt(value: Any) -> datetime | None:
    """Best-effort ISO datetime parse; never raises (bad date → None/omit)."""
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


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
    """A validated, registry-authoritative claim ready for the resolver.

    ``bucket`` is the GENERIC contradiction-bucket identity the resolver groups
    by — the resolver never inspects key/entity/qualifiers directly. v1 buckets
    by key alone; v2 buckets by (key, entity, canonical qualifiers).
    """

    canonical_key: str
    value: str  # canonical (normalized scalar for v1, canonical JSON for v2)
    scope: str  # authoritative cardinality, from the registry definition
    valid_from: datetime | None
    valid_to: datetime | None
    bucket: tuple = ()


def resolve_claim(metadata: dict | None) -> ResolvedClaim | None:
    """Parse + validate ``metadata['claim']`` into a :class:`ResolvedClaim`.

    Returns ``None`` (→ legacy behavior) for every unsafe case: no envelope,
    malformed envelope, unsupported ``schema_version``, unknown/unregistered
    key, missing required identity, or unusable value. Never raises.
    """
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get(CLAIM_METADATA_KEY)
    if not isinstance(raw, dict):
        return None

    version = raw.get("schema_version")
    if version == 1:
        return _resolve_v1(raw)
    if version == 2:
        return _resolve_v2(raw)
    logger.info("claim_envelope_unsupported_version", version=version)
    return None


def _resolve_v1(raw: dict) -> ResolvedClaim | None:
    try:
        env = ClaimEnvelope.model_validate(raw)
    except ValidationError:
        logger.info(
            "claim_envelope_invalid",
            reason="validation_error",
            raw_key=raw.get("key") if isinstance(raw.get("key"), str) else None,
        )
        return None
    canonical = canonicalize_key(env.key)
    if canonical is None:
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
        bucket=("v1", canonical),
    )


def _resolve_v2(raw: dict) -> ResolvedClaim | None:
    canonical = canonicalize_key(raw.get("key"))
    if canonical is None:
        return None
    definition = CLAIM_REGISTRY[canonical]
    if not definition.supports_v2:
        logger.info("claim_envelope_invalid", reason="v2_unsupported_key", raw_key=canonical)
        return None
    entity = canonical_entity_key(raw.get("entity_key"))
    if definition.entity_required and entity is None:
        logger.info("claim_envelope_invalid", reason="missing_entity", raw_key=canonical)
        return None
    quals = canonical_qualifiers(raw.get("qualifiers", {}), definition)
    if quals is None:
        logger.info("claim_envelope_invalid", reason="qualifiers", raw_key=canonical)
        return None
    value = canonical_value(raw.get("value"), definition)
    if value is None:
        logger.info("claim_envelope_invalid", reason="value", raw_key=canonical)
        return None
    return ResolvedClaim(
        canonical_key=canonical,
        value=value,
        scope=definition.scope,
        valid_from=_parse_dt(raw.get("valid_from")),
        valid_to=_parse_dt(raw.get("valid_to")),
        bucket=("v2", canonical, entity, quals),
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
    normalized = normalize_value(value)
    if canonical is None or normalized is None:
        return None
    env: dict[str, Any] = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "key": canonical,
        "value": normalized,  # store the normalized value (resolver compares on it)
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


def build_v2_envelope(raw_claim: Any) -> dict | None:
    """Validate a producer-supplied v2 claim and return a CLEAN storable
    envelope, or ``None`` if it is not authoritative.

    Storing a re-canonicalized envelope (canonical entity, normalized
    qualifiers) keeps persisted data deterministic; the resolver re-validates on
    read regardless. The structured value is stored verbatim (already proven
    canonical-safe). Used by the structured memory-candidate path.
    """
    if not isinstance(raw_claim, dict):
        return None
    canonical = canonicalize_key(raw_claim.get("key"))
    if canonical is None:
        return None
    definition = CLAIM_REGISTRY[canonical]
    if not definition.supports_v2:
        return None
    entity = canonical_entity_key(raw_claim.get("entity_key"))
    if definition.entity_required and entity is None:
        return None
    quals = normalize_qualifiers(raw_claim.get("qualifiers", {}), definition)
    if quals is None:
        return None
    if canonical_value(raw_claim.get("value"), definition) is None:
        return None
    env: dict[str, Any] = {
        "schema_version": 2,
        "key": canonical,
        "entity_key": entity,
        "qualifiers": quals,
        "value": raw_claim.get("value"),
        "source": raw_claim.get("source")
        if isinstance(raw_claim.get("source"), str)
        else "structured_candidate",
    }
    vf, vt = _parse_dt(raw_claim.get("valid_from")), _parse_dt(raw_claim.get("valid_to"))
    if vf is not None:
        env["valid_from"] = vf.isoformat()
    if vt is not None:
        env["valid_to"] = vt.isoformat()
    conf = raw_claim.get("confidence")
    if isinstance(conf, (int, float)) and math.isfinite(conf):
        env["confidence"] = conf
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
