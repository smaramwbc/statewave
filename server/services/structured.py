"""Structured memory candidates.

A producer that already knows facts in structured form (a connector, a demo
agent, a structured candidate mapper) may attach them to an episode so Statewave
compiles them into **separate atomic memories** instead of one indivisible prose
blob — and so a stale atomic fact can be superseded without dragging down the
independent facts that shared its source episode.

Ingestion extension point (additive, namespaced, opaque to old readers):

    episode.payload["statewave"]["memory_candidates"] = [
        {"kind": "domain_fact", "text": "...", "metadata": {...}, "claim": {...}},
        ...
    ]

The raw ``payload`` body (timeline/source/audit) is untouched; candidates are an
additive sibling key.

Deterministic fallback policy (documented):
  1. Container missing or not a non-empty list  -> full legacy compiler path.
  2. ANY candidate missing valid text or a mappable kind -> full legacy path
     (never partially drop an episode).
  3. Candidate text + kind valid but its optional claim invalid -> emit the
     candidate as an UNKEYED memory (keep text/kind/metadata, drop the claim).
  4. Accepted set -> compile atomically, and do NOT also emit a catch-all
     episode_summary.

Active-context representation: an episode that was compiled from accepted
candidates is NOT injected verbatim into active context (see
``server.services.context``); its active atomic memories are the representation.
The raw episode stays available in timeline/admin/history. No textual redaction,
no word detection — purely the presence of accepted structured candidates.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Mapping

from server.core.config import settings
from server.db.tables import EpisodeRow, MemoryRow
from server.services.claims import (
    CLAIM_METADATA_KEY,
    _parse_dt,
    anchor_valid_from,
    build_claim_envelope,
    build_v2_envelope,
)
from server.services.compilers.heuristic import episode_valid_from
from server.services.memory_ttl import compute_valid_to

CANDIDATE_CONTAINER_KEY = "statewave"
CANDIDATE_LIST_KEY = "memory_candidates"

# Candidate kind -> stored MemoryKind. No enum change: "domain_fact" maps to the
# existing ``profile_fact`` kind. Unknown kinds make the container non-acceptable
# (full legacy fallback), never a silent drop.
_KIND_MAP = {
    "domain_fact": "profile_fact",
    "fact": "profile_fact",
    "profile_fact": "profile_fact",
    "episode_summary": "episode_summary",
    "procedure": "procedure",
}


def _container(payload: Any) -> Any:
    if isinstance(payload, dict):
        sw = payload.get(CANDIDATE_CONTAINER_KEY)
        if isinstance(sw, dict):
            return sw.get(CANDIDATE_LIST_KEY)
    return None


def accepted_candidates(payload: Any) -> list[dict] | None:
    """Validated candidate list, or ``None`` to use the full legacy path.

    Read-only and deterministic — used by both the compiler (to emit atomic
    memories) and context assembly (to recognize a structured episode). Returns
    ``None`` for rules 1 and 2 of the fallback policy.
    """
    cont = _container(payload)
    if not isinstance(cont, list) or not cont:
        return None
    for c in cont:
        if not isinstance(c, dict):
            return None
        text = c.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        kind = c.get("kind")
        if not isinstance(kind, str) or kind.strip().lower() not in _KIND_MAP:
            return None
    return cont


def is_structured_episode(payload: Any) -> bool:
    return accepted_candidates(payload) is not None


def _candidate_claim_metadata(
    claim: Any,
    *,
    default_valid_from: datetime | None = None,
    claim_keys: Mapping[str, Any] | None = None,
) -> dict | None:
    """Validate a candidate's optional claim into a clean stored envelope, or
    ``None`` (rule 3 → unkeyed). Producers never define authoritative scope.

    ``default_valid_from`` anchors claims that do not carry their own
    ``valid_from``: without one, ``_claim_cmp`` falls through to ``created_at``
    — which ties for rows compiled in one transaction and then breaks the tie
    on random UUIDs, making contradiction resolution non-deterministic. The
    episode's temporal anchor (``episode_valid_from``) is what the produced
    MemoryRow already records as its own ``valid_from``, so the claim and its
    row stay consistent. A producer-supplied ``valid_from`` always wins.
    """
    if not isinstance(claim, dict):
        return None
    version = claim.get("schema_version")
    if version == 2:
        envelope = build_v2_envelope(claim, claim_keys)
        if (
            envelope
            and envelope[CLAIM_METADATA_KEY].get("valid_from") is None
        ):
            default = anchor_valid_from(_parse_dt(envelope[CLAIM_METADATA_KEY].get("valid_to")), default_valid_from)
            if default is not None:
                envelope[CLAIM_METADATA_KEY]["valid_from"] = default.isoformat()
        return envelope
    if version == 1:
        valid_to = _parse_dt(claim.get("valid_to"))
        return build_claim_envelope(
            claim.get("key"),
            claim.get("value"),
            extra_keys=claim_keys,
            valid_from=_parse_dt(claim.get("valid_from"))
            or anchor_valid_from(valid_to, default_valid_from),
            valid_to=valid_to,
            source=claim.get("source")
            if isinstance(claim.get("source"), str)
            else "structured_candidate",
        )
    return None



def compile_candidates(
    ep: EpisodeRow, claim_keys: Mapping[str, Any] | None = None
) -> list[MemoryRow] | None:
    """Compile an episode's accepted structured candidates into atomic memories,
    or ``None`` if the episode has none (caller uses the legacy path).

    No catch-all ``episode_summary`` is emitted. Existing candidate ``metadata``
    survives; a valid claim is attached, an invalid one is dropped (text kept).
    """
    cands = accepted_candidates(ep.payload)
    if cands is None:
        return None

    vf = episode_valid_from(ep)
    ttl = settings.kind_ttl_days
    out: list[MemoryRow] = []
    for c in cands:
        kind = _KIND_MAP[c["kind"].strip().lower()]
        text = c["text"]
        metadata: dict[str, Any] = dict(c.get("metadata") or {})
        claim_md = _candidate_claim_metadata(
            c.get("claim"), default_valid_from=vf, claim_keys=claim_keys
        )
        if claim_md:
            metadata.update(claim_md)
        confidence = c.get("confidence")
        if not isinstance(confidence, (int, float)):
            confidence = 0.9
        out.append(
            MemoryRow(
                id=uuid.uuid4(),
                subject_id=ep.subject_id,
                kind=kind,
                content=text,
                summary=text[:200],
                confidence=float(max(0.0, min(1.0, confidence))),
                valid_from=vf,
                valid_to=compute_valid_to(kind, vf, ttl),
                source_episode_ids=[ep.id],
                metadata_=metadata,
                status="active",
            )
        )
    return out
