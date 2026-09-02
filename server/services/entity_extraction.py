"""LLM-based entity extraction for the subject_entities store.

Phase 2 of the cross-session retrieval improvements needs an
entity-population mechanism for `subject_entities` (the table introduced
in migration 0028). A common approach uses spaCy NER for proper-noun
extraction; we use the same LiteLLM-routed model we already have for
compilation instead, because:

  - Zero new infra: the compile path already has an LLM provider
    configured + reachable; reusing it means no spaCy install, no model
    download in the Dockerfile, no new failure mode at boot.
  - Context-aware: spaCy's small English model misses domain-specific
    entities (product names, made-up identifiers) that an LLM picks up
    from surrounding context.
  - Cheap per call: ~50 input tokens + ~30 output tokens per memory
    (about $0.0002 at gpt-4o-mini). Backfilling 10K existing memories
    costs ~$2; per-compile-batch overhead is marginal.

Trade-off: LLM extraction has non-zero latency (~200-500ms per call vs
spaCy's sub-ms). The compile path already runs LLM calls in parallel
batches; we slot entity extraction into the same async fan-out so the
total compile wall-time grows by at most one extra round-trip.

`extract_entities` is the public entry point. It returns a list of
(text, kind) tuples ready for upsert into `subject_entities`. `kind`
mirrors spaCy NER labels (PERSON / ORG / GPE / DATE / PRODUCT / EVENT /
MISC) so the schema stays compatible with a future spaCy swap-in — only
the extractor changes, the storage + retrieval contracts stay intact.

Graceful degradation: if the LLM call fails for any reason, the function
logs at WARNING and returns an empty list. The compile transaction
commits anyway with no entity rows — Phase 3 retrieval (entity boost)
silently no-ops on subjects with no entities, falling back to the
pure-Phase-1 hybrid (semantic + BM25). One flaky compile doesn't break
the bench or production retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import structlog

from server.core.config import settings
from server.services.llm import acomplete

logger = structlog.stdlib.get_logger()


# Reuse the same LLM model the compiler uses — same provider, same
# credentials, same retry path. Reads from
# `settings.litellm_compile_model` (the compile-specific override,
# falling back to the default `settings.litellm_model` if unset). The
# extraction is small enough that gpt-4o-mini handles it well; ops can
# point this at a cheaper model than the compiler via env var without
# touching code.
def _extraction_model() -> str | None:
    return getattr(settings, "litellm_compile_model", None) or settings.litellm_model

EXTRACTION_SYSTEM_PROMPT = (
    "You are a precise named-entity extractor. Given a short fact about a "
    "person ('user') and their conversation history, extract every distinct "
    "entity (proper noun or compound noun phrase) the fact references. "
    "Return ONLY valid JSON with the exact shape requested."
)

_ENTITY_RULES = """An entity is:
  - A proper noun (person, place, organization, product, event, work)
  - A specific compound noun phrase that identifies one thing ("the navy blue blazer", "the 5K charity run")
  - A specific date or date range when it identifies a SPECIFIC event (skip generic dates like "yesterday")
  - A unique role/title that picks out one entity ("Senior Sales Engineer at Acme")

NOT entities:
  - Generic words ("food", "music", "exercise")
  - The user themselves ("user", "I", "me")
  - Common adjectives or qualifiers
  - Numbers or units in isolation

For each entity, output:
  - text: the canonical surface form (preserve casing as written)
  - kind: one of PERSON | ORG | GPE | LOC | PRODUCT | EVENT | WORK | DATE | MISC"""

EXTRACTION_USER_TEMPLATE = (
    """Extract every distinct named entity from the fact below.

"""
    + _ENTITY_RULES
    + """

FACT:
{text}

Return exactly this JSON:
{{"entities": [{{"text": "<entity>", "kind": "<label>"}}, ...]}}"""
)

BATCH_EXTRACTION_USER_TEMPLATE = (
    """Extract every distinct named entity from EACH numbered fact below.
Treat every fact independently — an entity mentioned in two facts appears in both results.

"""
    + _ENTITY_RULES
    + """

FACTS (one per line, each prefixed with its index):
{facts}

Return exactly this JSON, with one entry per fact index:
{{"results": {{"0": [{{"text": "<entity>", "kind": "<label>"}}, ...], "1": [...], ...}}}}"""
)


@dataclass(frozen=True)
class ExtractedEntity:
    """One entity extracted from a memory's content. `text` preserves the
    original casing for display; `normalized` is the dedup key (lowercase,
    whitespace-collapsed); `kind` is the spaCy-compatible NER label."""

    text: str
    normalized: str
    kind: str | None


_NORMALIZE_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Canonicalize an entity surface form for the dedup key. Lowercase
    + whitespace collapse + strip. This is the normalization applied for
    entity exact-match dedup before falling back to embedding cosine."""
    return _NORMALIZE_WS.sub(" ", text).strip().lower()


def _parse_entities(raw: str) -> list[ExtractedEntity]:
    """Pull the entities array out of the LLM's JSON output. Defensive:
    accepts code-fence wrappers, partial JSON, missing fields — anything
    parseable contributes; anything that isn't drops silently."""
    s = (raw or "").strip()
    # Strip code-fence wrappers
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        j = json.loads(s)
    except (ValueError, json.JSONDecodeError):
        # Last-ditch: scan for the first {...entities...} block
        m = re.search(r"\{[^{}]*\"entities\"\s*:[^{}]*\[[^\[\]]*\][^{}]*\}", s, re.DOTALL)
        if not m:
            return []
        try:
            j = json.loads(m.group(0))
        except (ValueError, json.JSONDecodeError):
            return []
    if not isinstance(j, dict):
        return []
    return _entities_from_list(j.get("entities"))


def _entities_from_list(raw_entities: object) -> list[ExtractedEntity]:
    """Validate + normalize one JSON entity array (shared by the single and
    batched parsers — the per-entity rules must never drift between them)."""
    if not isinstance(raw_entities, list):
        return []
    out: list[ExtractedEntity] = []
    seen_normalized: set[str] = set()
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        normalized = _normalize(text)
        if not normalized or normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        kind = item.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            kind = None
        out.append(
            ExtractedEntity(
                text=text.strip(),
                normalized=normalized,
                kind=kind.strip().upper() if kind else None,
            )
        )
    return out


async def extract_entities(text: str) -> list[ExtractedEntity]:
    """Return the entities extracted from one memory's content.

    Empty list on any failure (LLM unavailable, parse error, etc.) —
    the caller (compile-time hook) treats this as "no entities for this
    memory" and proceeds. The compile transaction is NOT rolled back if
    extraction fails for one row.

    Cheap to call per memory (~50 in + ~30 out tokens); the LLM call
    fans out alongside the existing compile fan-out so latency is
    parallel.
    """
    if not text or not text.strip():
        return []
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": EXTRACTION_USER_TEMPLATE.format(text=text.strip())},
    ]
    try:
        response_text = await acomplete(
            messages,
            model=_extraction_model(),
            max_tokens=400,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception:
        logger.warning("entity_extraction_failed", text_preview=text[:80], exc_info=True)
        return []
    return _parse_entities(response_text)


async def extract_entities_batch(texts: list[str]) -> list[list[ExtractedEntity]]:
    """Extract entities for several facts in ONE provider call.

    Per-memory extraction was ~60% of all provider round-trips on a large
    compile and the dominant amplifier of a provider latency regression
    (issue #380: ~745 calls per 411-episode pack). Batching cuts that to
    ~1/batch_size with identical per-entity rules (the rules block is
    shared with the single-fact prompt).

    Fail-open like `extract_entities`, in two stages: a structurally
    unusable batched response falls back to per-fact calls (which
    themselves fail-open to []); a well-formed response with a missing
    index yields [] for that fact only. Always returns exactly
    ``len(texts)`` lists, index-aligned.
    """
    if not texts:
        return []
    if len(texts) == 1:
        return [await extract_entities(texts[0])]

    # Collapse internal whitespace so multi-line fact content (procedure
    # steps like "2: run alembic") cannot mimic the index framing and
    # cross-attribute entities to the wrong fact.
    numbered = "\n".join(
        f"FACT {i}: {' '.join(t.split())}" for i, t in enumerate(texts)
    )
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": BATCH_EXTRACTION_USER_TEMPLATE.format(facts=numbered)},
    ]
    try:
        response_text = await acomplete(
            messages,
            model=_extraction_model(),
            max_tokens=min(300 + 250 * len(texts), 4000),
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Transport/provider failure (timeout, outage): per-fact calls would
        # hit the SAME dead provider one at a time, sequentially, while the
        # caller holds a concurrency slot — up to batch_size x timeout of
        # stall per group. Entities are best-effort; return nothing instead.
        logger.warning(
            "entity_batch_extraction_unavailable",
            batch_size=len(texts),
            exc_info=True,
        )
        return [[] for _ in texts]

    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (response_text or "").strip())
        parsed = json.loads(cleaned)
        results = parsed.get("results") if isinstance(parsed, dict) else None
        if not isinstance(results, dict):
            raise ValueError("batched extraction response has no results mapping")
    except Exception:
        # The provider ANSWERED but the batched shape is unusable — the
        # per-fact prompt is more robust, and the provider is demonstrably
        # up, so falling back per fact is worth the extra calls.
        logger.warning(
            "entity_batch_extraction_failed",
            batch_size=len(texts),
            exc_info=True,
        )
        return [await extract_entities(t) for t in texts]

    return [_entities_from_list(results.get(str(i))) for i in range(len(texts))]
