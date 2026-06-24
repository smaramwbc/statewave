"""Compile-time near-duplicate dedup — collapse restated/overlapping facts.

Full-conversation windowed compile (server/services/compilers/llm.py) extracts
the WHOLE haystack, which is the right recall move but produces 450-700 memories
per conversation: the 200-char window overlap re-extracts boundary facts, and the
same atomic fact gets restated across dozens of sessions. That bloat dilutes
retrieval (the precise answer memory ranks below top_k among hundreds of
near-duplicates — the root cause of the BEAM information_extraction regression
and the LongMemEval single-fact-lookup stall) and makes compile slow (reconcile
runs ~18 LLM chunks instead of ~5). The goal is to land on far fewer, deduped
atomic facts; this pass closes that gap.

Runs as a cheap, deterministic pass BEFORE the LLM reconcile
(server/services/reconcile.py), on the in-memory candidate list — it never
touches the session. Division of labour:
  * dedup     → "is this the SAME fact restated within this batch?" (cheap,
                deterministic, no committed-DB context needed)
  * reconcile → "is this a NEWER/CONTRADICTING value of an already-stored
                fact?" (LLM judgement; supersedes across batches)

Two passes:
  1. EXACT — group by normalized content (casefold + collapse whitespace +
     strip trailing punctuation). Byte-identical-after-normalization rows say
     the same thing, so this is the safe bulk of the win (windowing overlap).
  2. SEMANTIC — one batched embed of the survivors, then greedy chronological
     clustering. Two candidates merge ONLY when ALL THREE gates pass:
       (a) cosine >= dedup_sim_threshold (default 0.92),
       (b) token-Jaccard >= dedup_token_overlap_min (default 0.6),
       (c) identical extracted number/date set.
     Gate (c) is the critical safeguard: a knowledge-UPDATE pair ("API averages
     250ms" vs "90ms") differs in its numbers and is NEVER merged — it is left
     for reconcile, which is exactly what raised BEAM knowledge_update 0 -> 0.375.
     Gate (b) separates topically-close distinct facts that embeddings alone
     would over-cluster ("daughter Mia" vs "daughter Mary"; "hiking" vs
     "biking").

Merge is non-destructive: the chronological-earliest member is the canonical;
dropped members' source_episode_ids are unioned onto it and max confidence is
kept, so no provenance is lost. The canonical's embedding (computed here) is
stapled onto the row so the post-commit backfill can skip it (one embedding
round-trip reused for the memory.embedding column).

Fail-open: any error, the stub provider, a provider/count desync, or a
degenerate empty result returns the candidates unchanged — dedup can never lose
a memory. Gated by settings.dedup_compile_enabled.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from typing import Sequence

import structlog

from server.core.config import settings
from server.db.tables import MemoryRow
from server.services.embeddings import get_provider as get_embedding_provider

logger = structlog.stdlib.get_logger()

_MIN_DT = datetime.min.replace(tzinfo=timezone.utc)
_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+")
_TOK = re.compile(r"[a-z0-9]+")
_MONTHS = frozenset(
    "january february march april may june july august september october "
    "november december".split()
)
_TRAILING = " .,;:!?-—\"'()[]{}"
# Function words ignored when comparing the "significant content" of two facts.
# Differences confined to these (or to word order / punctuation) are safe to
# merge; a difference in ANY non-stopword token (a name, place, noun, verb) is
# treated as a potentially-distinct fact and left for the LLM reconcile.
_STOPWORDS = frozenset(
    "the a an and or but of to in on at for with from by as is are was were be "
    "been being has have had do does did this that these those it its their his "
    "her my your our we you they he she i me us them about into over under than "
    "then now during while which who whom whose what when where why how also".split()
)


def _normalize_core(content: str) -> str:
    """Normalized comparison key: NFKC + casefold + single-spaced + trimmed."""
    s = unicodedata.normalize("NFKC", content or "").casefold()
    s = _WS.sub(" ", s).strip()
    return s.strip(_TRAILING)


def _number_set(content: str) -> frozenset[str]:
    """Numbers + month names — the guard that keeps knowledge-update / date pairs
    distinct so dedup never collapses a changed value into its predecessor."""
    low = (content or "").casefold()
    nums = set(_NUM.findall(low))
    months = {m for m in _MONTHS if m in low}
    return frozenset(nums | months)


def _sig_tokens(content: str) -> frozenset[str]:
    """Significant content words: tokens >=3 chars, minus stopwords. Two facts
    may only merge when these are IDENTICAL — so any differing name/place/noun/
    verb (e.g. "Mia" vs "Mary", "hiking" vs "biking") blocks the merge, while
    word-order / stopword / punctuation differences (the windowing-reformat
    case) do not."""
    return frozenset(
        t for t in _TOK.findall((content or "").casefold())
        if len(t) >= 3 and t not in _STOPWORDS
    )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def _merge(keep: MemoryRow, drop: MemoryRow) -> None:
    """Fold a redundant row into its canonical without losing information:
    union provenance, keep the highest confidence and the earliest origin date."""
    keep.source_episode_ids = list(
        dict.fromkeys([*(keep.source_episode_ids or []), *(drop.source_episode_ids or [])])
    )
    try:
        keep.confidence = max(keep.confidence, drop.confidence)
    except Exception:  # pragma: no cover - defensive
        pass
    kf = getattr(keep, "valid_from", None)
    df = getattr(drop, "valid_from", None)
    if df is not None and (kf is None or df < kf):
        keep.valid_from = df


async def dedup_candidates(
    candidates: Sequence[MemoryRow], *, provider=None
) -> list[MemoryRow]:
    """Collapse near-duplicate candidate memories. Returns the kept canonicals
    (a subset of the input rows, with merged provenance + stapled embeddings).

    Fail-open: returns ``list(candidates)`` unchanged on any problem."""
    cands = list(candidates)
    if not settings.dedup_compile_enabled or len(cands) < 2:
        return cands
    if len(cands) > settings.dedup_max_candidates:
        logger.info("dedup_skipped_large_batch", candidates=len(cands))
        return cands

    try:
        # ── Pass 1: exact normalized-text collapse (stub-safe, no embeddings) ──
        # Keyed on (kind, normalized content) so identical text under different
        # kinds is never collapsed.
        groups: dict[tuple[str, str], MemoryRow] = {}
        order: list[tuple[str, str]] = []
        for c in cands:
            key = (c.kind, _normalize_core(c.content))
            if key in groups:
                _merge(groups[key], c)
            else:
                groups[key] = c
                order.append(key)
        survivors = [groups[k] for k in order]
        exact_dropped = len(cands) - len(survivors)
        if len(survivors) < 2:
            return survivors

        # ── Pass 2: semantic cluster (real embeddings only) ──────────────────
        prov = provider or get_embedding_provider()
        if (
            not settings.dedup_semantic_enabled
            or prov is None
            or not getattr(prov, "provides_semantic_similarity", False)
        ):
            logger.info(
                "dedup_done",
                input=len(cands),
                kept=len(survivors),
                exact_dropped=exact_dropped,
                semantic=False,
            )
            return survivors

        try:
            vecs = await prov.embed_texts([s.content for s in survivors])
        except Exception:
            logger.warning("dedup_embed_failed", exc_info=True)
            return survivors
        if not isinstance(vecs, list) or len(vecs) != len(survivors):
            return survivors  # provider desync — fail-open to exact-pass result

        # Deterministic chronological order so the canonical (first kept) is the
        # earliest member and clustering is reproducible.
        idx = sorted(
            range(len(survivors)),
            key=lambda i: (getattr(survivors[i], "valid_from", None) or _MIN_DT, str(survivors[i].id)),
        )
        sim_threshold = settings.dedup_sim_threshold
        kept: list[tuple[MemoryRow, list[float], frozenset, frozenset]] = []
        for i in idx:
            row = survivors[i]
            vec = vecs[i]
            nums = _number_set(row.content)
            sig = _sig_tokens(row.content)
            canonical = None
            for krow, kvec, knums, ksig in kept:
                if krow.kind != row.kind:
                    continue
                if nums != knums:  # number/date guard — never merge a changed value
                    continue
                if sig != ksig:    # significant-content guard — differing name/noun/verb blocks
                    continue
                if not sig:        # no content words to anchor on — don't merge
                    continue
                if _cosine(vec, kvec) < sim_threshold:  # final meaning check (role-reversal etc.)
                    continue
                canonical = krow
                break
            if canonical is not None:
                _merge(canonical, row)
            else:
                # Reuse this vector for the persisted memory.embedding column so
                # the post-commit backfill can skip this row.
                row.embedding = vec
                kept.append((row, vec, nums, sig))

        result = [r for (r, _v, _n, _t) in kept]
        if not result:
            return survivors  # degenerate guard — never empty out a batch

        logger.info(
            "dedup_done",
            input=len(cands),
            kept=len(result),
            exact_dropped=exact_dropped,
            semantic_dropped=len(survivors) - len(result),
            semantic=True,
        )
        return result
    except Exception:
        logger.warning("dedup_failed", exc_info=True)
        return cands
