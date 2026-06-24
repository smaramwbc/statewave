"""Context-aware compile reconcile — a "memory-update" step.

Phase 4 (context-aware compile) + Phase 5b (supersession / semantic dedup) of
the cross-session retrieval improvements, FUSED into one mechanism: a single
LLM call returns an ADD / UPDATE / DELETE / NONE event per candidate fact,
judged against the semantically nearest existing memories.

Why this is the highest-leverage missing piece
-----------------------------------------------
The LLM compiler was purely *additive*: it extracted facts from each episode
batch and never reconciled them against what was already stored. So a changed
value ("I live in Munich" → later "I moved to Berlin") left BOTH memories
active, and retrieval happily returned the stale one. That single gap is the
root cause of the near-zero GPT-only categories:

  * BEAM knowledge_update = 0.00
  * BEAM contradiction_resolution = 0.01
  * BEAM multi_session_reasoning = 0.00
  * LongMemEval knowledge_update / multi_session soft spots

Pipeline (called from ``server/api/memories._compile_one_batch`` AFTER the
compiler returns candidate rows, BEFORE they are inserted)
----------------------------------------------------------
  1. Read the subject's EXISTING active memories (committed). When that set is
     large, narrow it to the candidates' semantic neighbourhood via one
     batched embedding round-trip + indexed cosine lookups.
  2. Sort candidates chronologically (oldest first) and present existing
     memories (E0, E1, …) and new candidates (N0, N1, …) to ONE LLM call.
  3. The LLM returns, per candidate, exactly one decision:
       ADD       — genuinely new; keep it.
       DUPLICATE — already covered by a target (E# or earlier N#); drop it.
       UPDATE    — a newer value of the target's fact; keep it, retire target.
       DELETE    — asserts the target is now false; keep it, retire target.
  4. Apply: return the kept candidate rows + the set of EXISTING committed
     memory ids to mark superseded. Candidates retired by a later candidate
     are simply not inserted.

Robustness
----------
Best-effort and fail-OPEN: missing semantic embeddings, an LLM error, a
malformed response, or a degenerate all-dropped result all fall back to the
prior additive behaviour (every candidate ADDed, nothing superseded) so the
compile NEVER loses memories to a reconcile failure. Gated by
``settings.reconcile_compile_enabled``.

The reconcile LLM call routes through the central adapter with the compile
model (provider-neutral; gpt-4o-mini in the bench config), so it
introduces no new provider dependency.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

import structlog

from server.core.config import settings
from server.db import repositories as repo
from server.db.tables import MemoryRow
from server.services import llm as llm_adapter
from server.services.embeddings import get_provider as get_embedding_provider

logger = structlog.stdlib.get_logger()

# Sentinel for chronological sort of undated candidates (sorts first).
_MIN_DT = datetime.min.replace(tzinfo=timezone.utc)

_RECONCILE_SYSTEM_PROMPT = """\
You are a memory reconciliation engine for a long-term memory system. You keep a \
subject's stored memory consistent by deciding how each NEW candidate fact relates \
to the EXISTING memory and to the other new facts.

You are given two lists:
- EXISTING memories, labeled E0, E1, ... — already stored. (May be empty.)
- NEW candidate facts, labeled N0, N1, ... in CHRONOLOGICAL order, OLDEST FIRST. \
A higher index means more recent in time.

For EACH new candidate Ni choose EXACTLY ONE action:
- "ADD": Ni carries information not already captured by any existing memory or \
earlier new fact. Keep it.
- "DUPLICATE": Ni states the SAME fact as a "target" (an E# or an earlier N#), \
adding nothing new. Drop Ni — the target already covers it.
- "UPDATE": Ni is a NEWER or CORRECTED VALUE of the same attribute of the same \
entity as the "target" (e.g. target = "lives in Munich", Ni = "moved to Berlin in \
2024"; or target = "API averages 250ms", Ni = "API now averages 90ms"). Keep Ni \
and retire the target.
  * RUNNING/CUMULATIVE quantities: when target and Ni are two readings of the SAME \
one-direction tally (a page count, a times-done counter, a running total), ALSO \
return a "merged_content" field: ONE sentence stating the CURRENT ABSOLUTE value, \
resolving any relative delta against the target ("read 20 more pages" + target "on \
page 200" -> "on page 220"; "wore them again" + target "worn 5 times" -> "worn 6 \
times"), preserving the entity, unit and the as-of date. If Ni is already absolute, \
restate it cleanly. OMIT "merged_content" for ordinary value-replacements (Munich \
-> Berlin) where no carry-forward arithmetic is needed.
- "DELETE": Ni asserts the "target" fact is NO LONGER TRUE, was removed, or is \
contradicted (e.g. target = "uses Stripe", Ni = "stopped using Stripe and switched \
off card payments"). Keep Ni and retire the target.

Rules:
- "target" is a single label string (e.g. "E3" or "N1") for DUPLICATE / UPDATE / \
DELETE; it MUST be null for ADD.
- UPDATE/DELETE/DUPLICATE only when the two facts concern the SAME attribute of the \
SAME entity. Two different attributes are both kept (ADD) — e.g. "likes coffee" and \
"likes hiking" are unrelated; "has a daughter Mia" and "has a son Leo" are both true.
- Distinct concrete facts are ADD even when topically related. When in doubt, ADD — \
never discard information you are unsure about.
- A target must be STRICTLY OLDER than Ni: any E#, or an N# with a SMALLER index. \
Never target an equal or later N#.
- Do not retire a memory unless Ni genuinely duplicates, updates, or contradicts it.

Return ONLY a JSON object:
{"decisions": [{"i": 0, "action": "ADD", "target": null, "reason": "<short>"}, ...]}
with EXACTLY ONE entry per new candidate, ordered by i ascending. An UPDATE entry \
for a running/cumulative quantity MAY additionally carry "merged_content": "<current \
absolute value sentence>"."""

_LABEL_RE = re.compile(r"^([EN])(\d+)$")


def _compile_model() -> str:
    """The model the reconcile LLM call uses — the compile model, so reconcile
    shares the compiler's provider/credentials and never adds a dependency."""
    return getattr(settings, "litellm_compile_model", None) or settings.litellm_model


def _fmt_date(m: MemoryRow) -> str:
    vf = getattr(m, "valid_from", None)
    try:
        return vf.strftime("%Y-%m-%d") if vf is not None else "undated"
    except Exception:  # pragma: no cover - defensive
        return "undated"


async def _select_existing(
    session,
    subject_id: str,
    candidates: list[MemoryRow],
    *,
    tenant_id: str | None,
) -> list[MemoryRow]:
    """Return the existing active memories to reconcile against, capped to
    ``reconcile_max_existing``. When the subject already has more memories than
    the cap, narrow to the candidates' semantic neighbourhood (one batched
    embed + per-candidate indexed cosine lookups); otherwise feed them all.

    Reads ONLY committed rows — candidates are not yet in the session, so the
    "existing" view is never polluted by the batch being reconciled."""
    max_existing = settings.reconcile_max_existing
    existing = list(
        await repo.list_active_memories_by_subject(
            session, subject_id, tenant_id=tenant_id, limit=500
        )
    )
    if len(existing) <= max_existing:
        return existing

    provider = get_embedding_provider()
    if provider is None or not getattr(provider, "provides_semantic_similarity", False):
        # No usable embeddings — fall back to the most recent slice. The
        # repo returns created_at ASC, so the tail is the newest.
        return existing[-max_existing:]

    try:
        cand_embeddings = await provider.embed_texts([c.content for c in candidates])
        relevant: dict[uuid.UUID, MemoryRow] = {}
        by_id = {m.id: m for m in existing}
        top_k = settings.reconcile_top_k_per_candidate
        for emb in cand_embeddings:
            rows = await repo.search_memories_by_embedding(
                session, subject_id, emb, tenant_id=tenant_id, limit=top_k
            )
            for row, _distance in rows:
                if row.id in by_id:
                    relevant[row.id] = by_id[row.id]
        if relevant:
            return list(relevant.values())[:max_existing]
    except Exception:
        logger.warning("reconcile_existing_select_failed", subject_id=subject_id, exc_info=True)
    return existing[-max_existing:]


def _build_user_message(ctx_rows: list[MemoryRow], chunk_rows: list[MemoryRow]) -> str:
    existing_block = "\n".join(
        f"E{k}: {m.content}" for k, m in enumerate(ctx_rows)
    ) or "(none)"
    candidate_block = "\n".join(
        f"N{i}: [{_fmt_date(m)}] {m.content}" for i, m in enumerate(chunk_rows)
    )
    return (
        "EXISTING memories:\n"
        f"{existing_block}\n\n"
        "NEW candidate facts (chronological, oldest first):\n"
        f"{candidate_block}\n\n"
        f"Return a decision for each of the {len(chunk_rows)} new candidate(s)."
    )


def _parse_label(target: Any, *, n_existing: int, n_candidates: int) -> tuple[str, int] | None:
    """Parse an 'E#'/'N#' target into ('E'|'N', index), bounds-checked. None if
    malformed or out of range."""
    if not isinstance(target, str):
        return None
    match = _LABEL_RE.match(target.strip())
    if not match:
        return None
    kind, idx_s = match.group(1), match.group(2)
    idx = int(idx_s)
    if kind == "E" and 0 <= idx < n_existing:
        return ("E", idx)
    if kind == "N" and 0 <= idx < n_candidates:
        return ("N", idx)
    return None


async def reconcile_compile_batch(
    session,
    subject_id: str,
    candidates: Sequence[MemoryRow],
    *,
    tenant_id: str | None = None,
) -> tuple[list[MemoryRow], set[uuid.UUID]]:
    """Reconcile freshly-compiled candidate memories against existing memory.

    Returns ``(kept_rows, superseded_existing_ids)``:
      * ``kept_rows`` — candidates to insert (ADD/UPDATE/DELETE survivors), in
        chronological order; DUPLICATE candidates and candidates retired by a
        later candidate are excluded.
      * ``superseded_existing_ids`` — ids of EXISTING committed memories the
        caller must mark superseded.

    Fail-open: on any problem returns ``(list(candidates), set())`` so the
    compile degrades to the prior additive behaviour rather than losing rows.
    """
    candidates = list(candidates)
    if not candidates:
        return [], set()
    # Bound the prompt — an unusually large batch skips reconcile rather than
    # sending a giant prompt (resolve_conflicts + a later drain still apply).
    if len(candidates) > settings.reconcile_max_candidates:
        logger.info(
            "reconcile_skipped_large_batch",
            subject_id=subject_id,
            candidates=len(candidates),
            cap=settings.reconcile_max_candidates,
        )
        return candidates, set()

    # Chronological order (oldest first) so "target must be strictly older"
    # maps onto "smaller index". Stable on id for equal timestamps.
    candidates.sort(key=lambda m: (getattr(m, "valid_from", None) or _MIN_DT, str(m.id)))

    try:
        existing = await _select_existing(
            session, subject_id, candidates, tenant_id=tenant_id
        )
    except Exception:
        logger.warning("reconcile_existing_read_failed", subject_id=subject_id, exc_info=True)
        return candidates, set()

    # Nothing stored AND a single candidate → nothing to reconcile against.
    if not existing and len(candidates) == 1:
        return candidates, set()

    existing_ids = {m.id for m in existing}
    supersede_existing: set[uuid.UUID] = set()
    accepted_ids: list[uuid.UUID] = []           # candidate ids kept, in order
    dropped_ids: set[uuid.UUID] = set()          # candidate ids retired/duplicate
    cand_by_id: dict[uuid.UUID, MemoryRow] = {m.id: m for m in candidates}
    max_ctx = settings.reconcile_max_existing
    chunk_size = max(1, settings.reconcile_chunk_size)

    # Process candidates in chronological chunks. Each chunk is reconciled
    # against the committed existing memory PLUS the candidates accepted so far,
    # so duplicates / knowledge-updates are caught across the whole batch (and
    # across windows of a full-conversation compile) without sending one giant
    # prompt.
    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        live_accepted = [cand_by_id[i] for i in accepted_ids if i not in dropped_ids]
        # Context: always keep all committed existing; fill the rest of the
        # budget with the most-recent accepted candidates (likeliest dedup /
        # update targets within a conversation).
        room = max(0, max_ctx - len(existing))
        ctx_rows = existing + (live_accepted[-room:] if room else [])

        try:
            parsed = await llm_adapter.acomplete_json(
                [
                    {"role": "system", "content": _RECONCILE_SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_message(ctx_rows, chunk)},
                ],
                model=_compile_model(),
                temperature=0.0,
                max_tokens=min(settings.litellm_compile_max_tokens, 4000),
            )
        except Exception:
            logger.warning("reconcile_llm_failed", subject_id=subject_id, exc_info=True)
            # Keep this chunk wholesale (fail-open) and continue.
            accepted_ids.extend(m.id for m in chunk)
            continue

        decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
        if not isinstance(decisions, list) or not decisions:
            logger.warning("reconcile_no_decisions", subject_id=subject_id)
            accepted_ids.extend(m.id for m in chunk)
            continue

        # Index decisions by chunk-local i.
        by_i: dict[int, dict] = {}
        for entry in decisions:
            if isinstance(entry, dict) and isinstance(entry.get("i"), int):
                by_i[entry["i"]] = entry

        n_ctx = len(ctx_rows)
        n_chunk = len(chunk)
        for j, cand in enumerate(chunk):
            entry = by_i.get(j)
            action = str(entry.get("action", "ADD")).upper() if entry else "ADD"
            if action == "ADD" or not entry:
                accepted_ids.append(cand.id)
                continue
            target = _parse_label(
                entry.get("target"), n_existing=n_ctx, n_candidates=n_chunk
            )
            if target is None:
                # Missing/invalid target degrades to ADD.
                accepted_ids.append(cand.id)
                continue
            tkind, tidx = target
            if action == "DUPLICATE":
                # Redundant with an older target → drop this candidate.
                dropped_ids.add(cand.id)
            elif action in ("UPDATE", "DELETE"):
                # Keep the candidate (newer value / contradiction is itself a
                # fact); retire the target.
                accepted_ids.append(cand.id)
                trow = None
                if tkind == "E":
                    trow = ctx_rows[tidx]
                    if trow.id in existing_ids:
                        supersede_existing.add(trow.id)
                    else:
                        # Target is a previously-accepted candidate → drop it.
                        dropped_ids.add(trow.id)
                else:  # N#: target must be strictly older within this chunk.
                    if tidx < j:
                        trow = chunk[tidx]
                        dropped_ids.add(trow.id)
                # Cumulative/latest-value merge (opt-in, count-neutral): carry the
                # CURRENT absolute value into the surviving candidate so the fixed
                # answer model reads it off instead of tracking supersession across
                # the haystack. Fail-open — any problem leaves the candidate as the
                # extractor wrote it (today's exact behavior).
                if (
                    action == "UPDATE"
                    and trow is not None
                    and getattr(settings, "reconcile_cumulative_merge", False)
                ):
                    merged = entry.get("merged_content")
                    if isinstance(merged, str) and merged.strip():
                        cand.content = merged.strip()
                        cand.summary = merged.strip()[:200]
                        try:  # migrate retired row's provenance to the survivor
                            cand.source_episode_ids = list(
                                dict.fromkeys(
                                    list(cand.source_episode_ids or [])
                                    + list(trow.source_episode_ids or [])
                                )
                            )
                        except Exception:  # pragma: no cover - defensive
                            pass
            else:
                # Unknown action → treat as ADD.
                accepted_ids.append(cand.id)

    kept = [cand_by_id[i] for i in accepted_ids if i not in dropped_ids]

    # Degenerate guard: only fall back when dropping everything would leave the
    # subject with NOTHING — no existing memory AND no kept candidate. Dropping
    # all candidates as duplicates of EXISTING memories is legitimate (the
    # information is already stored), so we only protect against total loss on a
    # subject that had no prior memory.
    if not kept and not existing:
        logger.warning("reconcile_all_dropped_fallback", subject_id=subject_id)
        return candidates, set()

    logger.info(
        "reconcile_done",
        subject_id=subject_id,
        candidates=len(candidates),
        kept=len(kept),
        dropped=len(dropped_ids),
        superseded_existing=len(supersede_existing),
        existing_considered=len(existing),
    )
    return kept, supersede_existing
