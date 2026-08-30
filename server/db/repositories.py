"""Data-access layer. All SQL lives here.

Tenant scoping: when tenant_id is provided, all queries are filtered to
that tenant. When tenant_id is None (single-tenant mode), no filter is
applied — preserving backward compatibility for local/single-tenant use.
"""

from __future__ import annotations

import uuid
from typing import Sequence

from sqlalchemy import delete, func, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from datetime import datetime

from server.db.tables import (
    EpisodeRow,
    MemoryRow,
    QueryEmbeddingCacheRow,
    ReceiptRow,
    ResolutionRow,
    SubjectEntityRow,
    SubjectHealthCacheRow,
    TenantConfigRow,
)


def _unexpired(stmt):
    """Restrict a Memory query to rows whose `valid_to` has not passed.

    Memories with a NULL `valid_to` are forever-valid (no TTL configured
    for their kind, no conflict-supersession deadline). Memories with a
    `valid_to` in the future are still authoritative. Memories whose
    `valid_to` has passed are excluded — even if the hourly TTL cleanup
    pass has not yet tombstoned them, so retrieval honours the bound
    immediately on expiry instead of waiting up to an hour for the
    backstop sweep. See `server.services.memory_ttl` for the cleanup
    side and the v0.7 memory-TTL design notes.
    """
    return stmt.where(or_(MemoryRow.valid_to.is_(None), MemoryRow.valid_to > func.now()))


def _tenant_filter(stmt, column, tenant_id: str | None):
    """Apply tenant filter to a query when tenant_id is set."""
    if tenant_id is not None:
        return stmt.where(column == tenant_id)
    return stmt


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


async def insert_episode(session: AsyncSession, row: EpisodeRow) -> EpisodeRow:
    """Insert an episode, idempotently.

    With an `idempotency_key`, a row that collides with an existing
    (tenant_id, subject_id, idempotency_key) is NOT duplicated — the existing
    episode is returned unchanged. This makes re-ingest (re-running a seed,
    retrying a webhook) a no-op instead of inflating the subject. Episodes
    without a key are always inserted (live-chat ingest, legacy clients).
    """
    if not row.idempotency_key:
        session.add(row)
        await session.flush()
        return row
    # The unique index is the atomic arbiter — a SAVEPOINT lets us catch the
    # conflict without poisoning the surrounding transaction, then return the
    # winner. This is race-safe under the connectors' concurrent ingest.
    try:
        async with session.begin_nested():
            session.add(row)
            await session.flush()
        return row
    except IntegrityError:
        if row in session:
            session.expunge(row)
        existing = (
            await session.execute(
                select(EpisodeRow).where(
                    EpisodeRow.subject_id == row.subject_id,
                    EpisodeRow.idempotency_key == row.idempotency_key,
                    EpisodeRow.tenant_id.is_(None)
                    if row.tenant_id is None
                    else EpisodeRow.tenant_id == row.tenant_id,
                )
            )
        ).scalars().first()
        if existing is None:
            raise  # a non-idempotency constraint failed — don't swallow it
        return existing


async def list_episodes_by_subject(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
    newest_first: bool = False,
) -> Sequence[EpisodeRow]:
    # Order by `occurred_at` (the source-event time) so backfilled episodes
    # land in their real chronological position. `created_at` is the
    # tiebreak so simultaneous events from a single ingest batch retain
    # their insertion order in the response.
    #
    # `newest_first=True` selects the most-recent `limit` episodes (then
    # returns them in ascending chronological order). Callers that want a
    # bounded "recent activity" window MUST use it: with the default ascending
    # order, a `limit` smaller than the subject's lifetime episode count
    # returns the OLDEST `limit` rows — the opposite of recent.
    #
    # `offset` is applied within that same ordering, i.e. it pages through
    # the newest-first window when `newest_first=True`, or the
    # oldest-first window otherwise — it does not change which end of the
    # timeline paging starts from.
    if newest_first:
        stmt = (
            select(EpisodeRow)
            .where(EpisodeRow.subject_id == subject_id)
            .order_by(EpisodeRow.occurred_at.desc(), EpisodeRow.created_at.desc(), EpisodeRow.id.desc())
            .limit(limit)
            .offset(offset)
        )
        stmt = _tenant_filter(stmt, EpisodeRow.tenant_id, tenant_id)
        result = await session.execute(stmt)
        rows = list(result.scalars().all())
        rows.reverse()  # newest-`limit` fetched desc → return ascending
        return rows

    stmt = (
        select(EpisodeRow)
        .where(EpisodeRow.subject_id == subject_id)
        .order_by(EpisodeRow.occurred_at.asc(), EpisodeRow.created_at.asc(), EpisodeRow.id.asc())
        .limit(limit)
        .offset(offset)
    )
    stmt = _tenant_filter(stmt, EpisodeRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.scalars().all()


async def list_episodes_by_session(
    session: AsyncSession,
    subject_id: str,
    session_id: str,
    *,
    tenant_id: str | None = None,
    limit: int = 100,
) -> Sequence[EpisodeRow]:
    # Scoped to a single session so callers (e.g. handoff) always see the
    # active session's episodes regardless of how many lifetime episodes the
    # subject has. Same chronological ordering contract as
    # `list_episodes_by_subject`: `occurred_at` first so backfilled events
    # land in their real position, `created_at` as the ingest-order tiebreak.
    stmt = (
        select(EpisodeRow)
        .where(EpisodeRow.subject_id == subject_id)
        .where(EpisodeRow.session_id == session_id)
        .order_by(EpisodeRow.occurred_at.asc(), EpisodeRow.created_at.asc(), EpisodeRow.id.asc())
        .limit(limit)
    )
    stmt = _tenant_filter(stmt, EpisodeRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.scalars().all()


async def list_uncompiled_episodes(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    limit: int = 500,
) -> Sequence[EpisodeRow]:
    """Fetch episodes that have never been compiled.

    The compile route paginates with `settings.compile_batch_size`; this
    `limit` default is only the floor for ad-hoc callers (tests, scripts).
    """
    stmt = (
        select(EpisodeRow)
        .where(EpisodeRow.subject_id == subject_id)
        .where(EpisodeRow.last_compiled_at.is_(None))
        .order_by(EpisodeRow.created_at.asc())
        .limit(limit)
    )
    stmt = _tenant_filter(stmt, EpisodeRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.scalars().all()


async def count_uncompiled_episodes(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
) -> int:
    """Count episodes for a subject that have not been compiled yet.

    Powers the `remaining_episodes` / `has_more` drain signal on
    `CompileMemoriesResponse` (issue #134). Bare `COUNT(*)` — no cap;
    the count is authoritative and reflects the real backlog.
    """
    stmt = (
        select(func.count())
        .select_from(EpisodeRow)
        .where(EpisodeRow.subject_id == subject_id)
        .where(EpisodeRow.last_compiled_at.is_(None))
    )
    stmt = _tenant_filter(stmt, EpisodeRow.tenant_id, tenant_id)
    return await session.scalar(stmt) or 0


async def mark_episodes_compiled(
    session: AsyncSession,
    episode_ids: list[uuid.UUID],
) -> None:
    """Mark episodes as compiled so they won't be reprocessed."""
    if not episode_ids:
        return
    stmt = (
        update(EpisodeRow)
        .where(EpisodeRow.id.in_(episode_ids))
        .values(last_compiled_at=text("now()"))
    )
    await session.execute(stmt)


async def get_episodes_by_ids(
    session: AsyncSession,
    ids: list[uuid.UUID],
    *,
    tenant_id: str | None = None,
) -> Sequence[EpisodeRow]:
    if not ids:
        return []
    stmt = select(EpisodeRow).where(EpisodeRow.id.in_(ids))
    stmt = _tenant_filter(stmt, EpisodeRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.scalars().all()


async def delete_episodes_by_subject(
    session: AsyncSession, subject_id: str, *, tenant_id: str | None = None
) -> int:
    stmt = delete(EpisodeRow).where(EpisodeRow.subject_id == subject_id)
    stmt = _tenant_filter(stmt, EpisodeRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.rowcount  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Memories
# ---------------------------------------------------------------------------


async def insert_memory(session: AsyncSession, row: MemoryRow) -> MemoryRow:
    session.add(row)
    await session.flush()
    return row


async def search_memories(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    kind: str | None = None,
    query: str | None = None,
    limit: int = 20,
) -> Sequence[MemoryRow]:
    stmt = (
        select(MemoryRow)
        .where(MemoryRow.subject_id == subject_id)
        .where(MemoryRow.status == "active")
    )
    stmt = _unexpired(stmt)
    stmt = _tenant_filter(stmt, MemoryRow.tenant_id, tenant_id)
    if kind:
        stmt = stmt.where(MemoryRow.kind == kind)
    if query:
        stmt = stmt.where(MemoryRow.content.ilike(f"%{query}%"))
    stmt = stmt.order_by(MemoryRow.created_at.desc(), MemoryRow.id.desc()).limit(limit)
    result = await session.execute(stmt)
    return result.scalars().all()


async def list_memories_by_subject(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> Sequence[MemoryRow]:
    stmt = (
        select(MemoryRow)
        .where(MemoryRow.subject_id == subject_id)
        .order_by(MemoryRow.created_at.asc(), MemoryRow.id.asc())
        .limit(limit)
        .offset(offset)
    )
    stmt = _tenant_filter(stmt, MemoryRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.scalars().all()


async def delete_memories_by_subject(
    session: AsyncSession, subject_id: str, *, tenant_id: str | None = None
) -> int:
    stmt = delete(MemoryRow).where(MemoryRow.subject_id == subject_id)
    stmt = _tenant_filter(stmt, MemoryRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.rowcount  # type: ignore[return-value]


async def list_active_memories_by_subject(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    limit: int = 500,
) -> Sequence[MemoryRow]:
    """Fetch active memories for a subject (for conflict resolution)."""
    stmt = (
        select(MemoryRow)
        .where(MemoryRow.subject_id == subject_id)
        .where(MemoryRow.status == "active")
        .order_by(MemoryRow.created_at.asc(), MemoryRow.id.asc())
        .limit(limit)
    )
    stmt = _unexpired(stmt)
    stmt = _tenant_filter(stmt, MemoryRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.scalars().all()


async def mark_memories_superseded(
    session: AsyncSession,
    memory_ids: list[uuid.UUID],
) -> None:
    """Mark memories as superseded (conflict resolution)."""
    if not memory_ids:
        return
    stmt = (
        update(MemoryRow)
        .where(MemoryRow.id.in_(memory_ids))
        .values(status="superseded", updated_at=text("now()"))
    )
    await session.execute(stmt)


async def superseded_only_episode_ids(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
) -> set[uuid.UUID]:
    """Episode IDs whose backing memories are ALL non-active.

    Phase-1 leak fix: a correctly-superseded fact must not resurface verbatim
    via its originating episode in the "Recent interactions" context section.
    An episode that still backs at least one *active* memory — or that backs no
    memory at all — is NOT returned (i.e. it is kept), so partially-superseded
    and raw uncompiled episodes are preserved exactly as today. Only an episode
    referenced solely by superseded/tombstoned memories is reported obsolete.

    Pure, deterministic set logic over committed rows — no embeddings, no LLM,
    stub-safe — so it suppresses leaks regardless of which compile batch
    produced the supersession. Read-only.
    """
    stmt = select(MemoryRow.status, MemoryRow.source_episode_ids).where(
        MemoryRow.subject_id == subject_id
    )
    stmt = _tenant_filter(stmt, MemoryRow.tenant_id, tenant_id)
    result = await session.execute(stmt)

    alive: set[uuid.UUID] = set()
    dead: set[uuid.UUID] = set()
    for status, episode_ids in result.all():
        if not episode_ids:
            continue
        (alive if status == "active" else dead).update(episode_ids)
    # Referenced only by non-active memories, never by an active one.
    return dead - alive


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


async def search_memories_by_embedding(
    session: AsyncSession,
    subject_id: str,
    query_embedding: list[float],
    *,
    tenant_id: str | None = None,
    kind: str | None = None,
    limit: int = 20,
) -> list[tuple[MemoryRow, float]]:
    """Find memories by cosine distance. Returns (row, distance) tuples.

    Uses pgvector's native `<=>` cosine-distance operator. The HNSW index
    on `memories.embedding` (created in migration 0013) makes this an
    indexed nearest-neighbor lookup — sub-millisecond at our corpus
    sizes — instead of the previous fetch-all-and-cosine-in-Python path.

    Distance is in [0, 2] where 0 is identical and 2 is opposite (cosine
    distance, not similarity); callers convert to [0, 1] similarity if
    they need it. Lower is better; the query orders ascending and limits.
    """
    distance_expr = MemoryRow.embedding.cosine_distance(query_embedding)
    stmt = (
        select(MemoryRow, distance_expr.label("distance"))
        .where(MemoryRow.subject_id == subject_id)
        .where(MemoryRow.status == "active")
        .where(MemoryRow.embedding.isnot(None))
    )
    stmt = _unexpired(stmt)
    stmt = _tenant_filter(stmt, MemoryRow.tenant_id, tenant_id)
    if kind:
        stmt = stmt.where(MemoryRow.kind == kind)
    stmt = stmt.order_by(distance_expr).limit(limit)
    result = await session.execute(stmt)
    return [(row, float(distance)) for row, distance in result.all()]


async def search_memories_hybrid(
    session: AsyncSession,
    subject_id: str,
    query_text: str,
    query_embedding: list[float],
    *,
    tenant_id: str | None = None,
    kind: str | None = None,
    limit: int = 20,
    semantic_weight: float = 1.0,
    bm25_weight: float = 1.0,
    entity_weight: float = 1.0,
    use_entity_boost: bool = True,
    candidate_multiplier: int = 4,
    entity_max_distance: float = 0.5,
) -> list[tuple[MemoryRow, float, dict[str, float]]]:
    """Hybrid retrieval: semantic cosine + Postgres BM25 ts_rank_cd +
    cross-session entity boost, blended additively after sigmoid-style
    normalization of each signal. Returns (row, final_score,
    score_breakdown) tuples, ordered by final_score descending.

    Why this exists: blending THREE retrieval signals — semantic
    similarity, BM25 keyword rank, and an entity boost — recalls more
    of the right memories than any single signal alone. Pure-pgvector
    retrieval — what `search_memories_by_embedding` does — under-finds
    memories whose surface form matches the question verbatim AND
    memories that share an entity with the question but weren't picked
    up by either semantic or BM25. The entity-boost lane is what
    bridges across sessions: an entity ("Caroline", "the navy blazer")
    is the natural join key when a question references something
    mentioned in a memory the embedding cosine missed.

    Scoring:
      semantic_sim   = 1 - cosine_distance / 2     # [0, 1], 1 = identical
      bm25_norm      = ts_rank / (1 + ts_rank)     # [0, 1), sigmoid-ish
      entity_norm    = max boost from any matched entity              # [0, 1]
                       (boost per entity = 1 - cosine_distance, capped)
      final_score    = (semantic_weight * semantic_sim +
                        bm25_weight    * bm25_norm    +
                        (entity_weight * entity_norm if use_entity_boost else 0)) /
                       (sum of active weights)

    The defaults give equal weighting to semantic, BM25, and entity.
    Operators can shift the blend via the weight kwargs — useful for
    ablations + bench sweeps. Setting `use_entity_boost=False` reverts
    to pure Phase-1 (semantic + BM25) behavior, which is what the
    fallback path uses when the entity store is empty for the subject.

    Entity boost mechanics (Phase 3):
      1. Look up entities for the subject whose embedding cosine is
         within `entity_max_distance` of the query embedding.
      2. For each matched entity, collect its `linked_memory_ids`.
      3. For each memory in the candidate set, the entity score is the
         MAX boost across all matched entities pointing at it (so an
         entity that's an exact query match boosts more than one that's
         a paraphrase). Capped at 1.0.
      4. Memories pointed at by a matched entity that AREN'T in the
         semantic+BM25 candidate set get pulled in too (this is the
         cross-session connective tissue that lifts multi-session recall).

    `candidate_multiplier` controls how many candidates we draw from
    each lane before re-blending (default 4× the final `limit`).
    """
    candidate_limit = max(limit * candidate_multiplier, limit)

    # ── Semantic candidates ─────────────────────────────────────────────
    distance_expr = MemoryRow.embedding.cosine_distance(query_embedding)
    sem_stmt = (
        select(MemoryRow, distance_expr.label("distance"))
        .where(MemoryRow.subject_id == subject_id)
        .where(MemoryRow.status == "active")
        .where(MemoryRow.embedding.isnot(None))
    )
    sem_stmt = _unexpired(sem_stmt)
    sem_stmt = _tenant_filter(sem_stmt, MemoryRow.tenant_id, tenant_id)
    if kind:
        sem_stmt = sem_stmt.where(MemoryRow.kind == kind)
    sem_stmt = sem_stmt.order_by(distance_expr).limit(candidate_limit)
    sem_rows = (await session.execute(sem_stmt)).all()

    # ── BM25 candidates ─────────────────────────────────────────────────
    # ts_rank_cd over the generated content_tsvector column (migration 0027).
    # `plainto_tsquery` is the safe-for-untrusted-input variant — it
    # treats the query as a phrase, no operator injection.
    bm25_rank_expr = func.ts_rank_cd(
        text("memories.content_tsvector"),
        func.plainto_tsquery("english", query_text),
    ).label("bm25_rank")
    bm25_filter = text("memories.content_tsvector @@ plainto_tsquery('english', :q)")
    bm25_stmt = (
        select(MemoryRow, bm25_rank_expr)
        .where(MemoryRow.subject_id == subject_id)
        .where(MemoryRow.status == "active")
        .where(bm25_filter)
    )
    bm25_stmt = _unexpired(bm25_stmt)
    bm25_stmt = _tenant_filter(bm25_stmt, MemoryRow.tenant_id, tenant_id)
    if kind:
        bm25_stmt = bm25_stmt.where(MemoryRow.kind == kind)
    bm25_stmt = bm25_stmt.order_by(bm25_rank_expr.desc()).limit(candidate_limit)
    bm25_rows = (await session.execute(bm25_stmt, {"q": query_text})).all()

    # ── Entity boost lane (Phase 3) ─────────────────────────────────────
    # Look up subject entities matching the query embedding; for each,
    # collect the memories they point at. A memory's entity boost is the
    # max (1 - distance) across all matched entities pointing at it.
    entity_boost_by_memory_id: dict[uuid.UUID, float] = {}
    pulled_in_by_entity: dict[uuid.UUID, MemoryRow] = {}
    if use_entity_boost:
        entity_matches = await search_entities_by_embedding(
            session,
            subject_id,
            query_embedding,
            tenant_id=tenant_id,
            limit=candidate_limit,
            max_distance=entity_max_distance,
        )
        if entity_matches:
            # Per-entity boost (higher when query is closer to the entity).
            for entity_row, distance in entity_matches:
                per_entity_boost = max(0.0, 1.0 - float(distance))
                for mem_id in entity_row.linked_memory_ids:
                    prior = entity_boost_by_memory_id.get(mem_id, 0.0)
                    if per_entity_boost > prior:
                        entity_boost_by_memory_id[mem_id] = per_entity_boost
            # Pull in entity-only memories not already in semantic / BM25
            # candidate sets. (One DB roundtrip for all of them.)
            sem_ids = {row.id for row, _d in sem_rows}
            bm25_ids = {row.id for row, _r in bm25_rows}
            missing_ids = [
                mid
                for mid in entity_boost_by_memory_id
                if mid not in sem_ids and mid not in bm25_ids
            ]
            if missing_ids:
                pull_stmt = (
                    select(MemoryRow)
                    .where(MemoryRow.id.in_(missing_ids))
                    .where(MemoryRow.subject_id == subject_id)
                    .where(MemoryRow.status == "active")
                )
                pull_stmt = _unexpired(pull_stmt)
                pull_stmt = _tenant_filter(pull_stmt, MemoryRow.tenant_id, tenant_id)
                if kind:
                    pull_stmt = pull_stmt.where(MemoryRow.kind == kind)
                pull_stmt = pull_stmt.limit(candidate_limit)
                for row in (await session.execute(pull_stmt)).scalars().all():
                    pulled_in_by_entity[row.id] = row

    # ── Build the merged candidate set ──────────────────────────────────
    # Map id → (row, semantic_sim, bm25_norm, entity_norm). Each lane
    # contributes independently; missing-signal scores stay 0.
    by_id: dict[uuid.UUID, dict] = {}
    for row, distance in sem_rows:
        sim = max(0.0, 1.0 - float(distance) / 2.0)
        by_id[row.id] = {"row": row, "semantic": sim, "bm25": 0.0, "entity": 0.0}
    for row, raw_rank in bm25_rows:
        norm = float(raw_rank) / (1.0 + float(raw_rank))
        if row.id in by_id:
            by_id[row.id]["bm25"] = norm
        else:
            by_id[row.id] = {"row": row, "semantic": 0.0, "bm25": norm, "entity": 0.0}
    for mem_id, row in pulled_in_by_entity.items():
        if mem_id not in by_id:
            by_id[mem_id] = {"row": row, "semantic": 0.0, "bm25": 0.0, "entity": 0.0}
    for mem_id, boost in entity_boost_by_memory_id.items():
        if mem_id in by_id:
            by_id[mem_id]["entity"] = boost

    # ── Blend + rank ────────────────────────────────────────────────────
    active_entity_w = entity_weight if use_entity_boost else 0.0
    weight_sum = max(semantic_weight + bm25_weight + active_entity_w, 1e-9)
    scored: list[tuple[MemoryRow, float, dict[str, float]]] = []
    for entry in by_id.values():
        final = (
            semantic_weight * entry["semantic"]
            + bm25_weight * entry["bm25"]
            + active_entity_w * entry["entity"]
        ) / weight_sum
        scored.append(
            (
                entry["row"],
                final,
                {
                    "semantic": entry["semantic"],
                    "bm25": entry["bm25"],
                    "entity": entry["entity"],
                    "combined": final,
                },
            )
        )
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:limit]


# ---------------------------------------------------------------------------
# Subject entities (Phase 2 — cross-session entity-boost retrieval)
# ---------------------------------------------------------------------------


async def upsert_entity_with_link(
    session: AsyncSession,
    *,
    subject_id: str,
    tenant_id: str | None,
    entity_text: str,
    entity_normalized: str,
    entity_kind: str | None,
    embedding: list[float] | None,
    memory_id: uuid.UUID,
    dedup_cosine_threshold: float = 0.95,
) -> SubjectEntityRow:
    """Insert or merge an entity row for (subject_id, normalized text /
    cosine-similar embedding), appending memory_id to linked_memory_ids.

    Dedup order:
      1. Exact match on (subject_id, entity_normalized) — cheapest.
         If found, append memory_id and return.
      2. If miss AND embedding provided, cosine-distance probe against
         all entities for the same subject; merge if any row is within
         `dedup_cosine_threshold` (default 0.95).
      3. Otherwise insert a new row with [memory_id] as the initial
         linkage.

    The cosine probe uses pgvector's `<=>` operator and reads the HNSW
    index from migration 0028 — sub-millisecond at our scales. The
    same-normalization fast path skips the probe entirely for the
    ~80% of duplicates that are trivial casing/whitespace variants.

    Idempotent on repeated memory_id linkage: if memory_id is already
    in linked_memory_ids, the append is a no-op (Postgres array
    `array_append` would duplicate; we guard with `NOT (memory_id =
    ANY(linked_memory_ids))`).
    """
    # Step 1: exact-match dedup
    exact_stmt = select(SubjectEntityRow).where(
        SubjectEntityRow.subject_id == subject_id,
        SubjectEntityRow.entity_normalized == entity_normalized,
    )
    if tenant_id is not None:
        exact_stmt = exact_stmt.where(SubjectEntityRow.tenant_id == tenant_id)
    exact = (await session.execute(exact_stmt)).scalar_one_or_none()
    if exact is not None:
        if memory_id not in exact.linked_memory_ids:
            # SQLAlchemy doesn't detect mutations to ARRAY columns
            # in-place; rebuild + reassign so the update flushes.
            exact.linked_memory_ids = [*exact.linked_memory_ids, memory_id]
        return exact

    # Step 2: semantic dedup (only if we have an embedding to compare with)
    if embedding is not None:
        distance_expr = SubjectEntityRow.embedding.cosine_distance(embedding)
        near_stmt = (
            select(SubjectEntityRow, distance_expr.label("distance"))
            .where(SubjectEntityRow.subject_id == subject_id)
            .where(SubjectEntityRow.embedding.isnot(None))
        )
        if tenant_id is not None:
            near_stmt = near_stmt.where(SubjectEntityRow.tenant_id == tenant_id)
        near_stmt = near_stmt.order_by(distance_expr).limit(1)
        near_row = (await session.execute(near_stmt)).first()
        if near_row is not None:
            existing_row, distance = near_row
            # Cosine distance ≤ (1 - threshold) ⇒ similarity ≥ threshold.
            if float(distance) <= (1.0 - dedup_cosine_threshold):
                if memory_id not in existing_row.linked_memory_ids:
                    existing_row.linked_memory_ids = [
                        *existing_row.linked_memory_ids,
                        memory_id,
                    ]
                return existing_row

    # Step 3: insert fresh
    fresh = SubjectEntityRow(
        subject_id=subject_id,
        tenant_id=tenant_id,
        entity_text=entity_text,
        entity_normalized=entity_normalized,
        entity_kind=entity_kind,
        embedding=embedding,
        linked_memory_ids=[memory_id],
    )
    session.add(fresh)
    return fresh


async def search_entities_by_embedding(
    session: AsyncSession,
    subject_id: str,
    query_embedding: list[float],
    *,
    tenant_id: str | None = None,
    limit: int = 20,
    max_distance: float = 0.5,
) -> list[tuple[SubjectEntityRow, float]]:
    """Find entities for a subject whose embedding is close to the query.
    Returns (entity_row, cosine_distance) tuples ordered by distance asc.

    `max_distance` is the cosine-distance cutoff (default 0.5 → cosine
    similarity ≥ 0.5). Tighter than 0.5 misses paraphrased entities;
    looser pollutes the boost lane with weak matches. Phase 3 retrieval
    callers can tune this per-bench.
    """
    distance_expr = SubjectEntityRow.embedding.cosine_distance(query_embedding)
    stmt = (
        select(SubjectEntityRow, distance_expr.label("distance"))
        .where(SubjectEntityRow.subject_id == subject_id)
        .where(SubjectEntityRow.embedding.isnot(None))
        .where(distance_expr <= max_distance)
    )
    if tenant_id is not None:
        stmt = stmt.where(SubjectEntityRow.tenant_id == tenant_id)
    stmt = stmt.order_by(distance_expr).limit(limit)
    rows = (await session.execute(stmt)).all()
    return [(row, float(dist)) for row, dist in rows]


async def delete_entities_by_subject(
    session: AsyncSession, subject_id: str, *, tenant_id: str | None = None
) -> int:
    """Wipe every entity row for a subject. Mirrors
    `delete_memories_by_subject` — used by the same cleanup path so
    delete_subject(subject_id) removes both memories AND entities."""
    stmt = delete(SubjectEntityRow).where(SubjectEntityRow.subject_id == subject_id)
    stmt = _tenant_filter(stmt, SubjectEntityRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.rowcount  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Subject listing
# ---------------------------------------------------------------------------


async def list_subjects(
    session: AsyncSession,
    *,
    tenant_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Return distinct subject IDs with episode and memory counts."""
    # Base episode query, tenant-scoped
    ep_base = select(EpisodeRow.subject_id, func.count().label("episode_count"))
    if tenant_id is not None:
        ep_base = ep_base.where(EpisodeRow.tenant_id == tenant_id)
    ep_count = ep_base.group_by(EpisodeRow.subject_id).subquery()

    # Base memory query, tenant-scoped
    mem_base = select(MemoryRow.subject_id, func.count().label("memory_count"))
    if tenant_id is not None:
        mem_base = mem_base.where(MemoryRow.tenant_id == tenant_id)
    mem_count = mem_base.group_by(MemoryRow.subject_id).subquery()

    # UNION of subject_ids from both tables (tenant-scoped)
    ep_subjects = select(EpisodeRow.subject_id)
    mem_subjects = select(MemoryRow.subject_id)
    if tenant_id is not None:
        ep_subjects = ep_subjects.where(EpisodeRow.tenant_id == tenant_id)
        mem_subjects = mem_subjects.where(MemoryRow.tenant_id == tenant_id)
    all_subjects = ep_subjects.union(mem_subjects).subquery()

    stmt = (
        select(
            all_subjects.c.subject_id,
            func.coalesce(ep_count.c.episode_count, 0).label("episode_count"),
            func.coalesce(mem_count.c.memory_count, 0).label("memory_count"),
        )
        .outerjoin(ep_count, all_subjects.c.subject_id == ep_count.c.subject_id)
        .outerjoin(mem_count, all_subjects.c.subject_id == mem_count.c.subject_id)
        .where(all_subjects.c.subject_id.not_like("_snapshot/%"))
        .where(all_subjects.c.subject_id.not_like("_bootstrap_tmp/%"))
        .order_by(all_subjects.c.subject_id)
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return [
        {
            "subject_id": row.subject_id,
            "episode_count": row.episode_count,
            "memory_count": row.memory_count,
        }
        for row in result.all()
    ]


async def count_subjects(
    session: AsyncSession,
    *,
    tenant_id: str | None = None,
) -> int:
    """Return the unpaginated count of public subject IDs."""
    ep_subjects = select(EpisodeRow.subject_id)
    mem_subjects = select(MemoryRow.subject_id)
    if tenant_id is not None:
        ep_subjects = ep_subjects.where(EpisodeRow.tenant_id == tenant_id)
        mem_subjects = mem_subjects.where(MemoryRow.tenant_id == tenant_id)
    all_subjects = ep_subjects.union(mem_subjects).subquery()

    stmt = (
        select(func.count())
        .select_from(all_subjects)
        .where(all_subjects.c.subject_id.not_like("_snapshot/%"))
        .where(all_subjects.c.subject_id.not_like("_bootstrap_tmp/%"))
    )
    total = await session.scalar(stmt)
    return int(total or 0)


# ---------------------------------------------------------------------------
# Resolutions
# ---------------------------------------------------------------------------


async def upsert_resolution(session: AsyncSession, row: ResolutionRow) -> ResolutionRow:
    """Insert or update a resolution (keyed by subject_id + session_id + tenant_id)."""
    # Check for existing resolution on same subject+session
    stmt = select(ResolutionRow).where(
        ResolutionRow.subject_id == row.subject_id,
        ResolutionRow.session_id == row.session_id,
    )
    if row.tenant_id is not None:
        stmt = stmt.where(ResolutionRow.tenant_id == row.tenant_id)
    else:
        stmt = stmt.where(ResolutionRow.tenant_id.is_(None))

    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing:
        existing.status = row.status
        existing.resolution_summary = row.resolution_summary
        existing.resolved_at = row.resolved_at
        existing.metadata_ = row.metadata_
        await session.flush()
        return existing

    session.add(row)
    await session.flush()
    return row


async def list_resolutions(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> Sequence[ResolutionRow]:
    """List resolutions for a subject, optionally filtered by status."""
    stmt = select(ResolutionRow).where(ResolutionRow.subject_id == subject_id)
    stmt = _tenant_filter(stmt, ResolutionRow.tenant_id, tenant_id)
    if status:
        stmt = stmt.where(ResolutionRow.status == status)
    # `id` breaks ties between rows sharing an updated_at (bulk writes in one
    # transaction share a single now()), so OFFSET pages never skip or repeat rows.
    stmt = (
        stmt.order_by(ResolutionRow.updated_at.desc(), ResolutionRow.id.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await session.execute(stmt)
    return result.scalars().all()


async def get_resolved_session_ids(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
) -> set[str]:
    """Return session_ids that are marked as resolved for a subject."""
    stmt = (
        select(ResolutionRow.session_id)
        .where(ResolutionRow.subject_id == subject_id)
        .where(ResolutionRow.status == "resolved")
    )
    stmt = _tenant_filter(stmt, ResolutionRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return {row[0] for row in result.all()}


async def get_open_session_ids(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
) -> set[str]:
    """Return session_ids that have an open (unresolved) resolution."""
    stmt = (
        select(ResolutionRow.session_id)
        .where(ResolutionRow.subject_id == subject_id)
        .where(ResolutionRow.status == "open")
    )
    stmt = _tenant_filter(stmt, ResolutionRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return {row[0] for row in result.all()}


async def delete_resolutions_by_subject(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
) -> int:
    """Delete all resolutions for a subject (used in subject deletion)."""
    stmt = delete(ResolutionRow).where(ResolutionRow.subject_id == subject_id)
    stmt = _tenant_filter(stmt, ResolutionRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.rowcount


# ---------------------------------------------------------------------------
# Health cache
# ---------------------------------------------------------------------------


async def get_health_cache(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
) -> SubjectHealthCacheRow | None:
    """Get cached health state for a subject, scoped to the tenant.

    Without the tenant filter a caller could read another tenant's cached
    health for the same subject_id (the cache row identity is
    (tenant_id, subject_id); see migration 0024)."""
    stmt = select(SubjectHealthCacheRow).where(SubjectHealthCacheRow.subject_id == subject_id)
    stmt = _tenant_filter(stmt, SubjectHealthCacheRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def upsert_health_cache(
    session: AsyncSession,
    subject_id: str,
    state: str,
    score: int,
    *,
    tenant_id: str | None = None,
) -> None:
    """Update or insert cached health state."""
    from datetime import datetime, timezone

    # Tenant-scoped lookup: without it, tenant B's upsert would find and
    # overwrite tenant A's row for the same subject_id instead of inserting
    # its own.
    existing = await get_health_cache(session, subject_id, tenant_id=tenant_id)
    if existing:
        existing.last_state = state
        existing.last_score = score
        existing.updated_at = datetime.now(timezone.utc)
    else:
        row = SubjectHealthCacheRow(
            subject_id=subject_id,
            tenant_id=tenant_id,
            last_state=state,
            last_score=score,
            updated_at=datetime.now(timezone.utc),
        )
        session.add(row)
    await session.flush()


async def delete_health_cache_by_subject(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
) -> None:
    """Delete health cache for a subject (used in subject deletion), scoped to
    the tenant so one tenant's deletion can't drop another tenant's cache row
    for the same subject_id."""
    stmt = delete(SubjectHealthCacheRow).where(SubjectHealthCacheRow.subject_id == subject_id)
    stmt = _tenant_filter(stmt, SubjectHealthCacheRow.tenant_id, tenant_id)
    await session.execute(stmt)


# ---------------------------------------------------------------------------
# Query embedding cache (cross-machine L2)
# ---------------------------------------------------------------------------


async def query_cache_get(
    session: AsyncSession,
    text_key: str,
    model: str,
) -> list[float] | None:
    """Return a cached query embedding if present and not expired, else None."""
    from datetime import datetime, timezone

    stmt = (
        select(QueryEmbeddingCacheRow.embedding)
        .where(QueryEmbeddingCacheRow.text_key == text_key)
        .where(QueryEmbeddingCacheRow.model == model)
        .where(QueryEmbeddingCacheRow.expires_at > datetime.now(timezone.utc))
    )
    result = await session.execute(stmt)
    row = result.first()
    if row is None:
        return None
    embedding = row[0]
    # pgvector adapter returns numpy.ndarray; coerce to list[float] so the
    # caller doesn't have to think about it.
    return [float(x) for x in embedding] if embedding is not None else None


async def query_cache_set(
    session: AsyncSession,
    text_key: str,
    model: str,
    embedding: list[float],
    ttl_seconds: int,
) -> None:
    """Upsert an entry into the query embedding cache.

    Also opportunistically prunes entries that have been expired for at
    least 7 days, to bound table growth without needing a cron job.
    """
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=ttl_seconds)

    # Upsert by (text_key, model). Refresh expires_at on hit so popular
    # queries stay warm.
    upsert = text(
        """
        INSERT INTO query_embedding_cache (text_key, model, embedding, expires_at, created_at)
        VALUES (:text_key, :model, CAST(:embedding AS vector), :expires_at, :created_at)
        ON CONFLICT (text_key, model) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            expires_at = EXCLUDED.expires_at
        """
    )
    # pgvector accepts the bracketed-list TEXT format on input.
    embedding_str = "[" + ",".join(repr(float(x)) for x in embedding) + "]"
    await session.execute(
        upsert,
        {
            "text_key": text_key,
            "model": model,
            "embedding": embedding_str,
            "expires_at": expires_at,
            "created_at": now,
        },
    )

    # Opportunistic cleanup — drop entries that have been expired for a
    # week. Bounded work per call (DELETE ... WHERE indexed column < $),
    # no cron required.
    cleanup_threshold = now - timedelta(days=7)
    cleanup_stmt = delete(QueryEmbeddingCacheRow).where(
        QueryEmbeddingCacheRow.expires_at < cleanup_threshold
    )
    await session.execute(cleanup_stmt)


# ---------------------------------------------------------------------------
# Receipts (issue #49)
# ---------------------------------------------------------------------------


async def insert_receipt(session: AsyncSession, row: ReceiptRow) -> ReceiptRow:
    """Append-only insert. Service code is the sole writer; nothing in
    this repository module exposes UPDATE or DELETE against receipts.
    Operators wanting hard tamper-evidence should additionally restrict
    DB role privileges to INSERT+SELECT only."""
    session.add(row)
    await session.flush()
    return row


async def get_receipt_by_id(
    session: AsyncSession,
    receipt_id: str,
    *,
    tenant_id: str | None = None,
) -> ReceiptRow | None:
    """Fetch one receipt. Tenant-scoped when `tenant_id` is set so a
    tenant can never read another tenant's receipts even if it guesses
    the ULID."""
    stmt = select(ReceiptRow).where(ReceiptRow.receipt_id == receipt_id)
    stmt = _tenant_filter(stmt, ReceiptRow.tenant_id, tenant_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_receipts(
    session: AsyncSession,
    subject_id: str,
    *,
    tenant_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    cursor: str | None = None,
    limit: int = 50,
    include_tombstoned: bool = False,
) -> Sequence[ReceiptRow]:
    """List receipts for a subject, newest first.

    Pagination is cursor-based (not offset) — ULIDs sort lexically by
    creation time, so the cursor is simply the last `receipt_id` from
    the previous page. Offset pagination is unsafe for an append-only
    audit log where rows are continuously inserted.

    ``include_tombstoned`` defaults to False so the list returns the
    active audit trail. Tombstoned (retention-retired) receipts remain
    individually addressable via `get_receipt_by_id` for forensic
    lookup of "a receipt with id X was emitted and later retired."
    """
    # Order by the SAME key the cursor predicate uses (`receipt_id < cursor`
    # below). The ULID receipt_id encodes creation time, so receipt_id-desc is
    # newest-first as the docstring promises — and keyset pagination is only
    # sound when ORDER BY and the cursor comparison agree. Ordering by
    # created_at (a different clock: DB commit time vs the app-side ULID) while
    # cursoring on receipt_id silently dropped or duplicated rows across pages
    # whenever the two orderings disagreed (clock skew, multi-replica, or two
    # receipts in the same millisecond). `since`/`until` still filter created_at.
    stmt = (
        select(ReceiptRow)
        .where(ReceiptRow.subject_id == subject_id)
        .order_by(ReceiptRow.receipt_id.desc())
        .limit(limit)
    )
    stmt = _tenant_filter(stmt, ReceiptRow.tenant_id, tenant_id)
    if since is not None:
        stmt = stmt.where(ReceiptRow.created_at >= since)
    if until is not None:
        stmt = stmt.where(ReceiptRow.created_at <= until)
    if cursor is not None:
        stmt = stmt.where(ReceiptRow.receipt_id < cursor)
    if not include_tombstoned:
        stmt = stmt.where(ReceiptRow.status == "active")
    result = await session.execute(stmt)
    return result.scalars().all()


# ---------------------------------------------------------------------------
# Tenant configs (issue #49 — used by emission-decision; issue #50 will add
# policy_mode and require_caller_identity keys to the same document)
# ---------------------------------------------------------------------------


async def get_tenant_config(
    session: AsyncSession,
    tenant_id: str,
) -> TenantConfigRow | None:
    """Fetch the tenant's config row, or None if it has never been set.

    Callers should treat None as "all defaults" — the missing-row case
    is the dominant case (every tenant starts without an explicit
    config) and must not be a hot-path error path.
    """
    stmt = select(TenantConfigRow).where(TenantConfigRow.tenant_id == tenant_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()