"""SQLAlchemy ORM table definitions."""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Computed, DateTime, Float, Index, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, validates

# Embedding dimensionality must match the embedder output and the `vector(N)`
# type in the schema. Fixed at 1536 (text-embedding-3-small), matching the
# `vector(1536)` columns created in migration 0013.
EMBEDDING_DIMENSIONS = 1536


class Base(DeclarativeBase):
    pass


class EpisodeRow(Base):
    __tablename__ = "episodes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    tenant_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    source: Mapped[str] = mapped_column(String(256), nullable=False)
    type: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    provenance: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    # Client-supplied de-dup key. A partial unique index on
    # (tenant_id, subject_id, idempotency_key) WHERE idempotency_key IS NOT NULL
    # (migration 0025, NULLS NOT DISTINCT) makes ingest idempotent: re-seeding a
    # repo collapses to the same rows instead of duplicating them. NULL = no key
    # = always inserted (live-chat ingest, legacy clients).
    idempotency_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # When the source event actually happened. Distinct from `created_at`
    # (= ingest time) so backfilled connectors (Slack history, GitHub
    # issues from 2024, Zendesk imports, …) can preserve their real
    # timeline. Server-defaults to now() so legacy clients that don't
    # supply the field keep working unchanged.
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_compiled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    __table_args__ = (
        Index("ix_episodes_subject_created", "subject_id", "created_at"),
        Index("ix_episodes_subject_occurred", "subject_id", "occurred_at"),
        # Idempotent ingest: one episode per (tenant_id, subject_id,
        # idempotency_key). Partial (keyless episodes are unconstrained) and
        # NULLS NOT DISTINCT (a NULL tenant_id is a value, so single-tenant rows
        # still de-dup). Declared here so create_all-based tests get it too;
        # migration 0025 builds the same index on existing databases.
        Index(
            "ix_episodes_idempotency",
            "tenant_id",
            "subject_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
            postgresql_nulls_not_distinct=True,
        ),
    )


class MemoryRow(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    tenant_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")

    @validates("content", "summary")
    def _strip_nul_bytes(self, _key: str, value: str | None) -> str | None:
        # Postgres text cannot store NUL (U+0000). An LLM occasionally emits one
        # in generated content, and a single stray byte would raise
        # CharacterNotInRepertoireError and 500 the ENTIRE compile batch (losing
        # every memory in it). Strip NUL — and only NUL — so one bad byte can't
        # sink the batch. Valid UTF-8 is otherwise preserved untouched.
        if isinstance(value, str) and "\x00" in value:
            return value.replace("\x00", "")
        return value
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source_episode_ids: Mapped[list] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    # Per-memory capability tags consumed by the policy layer (issue #50).
    # Empty list = untagged = policy default-allow. See migration 0018 for
    # the GIN index that makes `sensitivity_labels && '{pii}'` cheap on the
    # hot path.
    sensitivity_labels: Mapped[list[str]] = mapped_column(
        ARRAY(String()), nullable=False, default=list
    )
    # Heuristic-derived label hints (issue #158). Populated automatically by
    # the auto-labeling pipeline at ingest time. ORTHOGONAL to sensitivity_labels:
    # the policy evaluator does not read this column and no destructive behaviour
    # (retention, redaction, refusal) keys off it. It exists so operators can
    # surface and promote suggestions into authoritative labels deliberately.
    # See migration 0022 for the GIN index backing the admin review endpoint.
    suggested_labels: Mapped[list[str]] = mapped_column(
        ARRAY(String()), nullable=False, default=list
    )
    # Stored as pgvector `vector(EMBEDDING_DIMENSIONS)` since migration 0013.
    # Reads/writes happen as `list[float]` — the pgvector SQLAlchemy adapter
    # serializes/deserializes transparently. Cosine search uses the SQL `<=>`
    # operator via repositories.search_memories_by_embedding (no Python-side
    # parsing or compute on the hot path).
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=True
    )
    # Postgres-generated tsvector for the BM25 lane of hybrid retrieval.
    # Mirrors migration 0027 (GENERATED ALWAYS ... STORED) so a `create_all`
    # schema (tests) matches the migrated production schema.
    content_tsvector: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_memories_subject_kind", "subject_id", "kind"),
        Index("ix_memories_content_tsvector", "content_tsvector", postgresql_using="gin"),
    )


class SubjectEntityRow(Base):
    """Per-subject entity store powering cross-session entity-boost retrieval.

    Phase 2 of the cross-session retrieval improvements. Each row represents a distinct
    entity (proper noun / compound noun phrase) extracted from any
    compiled memory for the subject. `linked_memory_ids` is the N-to-M
    edge to the memories the entity appears in — the retrieval lane
    looks up entities matching the question, then boosts every memory in
    any matched entity's `linked_memory_ids`. This is the mechanism that
    lets retrieval bridge sessions when the same entity is mentioned in
    multiple compiled memories without semantic similarity having to
    catch every variation.

    Populated at compile time (post-memory-insert hook) and at one-off
    backfill for memories that pre-date this column. Schema mirrors
    migration 0028 — see the migration docstring for index choices and
    dedup behavior.
    """

    __tablename__ = "subject_entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    subject_id: Mapped[str] = mapped_column(String(256), nullable=False)
    tenant_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    entity_text: Mapped[str] = mapped_column(Text, nullable=False)
    entity_normalized: Mapped[str] = mapped_column(Text, nullable=False)
    entity_kind: Mapped[str | None] = mapped_column(String(32), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=True
    )
    linked_memory_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("ix_subject_entities_subject_id", "subject_id"),
        Index(
            "ix_subject_entities_subject_normalized",
            "subject_id",
            "entity_normalized",
        ),
    )


class WebhookEventRow(Base):
    """Persistent webhook delivery queue.

    Events are written synchronously during the request, then delivered
    asynchronously with exponential backoff. After max_attempts, events
    are marked as 'dead_letter'.
    """

    __tablename__ = "webhook_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    event: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending"
    )  # pending | delivered | dead_letter
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    http_status: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("ix_webhook_events_status_next", "status", "next_attempt_at"),)


class SubjectSnapshotRow(Base):
    """Subject snapshot metadata for bootstrap/restore operations."""

    __tablename__ = "subject_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    source_subject_id: Mapped[str] = mapped_column(String(256), nullable=False)
    episode_count: Mapped[int] = mapped_column(Integer, nullable=False)
    memory_count: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (Index("ix_snapshots_name_version", "name", "version", unique=True),)


class CompileJobRow(Base):
    """Durable compile job tracking — survives restarts."""

    __tablename__ = "compile_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    tenant_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    memories_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RateLimitHitRow(Base):
    """Fixed-window rate limit counter — distributed across workers via Postgres."""

    __tablename__ = "rate_limit_hits"

    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    window_start: Mapped[int] = mapped_column(Integer, primary_key=True)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ResolutionRow(Base):
    """Tracks resolution state of support sessions."""

    __tablename__ = "resolutions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    tenant_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    resolution_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class SubjectHealthCacheRow(Base):
    """Caches last-known health state per (tenant_id, subject_id) for alert
    deduplication."""

    __tablename__ = "subject_health_cache"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    tenant_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    last_state: Mapped[str] = mapped_column(String(32), nullable=False)
    last_score: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Composite unique index ix_subject_health_cache_tenant_subject (migration
    # 0024, NULLS NOT DISTINCT) enforces one row per (tenant_id, subject_id).
    # Not a SQLAlchemy UniqueConstraint because NULLS NOT DISTINCT isn't
    # natively expressible — the migration owns it.


class ReceiptRow(Base):
    """State-assembly receipt — immutable per-retrieval audit artifact.

    See docs/state-assembly-receipts.md for the full schema and
    rationale. Append-only by service-layer convention; nothing in
    the server codebase issues UPDATE or DELETE against this table.
    """

    __tablename__ = "receipts"

    receipt_id: Mapped[str] = mapped_column(String(26), primary_key=True)
    parent_receipt_id: Mapped[str | None] = mapped_column(String(26), nullable=True)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    tenant_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    subject_id: Mapped[str] = mapped_column(String(256), nullable=False)
    query_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    context_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    context_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    policy_bundle_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    region: Mapped[str | None] = mapped_column(String(64), nullable=True)
    receipt_signature: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # v0.9 (issue #157): HMAC signing metadata. Key bytes never persist
    # in the DB — these columns reference what's in operator config.
    # Both nullable: pre-v0.9 receipts and unsigned-by-policy v0.9
    # receipts leave them null; the verifier reports `valid: null` /
    # `reason: "no_signature"` for those.
    receipt_signature_key_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    receipt_signature_algorithm: Mapped[str | None] = mapped_column(String(64), nullable=True)
    body: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # Embedded policy bundle YAML + hash captured at emission (issue #159).
    # Self-contained — replay can re-evaluate without consulting the live
    # `policy_bundles` row (which may have been deleted or overwritten).
    # NULL for pre-v0.9 receipts; the replay endpoint refuses those with
    # 422 ``missing_policy_snapshot``. See docs/replay.md.
    policy_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Row-level lifecycle state. `active` for fresh receipts; the v0.9
    # retention worker (`cleanup_expired_receipts`) transitions rows to
    # `tombstoned` once `created_at + tenant.receipt_retention_days` has
    # passed. Soft-delete only — the row persists for audit lookup.
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="active")
    tombstoned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index(
            "ix_receipts_tenant_subject_created",
            "tenant_id",
            "subject_id",
            "created_at",
        ),
        # Partial index — only active rows are visited by the retention
        # worker. Created in the alembic migration with raw SQL because
        # alembic 1.13's `op.create_index` doesn't surface the partial
        # `WHERE` clause; declared here for ORM parity.
        Index(
            "ix_receipts_active_tenant_created",
            "tenant_id",
            "created_at",
            postgresql_where=text("status = 'active'"),
        ),
    )


class TenantConfigRow(Base):
    """Per-tenant configuration document. JSONB so future knobs land
    without per-setting migrations. See docs/state-assembly-receipts.md
    for the keys that v1 reads."""

    __tablename__ = "tenant_configs"

    tenant_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class PolicyBundleRow(Base):
    """Immutable policy YAML bundle, content-addressed by `bundle_hash`.

    Identity of the *row* is the synthetic `id` UUID (added in 0019)
    so two tenants can install the same YAML independently — same
    `bundle_hash`, different `tenant_id`, distinct rows. Composite
    unique index on `(tenant_id, bundle_hash) NULLS NOT DISTINCT`
    enforces "one bundle per (scope, content)"; receipts continue to
    reference bundles by `bundle_hash` (with the tenant context
    available alongside on the receipt itself).
    """

    __tablename__ = "policy_bundles"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bundle_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    yaml_content: Mapped[str] = mapped_column(Text, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    tenant_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # The composite unique index `ix_policy_bundles_tenant_hash`
    # (created in migration 0019 with NULLS NOT DISTINCT) enforces
    # "one row per (tenant_id, bundle_hash)". Not declared here as a
    # SQLAlchemy UniqueConstraint because NULLS NOT DISTINCT isn't
    # natively expressible — the migration owns it.


class QueryEmbeddingCacheRow(Base):
    """Cross-machine cache of `embed_query(text)` results.

    Eliminates duplicate provider embedding round-trips when the same task
    text is asked across multiple Fly machines (each of which has its own
    in-process LRU cache). See migration 0014 for the table contract.

    Composite PK on (text_key, model) — same text under a different
    embedding model is a different cache entry, so model rotations don't
    return stale embeddings. No tenant scoping: query embeddings are
    universal (same text → same provider vector regardless of caller).
    """

    __tablename__ = "query_embedding_cache"

    text_key: Mapped[str] = mapped_column(Text, primary_key=True)
    model: Mapped[str] = mapped_column(Text, primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
