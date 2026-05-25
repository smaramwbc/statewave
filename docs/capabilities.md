# Statewave capabilities — full list

The lead [README](../README.md#capabilities) lists the top 8 capabilities most relevant on a first look. This file holds the complete inventory.

## Core runtime

- **Episode ingestion** — append-only raw event recording, single (`POST /v1/episodes`) or batch up to 100 (`POST /v1/episodes/batch`).
- **Pluggable compilers** — heuristic (regex) or LLM (any LiteLLM-supported provider) memory extraction; switch via `STATEWAVE_COMPILER_TYPE`.
- **Idempotent compilation** — recompiling the same subject produces no duplicates; safe to rerun.
- **Semantic search** — pgvector cosine similarity with text-search fallback when no embedding provider is configured.
- **Token-bounded context** — every context bundle respects a configurable token budget (`max_tokens`).
- **Ranked retrieval** — kind priority × recency × task relevance × temporal validity × semantic similarity.
- **Memory conflict resolution** — auto-supersede older overlapping memories.
- **Provenance** — every memory traces back to its source episodes; bundles carry the chain.
- **Subject management** — list subjects with episode/memory counts, inspect timelines, hard-delete all data per subject.

## Governance & audit

- **State-assembly receipts** — every `/v1/context` and `/v1/handoff` call can emit an immutable, ULID-addressable audit record of which memories + episodes influenced the bundle, with a SHA-256 hash of the bytes delivered to the agent. Tenant-scoped retrieval via `GET /v1/receipts`.
- **Per-memory sensitivity labels** — operator-supplied capability tags (`pii`, `financial`, `secret`, …) carried as a `TEXT[]` column with a GIN index; set via `PATCH /v1/memories/{id}/labels`.
- **Declarative policy engine** — YAML/JSON policy bundles with six predicates (label match, caller_type, caller_id) and two actions (`deny`, `redact`). Bundles are content-hashed and immutable, addressable by `bundle_hash`. Per-tenant `policy_mode` toggles between `log_only` (record decisions to receipts, no filtering) and `enforce` (drop denied memories before ranking).
- **Caller identity** — `caller_id` and `caller_type` on context/handoff requests feed the policy evaluator. Tenant config `require_caller_identity: true` 401s anonymous calls.
- **Per-tenant config** — `GET / PATCH /admin/tenants/{id}/config` for receipts emission, retention, policy_mode, caller-identity gating. PATCH-shape merge with optimistic concurrency via `expected_version`.

## Operations

- **Authentication** — optional API key via `X-API-Key` header.
- **Rate limiting** — per-IP fixed-window, distributed (Postgres-backed) or in-memory.
- **Multi-tenant** — optional `X-Tenant-ID` header with real query-scoped data isolation across all reads and writes.
- **Webhooks** — persistent HTTP callbacks with retries and dead-letter on episode, compile, and delete events; an optional event-type allowlist (`STATEWAVE_WEBHOOK_EVENTS`) restricts delivery to specific event types.
- **OpenTelemetry tracing** — optional spans on key operations (requires `[otel]` extra).
- **Structured logging** — `structlog` with JSON output in production, console in development.
- **Structured errors** — consistent JSON error format with request-ID correlation.

## Support-agent stack

- **Session-aware context** — active session boosted, resolved sessions deprioritized.
- **Resolution tracking** — mark issues open/resolved, surface resolution history.
- **Handoff context packs** — compact escalation briefs with health, SLA, and issue context.
- **Customer health scoring** — deterministic 0–100 score with explainable factors.
- **SLA tracking** — first-response time, resolution time, breach detection.
- **Proactive health alerts** — webhooks on health state transitions (degradation + recovery).
- **Repeat-issue detection** — surfaces prior resolutions when patterns recur.
