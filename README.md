# Statewave

[![CI](https://github.com/smaramwbc/statewave/workflows/CI/badge.svg)](https://github.com/smaramwbc/statewave/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

**Memory runtime for AI agents and AI-powered applications.**

### The problem

Most AI applications have no memory. Every conversation starts from scratch. Context is lost between sessions, decisions aren't remembered, and user history disappears the moment a session ends. Bolting on a vector database or dumping chat logs into a prompt doesn't solve this — it creates fragile, unstructured context that degrades as it scales.

### What Statewave does

Statewave gives your AI system **durable, structured memory** with a clear data lifecycle:

1. **Ingest** — record raw events (episodes) as they happen, append-only
2. **Compile** — extract typed, summarised memories with confidence scores and provenance
3. **Retrieve** — assemble ranked, token-bounded context bundles ready for your prompts
4. **Govern** — inspect subject timelines, trace every memory to its source, delete by subject

Everything is organised around **subjects** — a user, account, agent, repo, or any entity you track.

### Why Statewave

- **Your AI remembers** — preferences, decisions, history persist across sessions
- **Context is structured, not dumped** — ranked retrieval with token budgets, not raw chat log stuffing
- **Provenance is built in** — every memory traces back to its source episodes
- **You own it** — self-hosted, open source, no vendor lock-in
- **Framework-neutral** — works with any AI stack, any language, via REST API or typed SDKs

Statewave is **not** a chatbot framework, a vector database, a RAG pipeline, or a hosted service. It is infrastructure you run alongside your application.

> **Status:** v0.4.x — actively developed. Suitable for local development and early integration. See [current limitations](#current-limitations) below.

## 🎯 Live Demo

> **[▶ Try the interactive demo →](https://statewave-demo.vercel.app)**
>
> Two identical AI agents answer the same question — one has zero memory, one uses Statewave. See the difference in 10 seconds, no setup required.

## Documentation

| | |
|---|---|
| **[Getting started](https://github.com/smaramwbc/statewave-docs/blob/main/getting-started.md)** | Clone, run, ingest your first episode |
| [What is Statewave?](https://github.com/smaramwbc/statewave-docs/blob/main/product.md) | Product overview, use cases, limitations |
| [API v1 contract](https://github.com/smaramwbc/statewave-docs/blob/main/api/v1-contract.md) | Full endpoint reference |
| [Architecture overview](https://github.com/smaramwbc/statewave-docs/blob/main/architecture/overview.md) | System design and data flow |
| [Deployment guide](https://github.com/smaramwbc/statewave-docs/blob/main/deployment/guide.md) | Production deployment guidance |
| [Roadmap](https://github.com/smaramwbc/statewave-docs/blob/main/roadmap.md) | What's next |
| [Changelog](https://github.com/smaramwbc/statewave-docs/blob/main/CHANGELOG.md) | Release history |
| [Python SDK](https://github.com/smaramwbc/statewave-py) | Sync + async client, Pydantic models |
| [TypeScript SDK](https://github.com/smaramwbc/statewave-ts) | Fetch-based client, full type definitions |
| [Examples](https://github.com/smaramwbc/statewave-examples) | Quickstart, support agent, coding agent |

## Capabilities

- **Episode ingestion** — append-only raw event recording, single or batch (up to 100)
- **Pluggable compilers** — heuristic (regex) or LLM (OpenAI) memory extraction
- **Idempotent compilation** — recompiling the same subject produces no duplicates
- **Semantic search** — pgvector cosine similarity with text-search fallback
- **Token-bounded context** — context bundles respect a configurable token budget
- **Ranked retrieval** — kind priority × recency × task relevance × temporal validity × semantic similarity
- **Memory conflict resolution** — auto-supersede older overlapping memories
- **Provenance** — every memory traces back to its source episodes
- **Subject management** — list subjects with counts, inspect timelines, permanently delete all data by subject
- **Authentication** — optional API key via `X-API-Key` header
- **Rate limiting** — per-IP sliding window (in-memory, single-worker)
- **Multi-tenant** — optional `X-Tenant-ID` header (experimental)
- **Webhooks** — fire-and-forget HTTP callbacks on episode, compile, and delete events
- **OpenTelemetry tracing** — optional spans on key operations (requires `[otel]` extra)
- **Structured logging** — structlog with JSON output in production, console in development
- **Structured errors** — consistent JSON error format with request-ID correlation

## Quick start

```bash
# Start Postgres (pgvector)
docker compose up db -d

# Create virtualenv and install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run migrations
alembic upgrade head

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8100
```

The API is available at `http://localhost:8100`.

| Endpoint | Purpose |
|----------|---------|
| `http://localhost:8100/docs` | OpenAPI (Swagger) |
| `http://localhost:8100/redoc` | ReDoc |
| `GET /healthz` or `GET /health` | Liveness check |
| `GET /readyz` or `GET /ready` | Readiness check |

See the full [getting started guide](https://github.com/smaramwbc/statewave-docs/blob/main/getting-started.md) for step-by-step setup including environment configuration.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/episodes` | Ingest a single episode (append-only) |
| `POST` | `/v1/episodes/batch` | Ingest up to 100 episodes at once |
| `POST` | `/v1/memories/compile` | Compile memories from episodes (idempotent) |
| `GET` | `/v1/memories/search` | Search by kind, text, or semantic similarity |
| `POST` | `/v1/context` | Assemble ranked, token-bounded context bundle |
| `GET` | `/v1/timeline` | Chronological subject timeline |
| `GET` | `/v1/subjects` | List known subjects with episode/memory counts |
| `DELETE` | `/v1/subjects/{id}` | Permanently delete all data for a subject |

Full reference: [API v1 contract](https://github.com/smaramwbc/statewave-docs/blob/main/api/v1-contract.md).

## Configuration

All settings use the `STATEWAVE_` env prefix. Copy `.env.example` to `.env` to get started.

| Variable | Default | Description |
|----------|---------|-------------|
| `STATEWAVE_DATABASE_URL` | `postgresql+asyncpg://statewave:statewave@localhost:5432/statewave` | Postgres connection string |
| `STATEWAVE_DEBUG` | `false` | Enable debug logging |
| `STATEWAVE_COMPILER_TYPE` | `heuristic` | `heuristic` or `llm` |
| `STATEWAVE_EMBEDDING_PROVIDER` | `stub` | `stub`, `openai`, or `none` |
| `STATEWAVE_OPENAI_API_KEY` | — | Required for `llm` compiler and `openai` embeddings |
| `STATEWAVE_LLM_COMPILER_MODEL` | `gpt-4o-mini` | Model for LLM compiler |
| `STATEWAVE_OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model for OpenAI embeddings |
| `STATEWAVE_EMBEDDING_DIMENSIONS` | `1536` | Embedding vector dimensions |
| `STATEWAVE_API_KEY` | — | API key for auth (empty = open access) |
| `STATEWAVE_RATE_LIMIT_RPM` | `0` | Requests/min/IP (0 = disabled) |
| `STATEWAVE_WEBHOOK_URL` | — | Webhook callback URL (empty = disabled) |
| `STATEWAVE_WEBHOOK_TIMEOUT` | `5.0` | Webhook HTTP timeout in seconds |
| `STATEWAVE_TENANT_HEADER` | `X-Tenant-ID` | Header for multi-tenant isolation |
| `STATEWAVE_REQUIRE_TENANT` | `false` | Reject requests without tenant header |
| `STATEWAVE_DEFAULT_MAX_CONTEXT_TOKENS` | `4000` | Default token budget for context assembly |
| `STATEWAVE_CORS_ORIGINS` | `["*"]` | Allowed CORS origins |

## Running tests

```bash
# Unit tests (no DB required)
pytest tests/test_*.py -v

# Integration tests (requires Postgres)
PGPASSWORD=statewave createdb -h localhost -U statewave statewave_test
pytest tests/integration/ -v

# All tests
pytest tests/ -v
```

## Current limitations

Statewave is in active early development. Current caveats:

- **Single-node only** — no clustering or horizontal scaling
- **PostgreSQL required** — no alternative storage backends
- **Multi-tenant support is experimental** — header-based isolation works, but is not battle-tested
- **Rate limiting is in-memory** — resets on restart, single-worker only
- **Webhooks are fire-and-forget** — no retries, failures are logged but do not block
- **No built-in auth provider** — Statewave validates API keys you configure, but does not issue them
- **No streaming** — context responses are returned as complete JSON
- **No UI** — API-only; inspect via endpoints, OpenAPI docs, or direct DB access

## License

[AGPL-3.0](LICENSE)
