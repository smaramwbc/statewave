# Statewave

[![CI](https://github.com/smaramwbc/statewave/workflows/CI/badge.svg)](https://github.com/smaramwbc/statewave/actions/workflows/ci.yml)
[![License: AGPL-3.0 + Commercial](https://img.shields.io/badge/license-AGPL--3.0%20%2B%20Commercial-blue.svg)](LICENSING.md)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

**Memory runtime for AI agents and AI-powered applications.**

**Primary focus: support-agent workflows** — where structured memory clearly outperforms naive history stuffing and simple RAG. Statewave gives your support agent durable customer context across sessions, with provenance, ranked retrieval, and token budgets.

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
- **You own the storage** — self-hosted, open source, no vendor lock-in. Episodes and compiled memories live in your Postgres. The default heuristic compiler runs fully local; choose an LLM compiler or hosted embeddings if you want them. See [Privacy & Data Flow](https://github.com/smaramwbc/statewave-docs/blob/main/architecture/privacy-and-data-flow.md).
- **No GPU required** — the API process is CPU-only. GPUs only enter the picture if you self-host an LLM compiler or embedding model. See [Hardware & Scaling](https://github.com/smaramwbc/statewave-docs/blob/main/deployment/hardware-and-scaling.md).
- **Framework-neutral** — works with any AI stack, any language, via REST API or typed SDKs

Statewave is **not** a chatbot framework, a vector database, a RAG pipeline, or a hosted service. It is infrastructure you run alongside your application.

> **Status:** v0.6.1 — actively developed. Full support-agent intelligence stack: session-aware context, resolution tracking, handoff packs, health scoring, SLA tracking, proactive alerts. See [current limitations](#current-limitations) below.

## 🎯 Try it

> The interactive comparison demo is embedded directly in the website at **[statewave.ai](https://statewave.ai)** — open the chat widget to see two identical AI agents answer the same question, one stateless and one backed by Statewave.

## Documentation

| | |
|---|---|
| **[Getting started](https://github.com/smaramwbc/statewave-docs/blob/main/getting-started.md)** | Clone, run, ingest your first episode |
| [What is Statewave?](https://github.com/smaramwbc/statewave-docs/blob/main/product.md) | Product overview, use cases, limitations |
| [Why Statewave?](https://github.com/smaramwbc/statewave-docs/blob/main/why-statewave.md) | Technical comparison for support-agent workflows |
| [API v1 contract](https://github.com/smaramwbc/statewave-docs/blob/main/api/v1-contract.md) | Full endpoint reference |
| [Architecture overview](https://github.com/smaramwbc/statewave-docs/blob/main/architecture/overview.md) | System design and data flow |
| [Compiler modes](https://github.com/smaramwbc/statewave-docs/blob/main/architecture/compiler-modes.md) | Heuristic vs LLM — when to use which |
| [Privacy & data flow](https://github.com/smaramwbc/statewave-docs/blob/main/architecture/privacy-and-data-flow.md) | What stays local, what leaves your network |
| [Hardware & scaling](https://github.com/smaramwbc/statewave-docs/blob/main/deployment/hardware-and-scaling.md) | GPU is never required; scaling characteristics |
| [Deployment sizing guide](https://github.com/smaramwbc/statewave-docs/blob/main/deployment/sizing.md) | Hardware profiles by tier (local → enterprise) and topology patterns |
| [Capacity planning checklist](https://github.com/smaramwbc/statewave-docs/blob/main/deployment/capacity-planning.md) | Diagnostic flow + tuning order when load grows |
| [Deployment guide](https://github.com/smaramwbc/statewave-docs/blob/main/deployment/guide.md) | Production deployment guidance |
| [Roadmap](https://github.com/smaramwbc/statewave-docs/blob/main/roadmap.md) | What's next |
| [Changelog](https://github.com/smaramwbc/statewave-docs/blob/main/CHANGELOG.md) | Release history |
| [Python SDK](https://github.com/smaramwbc/statewave-py) | Sync + async client, Pydantic models |
| [TypeScript SDK](https://github.com/smaramwbc/statewave-ts) | Fetch-based client, full type definitions |
| [Examples](https://github.com/smaramwbc/statewave-examples) | Quickstart, support agent, coding agent |
| [Context quality eval](https://github.com/smaramwbc/statewave-examples/tree/main/eval-support-agent) | Automated assertions on context correctness |
| [Benchmark](https://github.com/smaramwbc/statewave-examples/tree/main/benchmark-support-agent) | Statewave vs history stuffing vs RAG |

## Capabilities

- **Episode ingestion** — append-only raw event recording, single or batch (up to 100)
- **Pluggable compilers** — heuristic (regex) or LLM (any LiteLLM-supported provider) memory extraction
- **Idempotent compilation** — recompiling the same subject produces no duplicates
- **Semantic search** — pgvector cosine similarity with text-search fallback
- **Token-bounded context** — context bundles respect a configurable token budget
- **Ranked retrieval** — kind priority × recency × task relevance × temporal validity × semantic similarity
- **Memory conflict resolution** — auto-supersede older overlapping memories
- **Provenance** — every memory traces back to its source episodes
- **Subject management** — list subjects with counts, inspect timelines, permanently delete all data by subject
- **Authentication** — optional API key via `X-API-Key` header
- **Rate limiting** — per-IP fixed-window, distributed (Postgres-backed) or in-memory
- **Multi-tenant** — optional `X-Tenant-ID` header with real query-scoped data isolation
- **Webhooks** — persistent HTTP callbacks with retries and dead-letter on episode, compile, and delete events
- **OpenTelemetry tracing** — optional spans on key operations (requires `[otel]` extra)
- **Structured logging** — structlog with JSON output in production, console in development
- **Structured errors** — consistent JSON error format with request-ID correlation
- **Session-aware context** — active session boosted, resolved sessions deprioritized
- **Resolution tracking** — mark issues open/resolved, surface resolution history
- **Handoff context packs** — compact escalation briefs with health, SLA, and issue context
- **Customer health scoring** — deterministic 0–100 score with explainable factors
- **SLA tracking** — first-response time, resolution time, breach detection
- **Proactive health alerts** — webhooks on health state transitions (degradation + recovery)
- **Repeat-issue detection** — surfaces prior resolutions when patterns recur

## Quick start

```bash
# Start Postgres (pgvector)
docker compose up db -d

# Create virtualenv and install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]"

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
| `POST` | `/v1/resolutions` | Track issue resolution state per session |
| `GET` | `/v1/resolutions` | List resolutions for a subject |
| `POST` | `/v1/handoff` | Generate compact handoff context pack |
| `GET` | `/v1/subjects/{id}/health` | Customer health score with explainable factors |
| `GET` | `/v1/subjects/{id}/sla` | SLA metrics — response time, resolution time, breaches |

Full reference: [API v1 contract](https://github.com/smaramwbc/statewave-docs/blob/main/api/v1-contract.md).

## Configuration

All settings use the `STATEWAVE_` env prefix. Copy `.env.example` to `.env` to get started.

> **For best results:** Set `STATEWAVE_COMPILER_TYPE=llm` and `STATEWAVE_EMBEDDING_PROVIDER=litellm` with an `STATEWAVE_LITELLM_API_KEY`. Statewave uses [LiteLLM](https://github.com/BerriAI/litellm) as its single provider abstraction, so you can use any supported provider — OpenAI, Anthropic, Azure, Ollama, Cohere, Gemini, Bedrock, Mistral, Groq, and 100+ others — by setting `STATEWAVE_LITELLM_MODEL` to any LiteLLM model identifier (e.g. `gpt-4o-mini`, `claude-3-haiku-20240307`, `ollama/llama3`, `azure/gpt-4`). The heuristic compiler still works without any LLM API key.

| Variable | Default | Description |
|----------|---------|-------------|
| `STATEWAVE_DATABASE_URL` | `postgresql+asyncpg://statewave:statewave@localhost:5432/statewave` | Postgres connection string |
| `STATEWAVE_DEBUG` | `false` | Enable debug logging |
| `STATEWAVE_COMPILER_TYPE` | `heuristic` | `heuristic` or `llm` |
| `STATEWAVE_EMBEDDING_PROVIDER` | `stub` | `stub`, `litellm`, or `none` |
| `STATEWAVE_LITELLM_API_KEY` | — | Provider-neutral API key (e.g. OpenAI `sk-...`, Anthropic `sk-ant-...`) — passed through to the provider chosen by `STATEWAVE_LITELLM_MODEL` |
| `STATEWAVE_LITELLM_MODEL` | `gpt-4o-mini` | Chat-completion model — any [LiteLLM identifier](https://docs.litellm.ai/docs/providers) (`claude-3-haiku-20240307`, `ollama/llama3`, `azure/gpt-4`, etc.) |
| `STATEWAVE_LITELLM_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model — any LiteLLM-supported (`cohere/embed-english-v3.0`, `voyage/voyage-large-2`, …) |
| `STATEWAVE_LITELLM_API_BASE` | — | Custom base URL (e.g. `http://localhost:11434` for Ollama, or a self-hosted OpenAI-compatible gateway) |
| `STATEWAVE_LITELLM_TIMEOUT_SECONDS` | `60` | Request timeout |
| `STATEWAVE_LITELLM_MAX_RETRIES` | `2` | Retries on transient errors |
| `STATEWAVE_EMBEDDING_DIMENSIONS` | `1536` | Embedding vector dimensions |
| `STATEWAVE_API_KEY` | — | API key for auth (empty = open access) |
| `STATEWAVE_RATE_LIMIT_RPM` | `0` | Requests/min/IP (0 = disabled) |
| `STATEWAVE_RATE_LIMIT_STRATEGY` | `distributed` | `distributed` (Postgres) or `memory` (in-process) |
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

Statewave is in active development (v0.6.1). Honest status:

- **Rate limiting is per-IP** — distributed (Postgres-backed), but keyed by IP only, not per-tenant or per-API-key yet
- **Multi-tenant is app-layer** — real query-scoped isolation (v0.5), no Postgres RLS yet
- **Single-node only** — no clustering, no horizontal scaling yet
- **PostgreSQL required** — no alternative storage backends
- **No built-in auth provider** — validates API keys you configure, doesn't issue them

See the [roadmap](https://github.com/smaramwbc/statewave-docs/blob/main/roadmap.md) for what's being fixed and when.

## Ecosystem

| Repo | Purpose |
|------|---------|
| **statewave** (this repo) | Core server — API, domain model, DB, services |
| [statewave-py](https://github.com/smaramwbc/statewave-py) | Python SDK (sync + async) |
| [statewave-ts](https://github.com/smaramwbc/statewave-ts) | TypeScript SDK |
| [statewave-docs](https://github.com/smaramwbc/statewave-docs) | Architecture, API contracts, ADRs |
| [statewave-examples](https://github.com/smaramwbc/statewave-examples) | Quickstarts, evals, benchmarks |
| [statewave-web](https://github.com/smaramwbc/statewave-web) | Marketing website + embedded interactive demo ([statewave.ai](https://statewave.ai)) |
| [statewave-admin](https://github.com/smaramwbc/statewave-admin) | Operator console (read-only) |

## Licensing

Statewave is **dual-licensed**:

- **[AGPLv3](LICENSE)** — for open-source / community use.
- **[Commercial license](COMMERCIAL-LICENSE.md)** — for proprietary, SaaS,
  embedded, hosted, or enterprise use.

This allows Statewave to stay open and community-driven while protecting
the project from unmanaged commercial hosting or closed-source
redistribution. If you want to use Statewave in a proprietary product,
SaaS platform, managed service, or enterprise environment without AGPL
obligations, contact us for a commercial license.

A startup-friendly commercial tier is available for early-stage companies
under a qualifying threshold.

- **Quick decision guide:** [LICENSING.md](LICENSING.md)
- **Commercial license overview:** [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)
- **Tiers (Community / Startup / Growth / Enterprise):** [docs/licensing.md](docs/licensing.md)
- **Trademark policy:** [TRADEMARKS.md](TRADEMARKS.md)
- **Contributing under dual licensing:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Contact:** [licensing@statewave.ai](mailto:licensing@statewave.ai)

> This repository describes Statewave's licensing model and is not legal
> advice. Consult qualified counsel before adopting Statewave in a
> commercial product.
