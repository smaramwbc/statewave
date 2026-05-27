<!-- HERO GRAPHIC: docs/img/hero.png landing in follow-up PR (issue #4) -->

# Statewave

[![CI](https://github.com/smaramwbc/statewave/workflows/CI/badge.svg)](https://github.com/smaramwbc/statewave/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/statewave)](https://pypi.org/project/statewave/)
[![Docker Pulls](https://img.shields.io/docker/pulls/statewavedev/statewave)](https://hub.docker.com/r/statewavedev/statewave)

Statewave is the open-source memory runtime that gives AI agents reproducible, provenance-tagged context — without sampling-noise from query-time retrieval.

_If Statewave is useful to you, a ⭐ on the repo helps others discover it._

<!-- QUICKSTART GIF: docs/img/quickstart.gif landing in follow-up PR (issue #4) -->

> **v0.9.2** — actively developed. [Changelog](https://github.com/smaramwbc/statewave-docs/blob/main/CHANGELOG.md) · [Roadmap](https://github.com/smaramwbc/statewave-docs/blob/main/roadmap.md) · [Limitations](#current-limitations)

## 🎯 Try it

The interactive comparison demo is embedded directly in the website at **[statewave.ai](https://statewave.ai)** — open the chat widget to see two identical AI agents answer the same question, one stateless and one backed by Statewave.

## [🚀 Skip to Getting Started →](#documentation)

## The problem

Most AI applications have no memory. Every conversation starts from scratch. Context is lost between sessions, decisions aren't remembered, and user history disappears the moment a session ends. Bolting on a vector database or dumping chat logs into a prompt doesn't solve this — it creates fragile, unstructured context that degrades as it scales.

## What Statewave does

Statewave gives your AI system **durable, structured memory** with a clear data lifecycle:

1. **Ingest** — record raw events (episodes) as they happen, append-only
2. **Compile** — extract typed, summarised memories with confidence scores and provenance
3. **Retrieve** — assemble ranked, token-bounded context bundles ready for your prompts
4. **Govern** — inspect subject timelines, trace every memory to its source, delete by subject

Everything is organised around **subjects** — a user, account, agent, repo, or any entity you track.

Statewave is **not** a chatbot framework, a vector database, a RAG pipeline, or a hosted service. It is infrastructure you run alongside your application.

## How it works

Statewave reads raw events, compiles them once per subject change into typed memories, and assembles a token-bounded context bundle on demand. Each bundle carries provenance back to its source episodes — the same query against the same subject at the same point in time always produces the same bytes. That determinism is what separates *compile-then-use* from query-time retrieval, where sampling noise leaks into every answer.

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="docs/img/how-statewave-works-mobile-dark.png">
  <source media="(max-width: 600px)" srcset="docs/img/how-statewave-works-mobile-light.png">
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/how-statewave-works-dark.png">
  <img alt="How Statewave works: events → compiled memories → ranked, token-bounded context bundle with provenance" src="docs/img/how-statewave-works-light.png" width="100%">
</picture>

Idempotent at every step — recompiling a subject produces no duplicates; reassembling a bundle for the same task at the same point in time returns the same bytes.

## Capabilities

The runtime essentials. [Full capability inventory →](docs/capabilities.md)

- **Compiled context bundles** — ranked, token-bounded, deterministic per subject and task
- **Provenance** — every memory traces back to its source episodes; receipts are content-hashed
- **Pluggable compilers** — heuristic (regex, fully local) or LLM (any [LiteLLM](https://github.com/BerriAI/litellm) provider)
- **Subject-organised** — `user:`, `repo:`, `account:`, or any entity prefix you choose
- **State-assembly receipts** — immutable, ULID-addressable record of which memories influenced each bundle. HMAC-SHA256 signature (v0.9), embedded policy snapshot (v0.9), and `POST /v1/receipts/{id}/replay` for "what would today's code say with the original rules?"
- **Sensitivity labels + policy engine** — declarative YAML policies (`deny` / `redact`) over per-memory tags (`pii`, `financial`, `secret`, …) with `log_only` and `enforce` modes. v0.9 adds advisory `suggested_labels` from heuristic detectors (pii.email/phone, financial.card, secret.token) with operator review + explicit promotion via the admin app
- **Multi-tenant** — query-scoped data isolation via `X-Tenant-ID` header, per-tenant config and policies. v0.9 adds per-tenant **region pinning** for residency: requests for a tenant pinned to `eu` are 403'd at any process running in another region
- **Self-hosted on Postgres + pgvector** — no vendor lock-in; runs on your infrastructure

## Why Statewave

- **Your AI remembers** — preferences, decisions, history persist across sessions
- **Context is structured, not dumped** — ranked retrieval with token budgets, not raw chat-log stuffing
- **Provenance is built in** — every memory traces back to its source episodes
- **You own the storage** — self-hosted, open source, no vendor lock-in. Episodes and compiled memories live in your Postgres. The default heuristic compiler runs fully local; choose an LLM compiler or hosted embeddings if you want them. See [Privacy & Data Flow](https://github.com/smaramwbc/statewave-docs/blob/main/architecture/privacy-and-data-flow.md).
- **No GPU required** — the API process is CPU-only. GPUs only enter the picture if you self-host an LLM compiler or embedding model. See [Hardware & Scaling](https://github.com/smaramwbc/statewave-docs/blob/main/deployment/hardware-and-scaling.md).
- **Framework-neutral** — works with any AI stack, any language, via REST API or typed SDKs

## Quickstart

```python
from statewave import StatewaveClient

with StatewaveClient("http://localhost:8100") as sw:
    sw.create_episode(subject_id="user-42", source="chat", type="message",
                      payload={"text": "Alice asked about pricing tiers"})
    sw.compile_memories("user-42")
    print(sw.get_context("user-42", task="answer pricing", max_tokens=1000).assembled_context)
```

That's the loop: **ingest → compile → use** — ranked, token-bounded context with provenance. Run the server below, or [self-host with Docker](DOCKER.md) / [Helm](helm/). Full SDK docs: [statewave-py](https://github.com/smaramwbc/statewave-py) (Python) · [statewave-ts](https://github.com/smaramwbc/statewave-ts) (TypeScript).

## Use cases

Three runnable examples in [statewave-examples](https://github.com/smaramwbc/statewave-examples):

- **[Customer support agent](https://github.com/smaramwbc/statewave-examples/tree/main/support-agent-python)** — returning customer recognised across sessions, ranked context with token budget, provenance tracing, handoff pack on escalation. *(Python · TypeScript)*
- **[Long-running coding agent](https://github.com/smaramwbc/statewave-examples/tree/main/coding-agent-python)** — multi-session project memory: tech stack, preferences, architecture decisions persist between conversations. *(Python · TypeScript)*
- **[Stateless vs memory-powered agent (live LLM)](https://github.com/smaramwbc/statewave-examples/tree/main/support-agent-llm)** — full loop with a real LLM (any LiteLLM provider) running side by side with and without Statewave so you can A/B the difference yourself. *(Python)*

Plus the minimal quickstart, docs-grounded support, eval suite (56 assertions across 23 tests), a benchmark, and **drop-in framework integrations** — [LangChain](https://github.com/smaramwbc/statewave-examples/tree/main/langchain-quickstart), [CrewAI](https://github.com/smaramwbc/statewave-examples/tree/main/crewai-quickstart), and [AutoGen](https://github.com/smaramwbc/statewave-examples/tree/main/autogen-quickstart) — in the [examples repo](https://github.com/smaramwbc/statewave-examples).

## Run the server

```bash
git clone https://github.com/smaramwbc/statewave && cd statewave
docker compose up -d
```

Brings up Postgres (pgvector) + the API; migrations run automatically on container start. The API is available at `http://localhost:8100`.

By default the server boots in demo mode — stub hash-based embeddings + the heuristic compiler, no real semantic search. For LLM-backed behaviour, add a `.env` next to `docker-compose.yml` and re-run `docker compose up -d` to pick it up:

```bash
STATEWAVE_EMBEDDING_PROVIDER=litellm
STATEWAVE_LITELLM_API_KEY=sk-...            # any LiteLLM provider
STATEWAVE_LITELLM_MODEL=gpt-4o-mini
STATEWAVE_LITELLM_EMBEDDING_MODEL=text-embedding-3-small
```

| Endpoint | Purpose |
|----------|---------|
| `http://localhost:8100/docs` | OpenAPI (Swagger) |
| `http://localhost:8100/redoc` | ReDoc |
| `GET /healthz` or `GET /health` | Liveness check |
| `GET /readyz` or `GET /ready` | Readiness check |

Check `GET /readyz` to confirm your LLM key was picked up — if the `llm` check shows `"detail":"STATEWAVE_LITELLM_API_KEY is not set"`, the key isn't being read (re-check that `.env` sits next to `docker-compose.yml` and re-run `docker compose up -d`).

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

## Supported platforms

| Surface | Supported |
|---|---|
| Python | 3.11+ (CI runs 3.11; 3.12 / 3.13 expected to work but not gated in CI yet) |
| OS — server | Linux verified in CI; macOS + Windows usually fine for local dev but not CI-tested |
| Docker image | `linux/amd64` and `linux/arm64` |
| Database | PostgreSQL 14+ with [pgvector](https://github.com/pgvector/pgvector) ≥ 0.4.2 |
| LLM provider (compiler) | Any of [100+ LiteLLM-supported providers](https://docs.litellm.ai/docs/providers) — OpenAI, Anthropic, Azure, Bedrock, Ollama, Cohere, Gemini, Mistral, Groq, … |
| Embedding provider | Any LiteLLM-supported, plus `stub` (local heuristic, no API key) |
| SDKs | [Python](https://github.com/smaramwbc/statewave-py) (`pip install statewave`) · [TypeScript](https://github.com/smaramwbc/statewave-ts) (`npm install @statewavedev/sdk`) |

## FAQ

**How is this different from Mem0 / Zep?**
Mem0 is lean and fast but loses on multi-hop reasoning in our bench. Zep extracts a graph but in our LoCoMo run its retrieval surface returned the same thread summary regardless of the query. Statewave compiles the context once per subject change, with provenance. See [statewave-bench](https://github.com/smaramwbc/statewave-bench) for row-level data and a 20-minute reproducibility command against your own keys.

**Does it work with my model provider?**
Yes — Statewave uses [LiteLLM](https://github.com/BerriAI/litellm) so any of 100+ providers work (OpenAI, Anthropic, Azure, Bedrock, Ollama, Groq, Cohere, Gemini, Mistral, …). Set `STATEWAVE_LITELLM_MODEL` to any LiteLLM identifier.

**What's the license — can I use this commercially?**
Yes. Statewave (server + SDKs) is Apache-2.0 — a permissive license with an explicit patent grant. Use it freely in proprietary, hosted, or commercial products with no source-disclosure obligations. See [LICENSING.md](LICENSING.md).

**Can I self-host?**
Yes — that's the default. Docker Compose, Helm chart, or bare-metal. See [Deployment guide](https://github.com/smaramwbc/statewave-docs/blob/main/deployment/guide.md).

**Why does it cost more tokens per answer than Mem0?**
Compiled context bundles are denser than Mem0's fact-store retrieval — that's what buys the higher multi-hop accuracy. If your queries are mostly single-hop and you're cost-sensitive, Mem0 may be the right call. The bench tells you when each system wins.

## Connectors

Statewave is not limited to live chat transcripts. **Connectors** feed real-world events into Statewave as episodes, so your agents can build memory from repos, communities, docs, support tools, email, and workflows — without you hand-writing an ingest path for each source.

| Source | Memory shape | Status |
|---|---|---|
| MCP server | Copilot / Claude / Cursor / agent memory | ✅ shipped |
| GitHub | Issues, pull requests, reviews, releases → repo memory | ✅ shipped |
| Markdown | Local docs, ADRs, RFCs → decision memory | ✅ shipped |
| Slack | Channel + thread history (pull) + Events-API webhook (push, with opt-in DMs and group DMs) → team memory | ✅ shipped |
| n8n | Workflow runs, failures, per-node errors → workflow memory | ✅ shipped |
| Zapier | "Webhooks by Zapier" → push-mode helper for any zap | ✅ shipped |
| Discord | Server channel + thread history → community memory | ✅ shipped |
| Notion | Pages + opt-in body content + database scoping → decision memory | ✅ shipped |
| Zendesk / Intercom / Freshdesk | Tickets + replies + notes (pull) and real-time webhook receivers (push) → customer memory | ✅ shipped |
| Gmail | Query-scoped messages (pull, with History-API delta sync) and Cloud Pub/Sub push receiver → relationship memory | ✅ shipped |

Connectors live in their own repository so this core stays focused on the runtime. They are **modular** — install only what you need:

```bash
# Pick what you need — every package is independent
npm install @statewavedev/connectors-github
npm install @statewavedev/connectors-markdown
npm install @statewavedev/connectors-slack
npm install @statewavedev/connectors-n8n
npm install @statewavedev/connectors-zapier
npm install @statewavedev/mcp-server
```

A convenience meta-package `@statewavedev/connectors` re-exports the official connectors for the rare case where you want them all at once. **It is not required** for normal usage.

Quick examples (dry-run-first — nothing is ingested without your say-so):

```bash
statewave-connectors sync github \
  --repo smaramwbc/statewave \
  --subject repo:smaramwbc/statewave \
  --dry-run

statewave-connectors sync markdown \
  --path ./docs \
  --subject repo:smaramwbc/statewave \
  --dry-run

statewave-connectors mcp start
```

**Where to go next:**

- Connector ecosystem repo: [statewave-connectors](https://github.com/smaramwbc/statewave-connectors) *(Phase-1 packages — core, CLI, MCP server, GitHub, Markdown — landing soon; npm publication is a follow-up)*
- Connector docs: [statewave-docs/connectors](https://github.com/smaramwbc/statewave-docs/blob/main/connectors/index.md)
- SDKs (custom ingestion paths): [Python](https://github.com/smaramwbc/statewave-py) · [TypeScript](https://github.com/smaramwbc/statewave-ts)

> **No connector code lives in this repo.** Connectors talk to Statewave through the same public HTTP API documented above. If you don't need any of them, you don't install any of them.

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
| `STATEWAVE_WEBHOOK_EVENTS` | — | Comma-separated event-type allowlist (empty = deliver every event) |
| `STATEWAVE_RECEIPT_SIGNING_KEYS` | — | JSON `{"<key_id>": "<base64>"}` map of HMAC keys for receipt signing (≥32 bytes each). Never persisted to the DB; per-tenant active key id set via `tenant_configs.config.receipt_signing_key_id`. |
| `STATEWAVE_AUTO_LABELING_ENABLED` | `false` | Run heuristic detectors at compile time and stamp advisory `suggested_labels` on memories (v0.9). See [`docs/auto-labeling.md`](docs/auto-labeling.md). |
| `STATEWAVE_AUTO_LABELING_PROVIDER` | `heuristic` | Detector provider. Only `heuristic` is valid in v0.9; the switch is reserved for future LLM-based classifiers. |
| `STATEWAVE_REGION` | — | Region this server process is running in. When set, requests for tenants pinned to a different region are refused with HTTP 403 `residency.mismatch` (v0.9). Empty = single-region mode, residency disabled. See [`docs/residency.md`](docs/residency.md). |
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

Statewave is in active development (v0.9.2). Honest status:

- **Rate limiting is per-IP** — distributed (Postgres-backed), but keyed by IP only, not per-tenant or per-API-key yet
- **Multi-tenant is app-layer** — query-scoped data isolation (v0.5) + per-tenant config / policy bundles / receipts (v0.8) + HMAC-signed audit + tenant region pin (v0.9), but no Postgres RLS yet
- **No built-in auth provider** — validates API keys you configure, doesn't issue them
- **No admin-action identity yet** — the v0.9 auto-labeling promote endpoint stamps `promoted_at` + `labels` on every promotion, but `promoted_by` is `null` until an admin-identity layer ships
- **No visual policy editor** — policy bundles are authored as YAML in git and uploaded through the admin workflow; a no-YAML form-based editor is deferred to a future admin release
- **No federated cross-region audit** — v0.9 ships strict per-region isolation by design; operators running multiple regions cannot search receipts or memories across regions from a single console yet. Federated audit should be added explicitly later, not through implicit cross-region access
- **Replay is "current code + original policy"** — v0.9 receipt replay re-runs the original retrieval against today's memories using the original policy bundle (frozen on every v0.9+ receipt). Byte-for-byte historical reproduction needs memory snapshots, which are deferred

See the [roadmap](https://github.com/smaramwbc/statewave-docs/blob/main/roadmap.md) for what's being fixed and when.

> **Horizontal scaling.** Statewave runs multi-replica in production (Fly multi-machine + Helm HPA both verified). The policy bundle cache that previously assumed single-process was dropped in v0.8 (#77) precisely so multi-replica deploys behave correctly under enforce mode. Heavy load is bottlenecked by Postgres + your embedding provider, not the API.

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
| [State-assembly receipts](docs/state-assembly-receipts.md) | Receipt schema, HMAC signing, verify semantics |
| [Receipt replay](docs/replay.md) | v0.9 — re-run a historical retrieval against current memories + original policy |
| [Auto-labeling](docs/auto-labeling.md) | v0.9 — heuristic detectors + review/promote workflow |
| [Residency](docs/residency.md) | v0.9 — per-region deployment + tenant pinning + ops runbook |
| [Roadmap](https://github.com/smaramwbc/statewave-docs/blob/main/roadmap.md) | What's next |
| [Changelog](https://github.com/smaramwbc/statewave-docs/blob/main/CHANGELOG.md) | Release history |

## Community

Statewave is built in the open. Where to post what:

- **[GitHub Discussions](https://github.com/smaramwbc/statewave/discussions)** — questions, ideas, RFCs, integrations, roadmap input, what you're building
- **[GitHub Issues](https://github.com/smaramwbc/statewave/issues)** — confirmed, reproducible bugs and concrete implementation tasks
- **Security vulnerabilities** — email [security@statewave.ai](mailto:security@statewave.ai) (do not post publicly; see [SECURITY.md](SECURITY.md))
- **Commercial / private licensing terms** — email [licensing@statewave.ai](mailto:licensing@statewave.ai) (general licensing *questions* are welcome in Discussions)

The full community guide — categories, RFC process, moderation — lives in [statewave-docs/community/discussions.md](https://github.com/smaramwbc/statewave-docs/blob/main/community/discussions.md). Copy-paste templates for questions, feature requests, RFCs, and show-and-tell are in [discussion-templates.md](https://github.com/smaramwbc/statewave-docs/blob/main/community/discussion-templates.md).

## Ecosystem

| Project | Description |
|---|---|
| **Server** (this repo) | Core server — API, domain model, DB, services |
| [Python SDK](https://github.com/smaramwbc/statewave-py) | `pip install statewave` — sync + async client |
| [TypeScript SDK](https://github.com/smaramwbc/statewave-ts) | `npm install @statewavedev/sdk` — fetch-based client |
| [Connectors](https://github.com/smaramwbc/statewave-connectors) | `@statewavedev/connectors-*` — GitHub, Markdown/docs, MCP server, modular packages |
| [Docs](https://github.com/smaramwbc/statewave-docs) | Architecture, API contracts, ADRs |
| [Examples](https://github.com/smaramwbc/statewave-examples) | Quickstarts, evals, benchmarks |
| [Website + demo](https://github.com/smaramwbc/statewave-web) | Marketing website + embedded interactive demo ([statewave.ai](https://statewave.ai)) |
| [Admin](https://github.com/smaramwbc/statewave-admin) | Operator console (read-only) |

## Licensing

Statewave is licensed under the [Apache License, Version 2.0](LICENSE).

You may use, modify, and distribute it — including in proprietary or
hosted products — without a separate agreement. Apache-2.0 includes an
explicit patent grant from contributors.

- **License:** [LICENSE](LICENSE) (Apache-2.0)
- **Licensing overview:** [LICENSING.md](LICENSING.md)
- **Trademark policy:** [TRADEMARKS.md](TRADEMARKS.md)
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Enterprise support (SLA, indemnity, procurement):**
  [licensing@statewave.ai](mailto:licensing@statewave.ai)
