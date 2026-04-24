# Statewave

**Memory OS — trusted context runtime for AI agents and applications.**

Statewave helps developers build AI systems that remember across sessions, compile durable memories, retrieve trusted context, and govern data by subject.

## Capabilities

- **Batch & single episode ingestion** — append-only raw event recording, up to 100 per batch
- **Pluggable compilers** — heuristic (regex) or LLM (OpenAI chat) memory extraction
- **Idempotent compilation** — recompiling the same subject produces no duplicates
- **Embedding generation** — OpenAI or stub providers, stored on memories
- **Semantic search** — pgvector cosine similarity with text-search fallback
- **Token-bounded context** — context bundles respect a configurable token budget
- **Ranked retrieval** — kind priority × recency × task relevance × temporal validity × semantic similarity
- **Memory conflict resolution** — auto-supersede older overlapping memories
- **Provenance** — every memory traces back to its source episodes
- **Subject management** — list, inspect, and permanently delete subject data
- **Authentication** — optional API key via `X-API-Key` header
- **Rate limiting** — per-IP sliding window
- **Webhooks** — event hooks for episode, compile, and delete events
- **OpenTelemetry tracing** — optional spans on key operations
- **Structured errors** — consistent JSON error format with request ID correlation
- **Structured logging** — structlog with JSON in prod, console in dev

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
- OpenAPI docs: `http://localhost:8100/docs`
- ReDoc: `http://localhost:8100/redoc`
- Liveness: `GET /healthz`
- Readiness: `GET /readyz`

## API overview

| Method | Path | Description |
|--------|------|-------------|
| POST | /v1/episodes | Ingest a raw episode (append-only) |
| POST | /v1/episodes/batch | Ingest up to 100 episodes at once |
| POST | /v1/memories/compile | Compile memories from episodes (idempotent) |
| GET | /v1/memories/search | Search memories by kind, text, or semantic similarity |
| POST | /v1/context | Assemble ranked, token-bounded context |
| GET | /v1/timeline | Get chronological subject timeline |
| GET | /v1/subjects | List known subjects with counts |
| DELETE | /v1/subjects/{id} | Delete all data for a subject |

## Running tests

```bash
# Unit tests (no DB required)
pytest tests/test_*.py -v

# Integration tests (requires Postgres)
# Create the test database once:
PGPASSWORD=statewave createdb -h localhost -U statewave statewave_test
# Run:
pytest tests/integration/ -v

# All tests
pytest tests/ -v
```

## Configuration

All settings can be set via environment variables with `STATEWAVE_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `STATEWAVE_DATABASE_URL` | `postgresql+asyncpg://...` | Postgres connection string |
| `STATEWAVE_DEBUG` | `false` | Enable debug logging |
| `STATEWAVE_COMPILER_TYPE` | `heuristic` | `heuristic` or `llm` |
| `STATEWAVE_EMBEDDING_PROVIDER` | `stub` | `stub`, `openai`, or `none` |
| `STATEWAVE_OPENAI_API_KEY` | — | Required for `llm` compiler and `openai` embeddings |
| `STATEWAVE_API_KEY` | — | API key for auth (empty = open access) |
| `STATEWAVE_RATE_LIMIT_RPM` | `0` | Requests/min/IP (0 = disabled) |
| `STATEWAVE_WEBHOOK_URL` | — | Webhook callback URL (empty = disabled) |
| `STATEWAVE_DEFAULT_MAX_CONTEXT_TOKENS` | `4000` | Default token budget |
| `STATEWAVE_CORS_ORIGINS` | `["*"]` | Allowed CORS origins |

See the full [configuration reference](../statewave-docs/api/v1-contract.md) for all options.

## License

AGPL-3.0
