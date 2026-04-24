# Statewave

**Memory OS — trusted context runtime for AI agents and applications.**

Statewave helps developers build AI systems that remember across sessions, compile durable memories, retrieve trusted context, and govern data by subject.

## v0.2 guarantees

- **Idempotent compilation** — recompiling the same subject produces no duplicates
- **Token-bounded context** — context bundles respect a configurable token budget
- **Ranked retrieval** — memories are scored by kind priority, recency, and task relevance
- **Provenance** — every memory traces back to its source episodes
- **Subject deletion** — all data for a subject can be permanently removed

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
| POST | /v1/memories/compile | Compile memories from episodes (idempotent) |
| GET | /v1/memories/search | Search memories by kind or text |
| POST | /v1/context | Assemble ranked, token-bounded context |
| GET | /v1/timeline | Get chronological subject timeline |
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
| `STATEWAVE_CORS_ORIGINS` | `["*"]` | Allowed CORS origins |
| `STATEWAVE_DEFAULT_MAX_CONTEXT_TOKENS` | `4000` | Default token budget |
| `STATEWAVE_COMPILER_TYPE` | `heuristic` | Memory compiler backend |

## License

AGPL-3.0
