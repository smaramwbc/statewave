# Statewave

**Memory OS — trusted context runtime for AI agents and applications.**

Statewave helps developers build AI systems that remember across sessions, compile durable memories, retrieve trusted context, and govern data by subject.

## Quick start

```bash
# Start Postgres (pgvector)
docker compose up db -d

# Install dependencies
pip install -e ".[dev]"

# Run migrations
alembic upgrade head

# Start the server
python -m server.main
```

The API is available at `http://localhost:8100`. OpenAPI docs at `/docs`.

## API overview

| Method | Path | Description |
|--------|------|-------------|
| POST | /v1/episodes | Ingest a raw episode |
| POST | /v1/memories/compile | Compile memories from episodes |
| GET | /v1/memories/search | Search memories |
| POST | /v1/context | Assemble context bundle |
| GET | /v1/timeline | Get subject timeline |
| DELETE | /v1/subjects/{id} | Delete all data for a subject |

## Running tests

```bash
pytest
```

## License

AGPL-3.0
