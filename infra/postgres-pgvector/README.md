# Fly Postgres + pgvector — deployment runbook

The default `flyio/postgres-flex:17.2` image does not include pgvector. Migration `0013_pgvector_native` requires it. This directory holds a thin Dockerfile that adds `postgresql-17-pgvector` on top of the official Fly Postgres image.

## When to use this

You need this when:

- You're applying migration `0013_pgvector_native` for the first time, **and**
- Your Postgres app is on `flyio/postgres-flex:17.2` (or any image without pgvector preinstalled — check with `flyctl ssh console -a <pg-app> -C "ls /usr/share/postgresql/17/extension/vector*"`).

You don't need this if your Postgres image already bundles pgvector (most managed providers — Supabase, Neon, Crunchy — and newer Fly Postgres images do).

## One-time setup: build & push the image

```bash
# Authenticate Docker against Fly's registry
flyctl auth docker

# Build for the runtime architecture (Fly machines are amd64)
docker buildx build \
  --platform linux/amd64 \
  -t registry.fly.io/statewave-pg:pgvector \
  --push \
  infra/postgres-pgvector/
```

Confirm it landed:

```bash
flyctl image show -a statewave-pg
# (still shows old image)
```

## Deploy: swap the running Postgres image

This is the **risky step** — it restarts the Postgres machine. Plan a maintenance window. With a single-machine cluster (which `statewave-pg` currently is), expect ~30s of database unavailability while the machine recreates with the new image.

```bash
flyctl image update \
  -a statewave-pg \
  --image registry.fly.io/statewave-pg:pgvector \
  --yes
```

After the machine comes back, verify pgvector is now installed:

```bash
flyctl ssh console -a statewave-api -C "python -c \"
import asyncio
async def main():
    from server.db.engine import get_session_factory
    from sqlalchemy import text
    sf = get_session_factory()
    async with sf() as s:
        r = (await s.execute(text(\\\"SELECT name, default_version FROM pg_available_extensions WHERE name='vector'\\\"))).fetchall()
        print(r)
asyncio.run(main())
\""
# Expected: [('vector', '0.x.x')]
```

## Apply migration 0013

The migration itself does `CREATE EXTENSION IF NOT EXISTS vector` then `ALTER TABLE memories ALTER COLUMN embedding TYPE vector(1536)` and creates the HNSW index. With ~250 memories on `statewave-support-docs` it completes in under a second; on a corpus with millions of rows it can take minutes (HNSW build is the slow step) — the migration sets `statement_timeout = '20min'` to handle that.

```bash
# From a host that can reach the Fly Postgres internal address
# (or from within the statewave-api container via flyctl ssh):
flyctl ssh console -a statewave-api -C "alembic upgrade head"
```

After the migration the `statewave-api` machines need a restart to pick up the new column type via SQLAlchemy:

```bash
flyctl machine restart -a statewave-api
```

## Verify

```bash
# 1. Schema is correct
flyctl ssh console -a statewave-api -C "python -c \"
import asyncio
async def main():
    from server.db.engine import get_session_factory
    from sqlalchemy import text
    sf = get_session_factory()
    async with sf() as s:
        ext = (await s.execute(text(\\\"SELECT extname FROM pg_extension WHERE extname='vector'\\\"))).fetchall()
        col = (await s.execute(text(\\\"SELECT data_type FROM information_schema.columns WHERE table_name='memories' AND column_name='embedding'\\\"))).fetchall()
        idx = (await s.execute(text(\\\"SELECT indexname FROM pg_indexes WHERE indexname='ix_memories_embedding_hnsw'\\\"))).fetchall()
        print('vector_extension:', ext)
        print('embedding_column_type:', col)  # USER-DEFINED (vector)
        print('hnsw_index:', idx)
asyncio.run(main())
\""

# 2. Semantic search works (should be sub-second after this)
curl -sf -X POST -H \"X-API-Key: \$STATEWAVE_API_KEY\" \\
  -H \"Content-Type: application/json\" \\
  -d '{\"subject_id\":\"statewave-support-docs\",\"task\":\"What database does Statewave use?\",\"max_tokens\":300}' \\
  https://statewave-api.fly.dev/v1/context | jq .token_estimate

# 3. Run the docs eval — doc_match_rate should hold or improve
cd statewave-examples/eval-docs-support
STATEWAVE_URL=https://statewave-api.fly.dev STATEWAVE_API_KEY=... python eval_docs_support.py
```

## Rollback

If something breaks, the migration is reversible:

```bash
flyctl ssh console -a statewave-api -C "alembic downgrade 0012_add_health_cache"
```

This casts the `vector` column back to `TEXT` and drops the HNSW index. The old code path (Python-side cosine compute) was already removed in this release — so for a full rollback you'd also need to deploy the previous `statewave-api` image (the one before commit XXXX).

You can leave the pgvector-bundled Postgres image in place after a rollback — having pgvector available with no code using it is harmless.

## Why a custom image and not apt-install in the running container

Fly machines are immutable: `apt-get install` inside a running machine works **until the next restart**, when the original image is pulled fresh. A custom image is the only path that survives restarts. The runbook above is the boring, stable answer; the apt-install trick is fine for quick verification but unsafe as the production state.
