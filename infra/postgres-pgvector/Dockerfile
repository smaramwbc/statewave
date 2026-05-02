# Fly Postgres image with pgvector — required for migration 0013 onward.
#
# Build:
#   docker build -t registry.fly.io/statewave-pg:pgvector \
#     infra/postgres-pgvector/
#   flyctl auth docker
#   docker push registry.fly.io/statewave-pg:pgvector
#
# Deploy on the existing statewave-pg machine:
#   flyctl image update -a statewave-pg --image registry.fly.io/statewave-pg:pgvector --yes
#
# After the image swaps, run:
#   psql ... -c 'CREATE EXTENSION IF NOT EXISTS vector;'
#   alembic upgrade head      # applies migration 0013 (ALTER TYPE + HNSW index)
#
# See infra/postgres-pgvector/README.md for the full runbook including
# rollback steps.

FROM flyio/postgres-flex:17.2

# Install the pgvector package from the official PostgreSQL apt repo —
# already configured in the base image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-17-pgvector \
    && rm -rf /var/lib/apt/lists/*

# Verify the extension files are in place. Build fails loudly if not.
RUN test -f /usr/share/postgresql/17/extension/vector.control \
    || (echo "ERROR: vector.control not found after package install" && exit 1)
