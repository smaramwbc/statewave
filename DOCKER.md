# Statewave server

Open-source memory runtime for AI agents — episodes in, distilled memories out.

[![Image](https://img.shields.io/docker/image-size/statewavedev/statewave/latest?label=image)](https://hub.docker.com/r/statewavedev/statewave)
[![Pulls](https://img.shields.io/docker/pulls/statewavedev/statewave)](https://hub.docker.com/r/statewavedev/statewave)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/smaramwbc/statewave/blob/main/LICENSE)

Multi-arch (`linux/amd64`, `linux/arm64`), built with provenance + SBOM and signed via Sigstore.

## Quickstart with Docker Compose

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: statewave
      POSTGRES_PASSWORD: statewave
      POSTGRES_DB: statewave
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U statewave"]
      interval: 2s
      timeout: 5s
      retries: 10

  api:
    image: statewavedev/statewave:latest
    ports: ["8100:8100"]
    environment:
      STATEWAVE_DATABASE_URL: postgresql+asyncpg://statewave:statewave@db:5432/statewave
      # LLM provider — pick one and set the matching key.
      STATEWAVE_LITELLM_API_KEY: sk-...
    depends_on:
      db:
        condition: service_healthy

volumes:
  pgdata:
```

```sh
docker compose up -d
curl http://localhost:8100/healthz
```

## Pin a version

```sh
STATEWAVE_VERSION=0.7.0 docker compose up -d
```

## Handling port conflicts

Local deployments frequently encounter port conflicts if services like PostgreSQL or other applications already use ports 5432 (database), 8100 (API) or 8080 (admin console). Override the host ports using environment variables. Only the host-side port changes — traffic between the containers (e.g. admin → API) stays on the internal compose network and is unaffected:

```sh
# Use custom ports instead of defaults
STATEWAVE_DB_HOST_PORT=5433 STATEWAVE_ADMIN_HOST_PORT=8081 docker compose up -d
```

Or set them in a `.env` file next to `docker-compose.yml`:

```sh
# .env
STATEWAVE_DB_HOST_PORT=5433
STATEWAVE_ADMIN_HOST_PORT=8081
```

Then start as usual:

```sh
docker compose up -d
```

### Service port reference

| Service | Container port | Default host port | Override variable |
|---------|---|---|---|
| Database (PostgreSQL) | 5432 | 5432 | `STATEWAVE_DB_HOST_PORT` |
| Admin console | 8080 | 8080 | `STATEWAVE_ADMIN_HOST_PORT` |
| API | 8100 | 8100 | `STATEWAVE_API_HOST_PORT` |

### Troubleshooting port conflicts

**Error: `bind: address already in use`**

1. **Identify the conflicting process:**
   ```sh
   # On macOS/Linux
   lsof -i :5432  # Check database port
   lsof -i :8100  # Check API port
   lsof -i :8080  # Check admin port
   ```

2. **Choose one of:**
   - **Kill the other process** (if it's not needed)
   - **Change the Statewave port** (recommended for local dev) — set `STATEWAVE_DB_HOST_PORT`, `STATEWAVE_API_HOST_PORT` and/or `STATEWAVE_ADMIN_HOST_PORT`
   - **Change the other service's port** (if you control it)

3. **Start Statewave with the new ports:**
   ```sh
   STATEWAVE_DB_HOST_PORT=5433 STATEWAVE_ADMIN_HOST_PORT=8081 docker compose up -d
   ```

4. **Update anything that targets the remapped host ports.** In-cluster
   addresses are unchanged — only host-side clients need the new port:
   ```sh
   # API remapped to host port 8101 instead of 8100
   curl http://localhost:8101/healthz

   # A client connecting directly to the DB on a custom host port (5433):
   export STATEWAVE_DATABASE_URL=postgresql+asyncpg://statewave:statewave@localhost:5433/statewave
   ```

### Environment variable precedence

- Command-line variables override `.env` file variables: `STATEWAVE_DB_HOST_PORT=5433 docker compose up -d` takes precedence over `STATEWAVE_DB_HOST_PORT=5434` in `.env`
- If neither is set, defaults are used (5432 for DB, 8100 for API, 8080 for admin)

## Tags

| Tag | Meaning |
|---|---|
| `latest` | Tip of `main` |
| `X.Y.Z` | Semver release |
| `X.Y` | Latest in the minor line |
| `X` | Latest in the major line |
| `sha-<7>` | Specific commit |

## Verify the build attestation

```sh
gh attestation verify \
  oci://docker.io/statewavedev/statewave:latest \
  --owner smaramwbc
```

## Source & docs

- Repository: <https://github.com/smaramwbc/statewave>
- Documentation: <https://statewave.ai>
- License: Apache-2.0
