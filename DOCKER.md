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
