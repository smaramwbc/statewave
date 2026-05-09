# Statewave Helm chart

Deploys the Statewave API on Kubernetes.

> **Scope:** API-only. This chart does **not** deploy Postgres, the Statewave admin console, or any LLM / embedding model server. Bring your own Postgres (with the pgvector extension) and point `database.url` at it.
>
> **Companion guide:** see [`deployment/kubernetes.md`](https://github.com/smaramwbc/statewave-docs/blob/main/deployment/kubernetes.md) in `statewave-docs` for the full deployment walkthrough, secret-management patterns, and troubleshooting.

## Prerequisites

- Kubernetes 1.24+
- Helm 3.10+
- A Postgres instance reachable from the cluster, with the `pgvector` extension installed. Managed options: Neon, Supabase, RDS, Cloud SQL. In-cluster option: any chart that uses the `pgvector/pgvector:pg16` image (or installs `CREATE EXTENSION vector` against a stock Postgres).

## Quick start

```bash
helm install statewave \
  oci://ghcr.io/smaramwbc/charts/statewave \
  --version 0.1.0 \
  --set database.url='postgresql+asyncpg://user:pass@db.example.com:5432/statewave' \
  --set llm.apiKey='sk-…' \
  --set auth.apiKey='replace-me'
```

> The chart is also available from this repo as a directory:
> ```bash
> helm install statewave ./helm/statewave \
>   --set database.url='postgresql+asyncpg://…' \
>   --set llm.apiKey='sk-…'
> ```

The first install runs a Helm pre-install Job (`alembic upgrade head`) before any API pod admits traffic. Upgrades repeat the migration as a pre-upgrade Job.

## Configuration

All values are documented inline in [`values.yaml`](values.yaml). Highlights:

| Value | Default | Notes |
|---|---|---|
| `image.tag` | `""` (falls back to `Chart.AppVersion`) | Pin a digest in production. |
| `replicaCount` | `1` | See connection-budget math in the [Horizontal Scaling Guide](https://github.com/smaramwbc/statewave-docs/blob/main/deployment/horizontal-scaling.md) before raising. |
| `database.url` / `database.existingSecret` | — | One is **required**. |
| `compiler.type` | `llm` | `heuristic` for demo / no-LLM mode. |
| `embedding.provider` | `litellm` | `stub` for demo / no-embedding mode. |
| `llm.apiKey` / `llm.existingSecret` | — | Required when `compiler.type=llm` or `embedding.provider=litellm`. |
| `auth.apiKey` / `auth.existingSecret` | — | Strongly recommended in production. |
| `rateLimit.rpm` | `0` (off) | Per-IP. Postgres-backed, correct across replicas. |
| `cors.origins` | `["*"]` | Lock down for production. |
| `service.type` | `ClusterIP` | Set to `LoadBalancer` only if you skip the Ingress. |
| `ingress.enabled` | `false` | When enabled, **raise the proxy timeouts to ≥ 60s** — `/v1/context` cold-starts can take that long. |
| `migrationJob.enabled` | `true` | Disable only if you run migrations out-of-band. |
| `autoscaling.enabled` | `false` | HPA on CPU. Recompute the connection budget when raising `maxReplicas`. |
| `supportPack.autoUpdate` | `false` | Off by default for self-hosted operators (the bundled docs pack is statewave.ai-specific content). |

## Secret management

Two patterns supported. Pick **one** per credential:

### A. Inline (chart-managed Secret)

For dev / single-environment installs:

```bash
helm install statewave ./helm/statewave \
  --set database.url='postgresql+asyncpg://…' \
  --set llm.apiKey='sk-…' \
  --set auth.apiKey='…'
```

The chart creates a single `<release>-credentials` Secret holding all inline values.

### B. External Secret reference (recommended)

For production. Keep credentials in your Secret manager (Sealed Secrets, External Secrets Operator, SOPS, AWS/GCP Secrets Manager + CSI driver, …) and point the chart at the resulting Secret:

```yaml
database:
  existingSecret: statewave-db
  existingSecretKey: STATEWAVE_DATABASE_URL

llm:
  existingSecret: statewave-llm
  existingSecretKey: STATEWAVE_LITELLM_API_KEY

auth:
  existingSecret: statewave-auth
  existingSecretKey: STATEWAVE_API_KEY
```

When all three are externalised, no chart-managed Secret is created.

## Multi-instance / horizontal scaling

Statewave coordinates across replicas via Postgres (compile queue, webhook DLQ, rate limit, L2 query embedding cache). Sticky sessions are unnecessary and reduce L1 cache hit rates.

Before raising `replicaCount` past 2–3, walk the connection-budget math:

```
required_db_connections = replicas × (pool_size + max_overflow) + headroom
                        = replicas × 15 + ~15
```

At higher replica counts, put a transaction-mode PgBouncer in front of Postgres. Full guidance: [`deployment/horizontal-scaling.md`](https://github.com/smaramwbc/statewave-docs/blob/main/deployment/horizontal-scaling.md).

## Probes

- **Liveness** — `GET /healthz` (process up).
- **Readiness** — `GET /readyz` (DB reachable, queue healthy, optional LLM check).

`/readyz` may flap briefly during DB restarts; the chart's `failureThreshold: 6` keeps a single flap from cycling the pod.

## Upgrades

```bash
helm upgrade statewave ./helm/statewave \
  --reuse-values \
  --set image.tag=0.7.1
```

The pre-upgrade Job runs `alembic upgrade head` before the rollout begins. Rolling upgrades require backwards-compatible schemas across one version (the project's standard policy) — see the [migration runbook](https://github.com/smaramwbc/statewave-docs/blob/main/deployment/migrations.md).

## Uninstall

```bash
helm uninstall statewave
```

Helm removes everything the chart created. **Postgres data is not touched** — that's the operator's responsibility (the chart never owned it).

## What this chart deliberately does not do

- **No bundled Postgres.** Statewave needs a managed/operated DB lifecycle (backups, PITR, vacuum tuning) that does not belong inside a single application chart.
- **No admin console.** The `statewave-admin` console is a separate deployable with its own auth surface; bundle it via a separate chart or overlay.
- **No model server.** Self-hosted vLLM / Ollama / TEI deployments belong to their own chart with GPU scheduling and a separate runbook.
- **No NetworkPolicy.** Cluster-wide policy is operator-defined; the chart would either be too permissive or too restrictive for any given environment.
