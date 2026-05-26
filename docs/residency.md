# Data residency

Per-region deployment with metadata-pinned tenants. Introduced in
**v0.9 (issue #161)**.

The application is the source of truth for "is this request allowed
here?" Residency is enforced at the request boundary in the
application layer, not at DNS or the load balancer — a misrouted
request that bypassed the LB hostname is still caught by this layer
before any DB access happens.

## Model

Two pieces of operator-managed metadata work together:

| Where it lives                          | What it means                                 |
|-----------------------------------------|-----------------------------------------------|
| `STATEWAVE_REGION` env var              | Which region THIS server process is running in |
| `tenant_configs.config.region` JSONB    | Which region a tenant is pinned to             |

A request is allowed iff one of:

1. `STATEWAVE_REGION` is unset (**single-region mode** — residency
   disabled, all requests pass through).
2. The tenant has no `region` pinned in its config (**unpinned
   tenant** — legacy or globally-mobile).
3. The tenant's pinned region equals `STATEWAVE_REGION` exactly
   (case-sensitive).

Anything else returns HTTP 403 with the standard error envelope:

```json
{
  "error": {
    "code": "residency.mismatch",
    "message": "tenant 'acme-eu' is pinned to region 'eu'; this request did not reach the right region. Retry against the regional endpoint."
  }
}
```

The response **only** echoes the tenant's pinned region. The
server's local region is NEVER returned on the wire — a 403 message
that named the server region would leak topology to a caller
probing the residency boundary.

## Why this shape

- **Application-layer enforcement** is the correctness boundary.
  DNS, anycast, and load balancers can misroute requests. The
  application is the single point that knows definitively whether a
  request can be served here.
- **Separate Postgres / pgvector per region** is the intended infra
  topology. The data plane is region-isolated by deployment, not by
  in-DB partitioning. Residency does not depend on the DB layer for
  correctness; the application layer is the second line of defence.
- **No in-DB region column on every row.** That would conflate
  residency with multi-tenancy. Residency is per-tenant, not
  per-row.
- **Total isolation** in v0.9. Cross-region admin reads are not
  allowed. Unified federated audit is a future feature, not implicit
  cross-region access — building it as an explicit separate surface
  later is safer than letting cross-region reads sneak in by default
  now.

## Pinning a tenant

Through the existing admin config endpoint:

```bash
curl -X PATCH https://eu.api.statewave.dev/admin/tenants/acme-eu/config \
  -H 'Content-Type: application/json' \
  -d '{"region": "eu"}'
```

Safety check: the admin endpoint **refuses pinning a tenant to a
region this server doesn't serve**, because the pin would lock the
tenant out of this deployment — every subsequent request from this
region would 403. The refusal looks like:

```json
{
  "error": {
    "code": "residency.invalid_pin",
    "message": "this server is running in region 'eu'; pinning a tenant to 'us' from here would lock the tenant out (every subsequent request would 403). Patch the pin from a server running in the target region, or set STATEWAVE_REGION to match if this server should serve it."
  }
}
```

To pin from elsewhere (e.g. a bulk-config migration run from a
single-region orchestrator), set `force_region_pin: true`:

```bash
curl -X PATCH https://orchestrator.example/admin/tenants/acme-us/config \
  -H 'Content-Type: application/json' \
  -d '{"region": "us", "force_region_pin": true}'
```

The orchestrator must run with `STATEWAVE_REGION=None` (single-region
mode) so the safety check is bypassed cleanly. The `force_region_pin`
field is request-only — it is not persisted on the JSONB.

## Receipt audit trail

Every state-assembly receipt emitted in multi-region mode carries the
local server's region in its `region` field (column + body). The
column has been on `receipts` since v0.8 — v0.9 finally populates it
from `STATEWAVE_REGION`. Receipts emitted in single-region mode keep
`region = null`, backwards-compatible with pre-v0.9 readers.

Combined with the HMAC signature (v0.9 #157), the policy snapshot
(v0.9 #159), and the replay endpoint, the residency stamp lets an
auditor answer end-to-end:

  - *Where* was this retrieval served from? (`receipt.region`)
  - *With what rules?* (`receipt.policy_snapshot.bundle_yaml`)
  - *Was the body tampered with?* (HMAC verify)
  - *Would current code make the same decision?* (replay diff)

## Ops runbook — spinning up a second region

The v0.9 design ships **code + config model + ops runbook only**. No
second region is deployed by this PR; this section is the
reproducible recipe for an operator standing one up.

1. **Provision regional infra** for the new region:
   - A region-local Postgres + pgvector deployment.
   - A region-local Statewave server fleet (Fly app, ECS service,
     k8s deployment — whatever the existing region uses).
   - A regional hostname (`us.api.statewave.dev`) pointing to the
     local fleet's load balancer.
   - Region-local secrets (DB creds, HMAC signing keys per
     `STATEWAVE_RECEIPT_SIGNING_KEYS`). Secrets MUST NOT be shared
     across regions — that would defeat the residency story for
     "EU tenant data must never be readable from US infra."

2. **Set `STATEWAVE_REGION`** on the new fleet to the agreed-on
   short identifier (`us`, `us-east`, `ap-south-1`, …).

3. **Apply migrations** against the new region's Postgres. Schema
   is identical across regions; no region-specific migrations exist.

4. **Smoke test in single-region mode first.** Bring up the new
   fleet without pinning any tenant; verify health, receipts
   emission with the new `region` stamp, and HMAC verification.

5. **Migrate tenants intended for the new region.** For each
   tenant:

   a. **Move the data first.** Export episodes + memories from the
      origin region (`POST /admin/memory/export`) and import them
      into the new region (`POST /admin/import`). The application
      layer treats imports identically across regions.

   b. **Pin the tenant in the new region.** Run the PATCH on the
      new region's API (so the safety check passes naturally):

      ```bash
      curl -X PATCH https://us.api.statewave.dev/admin/tenants/acme-us/config \
        -H 'Content-Type: application/json' \
        -d '{"region": "us"}'
      ```

   c. **Pin the tenant in the origin region too.** The origin
      region needs to know this tenant lives elsewhere now, so
      future stray requests are refused with 403 instead of silently
      hitting stale data. From an orchestrator (single-region
      mode):

      ```bash
      curl -X PATCH https://orchestrator/admin/tenants/acme-us/config \
        -H 'Content-Type: application/json' \
        -d '{"region": "us", "force_region_pin": true}'
      ```

   d. **Verify the cutover.** Requests for the tenant against the
      origin region should now 403 with `residency.mismatch`.
      Requests against the new region should succeed. Check
      `receipts.region` on a fresh emission to confirm.

6. **Purge stale data from the origin region** once you're
   confident the cutover is complete. This is destructive; do not
   automate it.

## Failure modes

- **DB blip during a residency check.** The middleware fails OPEN
  with a structured warning log (`residency_check_failed_failing_open`).
  Failing closed on transient DB errors would take down every
  tenant-scoped request on any blip; the data plane is already
  separated per region by infra, and this middleware is the second
  line of defence. The first line is the deployment topology
  itself.
- **A non-string `region` value in the JSONB** (e.g. set via direct
  SQL with a typo). Treated as malformed and refused — the check
  defaults to deny rather than silently allow.
- **Cross-region admin path.** Same 403 as the public API path —
  v0.9 design decision is total isolation. The same middleware
  enforces both.

## What residency does NOT do (v0.9)

- **Does not move data.** Pinning a tenant assumes the data is
  already in the target region. The export/import + pin workflow
  is the operator's responsibility.
- **Does not unify cross-region audit reads.** v0.9 ships total
  isolation. Federated audit search is a future feature, built as
  an explicit cross-region surface — never as implicit access.
- **Does not deploy a second region.** This PR is code + config
  model + tests + runbook. Actual region rollout is operator work
  using the runbook above.
