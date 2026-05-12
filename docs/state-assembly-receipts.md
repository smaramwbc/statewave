# State-assembly receipts — design & reference

## What is a receipt

A **state-assembly receipt** is a compact, immutable record of which
state entries (memories and episodes) influenced a single context
assembly. One receipt per assembly call; addressable by ULID; queryable
by id or by `(subject_id, time_range)`.

The accountability primitives the receipt records — provenance,
lineage, validity intervals, supersession, conflict resolution — already
exist in Statewave's data model. Before receipts they were
*reconstructable* from the database and logs; the receipt collapses
reconstruction into a single addressable artifact written at assembly
time.

Tracked in issue
[#49](https://github.com/smaramwbc/statewave/issues/49). Sibling work
on per-memory sensitivity labels and the policy layer that fills
`policy.filters_applied` is tracked in
[#50](https://github.com/smaramwbc/statewave/issues/50); v1 of this
feature ships with the policy fields present but the policy input
"silent" (no opinion).

## Receipt schema (v1)

Receipts use a **strict-superset** shape — every field is always
present, optional fields are nullable. A `mode` discriminator
(`"retrieval"` in v1, future values reserved for `as_of_replay` and
`eval_run`) means new assembly modes can extend the schema without a
union break.

```yaml
receipt_id:            # ULID — addressable, chainable
parent_receipt_id:     # nullable ULID — chains multi-step tasks
mode:                  # "retrieval" (v1) | reserved future values
query_id:              # caller-supplied or generated UUID
task_id:               # optional, for multi-call tasks
tenant_id:             # tenant the assembly ran under
subject_id:            # subject the bundle was assembled for
as_of:                 # timestamp the assembly resolved against
created_at:            # when the receipt itself was written

selected_entries:
  - memory_id:
    memory_version:
    valid_from:
    valid_to:
    supersession_status: # active | superseded | tombstoned
    source_episode_ids: []
    provenance_hash:     # content hash of the source provenance record
    fact_key:            # for conflict grouping
    conflict_status:     # none | merged | overridden | unresolved
    rank:                # final position in the bundle

policy:
  policy_bundle_hash:    # content hash of the policy YAML in effect (null in v1)
  filters_applied: []    # decisions that fired (empty in v1)
  filters_skipped: []    # decisions evaluated but did not match (empty in v1)
  mode:                  # log_only | enforce (always log_only in v1)

output:
  context_hash:            # canonical hash of the bytes delivered to the agent
  context_size_bytes:
  canonicalization_version: # so the hash is reproducible across releases
  token_estimate:

region:                   # nullable — populated from tenant config when set
receipt_signature:        # nullable — reserved for v2 HMAC tamper-evidence
```

## Emission control

Whether a receipt is emitted is decided by a single function that
takes `(request_flag, tenant_config, policy_decision) → emit | skip |
must_emit`. The inputs are consulted in this order:

1. **Per-request flag** (`emit_receipt: true` on the assembly call) —
   the primary opt-in. Callers pay the storage cost only when they
   ask. Default `false`.
2. **Per-tenant config** (`receipts: always | on_request | never` in
   `tenant_configs.config`) — set in the admin dashboard. `always`
   overrides per-request `false`; `never` suppresses everything.
   Compliance-grade tenants flip this on once. Default `on_request`.
3. **Per-policy force-on** (deferred until #50). v1 of this layer
   returns "no opinion" so the call collapses to inputs 1 and 2.
4. **Env kill-switch** (`STATEWAVE_RECEIPTS_DISABLED=true`) —
   emergency operational hygiene; not the everyday control.

The decision function lives in `server/services/receipts.py` and is
the only place this logic exists.

## Storage

Receipts live in a dedicated `receipts` table — not in
`webhook_events` (which has delivery-retry semantics that pollute the
audit story) and not as JSONB on the assembly response (lost after the
request). The table is **append-only by convention**: no service-code
path issues `UPDATE` or `DELETE`. Operators running compliance-grade
deployments should additionally grant the service role
`INSERT`-and-`SELECT`-only on `receipts`; this is documented in
[`docs/deployment-hardening.md`](./deployment-hardening.md) (separate
from v1 of this feature).

Retention is tenant-controlled via
`tenant_configs.config.receipt_retention_days`. `0` (default) means
forever; positive integers enable a future scheduled purge worker.
v1 ships the *surface*, not the worker — receipts accumulate
indefinitely unless the operator runs a manual purge.

## Failure mode

If receipt emission fails (DB error, JSON serialization edge case),
the assembly call **still succeeds** and returns the bundle with
`receipt_emitted: false`. Receipts are an audit artifact; they must
not break agent serving. Failures are surfaced via structured logs
(`receipt_emission_failed`) and via the `receipt_emitted` flag on the
response so callers can detect the gap.

## Surfaces that emit receipts

- `POST /v1/context` — primary assembly path.
- `POST /v1/handoff` — handoff briefings are a different assembly
  window but the same accountability story.

`POST /v1/llm/complete` does not embed memory and so does not emit
receipts.

## Read API

| Method + Path | Purpose |
|---|---|
| `GET /v1/receipts/{receipt_id}` | Fetch one receipt by id |
| `GET /v1/receipts?subject_id=&since=&until=&cursor=&limit=` | List receipts for a subject in a time window |

Both endpoints are tenant-scoped via the existing `X-Statewave-Tenant`
header.

## Negative tests

A receipt is only useful if it makes failure modes detectable. Each
test is a deterministic assertion against a receipt, with no access to
assembly internals:

1. Stale fact selected for a current query — `valid_to` is in the past
   but the entry appears in `selected_entries`.
2. Superseded memory influenced ranking — `supersession_status =
   superseded` appears in `selected_entries`.
3. Sensitive tombstoned memory resurrected — `supersession_status =
   tombstoned` appears in `selected_entries`.
4. Conflicting entries merged without a `conflict_status` flag — two
   entries share a `fact_key` but neither carries `conflict_status =
   merged`.
5. As-of-date query silently fell back to latest-state recall — `as_of`
   on the receipt differs from `as_of` requested.
6. Output context hash does not match the bytes the agent received —
   `output.context_hash` recomputed from `assembled_context` mismatches
   the stored hash.

## Out of scope for v1

- Sensitivity-label / policy layer ([#50](https://github.com/smaramwbc/statewave/issues/50)) — receipt fields exist, policy input returns "no opinion".
- Review-time redaction UI.
- Receipt-driven replay / time-travel debugging tooling.
- Cross-tenant receipt aggregation / fleet-wide audit views.
- HMAC signing of receipt bodies (column reserved, no signing path).
- Scheduled retention-purge worker (config field accepted, no worker).
