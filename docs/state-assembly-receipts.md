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
(`"retrieval"` for `/v1/context` + `/v1/handoff`, `"as_of_replay"`
for v0.9 replay receipts, future values reserved for `eval_run`)
means new assembly modes can extend the schema without a union break.

```yaml
receipt_id:            # ULID — addressable, chainable
parent_receipt_id:     # nullable ULID — chains multi-step tasks (set on replay receipts)
mode:                  # "retrieval" | "as_of_replay" | reserved future values
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
`tenant_configs.config.receipt_retention_days`. Absent or `null` means
forever (the default); positive integers enable the scheduled purge
worker (v0.9). Once a receipt is past its tenant's retention window,
the worker transitions its row to `status = "tombstoned"` (soft delete
— rows persist for forensic lookup of "an audit record existed and was
retired"). The default list endpoint hides tombstoned rows; pass
`?include_tombstoned=true` to surface them. Detail lookup
(`GET /v1/receipts/{id}`) always returns tombstoned rows with the
`status` and `tombstoned_at` fields populated.

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

## HMAC signing (v0.9)

Receipts can be tamper-evidently signed under an operator-provided
HMAC-SHA256 key. The body's `receipt_signature` slot (reserved in
v0.8) is populated when:

1. The operator has configured a non-empty
   `STATEWAVE_RECEIPT_SIGNING_KEYS` map at process startup —
   `{"<key_id>": "<base64-of-32-or-more-bytes>"}`. The server **never
   persists raw signing keys to the database**; they live only in
   this process's in-memory config, sourced from env or a
   secret-manager mount.
2. The tenant's `tenant_configs.config.receipt_signing_key_id`
   names a `key_id` that's present in that map.

When both are true, the body's canonical form (sorted keys, no
whitespace, UTF-8, `receipt_signature` excluded; tagged as
`hmac-sha256-canonical-v1`) is HMAC-SHA256-signed and the signature
stored alongside `receipt_signature_key_id` and
`receipt_signature_algorithm`. Both metadata fields are inside the
signature's coverage, so an attacker who swaps the key_id or
algorithm on a stored receipt invalidates the signature too.

**Failure modes are fail-open** — agent serving must never be
blocked by audit infrastructure:

- Invalid config at startup (bad JSON, key shorter than 32 bytes,
  base64 corruption) **fails the server boot**. Operators see the
  problem before any unsigned receipt slips out.
- Runtime signing failure (tenant points at a `key_id` the operator
  hasn't loaded in this process, or an unlikely hash error) **emits
  the receipt unsigned** with a structured warning
  (`receipt_signing_key_unavailable` or `receipt_signing_failed`).
  Verify reports `valid: null, reason: "key_unavailable"` for the
  former; `no_signature` for the latter.

**Rotation** works by adding a new `key_id` to operator config
alongside the old one, then updating
`tenant_configs.config.receipt_signing_key_id` to the new id. New
receipts sign with the new key; historical receipts remain
verifiable as long as the old `key_id` stays loaded. Removing the
old `key_id` flips its receipts to `valid: null, reason:
"key_unavailable"` — they're still inspectable, just not
re-verifiable on this binary.

`GET /v1/receipts/{id}/verify` returns:

```json
{
  "valid": true | false | null,
  "key_id": "<operator key_id>" | null,
  "algorithm": "hmac-sha256-canonical-v1" | null,
  "reason": "ok" | "signature_mismatch" | "key_unavailable" | "no_signature" | "unsupported_algorithm"
}
```

`valid: null` is the "we couldn't determine" verdict (no signature
on the row, or the key isn't loaded, or the algorithm string names
a variant this binary doesn't implement). `valid: false` is reserved
for "we checked the math and the signature doesn't cover the body."
The response **never** returns the key bytes, a hash of them, or a
length hint.

Pre-v0.9 receipts have no signature columns populated and verify
cleanly as `{valid: null, reason: "no_signature"}` — never a 500.

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

## Shipped after this doc was written

- **Sensitivity-label / policy layer** ([#50](https://github.com/smaramwbc/statewave/issues/50)) — shipped v0.8. `receipt.policy.filters_applied/skipped` populates from the active bundle; `policy_mode` toggles `log_only` vs `enforce` per tenant.
- **HMAC signing** ([#157](https://github.com/smaramwbc/statewave/issues/157)) — shipped v0.9. See HMAC signing section above.
- **Scheduled retention worker** ([#156](https://github.com/smaramwbc/statewave/issues/156)) — shipped v0.9. `cleanup_expired_receipts` tombstones rows past `tenant_configs.receipt_retention_days`; partial index keeps it cheap.
- **Receipt-driven replay** ([#159](https://github.com/smaramwbc/statewave/issues/159)) — shipped v0.9. `mode: "as_of_replay"` receipts emitted by `POST /v1/receipts/{id}/replay`; design + diff envelope in [`docs/replay.md`](replay.md).
- **Auto-labeling** ([#158](https://github.com/smaramwbc/statewave/issues/158)) — shipped v0.9. Heuristic detectors stamp advisory `suggested_labels` on memories; the policy evaluator never reads them. Operator review + promote via the admin app (#160). See [`docs/auto-labeling.md`](auto-labeling.md).
- **Per-tenant residency** ([#161](https://github.com/smaramwbc/statewave/issues/161)) — shipped v0.9. `STATEWAVE_REGION` + `tenant_configs.config.region` enforced at the application layer. See [`docs/residency.md`](residency.md).

## Still out of scope (v0.9)

- Review-time redaction UI for receipts.
- Cross-tenant receipt aggregation / fleet-wide audit views. Single-tenant audit ships; federated cross-region audit is an explicit future surface, not implicit access (see `docs/residency.md`).
- KMS / Vault-backed signing (architecture is compatible — a future PR swaps the key resolver behind the same `receipt_signing_keys` settings field; v0.9 reads keys from env / secret-manager mount).
- Asymmetric signatures (the `algorithm` field reserves space for `ed25519-canonical-v1` etc.; v0.9 ships HMAC only).
- Bulk re-signing of pre-v0.9 receipts (forward-only signing — they verify as `no_signature`).
- Byte-for-byte historical replay (memory snapshots). v0.9 ships `current code + original policy`; the data model leaves room for memory snapshots without a schema break.
