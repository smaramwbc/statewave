# Receipt replay

Re-run a historical state-assembly retrieval against **current
memories** using the **original policy bundle** captured on the
receipt. Emits a fresh `mode="as_of_replay"` receipt linked back to
the source, plus a diff envelope describing what changed.

Introduced in **v0.9 (issue #159)**. The runtime half of v0.9
governance — the ingest half is [auto-labeling](auto-labeling.md).

## Semantic: current code + original policy

Replay is **not** byte-for-byte reproduction. The user-approved
design splits responsibility between two halves:

| Source of truth | At replay time |
|------------------|----------------|
| Memories         | **Current** — added/removed/superseded since emission |
| Compiler & scoring code | **Current** — whatever is deployed now |
| Policy bundle    | **Original** — frozen on the receipt's `policy_snapshot` |

This lets an auditor answer the question Statewave's accountability
story was always meant to answer: *given the same rules my org had
at the time, would I make the same retrieval decision today?* — and
see explicitly where the answer would differ, separated into
"data drift" (memory churn) and "policy drift" (which is *zero* in
replay; the snapshot YAML is replayed verbatim even if the live
bundle has since been overwritten).

For true byte-for-byte historical reproduction we would need a
memory snapshot too — that lands in a future PR as a separate
feature. Replay's design intentionally leaves room for that
extension without a schema break.

## How it works

1. **Emission** — every v0.9+ receipt carries
   `policy_snapshot`, a JSONB envelope embedded both inside the
   signed body and into a denormalised column for fast indexing.
   This applies to both retrieval receipts (`/v1/context`) and
   handoff receipts (`/v1/handoff`) — both paths are replayable
   symmetrically.

   ```json
   {
     "bundle_hash": "<sha256>" | null,
     "bundle_yaml": "<verbatim YAML>" | null,
     "captured_at": "<ISO-8601 UTC>"
   }
   ```

   A null inner pair (`bundle_hash` AND `bundle_yaml` both null) is
   a *valid* snapshot recording "no policy bundle was active at
   emission" — that state replays cleanly against the no-policy
   fallback. A NULL **column** instead means "pre-v0.9 receipt,
   never captured a snapshot" and is not replayable.

2. **Replay** — `POST /v1/receipts/{id}/replay`:

   - Loads the original receipt (tenant-scoped 404).
   - Refuses with 422 if the receipt is unreplayable (see below).
   - Parses the snapshot's `bundle_yaml` into a PolicyBundle.
   - Calls the same `assemble_context()` path that `/v1/context`
     uses, with the snapshot bundle injected (bypassing the live
     `policy_bundles` row) and `mode="as_of_replay"`.
   - Emits a fresh receipt with `parent_receipt_id` pointing back
     at the source.
   - Computes the diff envelope by comparing the original and the
     new receipt bodies.

3. **Response**:

   ```json
   {
     "original_receipt_id": "...",
     "replay_receipt_id": "...",
     "diff": {
       "context_hash": {
         "original": "...",
         "replay":   "...",
         "changed":  true
       },
       "selected_entries": {
         "added":   [ <entry>, ... ],
         "removed": [ <entry>, ... ],
         "common":  <int>
       },
       "filters_applied": {
         "added":   [ <filter>, ... ],
         "removed": [ <filter>, ... ]
       }
     }
   }
   ```

   Entries are matched by their `memory_id` / `episode_id`, so a
   re-ranked-but-still-present entry is reported under `common`,
   not `added` + `removed`. Filters are matched by
   `(rule_id, memory_id, action)`.

## Refusals (HTTP 422)

The response envelope uses Statewave's standard error shape; the
machine-readable code is on `error.code`:

| `error.code`                            | When                                                                 |
|-----------------------------------------|----------------------------------------------------------------------|
| `unreplayable.missing_policy_snapshot`  | Pre-v0.9 receipt. The column is NULL — no snapshot was captured.    |
| `unreplayable.nested_replay`            | Receipt is itself a replay. v0.9 ships one level only.              |
| `unreplayable.invalid_snapshot`         | Snapshot YAML failed to parse. Tampering or corruption — see below. |

Receipt-not-found returns plain 404 with the same wire shape as
the read endpoints.

## Failure modes

* **Replay receipt write fails** — the diff envelope is still
  returned (with the original entries listed under `removed` so
  the caller can still see what was on the source). `replay_receipt_id`
  is null. Same fail-open contract as the rest of the receipt
  surface — agent serving must not be blocked by audit infra.

* **Snapshot YAML tampered** — returns 422 `invalid_snapshot`. The
  receipt body's HMAC signature would already have caught a
  rewrite of the canonical fields; replay's parse failure is the
  belt-and-suspenders backstop for someone who edits the column
  directly via psql.

* **Original bundle deleted from `policy_bundles`** — irrelevant.
  The snapshot is self-contained; replay does not consult the live
  bundles table.

## What replay does NOT do

* **Does not modify the original receipt.** The original is
  immutable; the replay is a new row pointing back at it.
* **Does not write to `sensitivity_labels` or memory state.**
  Replay is read-only on memories.
* **Does not run nested replays.** Replaying a replay returns 422.
  The audit trail is still recoverable by walking
  `parent_receipt_id` back to the original.
* **Does not reproduce historical memory state.** Memories that
  have been tombstoned since emission do not come back; new
  memories show up in `selected_entries.added`. This is
  intentional — see "Semantic" above.

## Operator workflow

A typical post-incident review:

```bash
# 1. Find the receipt for the retrieval under investigation.
curl /v1/receipts/01J5...

# 2. Replay it. The diff tells you exactly what changed.
curl -X POST /v1/receipts/01J5.../replay

# 3. Both receipts (original + replay) remain in /v1/receipts list,
#    linked by parent_receipt_id, so the audit trail is durable.
```

The signature on every receipt (v0.9 #157) plus the snapshot YAML
(v0.9 #159) means an auditor can verify after the fact that:
  1. The receipt body has not been tampered with (signature check), AND
  2. The exact policy rules used at emission are still recoverable
     (the YAML is right there in the receipt), AND
  3. Re-running the decision today produces a diff against the
     current memory state — separating "the rules changed" from
     "the data changed" in incident reviews.

## Pre-v0.9 receipts

Receipts emitted by Statewave ≤ v0.8 carry no `policy_snapshot`.
They remain queryable, list-able, and HMAC-verifiable (if signed),
but `POST /v1/receipts/{id}/replay` refuses them with 422
`unreplayable.missing_policy_snapshot`.

This is intentional: we cannot synthesise a snapshot retroactively
without guessing which policy bundle was active at the time, which
defeats the entire point. Fresh v0.9+ traffic carries snapshots,
and the operator's audit trail starts there.
