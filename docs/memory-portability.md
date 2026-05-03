# Memory portability — admin API reference

The Statewave server ships with a small admin API for moving memory in
and out of subjects: starter-pack import, support docs reseed, subject
clone, and full export/import. The Statewave Admin UI is the primary
consumer; the same endpoints can back CLI tooling.

All endpoints live under `/admin/memory/*` and are gated by the existing
`X-API-Key` middleware. There is **no** dependency on GitHub Actions,
Fly.io, Vercel, or any provider-specific runtime.

## Endpoints

| Method + Path | Purpose |
|---|---|
| `GET  /admin/memory/starter-packs` | List the bundled platform packs (manifest metadata only) |
| `POST /admin/memory/starter-packs/import` | Import a bundled pack as a new tenant-owned subject |
| `POST /admin/memory/support/reseed` | Rebuild the shared `statewave-support-docs` subject (idempotent) |
| `POST /admin/memory/clone` | Clone a subject's episodes/memories into a new one |
| `POST /admin/memory/export` | Return a versioned plaintext payload for one or more subjects |
| `POST /admin/memory/import` | Ingest a previously exported payload |
| `POST /admin/docs-pack/reseed` | **Deprecated alias** — backward-compatible shim for `/admin/memory/support/reseed` |

### Backward compatibility — `POST /admin/docs-pack/reseed`

Older operator scripts (and the pre-vendor-neutral admin UI) called
`POST /admin/docs-pack/reseed`. That path used to dispatch a GitHub Actions
workflow which ran the reseed CLI in CI and required a `GITHUB_TOKEN` plus
provider-specific repo/workflow settings.

**The vendor-locked implementation is gone.** The path is preserved as a
deprecated alias that delegates straight to `reseed_support_subject` — the
same in-process service the new `/admin/memory/support/reseed` endpoint
calls. There is no GitHub token, no workflow dispatch, no CI dependency.

| Property | Value |
|---|---|
| Auth | Same `X-API-Key` gate as every other `/admin/*` route |
| Body | Same as the canonical endpoint: `{ "reason": "optional, ≤200 chars" }` (also accepts an empty body) |
| Response | Same shape as `/admin/memory/support/reseed` — `subject_id`, `pack_id`, `pack_version`, `imported_episodes`, `imported_memories`, `reseeded_at`, `reason` |
| Target | Always the configured `support_subject_id` (default `statewave-support-docs`); per-visitor `demo_web_*__statewave-support` subjects are never touched |

Prefer `POST /admin/memory/support/reseed` in new code. The alias may be
removed in a future major version.

## `POST /admin/memory/clone`

Forks a subject's records into a new subject for experiments. The
original subject is **never** mutated — every cloned record gets new
storage-layer ids and provenance pointers back to the source.

### Auth

Same as every other `/admin/*` route: `X-API-Key` header. The optional
`X-Statewave-Operator-Email` header (set by an admin proxy with operator
identity) flows through to the `cloned_by` provenance field on every
copied record.

### Request body

```json
{
  "source_subject_id": "user-1",
  "target_subject_id": "user-1-fork",
  "target_display_name": "Fork experiment",
  "target_tenant_id": "tenant-a",
  "clone_scope": "episodes_memories_sources"
}
```

| Field | Required | Notes |
|---|---|---|
| `source_subject_id` | yes | Subject to clone. Reserved-prefix ids (`demo_web_*`) are accepted on the source side so operators can fork visitor subjects. |
| `target_subject_id` | no | Optional target id. Safe characters only: `^[A-Za-z0-9_.\-:]{1,128}$`. Reserved prefixes are rejected on the target side. Auto-generated as `{source}-clone-<8 hex>` when omitted. |
| `target_display_name` | no | Human-readable label, written to per-record metadata. |
| `target_tenant_id` | no | Tenant scoping for the new subject. |
| `clone_scope` | no | One of `episodes`, `memories`, `episodes_and_memories`, `episodes_memories_sources` (default). |

### Response

```json
{
  "status": "cloned",
  "source_subject_id": "user-1",
  "target_subject_id": "user-1-fork",
  "target_display_name": "Fork experiment",
  "clone_scope": "episodes_memories_sources",
  "episode_count": 42,
  "memory_count": 7,
  "source_count": 0,
  "cloned_at": "2026-05-03T10:00:00.000+00:00"
}
```

### Provenance

Every copied record gets the following fields on its `metadata` /
`provenance` JSON:

| Field | On |
|---|---|
| `cloned_from_subject_id` | episodes + memories |
| `cloned_at` | episodes + memories |
| `cloned_by` | episodes + memories (only when `X-Statewave-Operator-Email` is present) |
| `original_episode_id` | episodes |
| `original_memory_id` | memories |

### Error codes

| Status | Reason |
|---|---|
| `400` | Invalid input (bad subject id, target uses reserved `demo_web_` prefix) |
| `401` / `403` | Missing or invalid `X-API-Key` |
| `404` | Source subject not found (no episodes AND no memories under that id) |
| `409` | Target subject already has data — pick a different id |
| `422` | Pydantic validation failure (e.g. unknown `clone_scope` literal) |
| `500` | Unexpected failure |

### Logging

The `subject_cloned` log event carries `source_subject_id`,
`target_subject_id`, `clone_scope`, the `*_count` fields, and `cloned_by`.
**No memory content is logged** at any point in the clone path.

### Known limitations

- **Sources/citations** are not yet first-class cloneable records in the
  storage schema. The `episodes_memories_sources` scope is honoured
  syntactically and the response always returns `source_count: 0` today.
  When sources land as first-class records, this endpoint will start
  reporting non-zero counts without any contract change.

## Starter pack format

Packs live on disk under `server/starter_packs/<pack_id>/` and are
discovered via `server/starter_packs/index.json`. Each pack directory
contains:

```
<pack_id>/
├── manifest.json
├── episodes.jsonl    (one JSON object per line)
└── memories.jsonl    (one JSON object per line)
```

`manifest.json`:

```json
{
  "format": "statewave-starter-pack",
  "format_version": 1,
  "pack_id": "<id>",
  "display_name": "...",
  "description": "...",
  "version": "1.0.0",
  "created_at": "<ISO 8601>",
  "subject_id_suggestion": "...",
  "episode_count": 0,
  "memory_count": 0,
  "source_count": 0,
  "tags": [...]
}
```

## Export payload format

`POST /admin/memory/export` returns plaintext JSON with the following
top-level shape:

```json
{
  "format": "statewave-memory-payload",
  "format_version": 1,
  "export_id": "<hex>",
  "exported_at": "<ISO 8601>",
  "export_scope": "episodes_memories_sources",
  "subjects": [{ "original_subject_id": "...", "tenant_id": "...", "metadata": {...} }],
  "episodes": [{ "subject_id": "...", "source": "...", "type": "...", "payload": {...}, ... }],
  "memories": [{ "subject_id": "...", "kind": "...", "content": "...", ... }],
  "sources": [],
  "metadata": { "subject_count": N, "episode_count": N, ... }
}
```

`format_version` is required. The server rejects unknown top-level fields
to prevent silent contract drift.

The Statewave Admin UI immediately wraps this payload in an authenticated
`.swmem` container before saving to disk — see
[statewave-admin/README.md](../../statewave-admin/README.md) for the
container format. The server intentionally does **not** see the
passphrase.

## Security limits

| Setting | Default | Purpose |
|---|---|---|
| `STATEWAVE_MEMORY_IMPORT_MAX_BYTES` | 50 MiB | Hard cap on serialized payload size |
| `STATEWAVE_MEMORY_IMPORT_MAX_EPISODES` | 50 000 | Per-import episode count |
| `STATEWAVE_MEMORY_IMPORT_MAX_MEMORIES` | 50 000 | Per-import memory count |
| `STATEWAVE_MEMORY_IMPORT_MAX_SUBJECTS` | 100 | Subjects per export / import |

Subject ids are validated against `^[A-Za-z0-9_.\-:]{1,128}$` and the
reserved prefix `demo_web_` (used by the marketing widget's per-visitor
subjects) is refused.

## What's NOT done by the server

* The `.swmem` archive container is opened and resealed entirely in the
  admin client. The server only ever sees the decrypted payload over the
  authenticated `/admin/*` proxy.
* The server does not derive keys, store passphrases, or log any memory
  content — only subject ids, counts, and pack ids appear in stdout.
* Demo-pack content is intentionally minimal (a few episodes per pack).
  Replace it with richer curated content via the same starter-pack file
  layout when ready.
