# Memory templates

Memory templates are declarative, versioned scaffolds for the kinds of
information a team ingests over and over — a support handoff, a project
decision, an incident summary. A template fixes the *shape* of that
pattern once; callers then supply values and Statewave produces a
consistent, provenance-tagged episode.

Templates exist to make adoption easy without adding magic:

- **Pure data.** A template is a YAML file. No code runs inside it.
- **Deterministic.** Applying a template is plain string substitution —
  the same values always produce the same bytes.
- **Provenance is explicit.** Every episode a template produces records
  the template id and version, both in its `payload` and in
  `metadata.template`.
- **Composes, doesn't replace.** A template produces an ordinary
  episode. It flows through the normal ingest → compile → context
  pipeline; the compiler is not involved in templating and is not
  changed by it.

## Bundled templates

Five templates ship in `server/templates/`:

| Template id | Episode type | Purpose |
|---|---|---|
| `customer-support-handoff` | `support.handoff_note` | Escalation note when a conversation changes hands |
| `user-preference` | `profile.preference` | A durable preference an agent should respect |
| `project-decision` | `project.decision` | A lightweight ADR-style decision record |
| `incident-summary` | `incident.summary` | Post-incident summary for repeat-issue detection |
| `account-onboarding` | `account.onboarding` | Starting context for a new account or tenant |

## Template file format

A template is a YAML mapping with these keys:

| Key | Required | Description |
|---|---|---|
| `id` | yes | Stable identifier, `^[a-z0-9][a-z0-9-]*$`. Unique across all templates. |
| `version` | yes | Integer ≥ 1. Bump it whenever the field set or content scaffold changes. |
| `title` | yes | Human-readable name. |
| `description` | no | What the template is for. |
| `episode_type` | yes | The `type` stamped on every episode this template produces. |
| `fields` | yes | List of field definitions (at least one). |
| `content_template` | yes | The content scaffold; `{field_name}` placeholders are substituted on apply. |

Each entry in `fields`:

| Key | Required | Description |
|---|---|---|
| `name` | yes | Field name, `^[a-zA-Z0-9_]+$`. Referenced as `{name}` in `content_template`. |
| `type` | no | `string` (short, default) or `text` (multi-line). Advisory metadata for callers and UIs — both render as plain strings. |
| `required` | no | Whether `apply` rejects a request that omits this field. Defaults to `false`. |
| `description` | no | What the field holds. |

Every `{placeholder}` in `content_template` must name a declared field;
a template that violates this — or has duplicate field names, or a
duplicate `id` — fails the server at startup rather than silently.

## API

### `GET /v1/memory-templates`

Lists every template with its full field schema. Templates are
inspectable by design — callers can render their own forms from this.

### `GET /v1/memory-templates/{template_id}`

Returns one template. `404` if there is no such template.

### `POST /v1/memory-templates/{template_id}/apply`

Validates caller-supplied values against the template and ingests the
resulting episode.

```json
{
  "subject_id": "customer:globex",
  "session_id": "ticket-8842",
  "values": {
    "customer": "Globex Corp",
    "issue": "Duplicate charge on the May invoice",
    "next_owner": "Tier 2 billing"
  }
}
```

Validation is strict — an unknown field, a missing required field, or a
non-string value is rejected with `422`. `404` if the template does not
exist. On success the response is `201` with the created episode.

The episode's `payload`:

```json
{
  "template_id": "customer-support-handoff",
  "template_version": 1,
  "fields": { "customer": "Globex Corp", "issue": "...", "next_owner": "..." },
  "content": "Customer support handoff — Globex Corp.\n\nActive issue: ..."
}
```

`fields` records exactly what the caller supplied — an omitted optional
field is absent here, and renders as an empty string inside `content`.
`metadata.template` carries `{ "id": ..., "version": ... }` so the
episode's provenance is exact.

## Adding or extending a template

Adding a template is dropping a new `*.yaml` file into
`server/templates/`. It is picked up at the next server start.

Extending an existing template — new field, changed scaffold — should
**bump `version`**. The id stays stable so historical episodes remain
attributable; the version distinguishes which scaffold produced them.

Two deliberate extension points exist:

- **Field `type`.** Today `string` and `text` are advisory metadata.
  A later release can use the type to drive richer validation (dates,
  enums, numbers) without changing the file format or existing
  templates.
- **Template scope.** Templates currently form a single global,
  bundled set. Per-tenant or operator-supplied template catalogues
  would layer on top of the same `MemoryTemplate` schema and the same
  `apply` semantics — the bundled set is the primitive they compose
  with.
