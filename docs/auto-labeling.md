# Auto-labeling

Heuristic detectors that stamp advisory **suggested_labels** on
memories at compile time. Introduced in **v0.9 (issue #158)** as the
ingest-time half of Statewave's governance story; the runtime half is
the authoritative `sensitivity_labels` column the policy evaluator
reads on every assembly call.

## The two columns

Statewave tracks **two** label columns on `memories`:

| Column                | Source              | Read by policy? | Mutable by detectors? |
|-----------------------|---------------------|-----------------|------------------------|
| `sensitivity_labels`  | Tenant (explicit)   | ✅ Yes          | ❌ Never               |
| `suggested_labels`    | Auto-labeling       | ❌ Never        | ✅ At compile time     |

The split is deliberate. Suggestions are *advisory* — surfaced for
operator review but never used to gate retrieval, refuse a request,
or change retention. A noisy detector cannot tighten policy on real
traffic; the worst it can do is produce a noisy suggestion in the
admin UI. Promotion into the authoritative column is a deliberate,
audited operator action (admin UI / SDK call); v0.9 ships review
only and the promotion endpoint lands in a follow-up.

## Enabling

Auto-labeling is **off by default** so a v0.9 upgrade is a no-op for
existing tenants. To enable it process-wide:

```bash
STATEWAVE_AUTO_LABELING_ENABLED=true
STATEWAVE_AUTO_LABELING_PROVIDER=heuristic   # the only v0.9 provider
```

With the flag on, both compilers (`heuristic` and `llm`) run the
pipeline after MemoryRow construction. The detector pass is in-process
and adds a sub-millisecond hop per memory; there is no network call.

## Label schema

Suggested labels follow `<category>.<specific>`:

| Label              | What it flags                                              |
|--------------------|-------------------------------------------------------------|
| `pii.email`        | RFC-5322-ish email addresses                                |
| `pii.phone`        | E.164 international numbers or grouped national US/EU forms |
| `financial.card`   | 13–19 digit runs that pass the Luhn checksum                |
| `secret.token`     | Known-provider API keys (AWS, GitHub, OpenAI, Google,        |
|                    | Slack) or bearer JWTs                                       |

The catalogue is also returned by
`GET /admin/memories/with-suggested-labels` so an admin UI can build
filter dropdowns without hard-coding the list.

## Example: pipeline output

Given an episode containing

> "Reach me at alice@example.com or +1 415 555 0199. Card on file is
> 4111-1111-1111-1111."

a `MemoryRow` derived from that text will be stamped with:

```json
{
  "suggested_labels": ["financial.card", "pii.email", "pii.phone"],
  "sensitivity_labels": []
}
```

`sensitivity_labels` stays empty unless an operator promotes the
suggestions or the tenant set them explicitly.

## Reviewing suggestions

```http
GET /admin/memories/with-suggested-labels
    ?subject_id=<id>          # optional
    &tenant_id=<id>           # optional
    &label=pii.email          # optional — narrows to one detector
    &limit=50&offset=0
```

Response shape:

```json
{
  "memories": [
    {
      "id": "...",
      "subject_id": "...",
      "tenant_id": null,
      "kind": "profile_fact",
      "content": "alice@example.com",
      "summary": "alice email",
      "suggested_labels": ["pii.email"],
      "sensitivity_labels": [],
      "created_at": "..."
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0,
  "catalogue": [
    {"label": "pii.email",      "description": "Email address ..."},
    {"label": "pii.phone",      "description": "Phone number ..."},
    {"label": "financial.card", "description": "Credit-card-shaped ..."},
    {"label": "secret.token",   "description": "Known-provider API key ..."}
  ]
}
```

The filter is GIN-indexed (migration 0022), so the `label=` overlap
path is cheap even on millions of memories.

## What auto-labeling does **NOT** do (v0.9)

* **Does not refuse or filter retrieval.** The policy evaluator
  ignores `suggested_labels`.
* **Does not change retention.** Memory TTL and the receipt
  retention worker do not key off suggestions.
* **Does not redact content.** Detectors are read-only; the memory
  body is preserved verbatim.
* **Does not promote suggestions automatically.** Promotion into
  `sensitivity_labels` is operator-driven.

These constraints are tested in
`tests/integration/test_auto_labeling.py::test_auto_labeling_never_writes_sensitivity_labels`
and are load-bearing for the governance story.

## Adding a detector

1. Add a pure `detect(text) -> bool` predicate in
   `server/services/auto_labeling/detectors.py`.
2. Wrap it in a `Detector` dataclass with a `<category>.<specific>`
   label and a one-line description.
3. Append the dataclass to the `DETECTORS` tuple.
4. Add positive + negative unit tests in
   `tests/test_auto_labeling.py` and update the registry assertion
   if you intend the new label to be part of the v0.9 contract.

Detectors should bias toward **precision over recall**: a false
positive is a noisy admin row; a flood of false positives undermines
operator trust in the column. Real low-recall gaps are recoverable
by an operator labelling the row by hand.
