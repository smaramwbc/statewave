# Claim-keyed conflict resolution

Statewave resolves contradictory memories with a **hybrid** strategy. By
default — no flag, no opt-in — it adds a wording-independent path on top of the
existing lexical one. This page explains what claims are, how compilers emit
them, and how the resolver uses them.

> This is **not** comprehensive natural-language contradiction understanding.
> It is a deliberately narrow, registry-gated mechanism for a small set of
> single-valued attributes. When in doubt, Statewave omits the claim and falls
> back to the original behavior.

## Claims are optional compiler annotations

A memory may carry an optional **claim envelope** under `metadata_.claim`
describing *what* it asserts:

```json
{
  "claim": {
    "schema_version": 1,
    "key": "employment.current_employer",
    "value": "globex",
    "scope": "single",
    "source": "heuristic"
  }
}
```

It is purely additive: `metadata_` stays arbitrary JSONB, the claim rides inside
it, and a memory without a claim behaves exactly as before. There is **no schema
change, no migration, no backfill, and no required field**. A server upgrade
does not reinterpret or rewrite any stored memory; old and unkeyed memories keep
their legacy behavior, and supersession state only ever changes when a *new*
compile runs the resolver.

## The registry is authoritative

The canonical key vocabulary and each key's **cardinality** live in one
versioned registry (`server/services/claims.py`):

| Cardinality | Keys | Behavior |
|---|---|---|
| `single` | `identity.name`, `employment.current_employer`, `location.current_home`, `billing.primary_payment_processor` | one current value wins |
| `multi` | `tools.used`, `skills`, `preferences`, `payment.processors_used` | values coexist |

A small set of **approved aliases** (`employer`/`works_at`/`company` →
`employment.current_employer`, etc.) is normalized centrally. Anything else — an
unknown key, an unapproved alias, an unsupported `schema_version`, or a malformed
envelope — is **non-authoritative**: it never drives supersession. Cardinality
is *never* taken from caller or model input; the registry decides.

## Compilers omit claims when uncertain

Compilers optimize for **near-zero wrongful claims**, not coverage. Omission is
always preferred to guessing — the resolver is already safe when claims are
absent.

### Heuristic compiler (default, local, no LLM/GPU/credentials)

Coverage is intentionally narrow — only three keys, and only from unambiguous
present-tense first-person triggers:

| Trigger | Key |
|---|---|
| `my name is …` | `identity.name` |
| `i work at …` | `employment.current_employer` |
| `i live in …` | `location.current_home` |

Deliberately **rejected** (memory still emitted, no claim):

- `i'm` / `i am` for names (`I am Bob's friend`), `i'm at` / `i am at` for
  employer (`I'm at the gym`), `i'm from` / `i am from` for home (origin, not
  current residence);
- negation, uncertainty, and history markers (`not`, `might`, `used to`,
  `previously`, `until`, …);
- reported speech inside quotes;
- generic tool usage — `I use Stripe` is **not** a
  `billing.primary_payment_processor` claim;
- the team/group and use/prefer/favorite patterns (no canonical single-valued
  key).

### LLM compiler

The structured output schema gains an **optional** `claim` field. The model is
**untrusted**: it may *propose* a key/value (and explicit temporal bounds), but
Statewave then canonicalizes the key through the registry + approved aliases,
stamps the registry-authoritative scope, normalizes the value, and **drops**
anything it cannot confidently key. A malformed or unknown proposal never fails
the rest of the compile and never persists in an authoritative form. The
granular-extraction objective is unchanged — claims do not reduce or merge the
memories the model would otherwise emit.

## The resolver

Structured claims activate the hybrid resolver (`server/services/conflicts.py`).
For the active memories of a subject:

- **Claim path** (authoritative, per canonical key, single-valued claims only):
  same key + **different normalized value** + **overlapping validity** →
  newest wins (supersede the older). Non-overlapping windows **coexist** as
  history. The legacy path is told to skip these pairs so lexical overlap can
  never undo the temporal/cardinality decision.
- **Legacy path** (lexical Jaccard, plus the widening guard below): everything
  else — unkeyed, malformed, unknown-key, unsupported-version, multi-valued,
  mixed keyed/unkeyed, and single-valued *same-value* duplicates.

### Where the widening guard does not apply

The guard runs on the lexical path only. Two other routes can retire a
narrower memory, and both are deliberate carve-outs rather than oversights.

**The claim path.** When two memories carry the same registered claim key,
they occupy the same single-valued slot and the newer one wins. That is a
declared identity, supplied by the producer, and it is a stronger statement
than any overlap score: the point of declared cardinality is that the system
does not have to infer it. Overriding a declaration with a lexical heuristic
would undo that. So `location.current_home = Berlin` still retires
`"Alice lives in Berlin during the winter months"`, and a qualifier that
matters has to live in the claim rather than beside it.

**Reconcile.** The LLM reconciler can judge a widened restatement an UPDATE
and retire the narrow row. Constraining that means a prompt rule, which no
test can verify and which moves benchmark results. The recorded skips make
how often this happens a question the data can answer first.

## The widening guard

Jaccard overlap is symmetric, so on its own it cannot tell *"later and
different"* from *"later and emptier"*. A broadly worded restatement scores
just as high against a narrow rule as a genuine correction does, and
superseding on that score alone takes the narrow rule's condition out of every
retrieval path:

| | text | outcome before |
|---|---|---|
| T1 | Refunds are approved up to 500 EUR for orders under 30 days | superseded |
| T2 | Refunds are approved up to 500 EUR for orders | survives |

The two score 0.75 against the 0.6 `profile_fact` threshold, and *"under 30
days"* is gone.

The lexical pass therefore checks one more thing before superseding: **does the
newer statement add anything?** It compares two token views of each side —
content words (≥3 characters, stopwords removed) and numbers/month names — and
declines to supersede only when the newer side is a subset on **both** views
and a *strict* subset on at least one. The two views are checked separately
because short numbers never reach the content-word view: a lost `30` in *"under
30 days"* would otherwise look like a stopword edit.

Everything else supersedes exactly as before:

| Shape | Example | Subset? | Result |
|---|---|---|---|
| Value replacement | `Munich` → `Berlin` | neither side | supersedes |
| Narrowing | `Alice` → `Alice Chen` | newer is a superset | supersedes |
| Duplicate / reword | `I use Stripe` → `I use Stripe.` | equal, nothing strict | supersedes |
| **Widening** | drops `under 30 days` | newer is a strict subset | **coexist** |

A skipped pair is not abandoned: a later memory can still supersede the same
row, so a genuine correction arriving after a widening restatement still
retires the stale rule.

### Skips are recorded, not logged

Each refusal is written to `supersession_records` under the rule
`widening_skipped`, with the score and threshold it declined to act on and a
`details` object giving the *size* of the drop (never the dropped tokens —
that table holds no memory text). "How often does the guard fire, and on which
memories" is a query.

A skip row uses the same two id columns as a real supersession, so **every
reader must filter on the rule**. The rules that mean a memory was retired are
enumerated in `SUPERSEDING_RULES` (`server/services/supersession.py`); the
admin relationship endpoint filters every record lookup, including its chain
walk, on that set. Unfiltered, a skip row reports an active, retrievable memory
as superseded.

The LLM reconcile pass (`server/services/reconcile.py`) is **not** covered by
this guard — it can still retire a narrow memory in favour of a broader one by
its own judgement. Constraining that is a prompt change, which no test can
verify and which moves benchmark numbers; the recorded skips make the
deterministic side observable in data first.

### Temporal and cardinality safeguards

- **Cardinality**: `multi` keys never supersede (e.g. `I use Stripe` /
  `I use PayPal` under `payment.processors_used` coexist). Only `single` keys
  newest-win.
- **Temporal**: a missing `valid_to` is open-ended; touching intervals do not
  overlap, so *"Acme until 2020"* and *"Globex from 2020"* coexist — a current
  fact never deletes a historical one.
- **Determinism**: winner order is semantic `valid_from` (only when both claims
  supply it) → `created_at` → stable memory id. Input and DB return order never
  affect the result.

## Known limitations (initial key set)

- Only four single-valued keys are recognized; everything else is unkeyed and
  uses legacy behavior. The set is intentionally minimal and may grow only with
  evidence.
- The heuristic compiler emits claims for three of those keys and never invents
  temporal bounds; richer extraction is the LLM compiler's job.
- Value comparison is normalized-string equality, not entity resolution
  (`Acme` and `Acme Corp` are treated as different values).
- `employment.current_employer` is for *current* employment; explicitly
  historical statements are left unkeyed by the heuristic rather than forced
  into a current-employer key.
