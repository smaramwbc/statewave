# Server-side evals

Internal regression tools for measuring server behaviour. These hit a live Statewave deployment (admin endpoints included) and are intended for the maintainers of this repo, not for external developers — see [statewave-examples](https://github.com/smaramwbc/statewave-examples) for developer-facing demos.

## eval_docs_support.py

Measures retrieval/ranking quality for the `statewave-support-docs` subject across a canonical question set. For each question:

- **Doc match** — was at least one expected doc cited?
- **Term recall** — fraction of expected substantive terms in the retrieved context
- **Groundability** — at least 2 facts substantive enough for an LLM to ground an answer
- **Citation diversity** — unique doc paths cited across the whole question set

Use it before/after a server-side ranking change to confirm an improvement.

```bash
STATEWAVE_URL=https://statewave-api.fly.dev \
STATEWAVE_API_KEY=... \
python scripts/eval/eval_docs_support.py

# Snapshot for diffing later
python scripts/eval/eval_docs_support.py --out=before.json
# ...make a change, redeploy...
python scripts/eval/eval_docs_support.py --out=after.json
diff <(jq -S . before.json) <(jq -S . after.json)
```

Snapshot JSONs are intentionally not committed — they're per-tuning artifacts.
