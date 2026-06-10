# AGENTS.md — guide for contributors and coding agents

A short orientation for humans and AI coding agents (GitHub Copilot, Claude,
Cursor, …) working in the **statewave** core repo. For the full contribution
process and licensing, start with [CONTRIBUTING.md](CONTRIBUTING.md).

## What this repo is

Statewave is an open-source memory/context service for AI agents: it records
episodes, compiles them into durable memories, and serves compact, ranked,
token-bounded context over a `/v1` API. This repo is the server and reference
implementation. For the big picture, see the
[architecture overview](https://github.com/smaramwbc/statewave-docs/blob/main/architecture/overview.md).

## Setup, build, test

The canonical steps live in
[CONTRIBUTING.md](CONTRIBUTING.md#development-setup) and the
[README Quickstart](README.md#quickstart) — don't duplicate them, follow
them. In short:

```bash
docker compose up db -d
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]"
alembic upgrade head
pytest tests/
```

Run `ruff` and the test suite before opening a PR; `make test-cold` validates a
full fresh-install path.

**Tip for agents:** unit tests are hermetic with stub providers — run them
without external services via:

```bash
STATEWAVE_EMBEDDING_PROVIDER=stub STATEWAVE_COMPILER_TYPE=heuristic \
  pytest tests/ --ignore=tests/integration
```

## Conventions

- **Code style & testing:** see
  [statewave-docs/dev/conventions.md](https://github.com/smaramwbc/statewave-docs/blob/main/dev/conventions.md).
- **Packages version independently.** The server, the Python SDK (`statewave`),
  and the TypeScript SDK (`@statewavedev/sdk`) version on separate cadences; the
  compatibility axis is the `/v1` API contract, not a shared number. Don't try
  to align version strings across packages.
- **Some "proof" numbers are mirrored across repos** — test counts, eval
  assertion/test counts, and the support-workflow benchmark score. They have a
  single source of truth and a consistency check in `statewave-docs` that runs
  at release time. If you change one of these numbers, keep every surface in
  sync rather than editing a single file.
- **Keep claims accurate and modest.** Describe what the code does; back any
  performance or benchmark claim with a reproducible source, and avoid
  unqualified superlatives.
- **CI enforces a few cross-cutting invariants** — tenant-scoped repository
  queries, bounded pagination params, and a single shared tokenizer. If a guard
  test fails, satisfy the rule rather than weakening the test. See
  [CONTRIBUTING.md](CONTRIBUTING.md#invariants-enforced-by-ci).

## Pull requests

See [CONTRIBUTING.md](CONTRIBUTING.md#pull-request-process). Keep PRs focused,
add tests for behavior changes, and make sure `ruff` and `pytest` pass.

## Optional: give your agent memory of this repo (with Statewave itself)

This project dogfoods Statewave. The easiest way to give your assistant a
queryable project brain for this repo is the **Statewave IDE Companion**
extension for **VS Code / Cursor** (publisher `statewavedev`, available as a preview) — install it from
your editor's extensions marketplace. It exposes your workspace, docs, git
state, structure, and run-commands to Copilot / Cursor / Claude over MCP and
**registers the MCP server for you** (no manual config); it just needs a
Statewave server to talk to (a one-file `docker compose up`). See the
[extension README](https://github.com/smaramwbc/statewave-connectors/blob/main/packages/vscode-extension/README.md).

Prefer to wire it up by hand, or use another MCP client? Run the
[Statewave MCP server](https://github.com/smaramwbc/statewave-docs/blob/main/connectors/mcp.md)
(`@statewavedev/mcp-server`) directly and query subject `repo:smaramwbc/statewave`.
