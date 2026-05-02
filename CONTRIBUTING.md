# Contributing to Statewave

Thanks for your interest in contributing — Statewave is built in the open and
external contributions are very welcome.

This document covers the contribution process and the licensing implications
of contributing to a dual-licensed project.

## Ways to contribute

- **Bug reports** — open a [GitHub issue](https://github.com/smaramwbc/statewave/issues)
  with reproduction steps, expected vs. actual behavior, and version info.
- **Feature requests** — open an issue describing the use case and the
  problem you are trying to solve. Concrete use cases beat abstract feature
  ideas.
- **Pull requests** — see "Pull request process" below.
- **Documentation** — improvements to the [statewave-docs](https://github.com/smaramwbc/statewave-docs)
  repo are equally valuable.

## Development setup

See the [Quick start](README.md#quick-start) section in the README for the
canonical setup. In short:

```bash
docker compose up db -d
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,llm]"
alembic upgrade head
pytest tests/ -v
```

Please run `ruff` and the test suite locally before opening a PR.

## Pull request process

1. **Open an issue first** for non-trivial changes so we can align on
   approach before you invest in code.
2. **Branch from `main`**, keep PRs focused, and prefer small commits with
   clear messages.
3. **Add tests** for new behavior. Statewave is infrastructure — coverage
   matters more than feature volume.
4. **Update docs** if you change public API, configuration, or behavior.
5. **Pass CI** — lint, type checks, and tests must be green.
6. **Describe the change** in the PR body: motivation, approach, and any
   tradeoffs considered.

## Licensing of contributions

Statewave is dual-licensed under AGPLv3 and the Statewave Commercial
License (see [LICENSING.md](LICENSING.md)). By contributing — opening a
pull request, sending a patch, or otherwise submitting work — you agree
that your contribution may be distributed under that same dual-license
model: AGPLv3 **and** the Statewave Commercial License. You retain
copyright in your work; we just need the right to ship it under both
license tracks so the dual-licensing model keeps working.

To keep that arrangement on a clean footing, we use a lightweight CLA
workflow for non-trivial contributions:

- **Non-trivial contributions** — new features, refactors, behavioral
  changes, anything beyond a small fix — may require a **signed
  Contributor License Agreement (CLA)** before merge. A maintainer will
  share the CLA text and link when relevant; it's a one-time step per
  contributor.
- **Trivial patches** — typo fixes, documentation corrections, small
  patches — may be accepted without a full CLA at the maintainers'
  discretion.
- **DCO** (`git commit -s`, Developer Certificate of Origin) may be used
  as a lightweight interim process if a CLA workflow is not yet in place
  for your contribution. DCO is not a long-term substitute for the CLA
  where commercial relicensing is involved.

If your employer has rights to your work, please make sure they have
authorized the contribution before submitting.

## Code style

- Python 3.11+, formatted with `ruff` (settings in `pyproject.toml`).
- Type hints on public APIs.
- Match the surrounding code's conventions; prefer small, composable
  functions; prefer clear names over comments.

## Reporting security issues

Please **do not** open a public issue for security vulnerabilities. See
[SECURITY.md](SECURITY.md) for the coordinated disclosure process.

## Questions

- General questions: [GitHub Discussions](https://github.com/smaramwbc/statewave/discussions)
- Licensing questions: [licensing@statewave.ai](mailto:licensing@statewave.ai)
- Security: [security@statewave.ai](mailto:security@statewave.ai)
