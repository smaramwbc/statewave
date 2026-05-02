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

## Licensing of contributions (important)

Statewave is **dual-licensed** under AGPLv3 and a separate Statewave
Commercial License (see [LICENSING.md](LICENSING.md)).

To keep this dual-licensing model viable, **contributions must be
compatible with both licenses**. By submitting a pull request, issue patch,
or any other contribution, you agree that:

1. You are the original author of the contribution, or you have the right
   to submit it under the terms below.
2. Your contribution may be distributed under the GNU Affero General Public
   License v3.0.
3. Your contribution may **also** be distributed under the Statewave
   Commercial License (and other future commercial licenses offered by the
   Statewave project) without additional notice or compensation.
4. You retain copyright in your contribution; you grant the Statewave
   project the rights described above to license and re-license it under
   both the open-source and commercial tracks.

### CLA / DCO

For non-trivial contributions, we may ask you to:

- Sign a **Contributor License Agreement (CLA)** confirming the terms
  above; or
- Use the Linux Foundation–style **Developer Certificate of Origin (DCO)**
  by adding a `Signed-off-by:` trailer to your commits (`git commit -s`).

If a CLA is required for your PR, a maintainer will provide the exact
language and link before merge. Trivial contributions (typo fixes, small
doc edits) generally do not require a CLA, but we may still ask for a DCO
sign-off.

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
