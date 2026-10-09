# uv is build-time only: start.sh needs alembic and uvicorn, never uv. Declaring
# it as a stage lets the RUN steps below bind-mount the binary instead of
# COPYing it, so its 36MB never enters a layer of the shipped image (a later
# `rm` could not reclaim it). Pinned to the same release CI installs, so a build
# can never resolve the lockfile differently than CI or a developer did;
# tests/test_uv_pin_consistency.py fails if the two pins drift apart.
FROM ghcr.io/astral-sh/uv:0.12.24 AS uvbin

FROM python:3.11-slim

WORKDIR /app

# uv is the installer because uv.lock is the single source of truth for every
# dependency version that ships. Pinned via the stage above to the same release
# CI installs, so a build can never resolve it differently than CI did.
# pip byte-compiles on install and uv does not, so without this the image would
# ship zero .pyc files where it previously shipped thousands, and every fresh
# container would pay source compilation on its first import. This change is
# meant to be invisible to the running service, so keep the parity.
ENV UV_COMPILE_BYTECODE=1

# Install third-party dependencies first, from the manifest and lockfile alone,
# so this layer stays cached when only application code changes. `server/` is
# not present yet, so nothing installed here contains application code — this
# step is only about the dependencies.
#
# `--locked` rather than `--frozen` is deliberate: it fails the build when
# uv.lock has drifted from pyproject.toml. `--frozen` succeeds on a drifted
# lock and exports the stale pins, which would ship a dependency set nobody
# declared — the failure mode behind #327, invisible to startup, pytest and
# docker-smoke alike because the imports are function-local.
#
# `--no-emit-project` keeps the project itself out of the export. It has to:
# the export carries hashes, which puts the installer in hash-checking mode,
# and that mode rejects the editable project entry outright. The project is
# installed separately below, which is what the second step was always for.
# README.md is deliberately NOT copied here. On the previous pip-based layer it
# had to be, because that step built the project wheel and hatchling reads
# `readme = "README.md"`. This step only runs `uv export`, which reads
# pyproject.toml and uv.lock alone, so including the README would just bust the
# dependency layer on every README edit. The project build below gets it from
# `COPY . .`.
COPY pyproject.toml uv.lock ./
RUN --mount=from=uvbin,source=/uv,target=/bin/uv \
    uv export --locked --extra llm --no-emit-project --no-header -q -o /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY . .

# Now install the application package itself. This step is NOT redundant: the
# install above ran before `COPY . .`, so `server/` is missing from site-packages
# until this point. `start.sh` runs `alembic upgrade head`, and the alembic
# console script does not put the working directory on sys.path, so dropping
# this line makes the container exit with ModuleNotFoundError before uvicorn
# starts. `--no-deps` because the dependencies are already installed above.
#
# Plain `uv pip install` is also deliberate over `uv pip sync` / `--exact`:
# those prune anything not in the requirements, which in a system environment
# means removing pip and setuptools out from under the image.
RUN --mount=from=uvbin,source=/uv,target=/bin/uv \
    uv pip install --system --no-cache --no-deps . && chmod +x start.sh

EXPOSE 8100

CMD ["./start.sh"]
