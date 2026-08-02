FROM python:3.11-slim

WORKDIR /app

# Install third-party dependencies first, from the manifest alone, so this
# layer stays cached when only application code changes. `server/` is not
# present yet, so the wheel built here contains no application code — this step
# is only about the dependencies.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir ".[llm]"

COPY . .

# Now install the application package itself. This step is NOT redundant: the
# install above ran before `COPY . .`, so `server/` is missing from site-packages
# until this point. `start.sh` runs `alembic upgrade head`, and the alembic
# console script does not put the working directory on sys.path, so dropping
# this line makes the container exit with ModuleNotFoundError before uvicorn
# starts. `--no-deps` because the dependencies are already installed above.
RUN pip install --no-cache-dir --no-deps . && chmod +x start.sh

EXPOSE 8100

CMD ["./start.sh"]
