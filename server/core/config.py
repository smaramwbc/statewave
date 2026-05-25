"""Application configuration via environment variables."""

from __future__ import annotations

import json

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from server.core.webhook_events import parse_webhook_event_filter


class Settings(BaseSettings):
    """Central configuration — populated from env vars or .env file."""

    app_name: str = "statewave"
    debug: bool = False

    # Postgres
    database_url: str = "postgresql+asyncpg://statewave:statewave@localhost:5432/statewave"
    database_echo: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8100

    # CORS
    cors_origins: list[str] = ["*"]

    # Token estimation model
    tiktoken_model: str = "cl100k_base"

    # Context assembly defaults
    default_max_context_tokens: int = 4000

    # Compiler
    compiler_type: str = "heuristic"

    # Embeddings.
    #
    # The default `stub` produces deterministic hash vectors so a fresh
    # boot works without LLM credentials, BUT stub vectors are not
    # semantic — semantic search will not work usefully under the stub.
    # Set STATEWAVE_EMBEDDING_PROVIDER=litellm + STATEWAVE_LITELLM_API_KEY +
    # STATEWAVE_LITELLM_EMBEDDING_MODEL for real semantic similarity.
    # `none` disables embeddings entirely (no vector column writes).
    embedding_provider: str = "stub"  # "stub" | "litellm" | "none"
    embedding_dimensions: int = 1536

    # LiteLLM — single provider abstraction. See server/services/llm.py for
    # the provider-neutral env-var contract. LiteLLM dispatches to the
    # underlying SDK (OpenAI, Anthropic, Azure, Bedrock, Ollama, …) by
    # model identifier.
    litellm_api_key: str | None = None
    litellm_model: str = "gpt-4o-mini"  # any LiteLLM model identifier
    litellm_embedding_model: str = "text-embedding-3-small"
    litellm_api_base: str | None = None
    litellm_timeout_seconds: float = 60.0
    litellm_max_retries: int = 2
    litellm_temperature: float = 0.1

    # Authentication (empty = disabled / open access)
    api_key: str | None = None

    # Rate limiting (0 = disabled)
    rate_limit_rpm: int = 0
    rate_limit_strategy: str = "memory"  # "memory" (default) | "distributed"

    # Subject Snapshots (advanced bootstrap — disabled by default)
    enable_snapshots: bool = False

    # Compile job retention (hours, 0 = no cleanup)
    compile_job_retention_hours: int = 168  # 7 days

    # Compile pagination & drain (issue #134)
    # Each call to `POST /v1/memories/compile` processes at most
    # `compile_batch_size` uncompiled episodes — bounded per-call latency
    # keeps the sync route from timing out on large backlogs. The response
    # carries `has_more` and `remaining_episodes` so sync clients can loop.
    # Async mode (`async: true`) drains the subject by looping internally,
    # up to `compile_max_iterations` batches per job (defensive cap
    # against a runaway compiler that fails to advance `last_compiled_at`).
    compile_batch_size: int = 500
    compile_max_iterations: int = 2000

    # ── Memory TTL / expiry policies ────────────────────────────────
    # Per-kind expiry windows. Keys are MemoryKind values
    # ("profile_fact", "episode_summary", "procedure", "artifact_ref");
    # values are positive integers (days). Memories of a configured kind
    # get `valid_to = valid_from + days` stamped on insert; the cleanup
    # loop tombstones any active memory whose `valid_to` has passed,
    # and `/v1/context` retrieval filters out unexpired-but-not-yet-
    # cleaned-up rows so the bound is enforced even between cleanup runs.
    #
    # Empty dict (default) = no expiry for any kind (backwards compatible).
    # Missing kind = no expiry for that kind (a kind not listed is forever).
    # Per-subject / per-tenant TTL policies are out of scope for v0.7
    # — they belong to the policy layer (issue #50). v0.7 ships per-kind
    # globals only so the simple primitive lands first.
    #
    # Set via env: STATEWAVE_KIND_TTL_DAYS='{"episode_summary":30,"artifact_ref":7}'
    kind_ttl_days: dict[str, int] = Field(default_factory=dict)

    @field_validator("kind_ttl_days", mode="before")
    @classmethod
    def _parse_kind_ttl_days(cls, value):
        """Accept either a real dict (in-process construction / test fixtures)
        or a JSON-encoded string (the env-var path). Reject anything else
        eagerly so misconfiguration surfaces at startup, not at first
        memory insert."""
        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"STATEWAVE_KIND_TTL_DAYS is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(
                    "STATEWAVE_KIND_TTL_DAYS must decode to a JSON object "
                    f"(got {type(parsed).__name__})"
                )
        else:
            raise ValueError(
                f"STATEWAVE_KIND_TTL_DAYS must be a dict or JSON string, got {type(value).__name__}"
            )
        # Normalise + validate values. Reject zero / negative explicitly
        # — a zero-day TTL is almost certainly an operator mistake; the
        # right way to disable expiry for a kind is to leave it out of
        # the dict entirely.
        clean: dict[str, int] = {}
        for kind, days in parsed.items():
            if not isinstance(days, int) or isinstance(days, bool):
                raise ValueError(
                    f"STATEWAVE_KIND_TTL_DAYS[{kind!r}] must be an integer, got {type(days).__name__}"
                )
            if days <= 0:
                raise ValueError(
                    f"STATEWAVE_KIND_TTL_DAYS[{kind!r}] must be > 0; "
                    "remove the kind from the dict to disable expiry for it."
                )
            clean[str(kind)] = days
        return clean

    # Webhooks (empty = disabled)
    webhook_url: str | None = None
    webhook_timeout: float = 5.0
    # Event-type allowlist for webhook delivery, comma-separated. Empty =
    # deliver every event (the backward-compatible default). Example:
    #   STATEWAVE_WEBHOOK_EVENTS=memories.compiled,subject.deleted
    # Kept as the raw string so the env var is plain comma-separated;
    # read the parsed, validated list via the `webhook_event_filter`
    # property.
    webhook_events: str = ""

    @field_validator("webhook_events")
    @classmethod
    def _validate_webhook_events(cls, value: str) -> str:
        """Validate the event-type allowlist at construction.

        An unknown event type fails fast at startup instead of silently
        dropping every webhook weeks later. The raw string is returned
        unchanged; `parse_webhook_event_filter` does the checking."""
        parse_webhook_event_filter(value)
        return value

    @property
    def webhook_event_filter(self) -> list[str]:
        """Parsed + validated webhook event-type allowlist.

        An empty list means no filter is configured — every event is
        delivered. See `parse_webhook_event_filter`."""
        return parse_webhook_event_filter(self.webhook_events)

    # Multi-tenant (empty = single-tenant mode)
    tenant_header: str = "X-Tenant-ID"
    require_tenant: bool = False

    # Migration safety
    strict_schema: bool = False  # if True, refuse to start on schema mismatch

    # Statewave Support shared docs subject. Rebuilt by the vendor-neutral
    # `POST /admin/memory/support/reseed` endpoint, which imports the bundled
    # `statewave-support-agent` starter pack from `server/starter_packs/`.
    # No GitHub Actions / Fly / Vercel dependency.
    support_subject_id: str = "statewave-support-docs"
    support_starter_pack_id: str = "statewave-support-agent"

    # Memory import/export hard limits — defence against pathological payloads.
    # Apply uniformly across starter-pack imports, clone, and bulk import.
    memory_import_max_bytes: int = 50 * 1024 * 1024  # 50 MiB
    memory_import_max_episodes: int = 50_000
    memory_import_max_memories: int = 50_000
    memory_import_max_subjects: int = 100

    model_config = {"env_prefix": "STATEWAVE_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
