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
                raise ValueError(f"STATEWAVE_KIND_TTL_DAYS is not valid JSON: {exc.msg}") from exc
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

    # Receipt HMAC signing (v0.9, issue #157) — operator-provided keys
    # per key_id. Set via env (JSON object):
    #
    #     STATEWAVE_RECEIPT_SIGNING_KEYS='{"key-2026-01":"<base64-32B>",...}'
    #
    # Per-tenant active key_id is in tenant_configs.config.receipt_signing_key_id.
    # The server NEVER persists raw signing keys to the database — they
    # live only in this process's in-memory config, sourced from
    # env / secret-manager mount. The field is excluded from __repr__ so
    # an accidental settings dump can't leak the keys.
    receipt_signing_keys: dict[str, bytes] = Field(default_factory=dict, repr=False)

    @field_validator("receipt_signing_keys", mode="before")
    @classmethod
    def _parse_receipt_signing_keys(cls, value):
        """Parse + validate the signing-key map at startup. Failures here
        fail the server boot — better than a silent fallback to unsigned
        receipts because a base64 typo went unnoticed.

        Accepts (a) a real dict (in-process tests / env-decoded), (b) a
        JSON-encoded string (defensive — pydantic-settings does the
        JSON decode itself for complex env fields, but this branch
        keeps programmatic construction symmetrical with the env path),
        or (c) None/"" for no signing. Per-key value can be either
        base64 (env path) or raw `bytes` (in-process tests)."""
        import base64
        import json as _json

        if value is None or value == "":
            return {}
        if isinstance(value, dict):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = _json.loads(value)
            except _json.JSONDecodeError as exc:
                raise ValueError(
                    f"STATEWAVE_RECEIPT_SIGNING_KEYS is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(
                    "STATEWAVE_RECEIPT_SIGNING_KEYS must decode to a JSON "
                    f"object of key_id -> base64 key (got {type(parsed).__name__})"
                )
        else:
            raise ValueError(
                "STATEWAVE_RECEIPT_SIGNING_KEYS must be a JSON object or dict, "
                f"got {type(value).__name__}"
            )

        clean: dict[str, bytes] = {}
        for key_id, raw in parsed.items():
            if not isinstance(key_id, str) or not key_id:
                raise ValueError(
                    "STATEWAVE_RECEIPT_SIGNING_KEYS: every key_id must be a "
                    f"non-empty string (got {key_id!r})"
                )
            if isinstance(raw, (bytes, bytearray)):
                key_bytes = bytes(raw)
            elif isinstance(raw, str):
                try:
                    # validate=True rejects non-base64 chars; raises on bad input.
                    key_bytes = base64.b64decode(raw, validate=True)
                except Exception as exc:
                    raise ValueError(
                        f"STATEWAVE_RECEIPT_SIGNING_KEYS[{key_id!r}] is not "
                        "valid base64 (operator config should base64-encode "
                        "the raw secret bytes)"
                    ) from exc
            else:
                raise ValueError(
                    f"STATEWAVE_RECEIPT_SIGNING_KEYS[{key_id!r}] must be a "
                    "base64 string or raw bytes, got "
                    f"{type(raw).__name__}"
                )
            if len(key_bytes) < 32:
                # HMAC-SHA256 best practice: key length ≥ output length.
                # Refuse weaker keys at config-load rather than allow
                # weak signatures into production.
                raise ValueError(
                    f"STATEWAVE_RECEIPT_SIGNING_KEYS[{key_id!r}] is too "
                    f"short: {len(key_bytes)} bytes (minimum 32 for "
                    "HMAC-SHA256)."
                )
            clean[key_id] = key_bytes
        return clean

    # Auto-labeling (v0.9, issue #158) — heuristic detectors that stamp
    # advisory `suggested_labels` on memories at compile time. OFF by default
    # so the v0.9 → existing-tenant upgrade is a no-op until an operator opts
    # in. When enabled, the pipeline runs after MemoryRow construction and
    # writes to `memories.suggested_labels` only. The policy evaluator does
    # not read this column; promotion into authoritative `sensitivity_labels`
    # is a deliberate, audited operator action (admin UI / SDK call).
    #
    # `auto_labeling_provider` is forward-looking: in v0.9 the only supported
    # value is `heuristic` (regex + Luhn detectors). Future LLM-based
    # classifiers land as new provider strings without an API break.
    auto_labeling_enabled: bool = False
    auto_labeling_provider: str = "heuristic"  # "heuristic" only in v0.9

    @field_validator("auto_labeling_provider")
    @classmethod
    def _validate_auto_labeling_provider(cls, value: str) -> str:
        allowed = {"heuristic"}
        if value not in allowed:
            raise ValueError(
                f"STATEWAVE_AUTO_LABELING_PROVIDER must be one of {sorted(allowed)}, got {value!r}"
            )
        return value

    # Multi-tenant (empty = single-tenant mode)
    tenant_header: str = "X-Tenant-ID"
    require_tenant: bool = False

    # Data residency (v0.9, issue #161) — per-region deployment model.
    #
    # `region` declares which region THIS server process is running in
    # (e.g. "eu", "us", "ap"). The residency enforcement layer hard-
    # checks every tenant-scoped request against the tenant's pinned
    # region (`tenant_configs.config.region`): a request for a tenant
    # pinned to "us" arriving at a process with STATEWAVE_REGION="eu"
    # is rejected with HTTP 403 / `residency.mismatch`.
    #
    # Default `None` is single-region mode: no checks, no stamping.
    # Set this whenever you operate >1 region so the application
    # enforces residency at the request boundary instead of relying
    # on DNS / load-balancer hints (which can be misrouted).
    #
    # Naming is up to the operator; we recommend short lowercase
    # identifiers ("eu", "us-east", "ap-south-1") so the value is
    # grep-friendly across logs.
    region: str | None = None

    @field_validator("region")
    @classmethod
    def _validate_region(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        s = value.strip()
        if not s:
            return None
        # Keep the value tight so a leading-space typo doesn't silently
        # produce two distinct regions in audit logs.
        if s != value:
            raise ValueError("STATEWAVE_REGION must not have leading/trailing whitespace")
        if len(s) > 64:
            raise ValueError("STATEWAVE_REGION must be ≤64 characters")
        return s

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
