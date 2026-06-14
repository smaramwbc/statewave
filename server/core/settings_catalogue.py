"""Static catalogue of every setting the admin UI exposes.

This is the **single source of truth** for what's editable, what's a secret,
what category it falls into, whether it can hot-reload, and whether tenants
can override it. Adding a new env-driven setting here makes it appear on
the admin /settings page automatically.

The catalogue does NOT carry values — values live in env (read by pydantic
`Settings`) plus DB overrides (`system_settings` / `tenant_settings` tables).
The `get_setting()` helper combines them; `SettingSpec` describes what each
key MEANS.

NOT exposed (intentionally) — the admin UI never reads these and never
offers them for edit:

  * ``database_url`` / ``host`` / ``port`` — bootstrap-locked
  * ``region`` / ``strict_schema`` — bootstrap-locked
  * ``receipt_signing_keys`` — comment in core/config.py explicitly forbids
    DB persistence
  * ``database_echo`` / ``debug`` — local-dev only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Kind = Literal["string", "int", "float", "bool", "json", "string_or_null"]
Category = Literal[
    "backend",
    "auth",
    "llm",
    "embeddings",
    "webhooks",
    "rate_limits",
    "memory",
    "labeling",
    "advanced",
]


@dataclass(frozen=True)
class SettingSpec:
    """Per-setting metadata.

    Attributes:
        key: stable identifier used in API + DB (matches the pydantic
            Settings attribute name, lowercase snake_case).
        env_name: the legacy env var name (e.g. STATEWAVE_LITELLM_MODEL).
            Surfaced in the UI so operators can map UI → deployment env.
        category: which UI tab this lands in.
        kind: how the JSON value is validated + how the UI renders it.
        is_secret: when true, the API returns a redacted preview
            (`sk-•••abc`) instead of the raw value; the UI shows a
            write-only field. Audit log redacts likewise.
        hot_reloadable: when true, a PATCH takes effect on the next
            request without a restart. When false, the UI badges the
            row as "Requires restart" and the operator must redeploy.
        tenant_overridable: when true, the setting can also be overridden
            per-tenant via the tenant_settings table.
        description: human-readable explanation (markdown-safe).
        editable: when false, the value is shown read-only with an
            explanation. Used for env-locked settings that still need
            UI visibility (e.g. database_url).
    """

    key: str
    env_name: str
    category: Category
    kind: Kind
    is_secret: bool = False
    # Default False is the honest position: most settings are snapshotted
    # by a singleton at process startup (webhooks.configure, llm compiler,
    # rate-limit middleware, …). The override layer applies at startup
    # (see ``dynamic_settings.apply_db_overrides_to_settings``), so an
    # edit is live on next restart. Per-call-reading settings flip this
    # to True explicitly.
    hot_reloadable: bool = False
    tenant_overridable: bool = False
    description: str = ""
    editable: bool = True
    # Fixed set of legal values, when the setting is an enum-shaped string.
    # Rendered as a `<select>` in the admin UI (no typos possible) and
    # enforced server-side in ``dynamic_settings._validate_value``. None =
    # any value of `kind` is allowed.
    allowed_values: tuple[str, ...] | None = None
    # Per-kind numeric bounds. Both inclusive; either may be None for
    # one-sided checks. Surfaced to the UI so an HTML `<input type="number">`
    # can prevent out-of-range submission, and re-checked server-side.
    min_value: float | int | None = None
    max_value: float | int | None = None
    # Lightweight string pattern. Currently used to mark URL-shaped fields
    # so the UI can validate format on blur. Server-side this is advisory
    # — pydantic catches malformed URLs at use-time anyway.
    format: str | None = None  # "url" | None


# Lowercase key → SettingSpec. Key matches the pydantic Settings attribute.
CATALOGUE: dict[str, SettingSpec] = {
    # ─── Backend (read-only — bootstrap-locked) ──────────────────────────
    "debug": SettingSpec(
        key="debug",
        env_name="STATEWAVE_DEBUG",
        category="backend",
        kind="bool",
        hot_reloadable=False,
        description=(
            "Verbose logging. Off in production — debug logs can leak "
            "prompts, full episode payloads, and internal IDs into "
            "your log pipeline. Applied on next server restart. The "
            "production-readiness check fires when this is true."
        ),
    ),
    "database_url": SettingSpec(
        key="database_url",
        env_name="STATEWAVE_DATABASE_URL",
        category="backend",
        kind="string",
        is_secret=True,
        hot_reloadable=False,
        editable=False,
        description=(
            "Postgres connection string. Set at deploy time via env; "
            "editing at runtime would brick the running process."
        ),
    ),
    "host": SettingSpec(
        key="host",
        env_name="STATEWAVE_HOST",
        category="backend",
        kind="string",
        hot_reloadable=False,
        description=(
            "TCP bind interface (e.g. `0.0.0.0` to accept any host, "
            "`127.0.0.1` to restrict to loopback). Applied on next "
            "server restart."
        ),
    ),
    "port": SettingSpec(
        key="port",
        env_name="STATEWAVE_PORT",
        category="backend",
        kind="int",
        hot_reloadable=False,
        description="TCP bind port. Applied on next server restart.",
        min_value=1,
        max_value=65535,
    ),
    "region": SettingSpec(
        key="region",
        env_name="STATEWAVE_REGION",
        category="backend",
        kind="string_or_null",
        hot_reloadable=False,
        description=(
            "Server region label used by the residency layer "
            "(e.g. `eu`, `us-east`). Tenants pinned to a different "
            "region are rejected with 403. Applied on next restart "
            "so in-flight requests aren't re-routed mid-flight."
        ),
    ),
    "strict_schema": SettingSpec(
        key="strict_schema",
        env_name="STATEWAVE_STRICT_SCHEMA",
        category="backend",
        kind="bool",
        hot_reloadable=False,
        description=(
            "When true, the server refuses to boot if migrations are "
            "behind. Use in production to fail loudly on a missed "
            "deploy step instead of running on a stale schema."
        ),
    ),
    # ─── Auth & access ─────────────────────────────────────────────────────
    "api_key": SettingSpec(
        key="api_key",
        env_name="STATEWAVE_API_KEY",
        category="auth",
        kind="string_or_null",
        is_secret=True,
        hot_reloadable=False,
        description=(
            "API key the backend requires on every request. Empty = open "
            "access. Changes take effect on next request, but a wrong "
            "value can lock the admin out — pair with the deployment env "
            "as fallback."
        ),
    ),
    "require_tenant": SettingSpec(
        key="require_tenant",
        env_name="STATEWAVE_REQUIRE_TENANT",
        category="auth",
        kind="bool",
        hot_reloadable=False,
        description=(
            "When true, every request must carry the tenant header. "
            "Takes effect on restart."
        ),
    ),
    "tenant_header": SettingSpec(
        key="tenant_header",
        env_name="STATEWAVE_TENANT_HEADER",
        category="auth",
        kind="string",
        hot_reloadable=False,
        description="HTTP header carrying tenant id. Default: X-Tenant-ID.",
    ),
    "cors_origins": SettingSpec(
        key="cors_origins",
        env_name="STATEWAVE_CORS_ORIGINS",
        category="auth",
        kind="json",
        hot_reloadable=False,
        description=(
            "JSON array of allowed browser origins (e.g. "
            "`[\"https://admin.example.com\"]`). `[\"*\"]` allows any "
            "origin — fine for local dev, refuse in production. Applied "
            "on next restart. A wrong value can lock the admin frontend "
            "out of the backend — verify in a staging environment first."
        ),
    ),
    # ─── LLM ─────────────────────────────────────────────────────────────
    "litellm_model": SettingSpec(
        key="litellm_model",
        env_name="STATEWAVE_LITELLM_MODEL",
        category="llm",
        kind="string",
        tenant_overridable=True,
        description=(
            "LiteLLM model identifier (e.g. `gpt-4o-mini`, "
            "`anthropic/claude-3-5-sonnet-latest`)."
        ),
    ),
    "litellm_api_key": SettingSpec(
        key="litellm_api_key",
        env_name="STATEWAVE_LITELLM_API_KEY",
        category="llm",
        kind="string_or_null",
        is_secret=True,
        tenant_overridable=True,
        description="API key forwarded to the LLM provider via LiteLLM.",
    ),
    "litellm_api_base": SettingSpec(
        key="litellm_api_base",
        env_name="STATEWAVE_LITELLM_API_BASE",
        category="llm",
        kind="string_or_null",
        description=(
            "LiteLLM base URL. Set for Azure OpenAI, self-hosted Ollama, "
            "or a LiteLLM proxy."
        ),
        format="url",
    ),
    "litellm_timeout_seconds": SettingSpec(
        key="litellm_timeout_seconds",
        env_name="STATEWAVE_LITELLM_TIMEOUT_SECONDS",
        category="llm",
        kind="float",
        description="LLM request timeout in seconds.",
        min_value=1.0,
        max_value=600.0,
    ),
    "litellm_max_retries": SettingSpec(
        key="litellm_max_retries",
        env_name="STATEWAVE_LITELLM_MAX_RETRIES",
        category="llm",
        kind="int",
        description="Max retries on transient LLM provider errors.",
        min_value=0,
        max_value=10,
    ),
    "litellm_temperature": SettingSpec(
        key="litellm_temperature",
        env_name="STATEWAVE_LITELLM_TEMPERATURE",
        category="llm",
        kind="float",
        description="Sampling temperature for compile LLM calls.",
        min_value=0.0,
        max_value=2.0,
    ),
    "litellm_compile_max_tokens": SettingSpec(
        key="litellm_compile_max_tokens",
        env_name="STATEWAVE_LITELLM_COMPILE_MAX_TOKENS",
        category="llm",
        kind="int",
        description=(
            "Output-token cap for the LLM compiler. Cap, not target — "
            "reasoning models spend hidden tokens, lower values risk "
            "truncating their JSON output."
        ),
        min_value=64,
        max_value=128000,
    ),
    "compiler_type": SettingSpec(
        key="compiler_type",
        env_name="STATEWAVE_COMPILER_TYPE",
        category="llm",
        kind="string",
        description="Memory compiler: `heuristic` (deterministic) or `llm`.",
        allowed_values=("heuristic", "llm"),
    ),
    # ─── Embeddings ──────────────────────────────────────────────────────
    "embedding_provider": SettingSpec(
        key="embedding_provider",
        env_name="STATEWAVE_EMBEDDING_PROVIDER",
        category="embeddings",
        kind="string",
        hot_reloadable=False,
        description=(
            "Embedding backend: `stub` (deterministic, no semantic search), "
            "`litellm`, or `none` (disable embeddings). Singleton built at "
            "startup — restart required."
        ),
        allowed_values=("stub", "litellm", "none"),
    ),
    "litellm_embedding_model": SettingSpec(
        key="litellm_embedding_model",
        env_name="STATEWAVE_LITELLM_EMBEDDING_MODEL",
        category="embeddings",
        kind="string",
        tenant_overridable=True,
        description="LiteLLM embedding model (e.g. `text-embedding-3-small`).",
    ),
    # ─── Webhooks ────────────────────────────────────────────────────────
    "webhook_url": SettingSpec(
        key="webhook_url",
        env_name="STATEWAVE_WEBHOOK_URL",
        category="webhooks",
        kind="string_or_null",
        tenant_overridable=True,
        description=(
            "Destination URL for outbound webhooks. Empty = webhooks "
            "disabled."
        ),
        format="url",
    ),
    "webhook_timeout": SettingSpec(
        key="webhook_timeout",
        env_name="STATEWAVE_WEBHOOK_TIMEOUT",
        category="webhooks",
        kind="float",
        description="Webhook delivery timeout in seconds.",
        min_value=0.5,
        max_value=120.0,
    ),
    "webhook_events": SettingSpec(
        key="webhook_events",
        env_name="STATEWAVE_WEBHOOK_EVENTS",
        category="webhooks",
        kind="string",
        tenant_overridable=True,
        description=(
            "Comma-separated event-type allowlist. Empty = deliver every "
            "event (default)."
        ),
    ),
    # ─── Rate limits ─────────────────────────────────────────────────────
    "rate_limit_rpm": SettingSpec(
        key="rate_limit_rpm",
        env_name="STATEWAVE_RATE_LIMIT_RPM",
        category="rate_limits",
        kind="int",
        tenant_overridable=True,
        description="Requests-per-minute cap. 0 disables rate limiting.",
        min_value=0,
        max_value=1_000_000,
    ),
    "rate_limit_strategy": SettingSpec(
        key="rate_limit_strategy",
        env_name="STATEWAVE_RATE_LIMIT_STRATEGY",
        category="rate_limits",
        kind="string",
        hot_reloadable=False,
        description=(
            "`memory` (default, single-process) or `distributed` "
            "(Postgres-backed, multi-replica). Middleware is wired at "
            "startup — switching requires restart."
        ),
        allowed_values=("memory", "distributed"),
    ),
    # ─── Memory ──────────────────────────────────────────────────────────
    "compile_batch_size": SettingSpec(
        key="compile_batch_size",
        env_name="STATEWAVE_COMPILE_BATCH_SIZE",
        category="memory",
        kind="int",
        description=(
            "Max episodes per compile-call batch. Higher = fewer batches "
            "but riskier per-call latency."
        ),
        min_value=1,
        max_value=10_000,
    ),
    "compile_max_iterations": SettingSpec(
        key="compile_max_iterations",
        env_name="STATEWAVE_COMPILE_MAX_ITERATIONS",
        category="memory",
        kind="int",
        description=(
            "Defensive cap for async compile loops to prevent a "
            "non-advancing compiler from running unbounded."
        ),
        min_value=1,
        max_value=100_000,
    ),
    "compile_job_retention_hours": SettingSpec(
        key="compile_job_retention_hours",
        env_name="STATEWAVE_COMPILE_JOB_RETENTION_HOURS",
        category="memory",
        kind="int",
        description="Hours to keep finished compile_jobs rows. 0 = forever.",
        min_value=0,
        max_value=8760,
    ),
    "kind_ttl_days": SettingSpec(
        key="kind_ttl_days",
        env_name="STATEWAVE_KIND_TTL_DAYS",
        category="memory",
        kind="json",
        description=(
            "Per-kind expiry in days, e.g. "
            "`{\"episode_summary\": 30, \"artifact_ref\": 7}`. Missing "
            "kinds never expire."
        ),
    ),
    "memory_import_max_bytes": SettingSpec(
        key="memory_import_max_bytes",
        env_name="STATEWAVE_MEMORY_IMPORT_MAX_BYTES",
        category="memory",
        kind="int",
        description="Hard cap on a single import payload's serialized size.",
        min_value=1,
    ),
    "memory_import_max_episodes": SettingSpec(
        key="memory_import_max_episodes",
        env_name="STATEWAVE_MEMORY_IMPORT_MAX_EPISODES",
        category="memory",
        kind="int",
        description="Per-import episode count cap.",
    ),
    "memory_import_max_memories": SettingSpec(
        key="memory_import_max_memories",
        env_name="STATEWAVE_MEMORY_IMPORT_MAX_MEMORIES",
        category="memory",
        kind="int",
        description="Per-import memory count cap.",
    ),
    "memory_import_max_subjects": SettingSpec(
        key="memory_import_max_subjects",
        env_name="STATEWAVE_MEMORY_IMPORT_MAX_SUBJECTS",
        category="memory",
        kind="int",
        description="Per-import subject count cap.",
    ),
    # ─── Labeling ────────────────────────────────────────────────────────
    "auto_labeling_enabled": SettingSpec(
        key="auto_labeling_enabled",
        env_name="STATEWAVE_AUTO_LABELING_ENABLED",
        category="labeling",
        kind="bool",
        description=(
            "Stamp `suggested_labels` on memories at compile time. Advisory "
            "only — promotion to authoritative labels is a separate "
            "operator action."
        ),
    ),
    "auto_labeling_provider": SettingSpec(
        key="auto_labeling_provider",
        env_name="STATEWAVE_AUTO_LABELING_PROVIDER",
        category="labeling",
        kind="string",
        description="Currently only `heuristic` (regex + Luhn) is supported.",
        allowed_values=("heuristic",),
    ),
}


def get_spec(key: str) -> SettingSpec | None:
    """Return the catalogue entry for a key, or None if unknown."""
    return CATALOGUE.get(key)


def keys_by_category(category: Category) -> list[str]:
    """Stable-ordered keys in a category."""
    return [k for k, spec in CATALOGUE.items() if spec.category == category]


def is_secret(key: str) -> bool:
    """True iff the catalogue marks the setting as a secret. Defaults to
    treating unknown keys as secret (defence-in-depth)."""
    spec = CATALOGUE.get(key)
    return spec.is_secret if spec is not None else True


def redact(value: Any, *, key: str | None = None, kind: Kind | None = None) -> Any:
    """Redact a secret value for safe display in the API + audit log.

    For string secrets we show the last 3 characters prefixed with
    `sk-•••`, so operators recognise the right key without exposing the
    full value. For non-string secrets (rare) we collapse to the
    sentinel `"•••"`. Empty / null values pass through unredacted so
    the UI can clearly show "not set".
    """
    if value is None or value == "":
        return value
    inferred_kind = kind
    if inferred_kind is None and key is not None:
        spec = CATALOGUE.get(key)
        if spec is not None:
            inferred_kind = spec.kind
    if inferred_kind in ("string", "string_or_null") and isinstance(value, str):
        tail = value[-3:] if len(value) >= 4 else ""
        return f"•••{tail}" if tail else "•••"
    return "•••"
