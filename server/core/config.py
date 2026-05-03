"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


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

    # Embeddings
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

    # Webhooks (empty = disabled)
    webhook_url: str | None = None
    webhook_timeout: float = 5.0

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
