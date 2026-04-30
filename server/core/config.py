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
    embedding_provider: str = "stub"  # "stub" | "openai" | "none"
    embedding_dimensions: int = 1536
    openai_api_key: str | None = None  # backward compat; also set via OPENAI_API_KEY env
    openai_embedding_model: str = "text-embedding-3-small"

    # LLM compiler
    llm_compiler_model: str = "gpt-4o-mini"  # any litellm model string

    # Authentication (empty = disabled / open access)
    api_key: str | None = None

    # Rate limiting (0 = disabled)
    rate_limit_rpm: int = 0
    rate_limit_strategy: str = "distributed"  # "distributed" | "memory"

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

    model_config = {"env_prefix": "STATEWAVE_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
