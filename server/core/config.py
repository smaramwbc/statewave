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

    # Token estimation model
    tiktoken_model: str = "cl100k_base"

    # Context assembly defaults
    default_max_context_tokens: int = 4000

    model_config = {"env_prefix": "STATEWAVE_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
