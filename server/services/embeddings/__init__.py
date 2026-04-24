"""Embedding provider interface and registry.

All embedding providers implement the BaseEmbeddingProvider protocol.
The get_provider() factory returns the active provider based on config.
"""

from __future__ import annotations

from typing import Protocol


class BaseEmbeddingProvider(Protocol):
    """Protocol that all embedding providers must satisfy."""

    @property
    def dimensions(self) -> int:
        """The dimensionality of produced vectors."""
        ...

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Contract:
        - Returns a list of float vectors, one per input text.
        - Each vector has exactly `self.dimensions` elements.
        - May raise on transient errors; caller handles retries.
        - Empty input returns empty output.
        """
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Generate a single embedding for a search query.

        Some providers use different models/settings for queries vs documents.
        Default implementation delegates to embed_texts.
        """
        ...


_provider_instance: BaseEmbeddingProvider | None = None


def get_provider() -> BaseEmbeddingProvider | None:
    """Return the active embedding provider, or None if disabled.

    Returns None when embedding_provider == "none".
    Caches the instance after first call.
    """
    global _provider_instance

    from server.core.config import settings

    if settings.embedding_provider == "none":
        return None

    if _provider_instance is not None:
        return _provider_instance

    if settings.embedding_provider == "stub":
        from server.services.embeddings.stub import StubEmbeddingProvider

        _provider_instance = StubEmbeddingProvider(dimensions=settings.embedding_dimensions)
    elif settings.embedding_provider == "openai":
        from server.services.embeddings.openai import OpenAIEmbeddingProvider

        _provider_instance = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")

    return _provider_instance


def reset_provider() -> None:
    """Reset cached provider — useful for testing."""
    global _provider_instance
    _provider_instance = None
