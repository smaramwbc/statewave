"""Embedding provider interface and registry.

All embedding providers implement the BaseEmbeddingProvider protocol.
The get_provider() factory returns the active provider based on config.

The default provider is `stub` so a fresh `docker compose up` can boot
without any LLM credentials, but stub vectors are deterministic hash
output, NOT semantic. The factory logs a loud WARNING the first time
the stub is selected so an operator can see at a glance that semantic
search is degraded.
"""

from __future__ import annotations

import structlog

from typing import Protocol

logger = structlog.stdlib.get_logger()


class BaseEmbeddingProvider(Protocol):
    """Protocol that all embedding providers must satisfy."""

    @property
    def dimensions(self) -> int:
        """The dimensionality of produced vectors."""
        ...

    @property
    def provides_semantic_similarity(self) -> bool:
        """Whether the provider produces vectors with real semantic meaning.

        True for real embedding APIs (any LiteLLM-supported provider) where
        similar texts produce similar vectors. False for the hash-based stub,
        whose vectors are deterministic but semantically meaningless —
        callers should NOT use stub-vector cosine similarity as a relevance
        signal during ranking.
        """
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
        # One-shot loud warning per process. The stub produces
        # deterministic but semantically meaningless vectors — semantic
        # search ranks pseudo-randomly with the stub. Operators reading
        # logs see this immediately instead of debugging "why is search
        # ranking weird" against a hash function.
        logger.warning(
            "embedding_provider_stub_active",
            advice=(
                "Statewave is using the hash-based embedding stub. Vectors are "
                "deterministic but NOT semantic — /v1/memories/search?semantic=true "
                "and /v1/context relevance ranking will not work usefully. "
                "Set STATEWAVE_EMBEDDING_PROVIDER=litellm and a "
                "STATEWAVE_LITELLM_EMBEDDING_MODEL to enable real semantic search."
            ),
        )
    elif settings.embedding_provider == "litellm":
        from server.services.embeddings.litellm import LiteLLMEmbeddingProvider

        _provider_instance = LiteLLMEmbeddingProvider(
            model=settings.litellm_embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")

    return _provider_instance


def reset_provider() -> None:
    """Reset cached provider — useful for testing."""
    global _provider_instance
    _provider_instance = None
