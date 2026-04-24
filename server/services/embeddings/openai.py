"""OpenAI embedding provider — real semantic vectors via OpenAI API.

Requires:
- STATEWAVE_OPENAI_API_KEY set in environment
- STATEWAVE_EMBEDDING_PROVIDER=openai

Uses text-embedding-3-small by default (1536 dimensions).
"""

from __future__ import annotations

import structlog

logger = structlog.stdlib.get_logger()


class OpenAIEmbeddingProvider:
    """OpenAI API-based embedding provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ) -> None:
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set STATEWAVE_OPENAI_API_KEY or "
                "switch to STATEWAVE_EMBEDDING_PROVIDER=stub for local dev."
            )
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for the OpenAI embedding provider. "
                "Install it with: pip install 'statewave[openai]'"
            )
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # OpenAI supports batching natively
        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=self._dimensions,
        )
        logger.debug(
            "openai_embeddings_generated",
            count=len(texts),
            model=self._model,
            usage=response.usage.total_tokens if response.usage else None,
        )
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        return results[0]
