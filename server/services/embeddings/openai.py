"""LiteLLM embedding provider — supports OpenAI, Azure, Cohere, Bedrock, Ollama, etc.

Uses litellm.embedding() for unified multi-provider embedding generation.
Backward compatible: STATEWAVE_EMBEDDING_PROVIDER=openai still works.

Requires:
- pip install 'statewave[llm]'
- Appropriate API key for the chosen model (e.g. OPENAI_API_KEY)
"""

from __future__ import annotations

import os

import structlog

logger = structlog.stdlib.get_logger()


class OpenAIEmbeddingProvider:
    """LiteLLM-based embedding provider. Name kept for backward compat."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise ImportError(
                "litellm package is required for embeddings. "
                "Install with: pip install 'statewave[llm]'"
            )
        # Set API key for backward compat (STATEWAVE_OPENAI_API_KEY → OPENAI_API_KEY)
        if api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provides_semantic_similarity(self) -> bool:
        # Real embedding API — vectors carry semantic meaning, callers may
        # safely use cosine distance as a relevance signal.
        return True

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import litellm

        response = await litellm.aembedding(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        logger.debug(
            "litellm_embeddings_generated",
            count=len(texts),
            model=self._model,
            usage=response.usage.total_tokens if response.usage else None,
        )
        return [item["embedding"] for item in response.data]

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_texts([text])
        return results[0]
