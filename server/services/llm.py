"""Central LiteLLM adapter — the ONLY place in Statewave that imports LiteLLM.

All Statewave-internal LLM tasks (compilation, embeddings, readiness checks,
future eval/extraction work) route through the functions in this module.
Direct provider SDK use elsewhere is forbidden — see
`tests/test_llm_adapter_isolation.py` for the static check that enforces
this.

Why a central adapter:
  * Provider portability — swap any LiteLLM-supported provider (OpenAI,
    Anthropic, Azure, Bedrock, Ollama, Cohere, Gemini, Mistral, Groq, …)
    via configuration only, no code change at call sites.
  * One place to set timeouts, retries, error classification, and
    observability hooks.
  * One place to read provider/model env config — call sites take a
    `model: str | None = None` and the adapter resolves the default.
  * Testable surface — call sites mock `services.llm.acomplete` rather
    than litellm directly, decoupling tests from the underlying SDK.

Configuration — settings are sourced from `server.core.config.settings`,
populated from env vars with the `STATEWAVE_` prefix:

    STATEWAVE_LITELLM_API_KEY        provider-agnostic API key — passed
                                     through to LiteLLM unchanged
    STATEWAVE_LITELLM_MODEL          chat-completion model identifier
                                     (e.g. "gpt-4o-mini",
                                     "claude-3-haiku-20240307",
                                     "ollama/llama3", "azure/gpt-4")
    STATEWAVE_LITELLM_EMBEDDING_MODEL  embedding model identifier
    STATEWAVE_LITELLM_API_BASE       custom base URL (e.g.
                                     http://localhost:11434 for Ollama,
                                     or a self-hosted OpenAI-compatible
                                     gateway). Empty = provider default.
    STATEWAVE_LITELLM_TIMEOUT_SECONDS  request timeout (default 60s)
    STATEWAVE_LITELLM_MAX_RETRIES      retries on transient errors (default 2)
    STATEWAVE_LITELLM_TEMPERATURE      default temperature (default 0.1)

Public surface:

    acomplete(messages, *, model=None, temperature=None, max_tokens=None,
              response_format=None, timeout=None) -> str
    acomplete_json(messages, *, model=None, ...) -> dict
    aembed_texts(texts, *, model=None, dimensions=None) -> list[list[float]]
    aembed_query(text, *, model=None, dimensions=None) -> list[float]
    aping(timeout=10.0) -> bool

Errors:

    StatewaveLLMError    base exception
    LLMTimeoutError      request timed out
    LLMResponseError     provider returned an unparseable / malformed
                         response (e.g. JSON mode but invalid JSON)
    LLMProviderError     all other provider-level failures (auth, rate
                         limit, etc.) — original LiteLLM exception is
                         preserved as `__cause__`
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from server.core.config import settings

logger = structlog.stdlib.get_logger()


# ─── Errors ──────────────────────────────────────────────────────────


class StatewaveLLMError(Exception):
    """Base for all errors surfaced from the LLM adapter."""


class LLMTimeoutError(StatewaveLLMError):
    """The provider call timed out before a response."""


class LLMResponseError(StatewaveLLMError):
    """Provider returned an unparseable or malformed response."""


class LLMProviderError(StatewaveLLMError):
    """Any other provider-side failure (auth, rate limit, 5xx, etc.).
    The original LiteLLM exception is preserved as `__cause__`."""


# ─── Lazy LiteLLM import ─────────────────────────────────────────────


def _ensure_litellm() -> Any:
    """Lazy-import litellm so test files that don't exercise the LLM path
    (the vast majority) don't pull in the dependency at collection time."""
    try:
        import litellm  # noqa: WPS433 — intentional lazy import
    except ImportError as exc:  # pragma: no cover
        raise StatewaveLLMError(
            "litellm is required for LLM operations. "
            "Install with: pip install 'statewave[llm]'"
        ) from exc
    return litellm


def _classify(exc: BaseException) -> StatewaveLLMError:
    """Map a LiteLLM/network exception into our typed error hierarchy."""
    if isinstance(exc, asyncio.TimeoutError):
        return LLMTimeoutError("LLM request timed out")
    return LLMProviderError(str(exc) or type(exc).__name__)


def _common_kwargs() -> dict[str, Any]:
    """Per-call kwargs that come from settings — api_key, api_base.

    The api_key is passed explicitly to LiteLLM rather than mutating
    `os.environ`, so multiple Statewave instances in one process can
    target different providers without leaking credentials between them.
    """
    kw: dict[str, Any] = {}
    if settings.litellm_api_key:
        kw["api_key"] = settings.litellm_api_key
    if settings.litellm_api_base:
        kw["api_base"] = settings.litellm_api_base
    return kw


# ─── Chat completion ─────────────────────────────────────────────────


async def acomplete(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict | None = None,
    timeout: float | None = None,
) -> str:
    """Single chat completion. Returns the message content as a string.

    `response_format` is passed through to LiteLLM unchanged (e.g.
    `{"type": "json_object"}` for JSON mode on supporting providers).
    Use `acomplete_json` if you want the parsed dict back.
    """
    litellm = _ensure_litellm()
    chosen_model = model or settings.litellm_model
    temp = temperature if temperature is not None else settings.litellm_temperature
    timeout_s = timeout if timeout is not None else settings.litellm_timeout_seconds

    kwargs: dict[str, Any] = {
        "model": chosen_model,
        "messages": messages,
        "temperature": temp,
        "num_retries": settings.litellm_max_retries,
        **_common_kwargs(),
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format

    try:
        resp = await asyncio.wait_for(litellm.acompletion(**kwargs), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise LLMTimeoutError(f"LLM completion timed out after {timeout_s}s") from exc
    except Exception as exc:  # noqa: BLE001
        raise _classify(exc) from exc

    try:
        return resp.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError) as exc:
        raise LLMResponseError("LLM response missing expected choices/message") from exc


async def acomplete_json(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> Any:
    """Chat completion that returns parsed JSON.

    Sets `response_format={"type": "json_object"}` and parses the resulting
    string. Strips markdown fences if the model wraps the JSON in them.
    Raises LLMResponseError on invalid JSON.
    """
    raw = await acomplete(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        timeout=timeout,
    )
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"LLM returned invalid JSON: {cleaned[:200]}") from exc


# ─── Embeddings ──────────────────────────────────────────────────────


async def aembed_texts(
    texts: list[str],
    *,
    model: str | None = None,
    dimensions: int | None = None,
    timeout: float | None = None,
) -> list[list[float]]:
    """Batch embedding via LiteLLM. Empty input returns empty output."""
    if not texts:
        return []
    litellm = _ensure_litellm()
    chosen_model = model or settings.litellm_embedding_model
    dim = dimensions if dimensions is not None else settings.embedding_dimensions
    timeout_s = timeout if timeout is not None else settings.litellm_timeout_seconds

    kwargs: dict[str, Any] = {
        "model": chosen_model,
        "input": texts,
        "dimensions": dim,
        **_common_kwargs(),
    }

    try:
        resp = await asyncio.wait_for(litellm.aembedding(**kwargs), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise LLMTimeoutError(f"Embedding request timed out after {timeout_s}s") from exc
    except Exception as exc:  # noqa: BLE001
        raise _classify(exc) from exc

    logger.debug(
        "litellm_embeddings_generated",
        count=len(texts),
        model=chosen_model,
        usage=resp.usage.total_tokens if getattr(resp, "usage", None) else None,
    )
    try:
        return [item["embedding"] for item in resp.data]
    except (AttributeError, KeyError, TypeError) as exc:
        raise LLMResponseError("Embedding response missing data/embedding") from exc


async def aembed_query(
    text: str,
    *,
    model: str | None = None,
    dimensions: int | None = None,
    timeout: float | None = None,
) -> list[float]:
    """Single-query embedding — convenience wrapper over `aembed_texts`."""
    results = await aembed_texts(
        [text], model=model, dimensions=dimensions, timeout=timeout
    )
    return results[0]


# ─── Health ──────────────────────────────────────────────────────────


async def aping(timeout: float = 10.0) -> bool:
    """Lightweight provider-reachability check. Returns True on success.

    Uses a one-token completion to verify both auth and connectivity
    without burning meaningful tokens. Re-raises typed errors so callers
    can distinguish timeout vs auth failure if they want; otherwise just
    catch StatewaveLLMError.
    """
    await acomplete(
        [{"role": "user", "content": "ping"}],
        max_tokens=1,
        timeout=timeout,
    )
    return True


# ─── Public model accessors (used by cache layers etc.) ──────────────


def chat_model() -> str:
    """Resolved chat-completion model name. Used by callers that key
    caches by (text, model) and need to identify the model in use."""
    return settings.litellm_model


def embedding_model() -> str:
    """Resolved embedding model name. Same purpose as `chat_model`."""
    return settings.litellm_embedding_model


def embedding_dimensions() -> int:
    return settings.embedding_dimensions
