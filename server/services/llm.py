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
    acomplete_with_usage(messages, *, model=None, temperature=None,
              max_tokens=None, timeout=None) -> tuple[str, dict | None]
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

_GETTING_STARTED_TROUBLESHOOTING_URL = (
    "https://github.com/smaramwbc/statewave-docs/blob/main/getting-started.md#troubleshooting"
)


def litellm_api_key_configured() -> bool:
    """True when STATEWAVE_LITELLM_API_KEY is set to a non-empty value."""
    key = settings.litellm_api_key
    return bool(key and str(key).strip())


def llm_requires_api_key() -> bool:
    """False for providers that authenticate locally / need no API key.

    A LiteLLM ``ollama/*`` (or ``ollama_chat/*``) model talks to a local
    Ollama server with no credentials, so an unset
    STATEWAVE_LITELLM_API_KEY is the *expected* configuration there — not a
    misconfiguration. Without this, the missing-key warning and the
    ``/readyz`` "key is not set" message both fire spuriously for every
    local-Ollama operator (credit: @LPHuynh, #122).
    """
    return not settings.litellm_model.startswith("ollama")


def warn_if_llm_compiler_missing_api_key() -> None:
    """Log a one-shot startup warning when LLM compiler is selected without credentials.

    Operators sometimes select the LLM compiler (STATEWAVE_COMPILER_TYPE=llm)
    with an empty key and assume semantic search is live. It is not: without a
    reachable key, every compile call fails its provider round-trip and yields
    zero memories — the LLM compiler does NOT fall back to the regex
    (heuristic) compiler. Embeddings separately degrade to the non-semantic
    hash stub. For a keyless setup, use STATEWAVE_COMPILER_TYPE=heuristic
    (the .env.example default), which extracts memories locally.
    """
    if settings.compiler_type != "llm":
        return
    if litellm_api_key_configured():
        return
    if not llm_requires_api_key():
        # Local Ollama model — no key needed, nothing to warn about.
        return
    logger.warning(
        "llm_compiler_missing_api_key",
        missing_var="STATEWAVE_LITELLM_API_KEY",
        effect=(
            "compilation will produce zero memories (LLM calls fail with no "
            "key — there is no regex fallback); embeddings use the "
            "non-semantic hash stub"
        ),
        advice=(
            "Set STATEWAVE_LITELLM_API_KEY in .env (or deployment secrets) and "
            "restart the API process — or, for a keyless setup, set "
            "STATEWAVE_COMPILER_TYPE=heuristic to extract memories locally."
        ),
        docs=_GETTING_STARTED_TROUBLESHOOTING_URL,
    )


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


def _common_kwargs(model: str | None = None) -> dict[str, Any]:
    """Per-call kwargs that come from settings — api_key, api_base.

    The api_key is passed explicitly to LiteLLM rather than mutating
    `os.environ`, so multiple Statewave instances in one process can
    target different providers without leaking credentials between them.

    Provider-aware routing (v9, mixed-vendor deployments): if `model`
    begins with `claude-`, we DO NOT pass `api_key` — LiteLLM will read
    `ANTHROPIC_API_KEY` from the env automatically. For any other model
    (including OpenAI embedding models like text-embedding-3-small), we
    pass `settings.litellm_api_key` as before. This lets a single
    deployment run an Anthropic compiler against a Claude model AND
    keep an OpenAI embedding model on the same server without the OpenAI
    key being silently passed to the Anthropic API.
    """
    kw: dict[str, Any] = {}
    is_claude = bool(model and model.startswith("claude-"))
    if settings.litellm_api_key and not is_claude:
        kw["api_key"] = settings.litellm_api_key
    if settings.litellm_api_base:
        kw["api_base"] = settings.litellm_api_base
    return kw


# gpt-5 family and o-series reasoning models reject any explicit temperature
# other than the default (1) — passing 0.1 raises BadRequestError. Omit the
# param for them and let the model use its default.
_RESTRICTED_TEMP_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _omit_temperature(model: str | None) -> bool:
    m = (model or "").lower()
    if m.startswith("openai/"):
        m = m[len("openai/"):]
    return any(m.startswith(p) for p in _RESTRICTED_TEMP_PREFIXES)


def _reasoning_kwargs(model: str | None) -> dict[str, Any]:
    """``reasoning_effort`` for gpt-5/o-series reasoning models, when configured.

    Only emitted for reasoning models (the same set that rejects temperature), so
    non-reasoning chat models and the embedding path are unaffected. Setting it to
    "minimal" stops reasoning tokens from consuming the whole response budget —
    the cause of empty JSON on complex compile/reconcile prompts — and is far
    faster and cheaper.
    """
    if settings.litellm_reasoning_effort and _omit_temperature(model):
        return {"reasoning_effort": settings.litellm_reasoning_effort}
    return {}


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
        "num_retries": settings.litellm_max_retries,
        **_common_kwargs(chosen_model),
        **_reasoning_kwargs(chosen_model),
    }
    if not _omit_temperature(chosen_model):
        kwargs["temperature"] = temp
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


def _extract_usage(resp: Any) -> dict[str, int] | None:
    """Pull OpenAI-shape token usage off a LiteLLM completion response.

    Returns ``None`` when the provider omits usage (some local models do),
    so callers can render "tokens: n/a" rather than a misleading zero.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None

    def _as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    prompt = _as_int(getattr(usage, "prompt_tokens", None))
    completion = _as_int(getattr(usage, "completion_tokens", None))
    total = _as_int(getattr(usage, "total_tokens", None)) or (prompt + completion)
    if prompt == 0 and completion == 0 and total == 0:
        return None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


async def acomplete_with_usage(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> tuple[str, dict[str, int] | None]:
    """Like :func:`acomplete`, but also returns token usage when the
    provider reports it: ``(reply, usage_or_none)``.

    Powers the internal ``/v1/llm/complete`` route so first-party consoles
    (the admin "Chat with Memory") can show token stats. Kept as a separate
    function — rather than threading a flag through :func:`acomplete` — so
    the hot compilation path stays byte-for-byte unchanged; the small amount
    of duplicated call/error scaffolding is the deliberate cost of that
    isolation.
    """
    litellm = _ensure_litellm()
    chosen_model = model or settings.litellm_model
    temp = temperature if temperature is not None else settings.litellm_temperature
    timeout_s = timeout if timeout is not None else settings.litellm_timeout_seconds

    kwargs: dict[str, Any] = {
        "model": chosen_model,
        "messages": messages,
        "num_retries": settings.litellm_max_retries,
        **_common_kwargs(chosen_model),
        **_reasoning_kwargs(chosen_model),
    }
    if not _omit_temperature(chosen_model):
        kwargs["temperature"] = temp
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        resp = await asyncio.wait_for(litellm.acompletion(**kwargs), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise LLMTimeoutError(f"LLM completion timed out after {timeout_s}s") from exc
    except Exception as exc:  # noqa: BLE001
        raise _classify(exc) from exc

    try:
        reply = resp.choices[0].message.content or ""
    except (AttributeError, IndexError, KeyError) as exc:
        raise LLMResponseError("LLM response missing expected choices/message") from exc

    return reply, _extract_usage(resp)


async def acomplete_json(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> Any:
    """Chat completion that returns parsed JSON.

    Provider-aware: Anthropic models (`claude-*`) use forced **tool use**
    (LiteLLM passes through `tools` + `tool_choice` to Anthropic; the
    model is REQUIRED to call the `emit_json` tool with a valid JSON
    `payload` arg). All other providers use `response_format=json_object`,
    which OpenAI / Azure / etc. honor natively.

    Why the split: with `response_format=json_object` alone, Anthropic
    sometimes returns conversational preamble ("I'll break this down…")
    when the user message contains conversational content (e.g. an
    extraction prompt followed by a long chat transcript). Tool use is
    the only Anthropic-native way to FORCE structured output —
    `response_format` is a soft instruction the model can still ignore;
    `tool_choice={"type": "tool", "name": "..."}` is a hard constraint.

    Raises LLMResponseError if the response is missing the tool call
    (Anthropic) or if the resulting string fails to parse as JSON.
    """
    chosen_model = model or settings.litellm_model
    is_anthropic = chosen_model.startswith("claude-") or chosen_model.startswith("anthropic/")

    if is_anthropic:
        return await _acomplete_json_anthropic_tool_use(
            messages,
            model=chosen_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    raw = await acomplete(
        messages,
        model=chosen_model,
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


# Generic free-form tool the JSON-forced path uses. We don't constrain
# the schema beyond `type: object` so callers can stuff any shape into
# the `payload` — the compiler, e.g., expects `{"memories": [...]}`,
# while a future eval pipeline might want `{"score": 0.7}`. Schema
# enforcement happens at the caller boundary, not here.
_EMIT_JSON_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "emit_json",
        "description": (
            "Emit the structured JSON response for this task. ALWAYS call "
            "this function. Do not respond with plain text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "payload": {
                    "type": "object",
                    "description": "The full JSON payload requested by the user message.",
                    "additionalProperties": True,
                }
            },
            "required": ["payload"],
        },
    },
}


async def _acomplete_json_anthropic_tool_use(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
) -> Any:
    """Anthropic-specific JSON-forcing path via LiteLLM tool use.

    Forces the model to emit a single `emit_json` tool call whose
    `payload` argument is the structured response. Returns the parsed
    `payload` dict; raises LLMResponseError if Anthropic returns text
    or a missing/malformed tool call.
    """
    litellm = _ensure_litellm()
    temp = temperature if temperature is not None else settings.litellm_temperature
    timeout_s = timeout if timeout is not None else settings.litellm_timeout_seconds

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "num_retries": settings.litellm_max_retries,
        "tools": [_EMIT_JSON_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "emit_json"}},
        **_common_kwargs(model),
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    try:
        resp = await asyncio.wait_for(litellm.acompletion(**kwargs), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise LLMTimeoutError(f"LLM completion timed out after {timeout_s}s") from exc
    except Exception as exc:  # noqa: BLE001
        raise _classify(exc) from exc

    # LiteLLM normalizes Anthropic tool_use → OpenAI-shaped tool_calls.
    try:
        message = resp.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
    except (AttributeError, IndexError) as exc:
        raise LLMResponseError("Anthropic JSON tool: missing choices/message") from exc

    if not tool_calls:
        # Defensive: if the model ignored tool_choice and returned text,
        # report it as malformed (caller will surface the snippet).
        content = getattr(message, "content", "") or ""
        raise LLMResponseError(
            "Anthropic JSON tool: model returned text instead of tool_call: "
            f"{str(content)[:200]}"
        )

    call = tool_calls[0]
    raw_args = getattr(getattr(call, "function", None), "arguments", None)
    if not raw_args:
        raise LLMResponseError("Anthropic JSON tool: tool_call has no arguments")
    try:
        parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except json.JSONDecodeError as exc:
        raise LLMResponseError(
            f"Anthropic JSON tool: arguments unparseable: {str(raw_args)[:200]}"
        ) from exc
    if isinstance(parsed, dict) and "payload" in parsed:
        return parsed["payload"]
    # Defensive: some LiteLLM versions may pass arguments through
    # without the `payload` wrapper — accept the bare dict.
    return parsed


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
        **_common_kwargs(chosen_model),
    }
    # Embedder may target a dedicated endpoint (e.g. a local Qwen server) distinct
    # from the chat/extraction provider. A custom OpenAI-compatible endpoint
    # controls its own output width and may reject the `dimensions` param via
    # LiteLLM validation; the server is configured to emit
    # ``settings.embedding_dimensions`` directly, so drop it here.
    if settings.litellm_embedding_api_base:
        kwargs["api_base"] = settings.litellm_embedding_api_base
        kwargs.pop("dimensions", None)

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
        # The OpenAI-compatible embeddings API does NOT guarantee that
        # ``resp.data`` comes back in input order — each item carries an
        # ``index`` precisely so clients can re-order. Sort by it so that
        # ``result[i]`` corresponds to ``texts[i]`` (callers, e.g. the
        # background backfill, zip the result against the input ids). Fall
        # back to response order if a provider omits ``index``.
        data = list(resp.data)
        try:
            data.sort(key=lambda item: item["index"])
        except (KeyError, TypeError):
            pass
        return [item["embedding"] for item in data]
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


def _is_output_limit_error(exc: BaseException) -> bool:
    """True when a provider rejected the call only because the output/token cap
    was hit. The round-trip still reached the model, so auth + connectivity are
    proven — what `aping` cares about."""
    msg = str(exc).lower()
    return (
        "max_tokens" in msg
        or "max output" in msg
        or "output limit" in msg
        or "finish the message" in msg
        or "max_completion_tokens" in msg
    )


async def aping_with_overrides(
    *,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float = 10.0,
) -> tuple[bool, str | None]:
    """One-token reachability check against a CANDIDATE (key, model, base).

    Used by the admin settings ``test_probe`` to validate proposed LLM
    credentials without persisting them. Returns ``(ok, detail)`` so the
    caller can surface failures in the UI without a typed exception
    crossing the module boundary. The only place outside `services/llm.py`
    that LiteLLM-shaped errors leak.

    A reasoning-model max-tokens hit is treated as proven-reachable, same
    as :func:`aping`."""
    litellm = _ensure_litellm()
    kw: dict[str, Any] = {
        "model": model or settings.litellm_model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "num_retries": 0,
    }
    if api_key is not None:
        kw["api_key"] = api_key
    elif settings.litellm_api_key:
        kw["api_key"] = settings.litellm_api_key
    if api_base is not None:
        kw["api_base"] = api_base
    elif settings.litellm_api_base:
        kw["api_base"] = settings.litellm_api_base

    try:
        await asyncio.wait_for(litellm.acompletion(**kw), timeout=timeout)
    except asyncio.TimeoutError:
        return False, f"timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        classified = _classify(exc)
        # max-tokens / output-limit means we DID reach the model — auth
        # and connectivity are proven, so treat as ok.
        if isinstance(classified, LLMProviderError) and _is_output_limit_error(classified):
            return True, "reachable (output-limit hit on 1-token ping)"
        return False, f"{type(classified).__name__}: {classified}"
    return True, "ok"


async def aping(timeout: float = 10.0) -> bool:
    """Lightweight provider-reachability check. Returns True on success.

    Uses a one-token completion to verify both auth and connectivity without
    burning meaningful tokens. A reasoning model (o-series, gpt-5.x, ...) can
    exhaust that tiny budget on hidden reasoning tokens before emitting output,
    which the provider reports as a max-tokens/output-limit error — but that
    round-trip still PROVES auth + connectivity, so we treat it as reachable.
    Re-raises other typed errors so callers can distinguish timeout vs auth.
    """
    try:
        await acomplete(
            [{"role": "user", "content": "ping"}],
            max_tokens=1,
            timeout=timeout,
        )
    except LLMProviderError as exc:
        if _is_output_limit_error(exc):
            return True
        raise
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
