"""Internal LLM completion route used by Statewave's own demo/website flows.

This is intentionally narrow:

- Callers POST a list of chat messages plus optional generation knobs.
- The server picks the model and provider via `STATEWAVE_LITELLM_*` config —
  callers cannot specify them. This is the whole point: provider choice is
  a server concern, not a client concern.
- The endpoint is gated by the regular `X-API-Key` middleware. There is no
  separate auth scope; the existing key is sufficient because this isn't
  meant to be a public LLM proxy — it's the same trust boundary as the rest
  of `/v1/*`.
- The response is just the assistant text. No token counts, no usage
  metadata. The widget doesn't need them.

If you find yourself extending this endpoint with model selection, streaming,
function-calling, or per-call provider overrides, stop and ask whether you
really want a generic LLM API. That belongs behind a separate, more carefully
designed surface.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from server.core.config import settings
from server.schemas.requests import LLMCompleteRequest
from server.schemas.responses import LLMCompleteResponse
from server.services.llm import (
    LLMProviderError,
    LLMResponseError,
    LLMTimeoutError,
    StatewaveLLMError,
    acomplete,
)

router = APIRouter(tags=["llm"])


@router.post(
    "/v1/llm/complete",
    response_model=LLMCompleteResponse,
    summary="Internal chat completion via the configured LiteLLM provider",
)
async def complete_chat(body: LLMCompleteRequest) -> LLMCompleteResponse:
    if not settings.litellm_model:
        raise HTTPException(
            status_code=503,
            detail={"code": "llm_not_configured", "message": "LLM provider is not configured."},
        )

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    try:
        reply = await acomplete(
            messages,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )
    except LLMTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail={"code": "upstream_llm_timeout", "message": str(exc)},
        ) from exc
    except (LLMProviderError, LLMResponseError, StatewaveLLMError) as exc:
        # Don't echo the raw provider message back — it can include URLs,
        # model identifiers, or other internal config detail.
        raise HTTPException(
            status_code=502,
            detail={"code": "upstream_llm_error", "message": "Upstream LLM call failed."},
        ) from exc

    return LLMCompleteResponse(reply=reply)
