"""API request schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CreateEpisodeRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    source: str = Field(..., min_length=1, max_length=256)
    type: str = Field(..., min_length=1, max_length=128)
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = Field(None, max_length=256)
    # When the source event actually happened. Optional — when absent, the
    # database server-defaults to now() (= ingest time), which is the right
    # answer for live-chat ingest. Connectors that backfill historical data
    # (Slack history, GitHub issues, Zendesk imports) should always set
    # this so timeline-style queries see the real ordering.
    occurred_at: datetime | None = None


class BatchCreateEpisodesRequest(BaseModel):
    episodes: list[CreateEpisodeRequest] = Field(..., min_length=1, max_length=100)


class CompileMemoriesRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    async_mode: bool = Field(default=False, alias="async")


class SearchMemoriesRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    kind: str | None = None
    query: str | None = None
    limit: int = Field(20, ge=1, le=100)


class GetContextRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    task: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int | None = Field(None, ge=1, le=128000)
    session_id: str | None = Field(
        None,
        max_length=256,
        description="Current session ID — episodes in this session receive a relevance boost",
    )
    emit_receipt: bool | None = Field(
        None,
        description=(
            "Per-request opt-in for emitting a state-assembly receipt. "
            "Default behavior is governed by tenant config; see "
            "docs/state-assembly-receipts.md."
        ),
    )
    query_id: str | None = Field(
        None,
        max_length=64,
        description="Caller-supplied query id, recorded on the receipt for trace correlation.",
    )
    task_id: str | None = Field(
        None,
        max_length=64,
        description="Caller-supplied task id, recorded on the receipt for multi-call grouping.",
    )
    parent_receipt_id: str | None = Field(
        None,
        max_length=26,
        description="ULID of a parent receipt to chain this assembly to.",
    )
    caller_id: str | None = Field(
        None,
        max_length=256,
        description=(
            "Identity of the caller making the request — consumed by the "
            "sensitivity-label policy layer (#50). When the tenant config "
            "sets `require_caller_identity: true`, this and `caller_type` "
            "are mandatory and missing values return 401."
        ),
    )
    caller_type: str | None = Field(
        None,
        max_length=64,
        description=(
            "Category of caller (e.g. 'support_agent', 'marketing_tool', "
            "'eval_harness'). Used by policy predicates to express "
            "tool-class-level rules without per-caller policy authoring."
        ),
    )


class CreateResolutionRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    session_id: str = Field(..., min_length=1, max_length=256)
    status: str = Field("open", pattern=r"^(open|resolved|unresolved)$")
    resolution_summary: str | None = Field(None, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HandoffRequest(BaseModel):
    subject_id: str = Field(..., min_length=1, max_length=256)
    session_id: str = Field(
        ..., min_length=1, max_length=256, description="Session being handed off"
    )
    reason: str = Field("escalation", max_length=256, description="Why the handoff is happening")
    max_tokens: int | None = Field(None, ge=1, le=16000)
    emit_receipt: bool | None = Field(
        None,
        description=(
            "Per-request opt-in for emitting a state-assembly receipt. "
            "Default behavior is governed by tenant config; see "
            "docs/state-assembly-receipts.md."
        ),
    )
    query_id: str | None = Field(None, max_length=64)
    task_id: str | None = Field(None, max_length=64)
    parent_receipt_id: str | None = Field(None, max_length=26)
    caller_id: str | None = Field(None, max_length=256)
    caller_type: str | None = Field(None, max_length=64)


class SetMemoryLabelsRequest(BaseModel):
    """Body for PATCH /v1/memories/{memory_id}/labels.

    Replaces the memory's sensitivity_labels with the supplied list.
    Empty list clears all labels (memory becomes untagged → policy
    default-allow). See docs/sensitivity-labels.md for the label
    vocabulary recommendations and policy interaction.
    """

    sensitivity_labels: list[str] = Field(
        default_factory=list,
        max_length=32,
        description=(
            "Replacement label list. Capped at 32 entries to keep "
            "policy evaluation bounded — labels are not free-form "
            "metadata, they're an enumerable capability vocabulary."
        ),
    )


class LLMChatMessage(BaseModel):
    """Single chat-completion message. Mirrors the OpenAI/LiteLLM wire shape."""

    role: str = Field(..., pattern=r"^(system|user|assistant|tool)$")
    content: str = Field(..., max_length=16000)


class LLMCompleteRequest(BaseModel):
    """Request body for `POST /v1/llm/complete`.

    Intentionally narrow: callers (the website widget, internal demo flows)
    pass messages and optional generation knobs; **provider/model selection
    lives entirely in server config** (`STATEWAVE_LITELLM_MODEL` and
    friends). This is not a generic public LLM API.
    """

    messages: list[LLMChatMessage] = Field(..., min_length=1, max_length=50)
    max_tokens: int | None = Field(None, ge=1, le=4096)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
