"""LLM-backed memory compiler — uses LiteLLM for multi-provider support.

Extracts structured memories (profile facts, preferences, episode summaries,
procedures) from episode payloads using any LLM provider supported by LiteLLM
(OpenAI, Anthropic, Azure, Ollama, Cohere, Gemini, Bedrock, Mistral, Groq, etc.).

Falls back gracefully on parse errors or API failures.

Requires:
- STATEWAVE_COMPILER_TYPE=llm
- Model-appropriate API key (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY)
  or STATEWAVE_OPENAI_API_KEY for backward compatibility
- pip install 'statewave[llm]'
"""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Sequence

import structlog

from server.db.tables import EpisodeRow, MemoryRow
from server.services.compilers.heuristic import extract_payload_text

logger = structlog.stdlib.get_logger()

# Lazy import — litellm is optional (only needed when compiler_type=llm)
try:
    import litellm
except ImportError:
    litellm = None  # type: ignore[assignment]

_SYSTEM_PROMPT = """\
You are a memory extraction engine for an AI context system called Statewave.

Given a raw episode (a recorded interaction), extract structured memories.

Each memory must be one of these kinds:
- profile_fact: a concrete fact about the subject (name, role, location, preference, etc.)
- episode_summary: a concise summary of what happened in this interaction
- procedure: a step-by-step process, workflow, or instruction that was discussed

Return a JSON array of memory objects. Each object must have:
- "kind": one of the kinds above
- "content": the full memory text
- "summary": a one-sentence summary (max 200 chars)
- "confidence": a float 0.0–1.0 indicating extraction confidence

Rules:
- Extract ALL distinct facts; do not merge unrelated facts into one memory.
- Be precise and factual — never invent information not in the episode.
- If the episode contains no extractable memories, return an empty array [].
- Return ONLY the JSON array, no markdown fences or extra text.
"""


class LLMCompiler:
    """LiteLLM-based memory compiler. Supports any provider. Implements BaseCompiler protocol."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        if litellm is None:
            raise ImportError(
                "litellm package is required for the LLM compiler. "
                "Install with: pip install 'statewave[llm]'"
            )
        # Set API key for backward compat (STATEWAVE_OPENAI_API_KEY → OPENAI_API_KEY)
        if api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
        self._model = model
        self._executor = ThreadPoolExecutor(max_workers=2)

    def compile(self, episodes: Sequence[EpisodeRow]) -> list[MemoryRow]:
        memories: list[MemoryRow] = []
        for ep in episodes:
            memories.extend(self._compile_episode(ep))
        return memories

    def _compile_episode(self, ep: EpisodeRow) -> list[MemoryRow]:
        text = extract_payload_text(ep.payload)
        if not text:
            return []

        try:
            raw_memories = self._call_llm(text)
        except Exception:
            logger.warning("llm_compile_failed", episode_id=str(ep.id), exc_info=True)
            return []

        results: list[MemoryRow] = []
        for mem in raw_memories:
            kind = mem.get("kind", "episode_summary")
            if kind not in ("profile_fact", "episode_summary", "procedure"):
                kind = "episode_summary"

            content = mem.get("content", "")
            if not content:
                continue

            results.append(
                MemoryRow(
                    id=uuid.uuid4(),
                    subject_id=ep.subject_id,
                    kind=kind,
                    content=content,
                    summary=mem.get("summary", content[:200]),
                    confidence=min(max(float(mem.get("confidence", 0.7)), 0.0), 1.0),
                    valid_from=ep.created_at or datetime.now(timezone.utc),
                    source_episode_ids=[ep.id],
                    metadata_={"compiler": "llm", "model": self._model},
                    status="active",
                )
            )

        logger.info(
            "llm_compile_done",
            episode_id=str(ep.id),
            memories_extracted=len(results),
        )
        return results

    def _call_llm(self, text: str) -> list[dict[str, Any]]:
        """Call OpenAI and parse the JSON array response.

        Uses the synchronous OpenAI client. When called from an async context
        (e.g. the compile endpoint), the caller should use run_in_executor or
        the compile() method handles batching. The ThreadPoolExecutor ensures
        we don't block the event loop.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context — offload to thread pool
            future = self._executor.submit(self._call_llm_sync, text)
            # This is called from sync compile(), so we can't await.
            # Instead, the sync OpenAI call runs in the thread pool.
            return future.result(timeout=30)
        return self._call_llm_sync(text)

    def _call_llm_sync(self, text: str) -> list[dict[str, Any]]:
        """Synchronous LLM call via LiteLLM — supports any provider."""
        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract memories from this episode:\n\n{text[:8000]}"},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content or "[]"
        # Strip markdown fences if the model wraps them anyway
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []
        return parsed
