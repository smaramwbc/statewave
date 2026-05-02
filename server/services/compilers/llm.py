"""LLM-backed memory compiler — extracts structured memories from episodes.

Routes all LLM calls through the central adapter at `server.services.llm`,
which is the only place in Statewave that imports LiteLLM directly. The
compiler stays focused on batching + concurrency + result parsing; the
adapter owns provider selection, timeout, retries, and error mapping.

Optimized for speed:
- Batches small episodes into a single LLM call
- Runs multiple batches in parallel with concurrency control
- Falls back gracefully on parse errors or API failures

Requires:
- STATEWAVE_COMPILER_TYPE=llm
- LiteLLM-compatible model + credentials (see server/services/llm.py docstring
  for the env-var contract — provider-neutral)
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

import structlog

from server.db.tables import EpisodeRow, MemoryRow
from server.services import llm as llm_adapter
from server.services.compilers.heuristic import extract_payload_text

logger = structlog.stdlib.get_logger()

# ─── Configuration ───

_MAX_BATCH_CHARS = 6000  # Max total chars per LLM call (leaves room for prompt + response)
_MAX_CONCURRENCY = 4  # Max parallel LLM calls
_MAX_TOKENS = 3000  # Response token limit per batch

_SYSTEM_PROMPT = """\
You are a memory extraction engine for an AI context system called Statewave.

Given one or more raw episodes (recorded interactions, documentation sections, or other content), extract structured memories.

Each memory must be one of these kinds:
- profile_fact: a concrete, generalizable fact about the subject or system (e.g. "Statewave requires PostgreSQL", "Alice prefers email"). Must be a STATEMENT that holds in general, not a transient value.
- episode_summary: a concise summary of what happened in this interaction or what this section explains.
- procedure: a step-by-step process, workflow, or instruction that was discussed.

Return a JSON array of memory objects. Each object must have:
- "kind": one of the kinds above
- "content": the full memory text
- "summary": a one-sentence summary (max 200 chars)
- "confidence": a float 0.0–1.0 indicating extraction confidence
- "episode_index": the 0-based index of the episode this memory came from

Rules:
- Extract ALL distinct facts; do not merge unrelated facts into one memory.
- Be precise and factual — never invent information not in the episode.
- DO NOT extract values from inside code blocks, JSON examples, sample API responses, or curl/bash command examples as profile_facts. Those are illustrations of *shape*, not facts about the subject. For example, in `{"subject_id": "user-42", "memories_created": 5}`, "subject_id user-42" is a placeholder — it is NOT a profile fact about anyone. Skip example identifiers, sample values, placeholder names, and inline literals from documentation snippets.
- DO extract the surrounding *prose* explanation (e.g. "POST /v1/memories/compile returns memories_created and a memories array"). That is a generalizable fact.
- If an episode is mostly code or example data with no generalizable claims, return episode_summary describing what the section is about, not profile_facts cataloguing the example values.
- If an episode contains no extractable memories, skip it.
- Return ONLY the JSON array, no markdown fences or extra text.
"""

class LLMCompiler:
    """Async LLM memory compiler with batching + parallelism. Implements BaseCompiler protocol.

    All LLM calls route through `server.services.llm` — see that module's
    docstring for the provider-neutral env-var contract. `model` is a
    LiteLLM model identifier (e.g. "gpt-4o-mini",
    "claude-3-haiku-20240307", "ollama/llama3").
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model

    def compile(self, episodes: Sequence[EpisodeRow]) -> list[MemoryRow]:
        """Sync fallback — uses heuristic compiler."""
        from server.services.compilers.heuristic import HeuristicCompiler

        logger.warning("llm_compiler_sync_fallback")
        return HeuristicCompiler().compile(episodes)

    async def compile_async(self, episodes: Sequence[EpisodeRow]) -> list[MemoryRow]:
        """Async compile — batches episodes and processes in parallel."""
        # Extract text from each episode, skip empties
        episode_texts: list[tuple[EpisodeRow, str]] = []
        for ep in episodes:
            text = extract_payload_text(ep.payload)
            if text:
                episode_texts.append((ep, text[:4000]))  # Cap per-episode text

        if not episode_texts:
            return []

        # Group into batches by total character count
        batches = self._create_batches(episode_texts)
        logger.info("compile_batched", episodes=len(episode_texts), batches=len(batches))

        # Run batches in parallel with concurrency limit
        semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
        tasks = [self._process_batch(batch, semaphore) for batch in batches]
        batch_results = await asyncio.gather(*tasks)

        # Flatten results
        memories: list[MemoryRow] = []
        for result in batch_results:
            memories.extend(result)

        logger.info("compile_complete", total_memories=len(memories))
        return memories

    def _create_batches(
        self, episode_texts: list[tuple[EpisodeRow, str]]
    ) -> list[list[tuple[EpisodeRow, str]]]:
        """Group episodes into batches that fit within the char budget."""
        batches: list[list[tuple[EpisodeRow, str]]] = []
        current_batch: list[tuple[EpisodeRow, str]] = []
        current_chars = 0

        for ep, text in episode_texts:
            text_len = len(text)
            # If single episode exceeds budget, it goes in its own batch
            if text_len >= _MAX_BATCH_CHARS:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_chars = 0
                batches.append([(ep, text)])
                continue

            if current_chars + text_len > _MAX_BATCH_CHARS:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0

            current_batch.append((ep, text))
            current_chars += text_len

        if current_batch:
            batches.append(current_batch)

        return batches

    async def _process_batch(
        self,
        batch: list[tuple[EpisodeRow, str]],
        semaphore: asyncio.Semaphore,
    ) -> list[MemoryRow]:
        """Process a batch of episodes in a single LLM call."""
        async with semaphore:
            # Format the prompt with all episodes in this batch
            episode_blocks = []
            for i, (ep, text) in enumerate(batch):
                episode_blocks.append(f"--- Episode {i} ---\n{text}")
            combined_text = "\n\n".join(episode_blocks)

            try:
                raw_memories = await self._call_llm_async(combined_text, len(batch))
            except Exception:
                logger.warning("llm_batch_failed", episode_count=len(batch), exc_info=True)
                return []

            # Map memories back to their source episodes
            results: list[MemoryRow] = []
            for mem in raw_memories:
                # Determine which episode this memory belongs to
                ep_idx = mem.get("episode_index", 0)
                if not isinstance(ep_idx, int) or ep_idx < 0 or ep_idx >= len(batch):
                    ep_idx = 0
                source_ep = batch[ep_idx][0]

                kind = mem.get("kind", "episode_summary")
                if kind not in ("profile_fact", "episode_summary", "procedure"):
                    kind = "episode_summary"

                # The contract says `content` is a string, but gpt-4o-mini
                # occasionally returns a list (bullet array) — observed live
                # against api/v1-contract.md procedural sections. Coerce
                # defensively rather than crashing the compile call: a list of
                # steps joins cleanly into a single readable memory body.
                raw_content = mem.get("content", "")
                if isinstance(raw_content, list):
                    content = "\n".join(str(item) for item in raw_content)
                elif isinstance(raw_content, str):
                    content = raw_content
                else:
                    content = str(raw_content) if raw_content else ""
                if not content:
                    continue

                # Same defensive coercion for `summary` — same failure shape.
                raw_summary = mem.get("summary", content[:200])
                if isinstance(raw_summary, list):
                    summary = " ".join(str(item) for item in raw_summary)[:200]
                elif isinstance(raw_summary, str):
                    summary = raw_summary
                else:
                    summary = str(raw_summary)[:200] if raw_summary else content[:200]

                results.append(
                    MemoryRow(
                        id=uuid.uuid4(),
                        subject_id=source_ep.subject_id,
                        kind=kind,
                        content=content,
                        summary=summary,
                        confidence=min(max(float(mem.get("confidence", 0.7)), 0.0), 1.0),
                        valid_from=source_ep.created_at or datetime.now(timezone.utc),
                        source_episode_ids=[source_ep.id],
                        metadata_={"compiler": "llm", "model": self._model},
                        status="active",
                    )
                )

            logger.info("llm_batch_done", episodes=len(batch), memories_extracted=len(results))
            return results

    async def _call_llm_async(self, text: str, episode_count: int) -> list[dict[str, Any]]:
        """Async LLM call via the central LiteLLM adapter.

        Returns the parsed memory-list. Routing through `server.services.llm`
        gives us provider portability plus standardized timeout/retry/error
        mapping.
        """
        # Adjust max tokens based on batch size
        max_tokens = min(_MAX_TOKENS, 500 * episode_count)

        try:
            parsed = await llm_adapter.acomplete_json(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Extract memories from these {episode_count} episode(s)."
                            " Return a JSON object with a single key `memories`"
                            f" whose value is the array.\n\n{text}"
                        ),
                    },
                ],
                model=self._model,
                temperature=0.1,
                max_tokens=max_tokens,
            )
        except llm_adapter.StatewaveLLMError as exc:
            # Same surface as the previous httpx-based path: caller
            # (_process_batch) catches generic Exception and falls
            # through to an empty memory list.
            raise RuntimeError(str(exc)) from exc

        # acomplete_json forces response_format=json_object, so the
        # provider returns a dict at top level. Some providers / older
        # behavior may return a bare list — accept both.
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("memories", "items", "results"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            # Single-memory dict — wrap as a one-element list.
            if "kind" in parsed and "content" in parsed:
                return [parsed]
        return []
