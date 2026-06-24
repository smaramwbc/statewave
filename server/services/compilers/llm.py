"""LLM-backed memory compiler — extracts structured memories from episodes.

Routes all LLM calls through the central adapter at `server.services.llm`,
which is the only place in Statewave that imports LiteLLM directly. The
compiler stays focused on batching + concurrency + result parsing; the
adapter owns provider selection, timeout, retries, and error mapping.

Optimized for speed:
- Batches small episodes into a single LLM call
- Runs multiple batches in parallel with concurrency control
- A failed provider round-trip (auth, timeout, 5xx, missing key) is logged
  at WARNING and re-raised as `CompilationError` — extraction could not RUN,
  which is distinct from a batch that legitimately extracted nothing. The
  caller (server/api/memories.py) leaves the affected episodes uncompiled
  and surfaces the failure rather than consuming the episodes for a run that
  produced no memories (issue #201).

Requires:
- STATEWAVE_COMPILER_TYPE=llm
- LiteLLM-compatible model + credentials (see server/services/llm.py docstring
  for the env-var contract — provider-neutral)
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Sequence

import structlog

from datetime import datetime

from server.core.config import settings
from server.db.tables import EpisodeRow, MemoryRow
from server.services import llm as llm_adapter
from server.services.auto_labeling import apply_suggestions
from server.services.claims import (
    CLAIM_REGISTRY,
    SCOPE_SINGLE,
    build_claim_envelope,
)
from server.services.compilers.errors import CompilationError
from server.services.compilers.heuristic import episode_valid_from, extract_payload_text
from server.services.memory_ttl import compute_valid_to

logger = structlog.stdlib.get_logger()

# ─── Configuration ───

_MAX_BATCH_CHARS = 6000  # Max total chars per LLM call (leaves room for prompt + response)
_MAX_CONCURRENCY = 4  # Max parallel LLM calls
# Response token limit per batch. The old 3000 ceiling combined with a
# 500-per-episode allocation produced two failure modes:
#   - Single-episode batches got 500 tokens, far too tight for a dense
#     conversation: the LLM emitted ~10 memories with content + summary,
#     hit the cap mid-string, and acomplete_json's strict json.loads
#     raised LLMResponseError. The whole batch was discarded.
#   - Multi-episode batches got 500 * episode_count but capped at 3000,
#     still tight for ~15+ memories per batch.
# The ceiling is configurable (settings.litellm_compile_max_tokens, default
# 16000) and is a CAP, not a target — only generated tokens are billed. We floor
# the per-call budget at half the ceiling even for a single-episode batch. That
# headroom is essential for reasoning models (o-series, gpt-5.x, ...): they spend
# part of the budget on hidden reasoning tokens before the JSON, so the old tight
# 1500/episode cap truncated their output into invalid JSON (compile 502).
_TOKENS_PER_EPISODE = 1500  # Per-episode allocation, floored below at half the ceiling

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
- PRESERVE SPECIFIC DETAILS VERBATIM. This is the most important rule. Keep the exact concrete details the source states — colors, proper names, titles, places, exact phrases, sentiment/emotion words, quantities, dates — in the memory `content`. NEVER generalize a specific into a vague theme: "painted a sunset with a pink sky and purple autumn tones" must NOT become "painted a landscape"; "savor all the good vibes at the grand opening" must NOT become "is excited about the opening"; "read 'Charlotte's Web'" must NOT become "enjoys reading"; "drives a Prius" must NOT become "has a car". If the source says it specifically, store it specifically — the specific wording is exactly what later questions ask for. When in doubt, keep MORE of the original detail, not less.
- DO NOT extract values from inside code blocks, JSON examples, sample API responses, or curl/bash command examples as profile_facts. Those are illustrations of *shape*, not facts about the subject. For example, in `{"subject_id": "user-42", "memories_created": 5}`, "subject_id user-42" is a placeholder — it is NOT a profile fact about anyone. Skip example identifiers, sample values, placeholder names, and inline literals from documentation snippets.
- DO extract the surrounding *prose* explanation (e.g. "POST /v1/memories/compile returns memories_created and a memories array"). That is a generalizable fact.
- If an episode is mostly code or example data with no generalizable claims, return episode_summary describing what the section is about, not profile_facts cataloguing the example values.
- If an episode contains no extractable memories, skip it.
- Return ONLY the JSON array, no markdown fences or extra text.

Temporal grounding:
- Every episode block is prefixed with a header line `--- Episode N | recorded YYYY-MM-DD (Weekday) ---`. That `recorded` date is the authoritative reference timestamp for everything in that episode unless a specific message carries its own more precise timestamp.
- If a message is prefixed with a bracketed timestamp like `[1:14 pm on 25 May, 2023]`, prefer that over the episode header for that message: it marks WHEN the speaker said this — and by extension, when any event they describe in present/past tense happened.
- For ANY memory you extract about a dated event, action, or state change (e.g. "ran a race", "attended a conference", "joined a group", "started a project", "moved cities", "got married"), the memory `content` MUST include the resolved absolute date.
- Resolve every relative phrase against the applicable reference date (a message's bracketed timestamp if present, otherwise the episode's `recorded` date): "today" / "this morning" / present tense -> the reference date itself; "yesterday" -> reference date minus 1 day; "last Saturday" -> the most recent Saturday before the reference date; "two days ago" -> reference date minus 2 days; "last year" -> the year before the reference date's year; "this weekend" / "the weekend" -> the Saturday–Sunday of the reference date's week. Render the resolved date as ISO-like prose ("on 2026-05-16" or "on 16 May 2026") in the memory `content`.
- NEVER invent, guess, or default a date. Do not emit any date that cannot be derived from either an explicit date in the text or the applicable reference date. Only if there is genuinely no reference date AND no absolute date in the text, omit the date rather than guess.
- This applies to BOTH profile_fact and episode_summary memories — a summary of a dated session should also lead with or include the session date.

Granularity — extract DETAILS, not just headlines:
- "Generalizable" does not mean "high-level". A specific concrete attribute about a subject IS a generalizable fact about them. "Melanie bought purple running shoes" is a valid profile_fact. "Caroline's favorite book is 'Becoming Nicole' by Amy Ellis Nutt" is a valid profile_fact. "Melanie's daughter's birthday is August 13" is a valid profile_fact.
- Extract each of these as distinct memories when they appear in the source — DO NOT collapse them into a vague "Caroline likes books" or "Melanie is into running".
- Specifically watch for and preserve:
    * Concrete objects + their attributes (colors, brand names, materials: "purple running shoes", "hand-painted bowl", "necklace from grandma in Sweden")
    * Motivations and reasons ("Melanie got into running to de-stress")
    * Quantities, durations, ages ("4 years", "10 years ago", "two weekends ago")
    * Specific titles, names, places ("'Becoming Nicole' by Amy Ellis Nutt", "Connected LGBTQ Activists", "lake sunrise")
    * Stated preferences and feelings ("the support group made Caroline feel accepted")
    * Relationships between people / things (who-mentors-whom, who-bought-what-for-whom)
- A profile_fact about a person can be ONE specific item — don't wait to find "enough" to summarize.
- Better to emit 30 concrete granular memories than 5 vague ones. The retrieval layer ranks them; the compiler's job is recall.

Capture BOTH speakers — including what the ASSISTANT said:
THIS IS A COMMONLY MISSED CATEGORY. Episodes are usually a dialogue between a user and an assistant. Extract facts conveyed by EITHER speaker, not just the user.
- When the ASSISTANT provides information, a recommendation, an answer, instructions, a diagnosis, or a resource, extract it as a memory that records WHAT the assistant said. These are frequently the exact answer to a later "what did you recommend / suggest / say about X / which … did you give me" question, and the model routinely drops them because they aren't facts "about the subject".
- Preserve the specifics of the assistant's contribution EXACTLY as given: named tools, product names, links/URLs, step lists, titles, quantities, dosages, settings.
- Attribute the speaker explicitly in the memory content so retrieval can distinguish them:
    * A fact the USER asserts about themselves → profile_fact about the user ("The user's daughter is named Mia.").
    * A recommendation / answer / instruction the ASSISTANT gives → record it as the assistant's contribution ("The assistant recommended trying yoga and a standing desk for the user's lower back pain on <date>." / "The assistant suggested the book 'Atomic Habits' by James Clear.").
- Use kind=procedure when the assistant gave a step-by-step process; otherwise episode_summary (or profile_fact when the assistant stated a durable fact the user will refer back to).
- Each memory is ONE atomic fact or ONE coherent recommendation — do not bundle several distinct assistant suggestions into a single memory; emit one per suggestion.

CRITICAL — Numerical specifications, metrics, and configuration values:
THIS IS THE MOST COMMONLY MISSED CATEGORY. The model treats numbers as transient context and fails to extract them. They are NOT transient — they are first-class facts about the subject's systems / projects / tools / workflows AND are commonly the answer to "what is X" / "how many Y" / "what's the current Z" questions. They are also frequently updated, so missing them silently breaks knowledge-update retrieval downstream.

ALWAYS extract as profile_fact with the EXACT value and the resolved date:
  * Response times, latencies, durations of system operations
  * Counts (commits, issues, items, deployments, users, requests)
  * Percentages, coverage, rates, conversions, success/failure rates
  * Deadlines, target dates, sprint boundaries, release windows
  * Versions (language, library, OS, schema)
  * IDs, keys (truncated for sensitive ones), model names, identifiers
  * Status / state values (deployed, blocked, in-review, enabled, disabled)
  * Thresholds, limits, quotas (rate limits, daily caps, budget ceilings)
  * Pricing, costs, financial figures
  * Performance / capacity numbers (memory, cpu, throughput)

Few-shot examples for technical-spec extraction:

Source text: "The dashboard API now averages 250ms response time due to the new caching layer."
Extract: {"kind":"profile_fact","content":"Dashboard API has an average response time of 250ms as of <date>, driven by the new caching layer."}

Source text: "We just merged commit 165 into main this morning."
Extract: {"kind":"profile_fact","content":"Project repository has 165 commits merged into the main branch as of <date>."}

Source text: "The production API key has a 1,200 daily call quota."
Extract: {"kind":"profile_fact","content":"Production API key daily call quota is 1,200 calls/day as of <date>."}

Source text: "Test coverage just hit 78%."
Extract: {"kind":"profile_fact","content":"API integration module test coverage is 78% as of <date>."}

Source text: "Sprint 1 deadline is April 5, 2024 — basic layout and navigation."
Extract: {"kind":"profile_fact","content":"Sprint 1 deadline is April 5, 2024, focused on basic layout and navigation."}

Source text: "We're running Python 3.12 and Postgres 16 in production."
Extract two profile_facts:
  {"kind":"profile_fact","content":"Production runs Python 3.12 as of <date>."}
  {"kind":"profile_fact","content":"Production runs Postgres 16 as of <date>."}

When a CHANGE is described ("we bumped it from X to Y", "switched from A to B", "increased to N"), extract the NEW value as the profile_fact (the latest state is the answer). If the prior value matters as a comparison reference, also extract it as a separate profile_fact noting it was the prior value.
"""

# Second-pass recall sweep. Diagnosis of the LoCoMo recall gap found
# ~54% of losses are first-pass extraction MISSES — a specific named object,
# title, date, place, count, or quote that was simply never compiled (the topic
# is captured, the identifying detail is dropped). This pass re-reads the same
# window, is shown what the first pass already captured, and emits COMPLETE
# atomic memories only for the missed specifics. Emitting whole standalone facts
# (not fragments) is the key guard against the message-level fragmentation that
# regressed earlier (-3.6).
_RECALL_SWEEP_PROMPT = """\
You are a RECALL AUDITOR for a memory extraction system. A first pass already \
extracted memories from a conversation, but it systematically DROPS specific \
identifying details while keeping the general topic. Your job is to catch exactly \
those misses.

You are given (1) the raw episode(s) with their recorded dates and (2) \
ALREADY_EXTRACTED — the facts the first pass captured. Re-read the source and \
find every specific detail that is NOT already fully captured, focusing on these \
high-value, commonly-dropped types:
- Named objects / titles / brands / proper nouns: book & movie & song titles, \
game names, product/brand names, distinctive physical objects WITH their \
attributes (color, material).
- Dates and times of events: resolve relative phrases ("last weekend", "two \
years ago", "the Friday before") against the episode's recorded date to an \
ABSOLUTE date.
- Place names: cities, states, countries, venues, specific locations.
- Counts, quantities, durations, ages, prices.
- The exact thing a person said / did / offered — including the ASSISTANT's \
recommendations and answers — and short anecdotes tied to a named person.

For each missed detail, emit ONE memory that states the COMPLETE fact (who + \
what + when), so it stands alone and directly answers a question. Rules:
- PRESERVE the specific noun/title/date/quantity VERBATIM — never generalize \
("read 'Charlotte's Web'", NOT "read a book"; "in Rome in June 2023", NOT \
"traveled abroad"; "a black and white flower bowl", NOT "pottery").
- Attribute each fact to the CORRECT person and preserve who-did-what-to-whom \
direction (a recommendation FROM the assistant TO the user, X bought Y for Z).
- Do NOT repeat anything already in ALREADY_EXTRACTED, and do NOT emit vague \
themes or fragments. A memory must be a complete, answerable statement.
- If a detail is genuinely already captured, skip it. When unsure whether a \
SPECIFIC was captured, include it — recall is the goal of this pass.
- Same JSON object format as the first pass for each memory: \
{"kind","content","summary","confidence","episode_index"}.
- Return ONLY a JSON object {"memories": [...]}. If nothing was missed, return \
{"memories": []}.
"""


# The single-valued canonical keys the model may PROPOSE. The registry remains
# authoritative for canonicalization, scope, and membership — the model never
# decides those (see _llm_claim_metadata).
_SINGLE_CLAIM_KEYS = sorted(k for k, spec in CLAIM_REGISTRY.items() if spec.scope == SCOPE_SINGLE)

_SYSTEM_PROMPT += (
    "\n\nOptional structured claim (advanced — usually OMITTED):\n"
    '- A memory MAY carry an optional "claim" object, but ONLY when it asserts a CURRENT,\n'
    "  single-valued fact about the subject that exactly matches one of these canonical keys:\n"
    + "".join(f"    * {k}\n" for k in _SINGLE_CLAIM_KEYS)
    + '- Shape: {"key": <one canonical key above>, "value": <the value>, and OPTIONALLY\n'
    '  "valid_from"/"valid_to" as ISO-8601 dates ONLY when the source states them explicitly}.\n'
    '- OMIT "claim" whenever you are not certain — omission is ALWAYS preferred to guessing.\n'
    "- Do NOT emit a claim for negated, hypothetical, uncertain, quoted/reported, historical, or\n"
    "  third-party statements. Distinguish a current single-valued state from a past one.\n"
    "- Generic tool usage is NOT a billing.primary_payment_processor claim: only assert that key when\n"
    '  the source explicitly says it is the PRIMARY or CURRENT payment processor. "I use Stripe"\n'
    "  alone must NOT produce a claim.\n"
    "- Use ONLY the keys listed; never invent keys, scopes, or cardinality. Claims are optional\n"
    "  annotations — do NOT reduce, merge, or skip the granular memories you would otherwise emit.\n"
)


# ── Compose-unit variant (opt-in via settings.compile_compose_unit) ──────────
# Same prompt as _SYSTEM_PROMPT, but the extraction UNIT becomes one complete,
# self-contained statement instead of atomized fragments. Diagnosis: at equal
# memory COUNT, Statewave memories are ~50% terser and fragment a
# single event across rows, so the answering memory ranks low and the answer
# model can't ground on it. Composing raises per-memory completeness — the one
# axis that differs at equal count — WITHOUT adding rows, so it cannot dilute
# retrieval the way the recall sweep did. Built by surgically swapping only the
# atomization directives; verbatim/temporal/numeric/code-block/claim rules are
# untouched, so the JSON contract and all downstream parsing are identical.
_COMPOSE_GRANULARITY_RULE = (
    "- UNIT OF A MEMORY: one COMPLETE statement per distinct event/topic, NOT one per noun. "
    "Each memory must be a STANDALONE sentence answerable with zero other context. FUSE the "
    "related parts of a single event into ONE memory: the person(s) involved + what was "
    "done/said + the object/title/value + the resolved absolute date + (for dialogue) "
    "who-said-it-to-whom. Do NOT split one event across rows — the degree, the certificate, "
    'and the date are ONE memory ("John earned a university degree, evidenced by a certificate '
    'of completion he shared on 2 April 2023"), not three terse rows.\n'
    "- Emit a SEPARATE memory only for a genuinely DIFFERENT event, attribute, or topic. If two "
    "parts assert DIFFERENT VALUES of the same attribute (two different dates, 250ms vs 90ms, "
    "Munich vs Berlin), keep them SEPARATE — never fuse a changed/updated value into its "
    "predecessor (reconciliation handles supersession downstream).\n"
    "- Aim for FEWER, RICHER memories — each ~1-2 full sentences carrying EVERY specific it "
    "covers — over many terse fragments. The same distinct facts, each made self-contained: a "
    "complete memory both ranks and answers better than a fragment. Composing changes the "
    "PACKAGING, never drops a detail."
)
_COMPOSE_ASSISTANT_RULE = (
    "- When the assistant gives a recommendation/answer and the user responds in the SAME "
    'exchange, capture the EXCHANGE in ONE statement preserving direction ("Sam suggested a '
    'yoga practice to Evan for stress relief, and Evan said he would try it and thanked Sam on '
    '<date>"), not two fragments that each drop the other side. Still emit separate memories for '
    "genuinely distinct suggestions."
)
_SYSTEM_PROMPT_COMPOSE = (
    _SYSTEM_PROMPT.replace(
        "Granularity — extract DETAILS, not just headlines:",
        "Granularity — COMPLETE statements, not atomized fragments (compose, don't atomize):",
    )
    .replace(
        '- A profile_fact about a person can be ONE specific item — don\'t wait to find "enough" to summarize.\n'
        "- Better to emit 30 concrete granular memories than 5 vague ones. The retrieval layer ranks them; the compiler's job is recall.",
        _COMPOSE_GRANULARITY_RULE,
    )
    .replace(
        "- Each memory is ONE atomic fact or ONE coherent recommendation — do not bundle several distinct assistant suggestions into a single memory; emit one per suggestion.",
        _COMPOSE_ASSISTANT_RULE,
    )
)
# Fail loud if a swap silently no-ops (prompt text drifted) — an inert COMPOSE
# variant would make the A/B meaningless.
assert _SYSTEM_PROMPT_COMPOSE != _SYSTEM_PROMPT, "compose-unit prompt swap matched nothing"
assert "compose, don't atomize" in _SYSTEM_PROMPT_COMPOSE


def _system_prompt() -> str:
    """Active extraction system prompt — the compose-unit variant when enabled."""
    return _SYSTEM_PROMPT_COMPOSE if settings.compile_compose_unit else _SYSTEM_PROMPT


def _window_text(
    text: str, max_total: int, window: int, overlap: int
) -> list[str]:
    """Split episode text into overlapping windows for full-content compile.

    Replaces the old `text[:4000]` truncation. Returns 1+ windows of up to
    `window` chars each, with `overlap` chars of carry-over so a fact split
    across a boundary still appears whole in one window. The total text
    considered is capped at `max_total` (cost bound). Windows prefer to break
    on a newline near the end so a message isn't sliced mid-sentence.
    """
    text = text[:max_total]
    if len(text) <= window:
        return [text] if text else []
    overlap = max(0, min(overlap, window // 2))
    windows: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + window, n)
        # Prefer a newline boundary in the last 20% of the window (but not when
        # this is the final slice — take it whole).
        if end < n:
            nl = text.rfind("\n", start + int(window * 0.8), end)
            if nl != -1 and nl > start:
                end = nl
        chunk = text[start:end].strip()
        if chunk:
            windows.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return windows


def _message_chunks(
    payload: dict, per_chunk: int, overlap: int, max_total_chars: int
) -> list[str] | None:
    """Split a chat payload into small COHERENT message groups for extraction.

    Returns a list of rendered chunk texts (each = `per_chunk` messages with
    `overlap` carried from the previous group for context), or None if the
    payload isn't message-shaped (caller falls back to char-windowing). Renders
    each message as "role: [timestamp] content" so the extractor keeps speaker +
    date context. Total content is capped at `max_total_chars` for cost.
    """
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list) or not messages:
        return None
    rendered: list[str] = []
    used = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or m.get("speaker") or ""
        content = m.get("content") or m.get("text") or ""
        if not content:
            continue
        ts = m.get("timestamp")
        line = f"{role}: [{ts}] {content}" if ts else f"{role}: {content}"
        used += len(line)
        if used > max_total_chars:
            break
        rendered.append(line)
    if not rendered:
        return None
    per_chunk = max(1, per_chunk)
    overlap = max(0, min(overlap, per_chunk - 1))
    step = max(1, per_chunk - overlap)
    chunks: list[str] = []
    for start in range(0, len(rendered), step):
        group = rendered[start : start + per_chunk]
        if group:
            chunks.append("\n".join(group))
        if start + per_chunk >= len(rendered):
            break
    return chunks


_NAMED_CONFIDENCE = {
    "very high": 0.95, "high": 0.9, "medium-high": 0.8, "medium": 0.6,
    "med": 0.6, "moderate": 0.6, "medium-low": 0.45, "low": 0.3, "very low": 0.1,
}


def _coerce_confidence(value: Any) -> float:
    """Confidence to a 0-1 float. The model (especially the recall-sweep pass)
    sometimes returns a word ('high') or a non-numeric string instead of a
    number; map known words, parse numeric strings, else default 0.7 — never
    crash the whole batch on one bad field."""
    if isinstance(value, bool):
        return 0.7
    if isinstance(value, (int, float)):
        return min(max(float(value), 0.0), 1.0)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _NAMED_CONFIDENCE:
            return _NAMED_CONFIDENCE[s]
        try:
            return min(max(float(s.rstrip("%")) / (100.0 if "%" in s else 1.0), 0.0), 1.0)
        except ValueError:
            return 0.7
    return 0.7


def _safe_dt(value: Any) -> datetime | None:
    """Parse an LLM-proposed ISO date, or None. Never raises; a bad date simply
    drops the temporal field rather than rejecting the whole claim."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _llm_claim_metadata(mem: dict) -> dict | None:
    """Validate an LLM-PROPOSED claim into an authoritative envelope, or None.

    The model is untrusted for canonicalization, scope, and registry membership:
    we only take its proposed key/value/temporal, then ``build_claim_envelope``
    canonicalizes the key through the registry + approved aliases, stamps the
    registry-authoritative scope, normalizes the value, and returns None for any
    unknown key or unusable value. An unknown/malformed proposal therefore never
    persists under ``metadata_.claim`` in an authoritative form — it is dropped.
    """
    raw = mem.get("claim")
    if not isinstance(raw, dict):
        return None
    key, value = raw.get("key"), raw.get("value")
    if not isinstance(key, str) or not isinstance(value, str):
        return None
    return build_claim_envelope(
        key,
        value,
        valid_from=_safe_dt(raw.get("valid_from")),
        valid_to=_safe_dt(raw.get("valid_to")),
        source="llm",
    )


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
        """Sync entry point — not supported for the LLM compiler.

        LLM extraction is fundamentally async (network round-trips,
        per-batch concurrency). The previous behaviour silently
        delegated to the regex-based `HeuristicCompiler`, which produced
        plausible-looking but lower-quality memories under
        STATEWAVE_COMPILER_TYPE=llm — exactly the silent-fallback
        pattern this module no longer carries.

        Callers must use `compile_async`; the `/v1/memories/compile`
        path already does (see server/api/memories.py: it calls
        `compile_async` whenever the active compiler defines it).
        """
        raise NotImplementedError(
            "LLMCompiler is async-only — use `compile_async`. The sync "
            "`compile()` no longer silently delegates to the heuristic "
            "compiler. See server/api/memories.py for the dispatch logic."
        )

    async def compile_async(self, episodes: Sequence[EpisodeRow]) -> list[MemoryRow]:
        """Async compile — batches episodes and processes in parallel.

        Returns the extracted memories (possibly empty when the episodes
        legitimately yield nothing). Raises `CompilationError` if any batch's
        provider round-trip fails, so the caller does not mistake a provider
        outage for an empty extraction and consume the episodes (issue #201).
        """
        # Structured episodes compile deterministically from producer-supplied
        # candidates — they never go to the LLM. Lazy import avoids a cycle.
        from server.services.structured import compile_candidates

        structured_memories: list[MemoryRow] = []
        episode_texts: list[tuple[EpisodeRow, str]] = []
        for ep in episodes:
            structured = compile_candidates(ep)
            if structured is not None:
                structured_memories.extend(structured)
                continue
            # Message-level chunking (opt-in): extract from small COHERENT
            # message groups instead of fixed char-windows. Char-windows split
            # mid-utterance (fragmenting facts) and, when large, let the model
            # abstract specifics away ("pink sky sunset" -> "a landscape").
            # Extracting atomic facts from small message-level units with the
            # same model preserves the specifics. Falls back
            # to char-windowing for non-chat payloads.
            msg_chunks = (
                _message_chunks(
                    ep.payload,
                    settings.compile_messages_per_chunk,
                    settings.compile_message_overlap,
                    settings.compile_max_episode_chars,
                )
                if settings.compile_message_level
                else None
            )
            if msg_chunks:
                for chunk in msg_chunks:
                    episode_texts.append((ep, chunk))
                continue
            text = extract_payload_text(ep.payload)
            if text:
                # Window the FULL episode text instead of truncating to a 4000-
                # char sliver. Long-context episodes (BEAM/LongMemEval sessions
                # run 100K-460K chars) were losing ~97% of their content before
                # extraction, so most facts were never compiled. Each window
                # becomes its own extraction unit; `_process_batch` maps every
                # resulting memory back to this same source episode. Bounded by
                # `compile_max_episode_chars` for cost control.
                for window in _window_text(
                    text,
                    settings.compile_max_episode_chars,
                    settings.compile_window_chars,
                    settings.compile_window_overlap,
                ):
                    episode_texts.append((ep, window))

        if not episode_texts:
            if structured_memories and settings.auto_labeling_enabled:
                apply_suggestions(structured_memories)
            return structured_memories

        # Group into batches by total character count
        batches = self._create_batches(episode_texts)
        logger.info("compile_batched", episodes=len(episode_texts), batches=len(batches))

        # Run batches in parallel with concurrency limit. Configurable so
        # full-conversation windowed compiles can fan out wide enough to finish
        # within client timeouts.
        semaphore = asyncio.Semaphore(
            getattr(settings, "compile_max_concurrency", None) or _MAX_CONCURRENCY
        )
        tasks = [self._process_batch(batch, semaphore) for batch in batches]
        batch_results = await asyncio.gather(*tasks)

        # Flatten results (structured candidates first, then LLM extractions)
        memories: list[MemoryRow] = list(structured_memories)
        for result in batch_results:
            memories.extend(result)

        # Auto-labeling runs post-extraction so a single code path stamps
        # `suggested_labels` regardless of whether the LLM batch was a
        # single or multi-episode call. Gated globally — a v0.9 upgrade
        # is a no-op for existing tenants until they opt in.
        if settings.auto_labeling_enabled and memories:
            apply_suggestions(memories)

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
            # Format the prompt with all episodes in this batch. Each block
            # is annotated with the episode's resolved reference timestamp
            # (`episode_valid_from` — the same anchor used for the memory's
            # `valid_from`), so the model resolves "today"/relative phrases
            # against the real episode date instead of inventing one. Without
            # this the model has no reference point and falls back to a
            # plausible-looking default (commonly the LoCoMo sample's
            # "25 May 2023") — see issue #115.
            episode_blocks = []
            for i, (ep, text) in enumerate(batch):
                ref_label = episode_valid_from(ep).strftime("%Y-%m-%d (%A)")
                episode_blocks.append(f"--- Episode {i} | recorded {ref_label} ---\n{text}")
            combined_text = "\n\n".join(episode_blocks)

            try:
                raw_memories = await self._call_llm_async(combined_text, len(batch))
            except Exception as exc:
                # A failed provider round-trip (auth, timeout, 5xx, no key)
                # means extraction could not RUN — it is NOT a legitimate
                # "zero memories" result. Raise so the caller leaves these
                # episodes uncompiled and surfaces the failure, instead of
                # silently consuming them (issue #201). Previously this
                # returned `[]`, which the compile route could not tell apart
                # from a real empty extraction.
                logger.warning("llm_batch_failed", episode_count=len(batch), exc_info=True)
                raise CompilationError(
                    f"LLM compilation failed for a batch of {len(batch)} episode(s): {exc}"
                ) from exc

            # Recall sweep (opt-in): a second pass that catches specific details
            # the first pass dropped. Shown the first-pass facts to avoid repeats;
            # emits complete atomic memories for the misses. Fail-open — a sweep
            # error must never lose the first-pass memories.
            if getattr(settings, "compile_recall_sweep", False):
                try:
                    swept = await self._call_recall_sweep(
                        combined_text, len(batch), raw_memories
                    )
                    for m in swept:
                        if isinstance(m, dict):
                            m["_recall_sweep"] = True
                    raw_memories = list(raw_memories) + swept
                    logger.info(
                        "recall_sweep_done",
                        episodes=len(batch),
                        first_pass=len(raw_memories) - len(swept),
                        swept=len(swept),
                    )
                except Exception:  # pragma: no cover - defensive; never break compile
                    logger.warning("recall_sweep_failed", exc_info=True)

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

                ep_valid_from = episode_valid_from(source_ep)
                # Optional, validated claim. A malformed/unknown proposal is
                # dropped and never fails the rest of the response.
                metadata: dict[str, Any] = {"compiler": "llm", "model": self._model}
                if mem.get("_recall_sweep"):
                    metadata["pass"] = "recall_sweep"
                try:
                    claim_md = _llm_claim_metadata(mem)
                except Exception:  # pragma: no cover - defensive; never break compile
                    claim_md = None
                if claim_md:
                    metadata.update(claim_md)
                results.append(
                    MemoryRow(
                        id=uuid.uuid4(),
                        subject_id=source_ep.subject_id,
                        kind=kind,
                        content=content,
                        summary=summary,
                        confidence=_coerce_confidence(mem.get("confidence", 0.7)),
                        valid_from=ep_valid_from,
                        valid_to=compute_valid_to(kind, ep_valid_from, settings.kind_ttl_days),
                        source_episode_ids=[source_ep.id],
                        metadata_=metadata,
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
        # Size the budget to the batch, but floor at half the ceiling so even a
        # single-episode batch leaves room for a reasoning model's hidden tokens
        # (a tight cap truncates the JSON → invalid → compile 502). The ceiling is
        # configurable and caps large batches below the model's output limit.
        ceiling = settings.litellm_compile_max_tokens
        max_tokens = min(ceiling, max(ceiling // 2, _TOKENS_PER_EPISODE * episode_count))

        try:
            parsed = await llm_adapter.acomplete_json(
                [
                    {"role": "system", "content": _system_prompt()},
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

    async def _call_recall_sweep(
        self,
        text: str,
        episode_count: int,
        first_pass: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Second extraction pass that catches specifics the first pass dropped.

        Shown the already-extracted facts so it only emits the misses. Same
        parsing as `_call_llm_async`. Returns [] on any provider error (the
        caller treats the sweep as best-effort and keeps the first-pass memories).
        """
        ceiling = settings.litellm_compile_max_tokens
        max_tokens = min(ceiling, max(ceiling // 2, _TOKENS_PER_EPISODE * episode_count))
        already = "\n".join(
            f"- {m.get('content', '')}" for m in first_pass if isinstance(m, dict)
        ) or "(nothing extracted yet)"
        try:
            parsed = await llm_adapter.acomplete_json(
                [
                    {"role": "system", "content": _RECALL_SWEEP_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"ALREADY_EXTRACTED (do not repeat these):\n{already}\n\n"
                            f"Now re-read these {episode_count} episode(s) and emit a JSON"
                            " object with a single key `memories` whose value is the array"
                            " of specific details the first pass MISSED.\n\n"
                            f"{text}"
                        ),
                    },
                ],
                model=self._model,
                temperature=0.1,
                max_tokens=max_tokens,
            )
        except llm_adapter.StatewaveLLMError as exc:
            raise RuntimeError(str(exc)) from exc

        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in ("memories", "items", "results"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            if "kind" in parsed and "content" in parsed:
                return [parsed]
        return []
