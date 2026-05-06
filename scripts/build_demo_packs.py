"""Rebuild the 5 bundled demo starter packs with denser, narrative-consistent
content that can sustain 8–10 visitor turns of recallable context.

Each persona pack (`server/starter_packs/demo-<persona>/`) is the source of
truth for what a visitor sees when they pick that persona on the marketing
website. The packs were initially hand-authored at 22 episodes / 14 memories
each — enough to establish identity but thin enough that visitors hit the
"out of context" wall quickly. This script extends each pack with new
episodes that deepen the established narrative (no persona redesign), then
runs the LLM compiler to produce a richer set of compiled memories.

Mechanism (mirrors `scripts/build_support_pack.py`):

  1. Read existing `episodes.jsonl` for each persona to anchor identity +
     timeline.
  2. Append NEW_EPISODES authored below — concrete names, IDs, dates,
     stack details, ticket numbers, commands, URLs, metrics. No filler.
  3. Wipe a per-persona temp subject (`_build_demo_pack_<persona>`).
  4. Import the combined episode set into the temp subject via
     `/admin/memory/import` (the same path the bundled packs are imported
     through at runtime).
  5. Run a synchronous LLM compile to extract compiled memories.
  6. Read episodes + memories back, rewrite memories'
     `source_episode_ids` from runtime UUIDs to the stable
     `provenance.content_hash` so the import remap on every visitor's
     fresh seed still works.
  7. Serialise to `episodes.jsonl` + `memories.jsonl`, write a fresh
     `manifest.json` with bumped version + counts.
  8. Delete the temp subject.

Usage:
    STATEWAVE_URL=http://localhost:8100 \
    python -m scripts.build_demo_packs [--persona PERSONA] [--keep-temp]

`--persona` rebuilds a single pack (useful for iteration). Default rebuilds
all five.

Env:
    STATEWAVE_URL          (default http://localhost:8100)
    STATEWAVE_API_KEY      (optional)
    DEMO_PACK_VERSION      (overrides the auto-version)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKS_ROOT = REPO_ROOT / "server" / "starter_packs"

PERSONAS = (
    "demo-support-agent",
    "demo-coding-assistant",
    "demo-sales-copilot",
    "demo-devops-agent",
    "demo-research-assistant",
)

# ─── Per-persona expansion content ──────────────────────────────────────────
#
# Each list contains NEW episodes to append to that persona's existing
# `episodes.jsonl`. Episodes are tuples of (source, type, payload). The
# payload's `content` is the narrative text the LLM compiler reads; other
# fields are structured metadata that show up in the inspector.
#
# Authored to deepen the established narrative — not to redesign it. Every
# episode references concrete details (named contacts, dates, ticket numbers,
# SHAs, metrics, URLs) so the compiled memories stay specific instead of
# turning into generic platitudes.
#
# Reference for the existing 22 episodes per persona: read the corresponding
# `server/starter_packs/demo-<persona>/episodes.jsonl`.

# Per-persona expansion content lives in `demo_pack_expansions.py` (kept in a
# separate module so the build orchestration here stays slim and so each
# persona's authored content is reviewable on its own).
from scripts.demo_pack_expansions import NEW_EPISODES  # noqa: E402


# ─── Build orchestration ────────────────────────────────────────────────────

INGEST_BATCH_SIZE = 25
COMPILE_TIMEOUT_S = 600.0


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_existing_episodes(pack_dir: Path) -> list[dict[str, Any]]:
    path = pack_dir / "episodes.jsonl"
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _new_episode_to_payload(source: str, type_: str, body: dict[str, Any]) -> dict[str, Any]:
    """Convert authored tuple data into the wire shape the import endpoint expects.

    Each new episode gets a deterministic `content_hash` based on its content +
    type, so a re-run produces stable IDs (no churn in the JSONL across rebuilds
    when the authored content is unchanged).
    """
    raw = json.dumps({"source": source, "type": type_, "payload": body}, sort_keys=True)
    h = _content_hash(raw)
    payload = dict(body)
    return {
        "id": h,
        "source": source,
        "type": type_,
        "payload": payload,
        "metadata": {},
        "provenance": {"content_hash": h, "authored": "build_demo_packs"},
    }


async def _health(client: httpx.AsyncClient, url: str) -> None:
    r = await client.get(f"{url}/healthz")
    r.raise_for_status()


async def _delete_subject(client: httpx.AsyncClient, url: str, sid: str) -> None:
    try:
        await client.delete(f"{url}/v1/subjects/{sid}")
    except httpx.HTTPError:
        pass


async def _import_payload(
    client: httpx.AsyncClient, url: str, *, target_subject_id: str, episodes: list[dict[str, Any]]
) -> None:
    payload = {
        "format": "statewave-memory-payload",
        "format_version": 1,
        "subjects": [
            {
                "original_subject_id": target_subject_id,
                "metadata": {"build": "demo-packs"},
            }
        ],
        "episodes": [{**e, "subject_id": target_subject_id} for e in episodes],
        "memories": [],
    }
    body = {"payload": payload, "target_subject_id": target_subject_id, "allow_reserved_target": False}
    r = await client.post(f"{url}/admin/memory/import", json=body, timeout=120.0)
    if not r.is_success:
        raise RuntimeError(f"import failed: {r.status_code} {r.text}")


async def _compile(client: httpx.AsyncClient, url: str, sid: str) -> dict[str, Any]:
    r = await client.post(
        f"{url}/v1/memories/compile",
        json={"subject_id": sid, "async": False},
        timeout=COMPILE_TIMEOUT_S,
    )
    if not r.is_success:
        raise RuntimeError(f"compile failed: {r.status_code} {r.text}")
    return r.json()


async def _fetch_all(
    client: httpx.AsyncClient, url: str, path: str, key: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    while True:
        r = await client.get(f"{url}{path}?limit=200&offset={offset}", timeout=60.0)
        r.raise_for_status()
        data = r.json()
        page = data.get(key, []) if isinstance(data, dict) else data
        if not page:
            break
        out.extend(page)
        if len(page) < 200:
            break
        offset += 200
    return out


def _episode_to_jsonl(ep: dict[str, Any]) -> dict[str, Any]:
    """Serialise a server-side episode into the bundled-pack JSONL row shape.

    Strips runtime fields (subject_id, tenant_id, raw timestamps) and ensures
    the `id` is the stable content_hash so a fresh import remaps cleanly.
    """
    prov = dict(ep.get("provenance") or {})
    h = prov.get("content_hash") or _content_hash(json.dumps(ep.get("payload") or {}, sort_keys=True))
    prov["content_hash"] = h
    return {
        "id": h,
        "source": ep.get("source"),
        "type": ep.get("type"),
        "payload": ep.get("payload") or {},
        "metadata": ep.get("metadata") or {},
        "provenance": prov,
        "created_at": ep.get("created_at"),
    }


def _memory_to_jsonl(mem: dict[str, Any], uuid_to_hash: dict[str, str]) -> dict[str, Any]:
    mapped_sources: list[str] = []
    for sid in mem.get("source_episode_ids") or []:
        h = uuid_to_hash.get(str(sid))
        if h:
            mapped_sources.append(h)
    return {
        "kind": mem.get("kind"),
        "content": mem.get("content"),
        "summary": mem.get("summary") or mem.get("content"),
        "confidence": mem.get("confidence", 0.9),
        "valid_from": mem.get("valid_from"),
        "valid_to": mem.get("valid_to"),
        "source_episode_ids": mapped_sources,
        "metadata": mem.get("metadata") or {},
        "status": mem.get("status") or "active",
    }


def _resolve_pack_version() -> str:
    override = os.environ.get("DEMO_PACK_VERSION")
    if override:
        return override.strip()
    today = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    return f"2.{today}"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":"), ensure_ascii=False))
            f.write("\n")


def _existing_manifest(pack_dir: Path) -> dict[str, Any]:
    p = pack_dir / "manifest.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_manifest(
    pack_dir: Path, *, persona: str, version: str, episode_count: int, memory_count: int
) -> None:
    existing = _existing_manifest(pack_dir)
    manifest = {
        "format": "statewave-starter-pack",
        "format_version": 1,
        "pack_id": persona,
        "display_name": existing.get("display_name") or persona,
        "description": existing.get("description") or f"{persona} demo pack.",
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subject_id_suggestion": existing.get("subject_id_suggestion") or persona,
        "episode_count": episode_count,
        "memory_count": memory_count,
        "source_count": 0,
        "tags": existing.get("tags") or ["starter-pack", "platform-bundled", "demo"],
    }
    (pack_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _load_existing_memories(pack_dir: Path) -> list[dict[str, Any]]:
    """Read the on-disk memories.jsonl. These are *curated anchor* memories
    — preserved across rebuilds so wow-moment facts the LLM compile might
    paraphrase or drop on a fresh extraction stay reliably retrievable."""
    path = pack_dir / "memories.jsonl"
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _normalise(text: str) -> str:
    """Cheap normalisation for content dedup: lowercase + collapse whitespace."""
    return " ".join((text or "").lower().split())


async def build_one(client: httpx.AsyncClient, url: str, persona: str, *, keep_temp: bool) -> None:
    pack_dir = PACKS_ROOT / persona
    if not pack_dir.is_dir():
        raise SystemExit(f"missing pack dir: {pack_dir}")

    print(f"\n=== {persona} ===")
    existing = _load_existing_episodes(pack_dir)
    print(f"  existing episodes on disk: {len(existing)}")
    existing_memories = _load_existing_memories(pack_dir)
    print(f"  existing curated memories on disk: {len(existing_memories)}")

    # Convert existing JSONL rows back to import-shaped episodes (id is the
    # content_hash, payload kept as-is). The combined set is what we feed into
    # the temp subject for compile.
    combined: list[dict[str, Any]] = []
    for ep in existing:
        combined.append(
            {
                "id": ep.get("id"),
                "source": ep.get("source"),
                "type": ep.get("type"),
                "payload": ep.get("payload") or {},
                "metadata": ep.get("metadata") or {},
                "provenance": ep.get("provenance") or {},
            }
        )

    # Idempotent expansion: dedup by stable content hash so a re-run on a
    # pack that already includes the expansion content doesn't double-add.
    existing_ids = {ep.get("id") for ep in combined if ep.get("id")}
    new_for_persona = NEW_EPISODES.get(persona) or []
    added = 0
    for src, type_, body in new_for_persona:
        ep = _new_episode_to_payload(src, type_, body)
        if ep["id"] in existing_ids:
            continue
        combined.append(ep)
        existing_ids.add(ep["id"])
        added += 1
    print(f"  new episodes to add (idempotent): {added} (would-be-dup skipped: {len(new_for_persona) - added})")

    print(f"  total episodes: {len(combined)}")

    temp_subject = f"_build_demo_pack_{persona}"
    await _delete_subject(client, url, temp_subject)
    print(f"  importing into {temp_subject}...")
    await _import_payload(client, url, target_subject_id=temp_subject, episodes=combined)

    print("  compiling memories (LLM)...")
    result = await _compile(client, url, temp_subject)
    print(f"    memories_created: {result.get('memories_created')}")

    print("  fetching back from temp subject...")
    eps = await _fetch_all(client, url, f"/admin/subjects/{temp_subject}/episodes", "episodes")
    mems = await _fetch_all(client, url, f"/admin/subjects/{temp_subject}/memories", "memories")
    print(f"    got {len(eps)} episodes, {len(mems)} memories")

    uuid_to_hash: dict[str, str] = {}
    for ep in eps:
        prov = ep.get("provenance") or {}
        h = prov.get("content_hash")
        if h:
            uuid_to_hash[str(ep["id"])] = h

    if not keep_temp:
        await _delete_subject(client, url, temp_subject)
        print(f"  deleted {temp_subject}")

    print("  writing JSONL files...")
    eps_records = [_episode_to_jsonl(ep) for ep in eps]
    llm_mem_records = [_memory_to_jsonl(m, uuid_to_hash) for m in mems]

    # Combine curated anchor memories (preserved from disk) with LLM-extracted
    # ones from this run, deduping by normalised content. The anchors win on
    # collision — they were authored deliberately to surface specific
    # wow-moment facts and shouldn't be replaced by a paraphrase.
    seen_content: set[str] = set()
    merged_memories: list[dict[str, Any]] = []
    anchor_kept = 0
    llm_added = 0
    for mem in existing_memories:
        key = _normalise(mem.get("content") or "")
        if not key or key in seen_content:
            continue
        seen_content.add(key)
        merged_memories.append(mem)
        anchor_kept += 1
    for mem in llm_mem_records:
        key = _normalise(mem.get("content") or "")
        if not key or key in seen_content:
            continue
        seen_content.add(key)
        merged_memories.append(mem)
        llm_added += 1
    print(f"  curated anchors kept: {anchor_kept}, LLM-extracted added: {llm_added}, total: {len(merged_memories)}")

    _write_jsonl(pack_dir / "episodes.jsonl", eps_records)
    _write_jsonl(pack_dir / "memories.jsonl", merged_memories)

    version = _resolve_pack_version()
    _write_manifest(
        pack_dir,
        persona=persona,
        version=version,
        episode_count=len(eps_records),
        memory_count=len(merged_memories),
    )
    print(f"  ✓ {persona} → {len(eps_records)} ep, {len(merged_memories)} mem at v{version}")


async def main_async(args: argparse.Namespace) -> int:
    server_url = os.environ.get("STATEWAVE_URL", "http://localhost:8100").rstrip("/")
    api_key = os.environ.get("STATEWAVE_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    targets = [args.persona] if args.persona else list(PERSONAS)
    invalid = [p for p in targets if p not in PERSONAS]
    if invalid:
        print(f"unknown persona(s): {invalid}", file=sys.stderr)
        return 2

    async with httpx.AsyncClient(headers=headers) as client:
        await _health(client, server_url)
        for persona in targets:
            await build_one(client, server_url, persona, keep_temp=args.keep_temp)
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--persona", choices=list(PERSONAS), help="Build only this persona.")
    p.add_argument("--keep-temp", action="store_true", help="Don't delete the temp build subject.")
    return p.parse_args()


def main() -> None:
    raise SystemExit(asyncio.run(main_async(parse_args())))


if __name__ == "__main__":
    main()
