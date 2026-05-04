"""Build the bundled Statewave Support starter pack from the live docs corpus.

This script regenerates `server/starter_packs/statewave-support-agent/{episodes,memories,manifest}.jsonl`
so the pack on disk matches what the docs-pack bootstrap would produce. A fresh
install that imports the bundled pack therefore lands on rich, docs-grounded
memory out of the box — no second step (`bootstrap_docs_pack.py --purge`)
needed, no `STATEWAVE_DOCS_PATH` mount required.

Mechanism:
  1. Read the curated docs (same allowlist used by `bootstrap_docs_pack`).
  2. Ingest every section as an episode under a temporary subject.
  3. Trigger compile on the temp subject (LLM compiler when configured).
  4. Read all episodes + memories back via admin pagination.
  5. Rewrite memory `source_episode_ids` from server-assigned UUIDs to each
     episode's stable `provenance.content_hash` so the pack's import path can
     remap them through the existing `id_map` mechanism on every fresh import.
  6. Serialise to JSONL, write a fresh manifest.json with bumped version + counts.
  7. Delete the temp subject.

Usage:
    python -m scripts.build_support_pack [--docs-path PATH] [--keep-temp]

Env:
    STATEWAVE_URL          (default http://localhost:8100)
    STATEWAVE_API_KEY      (optional)
    STATEWAVE_DOCS_PATH    (overrides --docs-path)
    SUPPORT_PACK_VERSION   (overrides the auto-version derived from PACK_VERSION)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Allow running as `python scripts/build_support_pack.py` from repo root
# in addition to `python -m scripts.build_support_pack`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.docs_loader import (  # noqa: E402
    MANIFEST,
    PACK_VERSION,
    DocSection,
    load_docs,
)

DEFAULT_DOCS_PATH = Path(__file__).resolve().parent.parent.parent / "statewave-docs"
PACK_DIR = (
    Path(__file__).resolve().parent.parent
    / "server"
    / "starter_packs"
    / "statewave-support-agent"
)
TEMP_SUBJECT = "_build_support_pack_tmp"
SOURCE = "statewave-docs"
EPISODE_TYPE = "doc_section"
INGEST_BATCH_SIZE = 50
PAGE_SIZE = 200  # admin listing max


def _section_to_episode(section: DocSection) -> dict:
    return {
        "subject_id": TEMP_SUBJECT,
        "source": SOURCE,
        "type": EPISODE_TYPE,
        "payload": section.to_episode_payload(),
        "provenance": section.to_episode_provenance(PACK_VERSION),
        "metadata": {"pack": "statewave-support-docs", "pack_version": PACK_VERSION},
    }


async def _health_check(client: httpx.AsyncClient, url: str) -> None:
    try:
        resp = await client.get(f"{url}/healthz", timeout=10.0)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: Cannot reach Statewave at {url}: {e}", file=sys.stderr)
        sys.exit(1)


async def _delete_subject(client: httpx.AsyncClient, url: str, subject: str) -> None:
    resp = await client.delete(f"{url}/v1/subjects/{subject}", timeout=30.0)
    if resp.status_code not in (200, 204, 404):
        print(
            f"  WARN: subject delete returned {resp.status_code}: {resp.text}",
            file=sys.stderr,
        )


async def _ingest(
    client: httpx.AsyncClient, url: str, sections: list[DocSection]
) -> int:
    total = 0
    for i in range(0, len(sections), INGEST_BATCH_SIZE):
        batch = sections[i : i + INGEST_BATCH_SIZE]
        resp = await client.post(
            f"{url}/v1/episodes/batch",
            json={"episodes": [_section_to_episode(s) for s in batch]},
            timeout=120.0,
        )
        if resp.status_code not in (200, 201):
            print(
                f"  ERROR ingest batch {i}-{i+len(batch)}: "
                f"{resp.status_code} {resp.text}",
                file=sys.stderr,
            )
            sys.exit(1)
        total += len(batch)
        print(f"  → ingested {total}/{len(sections)} sections")
    return total


async def _compile(client: httpx.AsyncClient, url: str) -> dict:
    # Compile is synchronous server-side; under the LLM compiler ~1-2s/section,
    # so the full corpus can run for several minutes. Mirror the timeout fix
    # already shipped in bootstrap_docs_pack.py.
    resp = await client.post(
        f"{url}/v1/memories/compile",
        json={"subject_id": TEMP_SUBJECT},
        timeout=900.0,
    )
    if resp.status_code not in (200, 201):
        print(f"  ERROR compile: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    return resp.json()


async def _fetch_all(
    client: httpx.AsyncClient, url: str, path: str, key: str
) -> list[dict]:
    out: list[dict] = []
    offset = 0
    while True:
        resp = await client.get(
            f"{url}{path}",
            params={"limit": PAGE_SIZE, "offset": offset},
            timeout=60.0,
        )
        if resp.status_code != 200:
            print(
                f"  ERROR fetch {path} offset={offset}: "
                f"{resp.status_code} {resp.text}",
                file=sys.stderr,
            )
            sys.exit(1)
        page = resp.json().get(key, [])
        if not page:
            break
        out.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return out


def _episode_to_jsonl_record(ep: dict) -> dict:
    """Render a runtime episode row into a starter-pack JSONL record.

    The episode's stable identity for re-import is its `provenance.content_hash`
    — that's what memories reference, so the import path can remap them via
    `id_map` to fresh UUIDs each time. The runtime UUID is dropped.
    """
    prov = ep.get("provenance") or {}
    content_hash = prov.get("content_hash")
    if not content_hash:
        # Should never happen for episodes produced from docs (the chunker
        # always populates content_hash) — fall back to UUID so we don't
        # silently lose the row.
        content_hash = ep["id"]
    return {
        "id": content_hash,
        "source": ep.get("source") or SOURCE,
        "type": ep.get("type") or EPISODE_TYPE,
        "created_at": ep.get("created_at"),
        "payload": ep.get("payload") or {},
        "metadata": ep.get("metadata") or {},
        "provenance": prov,
    }


def _memory_to_jsonl_record(mem: dict, uuid_to_hash: dict[str, str]) -> dict:
    """Remap a memory's `source_episode_ids` from runtime UUIDs to the same
    content-hash identities the JSONL episodes carry, so the pack's import
    path can re-resolve provenance to the freshly-minted UUIDs on every
    install."""
    mapped: list[str] = []
    for sid in mem.get("source_episode_ids") or []:
        h = uuid_to_hash.get(str(sid))
        if h:
            mapped.append(h)
        # Out-of-pack references (shouldn't happen for a fresh temp subject)
        # are silently skipped so the JSONL never carries a dangling pointer.
    return {
        "kind": mem["kind"],
        "content": mem["content"],
        "summary": mem.get("summary") or mem["content"],
        "confidence": mem.get("confidence", 0.9),
        "valid_from": mem.get("valid_from"),
        "valid_to": mem.get("valid_to"),
        "source_episode_ids": mapped,
        "metadata": mem.get("metadata") or {},
        "status": mem.get("status") or "active",
    }


def _resolve_pack_version() -> str:
    override = os.environ.get("SUPPORT_PACK_VERSION")
    if override:
        return override.strip()
    # Default: pin to the docs PACK_VERSION (currently 1) but bumped to a
    # marketing-friendly semver. Each rebuild stamps a fresh patch level
    # off the current UTC date so two builds on the same docs corpus
    # are distinguishable.
    today = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    return f"{PACK_VERSION}.{today}"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False))
            f.write("\n")


def _write_manifest(
    path: Path, *, version: str, episode_count: int, memory_count: int
) -> None:
    manifest = {
        "format": "statewave-starter-pack",
        "format_version": 1,
        "pack_id": "statewave-support-agent",
        "display_name": "Statewave Support",
        "description": (
            "Full Statewave Support docs pack — every curated section in "
            "statewave-docs chunked at heading boundaries, ingested as "
            "episodes, and compiled to memories. Importing this pack into "
            "the `statewave-support-docs` subject powers the docs-grounded "
            "support agent on day one with no extra bootstrap step."
        ),
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "subject_id_suggestion": "statewave-support-docs",
        "episode_count": episode_count,
        "memory_count": memory_count,
        "source_count": 0,
        "tags": ["starter-pack", "platform-bundled", "docs-grounded"],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


async def run(docs_path: Path, keep_temp: bool) -> None:
    server_url = os.environ.get("STATEWAVE_URL", "http://localhost:8100").rstrip("/")
    api_key = os.environ.get("STATEWAVE_API_KEY", "")

    print("=== Build Statewave Support starter pack ===")
    print(f"Docs path:  {docs_path}")
    print(f"Server:     {server_url}")
    print(f"Pack dir:   {PACK_DIR}")
    print()

    if not docs_path.is_dir():
        print(f"ERROR: docs path does not exist: {docs_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(MANIFEST)} curated docs...")
    sections = load_docs(docs_path)
    bytes_total = sum(len(s.body.encode("utf-8")) for s in sections)
    print(
        f"  Parsed {len(sections)} sections "
        f"({bytes_total/1024:.1f} KiB of body text)"
    )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    async with httpx.AsyncClient(headers=headers, timeout=120.0) as client:
        await _health_check(client, server_url)

        # Wipe any leftover temp subject from a prior run.
        await _delete_subject(client, server_url, TEMP_SUBJECT)

        print(f"\nIngesting {len(sections)} episodes (batches of {INGEST_BATCH_SIZE})...")
        await _ingest(client, server_url, sections)

        print("\nCompiling memories...")
        compile_result = await _compile(client, server_url)
        print(
            f"  ✓ Compiled {compile_result.get('compiled_memories', '?')} memories "
            f"from {compile_result.get('processed_episodes', len(sections))} episodes"
        )

        print("\nFetching back from temp subject...")
        episodes = await _fetch_all(
            client, server_url, f"/admin/subjects/{TEMP_SUBJECT}/episodes", "episodes"
        )
        memories = await _fetch_all(
            client, server_url, f"/admin/subjects/{TEMP_SUBJECT}/memories", "memories"
        )
        print(f"  Got {len(episodes)} episodes, {len(memories)} memories")

        # Build the runtime-uuid → content-hash map BEFORE cleanup so we can
        # rewrite memory provenance against stable section identities.
        uuid_to_hash: dict[str, str] = {}
        for ep in episodes:
            prov = ep.get("provenance") or {}
            h = prov.get("content_hash")
            if h:
                uuid_to_hash[str(ep["id"])] = h

        if not keep_temp:
            print("\nCleaning up temp subject...")
            await _delete_subject(client, server_url, TEMP_SUBJECT)
            print(f"  Deleted {TEMP_SUBJECT}")
        else:
            print(f"\nKeeping temp subject {TEMP_SUBJECT!r} (--keep-temp set)")

    print("\nWriting JSONL pack files...")
    episodes_records = [_episode_to_jsonl_record(ep) for ep in episodes]
    memories_records = [
        _memory_to_jsonl_record(m, uuid_to_hash) for m in memories
    ]
    _write_jsonl(PACK_DIR / "episodes.jsonl", episodes_records)
    _write_jsonl(PACK_DIR / "memories.jsonl", memories_records)

    version = _resolve_pack_version()
    _write_manifest(
        PACK_DIR / "manifest.json",
        version=version,
        episode_count=len(episodes_records),
        memory_count=len(memories_records),
    )

    print()
    print("=" * 64)
    print(f"✓ Pack rebuilt at v{version}")
    print(f"  episodes.jsonl  {len(episodes_records):4d} rows")
    print(f"  memories.jsonl  {len(memories_records):4d} rows")
    print(f"  manifest.json   v{version}")
    print("=" * 64)
    print(
        "\nNext: review the diff (`git diff server/starter_packs/statewave-support-agent/`),"
        " run pytest tests/test_admin_memory.py, then commit."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--docs-path",
        type=Path,
        default=None,
        help=f"Path to statewave-docs (default: env STATEWAVE_DOCS_PATH or {DEFAULT_DOCS_PATH})",
    )
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="Don't delete the temp build subject after pack is written (debug aid)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    docs_path = (
        args.docs_path
        or Path(os.environ.get("STATEWAVE_DOCS_PATH", str(DEFAULT_DOCS_PATH)))
    )
    asyncio.run(run(docs_path, keep_temp=args.keep_temp))
