"""Bootstrap the bundled demo-agent starter packs on a fresh server.

A fresh Statewave server has no demo data, so there is nothing for an
operator to "play with" out of the box and the marketing/demo persona
dropdown has nothing to show. This seeds the five bundled demo-agent
packs (`demo-support-agent`, `demo-coding-assistant`, `demo-sales-copilot`,
`demo-devops-agent`, `demo-research-assistant`) onto their canonical
`demo-<persona>` subject ids by calling the same admin starter-pack
import endpoint the operator UI uses.

This is the CORRECT path for full demo personas (~44 episodes each,
bundled episodes.jsonl + memories.jsonl — no compile needed). Do NOT
use `scripts/seed_demo_subjects.py` for this — that script seeds the
deliberately-minimal statewave-web hero visualization, not the demo
agents.

Idempotency: a pack whose subject already has episodes is skipped, so
this is safe to run on every container start (fresh installs get
seeded; restarts no-op; operator-added rows on other subjects are never
touched). Exit codes mirror `bootstrap_docs_pack`:
    0  at least one pack was freshly seeded
    2  every demo subject already populated — nothing to do
    1  error (server still serves; this is non-fatal at boot)

Usage:
    python -m scripts.bootstrap_demo_packs [--only id,id] [--force] [--dry-run]

Env:
    STATEWAVE_URL       (default http://localhost:8100)
    STATEWAVE_API_KEY   (optional; sent as X-API-Key when set)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INDEX_PATH = (
    Path(__file__).resolve().parent.parent / "server" / "starter_packs" / "index.json"
)
DEFAULT_URL = os.environ.get("STATEWAVE_URL", "http://localhost:8100")
API_KEY = os.environ.get("STATEWAVE_API_KEY", "")

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_NETWORK = (
    httpx.NetworkError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
)


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def _demo_pack_ids() -> list[str]:
    """Demo-agent pack ids from the bundled index (auto-picks new ones)."""
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return [
        p["pack_id"]
        for p in data.get("packs", [])
        if p.get("kind") == "demo_agent" and p.get("pack_id")
    ]


async def _retry(op: str, fn, *, attempts: int = 5, delay: float = 2.0) -> httpx.Response:
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            resp = await fn()
            if resp.status_code in _RETRYABLE_STATUS and i < attempts:
                raise httpx.HTTPStatusError(
                    f"{op}: retryable {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
            return resp
        except (_RETRYABLE_NETWORK + (httpx.HTTPStatusError,)) as e:  # noqa: PERF203
            last = e
            if i >= attempts:
                break
            await asyncio.sleep(min(delay * (2 ** (i - 1)), 30.0))
    raise RuntimeError(f"{op}: exhausted retries: {last!r}")


async def _populated_subjects(client: httpx.AsyncClient, url: str) -> set[str]:
    resp = await _retry(
        "list-subjects", lambda: client.get(f"{url}/v1/subjects", headers=_headers())
    )
    resp.raise_for_status()
    out: set[str] = set()
    for s in resp.json().get("subjects", []):
        if (s.get("episode_count") or 0) > 0:
            out.add(s.get("subject_id"))
    return out


async def _seed_one(client: httpx.AsyncClient, url: str, pack_id: str) -> None:
    # Clear any partial residue, then land the bundled pack on the
    # canonical subject id (merge = land on this exact id; default
    # create_copy would fork to a unique id the demo dropdown can't find).
    await _retry(
        f"delete {pack_id}",
        lambda: client.delete(f"{url}/v1/subjects/{pack_id}", headers=_headers()),
    )
    body = {
        "pack_id": pack_id,
        "target_subject_id": pack_id,
        "conflict_strategy": "merge",
    }
    resp = await _retry(
        f"import {pack_id}",
        lambda: client.post(
            f"{url}/admin/memory/starter-packs/import",
            headers=_headers(),
            json=body,
        ),
    )
    resp.raise_for_status()


async def run(url: str, only: list[str] | None, force: bool, dry_run: bool) -> int:
    pack_ids = _demo_pack_ids()
    if only:
        pack_ids = [p for p in pack_ids if p in set(only)]
    if not pack_ids:
        print("bootstrap_demo_packs: no demo_agent packs found in index.json")
        return 2

    seeded = 0
    async with httpx.AsyncClient(timeout=120.0) as client:
        populated = set() if force else await _populated_subjects(client, url)
        for pid in pack_ids:
            if pid in populated:
                print(f"  - {pid}: already populated — skipped")
                continue
            if dry_run:
                print(f"  - {pid}: would seed (dry-run)")
                continue
            await _seed_one(client, url, pid)
            print(f"  ✓ {pid}: seeded from bundled pack")
            seeded += 1

    if seeded == 0:
        print("bootstrap_demo_packs: all demo subjects already populated.")
        return 2
    print(f"bootstrap_demo_packs: seeded {seeded} demo pack(s).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL, help=f"API base (default {DEFAULT_URL})")
    ap.add_argument("--only", help="Comma-separated pack ids to seed (default: all demo packs)")
    ap.add_argument(
        "--force", action="store_true", help="Re-seed even if the subject already has data"
    )
    ap.add_argument("--dry-run", action="store_true", help="Report what would be seeded; no writes")
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    try:
        return asyncio.run(run(args.url, only, args.force, args.dry_run))
    except Exception as e:  # noqa: BLE001 — non-fatal at boot; report and exit 1
        print(f"bootstrap_demo_packs: error: {e!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
