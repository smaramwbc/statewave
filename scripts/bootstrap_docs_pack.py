"""Bootstrap the default Statewave support docs memory pack.

Reads the curated `statewave-docs` corpus, ingests each section as an
episode under subject `statewave-support-docs`, then compiles. The
result is a docs-grounded knowledge base that a Statewave-powered
support agent can query via `POST /v1/context`.

Usage:
    python -m scripts.bootstrap_docs_pack [--docs-path PATH] [--purge] [--dry-run]

Env:
    STATEWAVE_URL       (default http://localhost:8100)
    STATEWAVE_API_KEY   (optional)
    STATEWAVE_DOCS_PATH (overrides --docs-path)

Idempotency: by default, fails if the subject already has episodes.
Re-run with --purge to wipe and rebuild from scratch. Each episode
carries a content_hash in provenance so future incremental refresh
flows can diff section-by-section.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import httpx

# Allow running as `python scripts/bootstrap_docs_pack.py` from repo root
# in addition to `python -m scripts.bootstrap_docs_pack`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.docs_loader import (  # noqa: E402
    MANIFEST,
    PACK_VERSION,
    SUBJECT_ID,
    DocSection,
    load_docs,
)

DEFAULT_DOCS_PATH = Path(__file__).resolve().parent.parent.parent / "statewave-docs"
BATCH_SIZE = 50
SOURCE = "statewave-docs"
EPISODE_TYPE = "doc_section"

# HTTP statuses that warrant retry. These are the failure shapes a Fly
# rolling deploy produces when it lands on an in-flight request: 502
# (bad gateway while the upstream machine is being replaced), 503 (no
# machines accepting), 504 (request idle timeout while a machine is
# being replaced). 500 is included because compile failures sometimes
# surface as 500 from transient DB connection drops during deploy. 429
# covers rate-limit spikes.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRYABLE_NETWORK_EXCEPTIONS = (
    httpx.NetworkError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
)


async def _request_with_retry(
    op: str,
    fn,
    *,
    attempts: int = 6,
    initial_delay_s: float = 2.0,
    max_delay_s: float = 30.0,
) -> httpx.Response:
    """Run an httpx call with exponential backoff on transient failures.

    Catches the failure shape produced by a Fly rolling deploy that
    lands on an in-flight request: connection drops mid-stream
    (RemoteProtocolError), 502/503/504 from the platform, or a hung
    socket that times out. Without retry, the docs refresh silently
    leaves the support pack empty whenever a deploy collides — the pack
    only refreshes again on the next docs-repo push or manual rerun.

    Idempotency assumptions for the call sites that use this helper:
      - DELETE /v1/subjects/{id} is idempotent (404-on-missing accepted
        as success by `_purge`).
      - POST /v1/episodes/batch may produce duplicates if a prior call
        partially committed but the response was lost mid-flight; the
        downstream compile pass tolerates duplicate episodes gracefully
        and the workflow's verify-step only checks "episodes > 0".
      - POST /v1/memories/compile only processes uncompiled episodes,
        so a retry naturally resumes from where the killed call left
        off without re-doing work.

    Backoff: 2 → 4 → 8 → 16 → 30 → 30 seconds across 5 retries (~90s
    total wall-clock), which covers a typical Fly rolling deploy cycle
    (~30-60s per machine, two machines).
    """
    delay = initial_delay_s
    for attempt in range(1, attempts + 1):
        try:
            resp = await fn()
        except _RETRYABLE_NETWORK_EXCEPTIONS as e:
            if attempt == attempts:
                raise
            print(
                f"  {op} attempt {attempt}/{attempts} failed: "
                f"{type(e).__name__}: {e}; retrying in {delay:.1f}s",
                file=sys.stderr,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay_s)
            continue
        if resp.status_code in _RETRYABLE_STATUS and attempt < attempts:
            print(
                f"  {op} attempt {attempt}/{attempts} got HTTP "
                f"{resp.status_code}; retrying in {delay:.1f}s",
                file=sys.stderr,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay_s)
            continue
        return resp
    # Unreachable: the loop either returns or raises above.
    raise RuntimeError(f"{op}: retry loop exited without result")


def _section_to_episode(section: DocSection) -> dict:
    return {
        "subject_id": SUBJECT_ID,
        "source": SOURCE,
        "type": EPISODE_TYPE,
        "payload": section.to_episode_payload(),
        "provenance": section.to_episode_provenance(PACK_VERSION),
        "metadata": {"pack": "statewave-support-docs", "pack_version": PACK_VERSION},
    }


async def _health_check(client: httpx.AsyncClient, url: str) -> None:
    try:
        resp = await client.get(f"{url}/healthz")
        resp.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach Statewave at {url}: {e}", file=sys.stderr)
        sys.exit(1)


async def _existing_episode_count(client: httpx.AsyncClient, url: str) -> int:
    """Best-effort check via the timeline endpoint."""
    resp = await client.get(f"{url}/v1/timeline", params={"subject_id": SUBJECT_ID})
    if resp.status_code != 200:
        return 0
    return len(resp.json().get("episodes", []))


async def _purge(client: httpx.AsyncClient, url: str) -> None:
    resp = await _request_with_retry(
        "purge",
        lambda: client.delete(f"{url}/v1/subjects/{SUBJECT_ID}"),
    )
    if resp.status_code not in (200, 204, 404):
        print(
            f"  WARN: subject delete returned {resp.status_code}: {resp.text}",
            file=sys.stderr,
        )


async def _ingest_batched(
    client: httpx.AsyncClient,
    url: str,
    sections: list[DocSection],
    batch_size: int = BATCH_SIZE,
) -> int:
    total = 0
    for i in range(0, len(sections), batch_size):
        batch = sections[i : i + batch_size]
        body = {"episodes": [_section_to_episode(s) for s in batch]}
        resp = await _request_with_retry(
            f"ingest batch {i}-{i+len(batch)}",
            lambda body=body: client.post(f"{url}/v1/episodes/batch", json=body),
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
    # Compile is synchronous server-side: it walks every uncompiled episode
    # under the subject and (in LLM mode) runs the compiler against each
    # section. ~1-2s per section * 200+ sections puts the call well past the
    # client default's 120s — bumped to 600s so a full pack rebuild completes
    # even on the first run after a purge. The fly platform's request idle
    # timeout sits comfortably above this.
    resp = await _request_with_retry(
        "compile",
        lambda: client.post(
            f"{url}/v1/memories/compile",
            json={"subject_id": SUBJECT_ID},
            timeout=600.0,
        ),
    )
    if resp.status_code not in (200, 201):
        print(f"  ERROR compile: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    return resp.json()


async def run(docs_path: Path, purge: bool, dry_run: bool) -> None:
    server_url = os.environ.get("STATEWAVE_URL", "http://localhost:8100").rstrip("/")
    api_key = os.environ.get("STATEWAVE_API_KEY", "")

    print("=== Statewave default docs memory pack ===")
    print(f"Subject:      {SUBJECT_ID}")
    print(f"Pack version: v{PACK_VERSION}")
    print(f"Docs path:    {docs_path}")
    print(f"Server:       {server_url}")
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

    if dry_run:
        print()
        print("--- dry run: section preview ---")
        for s in sections[:6]:
            print(f"  [{s.doc_path}] {' › '.join(s.heading_path)}")
            preview = s.body.replace("\n", " ")[:80]
            print(f"      {preview}...")
        if len(sections) > 6:
            print(f"  ... and {len(sections) - 6} more")
        return

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    async with httpx.AsyncClient(headers=headers, timeout=120.0) as client:
        await _health_check(client, server_url)

        existing = await _existing_episode_count(client, server_url)
        if existing > 0 and not purge:
            print(
                f"\nERROR: subject {SUBJECT_ID!r} already has {existing} episodes.\n"
                "       Re-run with --purge to wipe and rebuild.",
                file=sys.stderr,
            )
            sys.exit(2)

        if existing > 0 and purge:
            print(f"Purging existing subject ({existing} episodes)...")
            await _purge(client, server_url)
        elif purge:
            print("Subject is empty — proceeding with fresh ingest (no purge needed).")

        print(f"\nIngesting {len(sections)} episodes (batches of {BATCH_SIZE})...")
        await _ingest_batched(client, server_url, sections)

        print("\nCompiling memories...")
        result = await _compile(client, server_url)
        memories_created = result.get("memories_created", 0)
        print(
            f"  ✓ Compiled {memories_created} memories from {len(sections)} episodes"
        )
        # Guard against the silent-failure mode that produces hallucinated
        # answers in the support widget: ingest reports success, compile
        # returns 200, but no memories were extracted (e.g. compiler
        # regression, payload-shape drift). Fail loudly here so CI catches
        # it before the broken pack is left in production.
        if len(sections) > 0 and memories_created == 0:
            print(
                "\nERROR: compile returned 0 memories despite ingesting "
                f"{len(sections)} episodes. The pack would answer poorly. "
                "Refusing to leave production in this state.",
                file=sys.stderr,
            )
            sys.exit(1)

    print("\nDone. The default support docs pack is ready.")
    print(f"Try: POST {server_url}/v1/context  with subject_id={SUBJECT_ID!r}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--docs-path",
        type=Path,
        default=Path(os.environ.get("STATEWAVE_DOCS_PATH", DEFAULT_DOCS_PATH)),
        help="Path to a checkout of statewave-docs (default: sibling dir)",
    )
    p.add_argument(
        "--purge",
        action="store_true",
        help="Delete existing episodes for the subject before ingesting",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk but skip all HTTP calls",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(run(args.docs_path, purge=args.purge, dry_run=args.dry_run))
