"""Bootstrap the default Statewave support docs memory pack.

Reads the curated `statewave-docs` corpus, ingests each section as an
episode, compiles, and publishes the result under subject
`statewave-support-docs`. The result is a docs-grounded knowledge base
that a Statewave-powered support agent can query via `POST /v1/context`.

Usage:
    python -m scripts.bootstrap_docs_pack [--docs-path PATH] [--purge] [--dry-run]

Env:
    STATEWAVE_URL       (default http://localhost:8100)
    STATEWAVE_API_KEY   (optional)
    STATEWAVE_DOCS_PATH (overrides --docs-path)

Build-then-swap (why this never empties production):
    The pack is built into a *staging* subject (`<subject>-staging`) and
    compiled there first. Only after the staging build is verified to hold
    memories is it swapped into the live subject via export/import — the
    live pack is replaced by validated data in one fast step (no compile in
    the critical window). A slow, flaky, or zero-memory rebuild therefore
    leaves the live pack untouched instead of purged-empty.

Async compile (why it no longer times out):
    Compile is kicked off with `async:true` (returns a job id immediately)
    and polled to completion, instead of one long synchronous request that
    an edge proxy idle-times-out (502) on a multi-minute pack rebuild.

Idempotency: by default, fails if the LIVE subject already has episodes.
Re-run with --purge to replace it. Each episode carries a content_hash in
provenance so future incremental refresh flows can diff section-by-section.
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
STAGING_SUBJECT_ID = f"{SUBJECT_ID}-staging"

# Async-compile job polling. The server drains the whole subject in the
# background; each poll is a cheap status read, so the long-request idle
# timeout that 502'd the old synchronous compile can no longer occur.
_COMPILE_POLL_INTERVAL_S = 5.0
_COMPILE_MAX_WAIT_S = 1500.0  # 25 min ceiling for a full pack rebuild
# One resubmission on a failed or orphaned compile job. Compile-start is
# idempotent (it only queues uncompiled episodes), so a retried start resumes
# where the dead job stopped instead of redoing work. Every historical
# refresh failure — a Fly deploy restarting the server mid-job (orphaning it
# into a poll timeout or a durable `failed` row) and the occasional opaque
# compile transient — recovered on exactly one manual workflow rerun; this
# folds that rerun into the script. The docs workflow's `timeout-minutes`
# budgets for both attempts.
_COMPILE_ATTEMPTS = 2
_COMPILE_PENDING_STATUSES = frozenset(
    {"pending", "queued", "running", "processing", "in_progress", "started"}
)
_COMPILE_FAILED_STATUSES = frozenset({"failed", "error", "cancelled", "canceled"})

# HTTP statuses that warrant retry. Standard transient-failure shapes
# emitted by any reverse proxy / load balancer in front of an HTTP
# service — 502 (upstream unavailable, e.g. during a rolling restart),
# 503 (no backends accepting), 504 (idle timeout). 500 is included
# because compile occasionally surfaces transient DB connection drops
# as a 500. 429 covers rate-limit spikes from upstream model providers.
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

    Catches the standard transient-failure shapes any reverse proxy
    produces when an upstream is briefly unavailable: connection drops
    mid-stream (RemoteProtocolError), 502/503/504 from the proxy, or a
    hung socket that times out. The most common trigger is a rolling
    server deployment landing on the in-flight request, but the same
    handling covers any transient network blip — without retry, the
    docs refresh sys.exits on the first hiccup.

    Idempotency assumptions for the call sites that use this helper:
      - DELETE /v1/subjects/{id} is idempotent (404-on-missing accepted
        as success by `_purge`).
      - POST /v1/episodes/batch may produce duplicates if a prior call
        partially committed but the response was lost mid-flight; only
        staging is ingested into, and staging is purged before each run.
      - POST /v1/memories/compile (async) is idempotent: it only queues
        uncompiled episodes, so a retried start resumes cleanly.

    Backoff: 2 → 4 → 8 → 16 → 30 → 30 seconds across 5 retries (~90s
    total wall-clock). Tuned to comfortably cover a typical rolling
    deployment cycle on any platform (Fly, Render, Railway, k8s, ECS,
    etc. — usually 30–90s for a single machine to be replaced).
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


def _section_to_episode(section: DocSection, subject_id: str) -> dict:
    return {
        "subject_id": subject_id,
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


async def _episode_count(client: httpx.AsyncClient, url: str, subject_id: str) -> int:
    """Best-effort episode count via the timeline endpoint."""
    resp = await client.get(f"{url}/v1/timeline", params={"subject_id": subject_id})
    if resp.status_code != 200:
        return 0
    return len(resp.json().get("episodes", []))


async def _purge(client: httpx.AsyncClient, url: str, subject_id: str) -> None:
    resp = await _request_with_retry(
        f"purge {subject_id}",
        lambda: client.delete(f"{url}/v1/subjects/{subject_id}"),
    )
    if resp.status_code not in (200, 204, 404):
        print(
            f"  WARN: delete {subject_id} returned {resp.status_code}: {resp.text}",
            file=sys.stderr,
        )


async def _ingest_batched(
    client: httpx.AsyncClient,
    url: str,
    sections: list[DocSection],
    subject_id: str,
    batch_size: int = BATCH_SIZE,
) -> int:
    total = 0
    for i in range(0, len(sections), batch_size):
        batch = sections[i : i + batch_size]
        body = {"episodes": [_section_to_episode(s, subject_id) for s in batch]}
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


async def _compile_async(client: httpx.AsyncClient, url: str, subject_id: str) -> dict:
    """Start an async compile, poll the job to completion, return its result.

    Replaces the single long synchronous compile (which an edge proxy
    idle-times-out → 502 on a multi-minute rebuild) with a quick start
    request plus cheap status polls. Returns the terminal job payload —
    its ``memories_created`` is the authoritative compile count. The
    per-subject ``memory_count`` in /v1/subjects is eventually-consistent
    and lags right after a large compile, so never gate on it here.

    A job that dies under us — the server redeployed mid-compile and the
    job row went ``failed`` or was orphaned into a poll timeout — is
    resubmitted once (see ``_COMPILE_ATTEMPTS``) before the script gives
    up; the resubmission resumes over the remaining uncompiled episodes.
    """
    for attempt in range(1, _COMPILE_ATTEMPTS + 1):
        result = await _compile_attempt(client, url, subject_id)
        if result is not None:
            return result
        if attempt < _COMPILE_ATTEMPTS:
            print(
                f"  compile attempt {attempt} died — resubmitting (start is "
                "idempotent over uncompiled episodes, the retry resumes where "
                "the dead job stopped)",
                file=sys.stderr,
            )
    print("  ERROR compile failed after resubmission — giving up", file=sys.stderr)
    sys.exit(1)


async def _compile_attempt(
    client: httpx.AsyncClient, url: str, subject_id: str
) -> dict | None:
    """One start+poll cycle. ``None`` = retryable death (job failed or
    orphaned); the caller decides whether another attempt remains."""
    resp = await _request_with_retry(
        "compile-start",
        lambda: client.post(
            f"{url}/v1/memories/compile",
            json={"subject_id": subject_id, "async": True},
        ),
    )
    if resp.status_code not in (200, 201, 202):
        print(f"  ERROR compile-start: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    job = resp.json()
    job_id = job.get("job_id")
    if not job_id:
        # Older server without async support returned an inline result —
        # the work is already done synchronously.
        return job

    waited = 0.0
    while waited < _COMPILE_MAX_WAIT_S:
        await asyncio.sleep(_COMPILE_POLL_INTERVAL_S)
        waited += _COMPILE_POLL_INTERVAL_S
        jr = await _request_with_retry(
            "compile-poll",
            lambda: client.get(f"{url}/v1/memories/compile/{job_id}"),
        )
        status = str(jr.json().get("status", "")).lower()
        if status in _COMPILE_FAILED_STATUSES:
            print(
                f"  ERROR compile job {job_id} ended in status {status!r}: "
                f"{jr.text}",
                file=sys.stderr,
            )
            return None
        if status not in _COMPILE_PENDING_STATUSES:
            return jr.json()  # terminal success — payload carries memories_created
    print(
        f"  ERROR compile job {job_id} did not finish within "
        f"{_COMPILE_MAX_WAIT_S:.0f}s",
        file=sys.stderr,
    )
    return None


async def _export(client: httpx.AsyncClient, url: str, subject_id: str) -> dict:
    resp = await _request_with_retry(
        f"export {subject_id}",
        lambda: client.get(f"{url}/admin/export/{subject_id}"),
    )
    if resp.status_code != 200:
        print(f"  ERROR export {subject_id}: {resp.status_code} {resp.text}", file=sys.stderr)
        sys.exit(1)
    return resp.json()


async def _import_into(
    client: httpx.AsyncClient, url: str, document: dict, target_subject_id: str
) -> dict:
    body = {
        "document": document,
        "target_subject_id": target_subject_id,
        "preserve_ids": False,
        # Export documents don't carry subject_entities; without this the
        # live pack has an empty entity store after every swap (issue #380).
        "rebuild_entities": True,
    }
    resp = await _request_with_retry(
        f"import -> {target_subject_id}",
        lambda: client.post(f"{url}/admin/import", json=body),
    )
    if resp.status_code not in (200, 201):
        print(
            f"  ERROR import -> {target_subject_id}: {resp.status_code} {resp.text}",
            file=sys.stderr,
        )
        sys.exit(1)
    return resp.json()


async def run(docs_path: Path, purge: bool, dry_run: bool) -> None:
    server_url = os.environ.get("STATEWAVE_URL", "http://localhost:8100").rstrip("/")
    api_key = os.environ.get("STATEWAVE_API_KEY", "")

    print("=== Statewave default docs memory pack ===")
    print(f"Subject:      {SUBJECT_ID}  (staging: {STAGING_SUBJECT_ID})")
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

        live_existing = await _episode_count(client, server_url, SUBJECT_ID)
        if live_existing > 0 and not purge:
            print(
                f"\nERROR: subject {SUBJECT_ID!r} already has {live_existing} "
                "episodes.\n       Re-run with --purge to replace it.",
                file=sys.stderr,
            )
            sys.exit(2)

        # 1. Build into staging (never touch live yet).
        print(f"\nBuilding staging pack {STAGING_SUBJECT_ID!r}...")
        await _purge(client, server_url, STAGING_SUBJECT_ID)
        print(f"Ingesting {len(sections)} episodes (batches of {BATCH_SIZE})...")
        await _ingest_batched(client, server_url, sections, STAGING_SUBJECT_ID)
        print("Compiling memories (async)...")
        compile_result = await _compile_async(client, server_url, STAGING_SUBJECT_ID)
        attempt_mem = int(compile_result.get("memories_created", 0) or 0)

        # 2. Verify staging BEFORE touching live — on the EXPORTED content,
        #    not the last attempt's `memories_created`: a resubmitted attempt
        #    that merely resumed a nearly-finished job legitimately reports a
        #    low (even zero) count while staging holds the full pack. A
        #    zero-memory export (compiler regression, payload drift) aborts
        #    here and the live pack is left exactly as it was.
        document = await _export(client, server_url, STAGING_SUBJECT_ID)
        staging_mem = len(document.get("memories", []) or [])
        print(
            f"  ✓ staging holds {staging_mem} memories from {len(sections)} episodes"
            f" (last attempt created {attempt_mem})"
        )
        if len(sections) > 0 and staging_mem == 0:
            print(
                "\nERROR: staging holds 0 memories despite ingesting "
                f"{len(sections)} episodes. Refusing to swap — the LIVE pack is "
                "untouched.",
                file=sys.stderr,
            )
            await _purge(client, server_url, STAGING_SUBJECT_ID)
            sys.exit(1)

        # 3. Swap staging -> live: replace live with the validated staging
        #    export in one fast import (no compile in the window).
        print(f"\nSwapping {STAGING_SUBJECT_ID!r} -> live {SUBJECT_ID!r}...")
        await _purge(client, server_url, SUBJECT_ID)
        result = await _import_into(client, server_url, document, SUBJECT_ID)
        imported = int(result.get("memories_imported", 0) or 0)
        print(f"  ✓ swapped {imported} memories into {SUBJECT_ID}")

        # 4. Drop staging. The import result is the authoritative count of what
        #    landed in live (no eventual-consistency lag to race).
        await _purge(client, server_url, STAGING_SUBJECT_ID)
        if imported == 0:
            print(
                f"\nERROR: live subject {SUBJECT_ID!r} has 0 memories after swap.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"\nDone. The default support docs pack is ready ({imported} memories).")
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
        help="Replace the live subject even if it already has episodes",
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
