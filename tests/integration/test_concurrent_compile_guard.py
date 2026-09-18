"""Concurrent compiles of one subject must not duplicate memories (#417).

Two compiles that read the same uncompiled episodes both compile them and
both write memories. Nothing serialised them: the uncompiled read took no
lock, the compiled mark was unconditional, and the async attach check was
a plain SELECT followed by an unguarded INSERT.

The severity is easy to understate, and that shapes these tests. Under the
default heuristic compiler the twins are near-identical, so the lexical
supersession pass retires one and only the row count looks wrong. Under a
compiler whose wording varies, which is the production and benchmark
configuration, the twins fall under the Jaccard threshold and BOTH stay
active and retrievable. `test_paraphrased_output_leaves_one_active` is the
one that fails for the reason users would actually notice.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytestmark = pytest.mark.anyio



_PHRASINGS = (
    "customer indicated a preference while discussing {ep}",
    "during {ep} the buyer mentioned wanting this enabled by default",
)


def _slow_paraphrasing_compiler(delay: float = 0.4):
    """A compiler that is slow and never repeats its wording.

    Both properties are load-bearing. The delay guarantees the two requests
    genuinely overlap, since an instant compiler lets them serialise by luck
    and the test then passes without the guard. The varying wording keeps the
    twins below the lexical supersession threshold, which is what leaves both
    copies active and retrievable rather than merely inflating the table.
    """
    import itertools
    import time

    from server.db.tables import MemoryRow

    counter = itertools.count()

    class _Compiler:
        def compile(self, episodes, *, claim_keys=None):
            n = next(counter)
            time.sleep(delay)  # sync compile runs in a worker thread
            # The two runs must share almost no vocabulary. Near-identical
            # wording scores above the lexical supersession threshold, one
            # copy is retired, and the test then passes while the defect is
            # still there: the table is inflated but retrieval looks fine.
            # Real LLM output varies at least this much between runs.
            phrasing = _PHRASINGS[n % len(_PHRASINGS)]
            return [
                MemoryRow(
                    subject_id=ep.subject_id,
                    kind="profile_fact",
                    content=phrasing.format(ep=ep.id),
                    summary="variant",
                    source_episode_ids=[ep.id],
                    metadata_={},
                    valid_from=ep.occurred_at or ep.created_at,
                )
                for ep in episodes
            ]

    return _Compiler()


async def _seed(client, subject_id: str, n: int = 4) -> None:
    for i in range(n):
        resp = await client.post(
            "/v1/episodes",
            json={
                "subject_id": subject_id,
                "source": "test",
                "type": "conversation",
                "payload": {"text": f"user prefers setting number {i} enabled"},
            },
        )
        assert resp.status_code in (200, 201), resp.text


async def _memory_rows(session_factory, subject_id: str):
    from sqlalchemy import select

    from server.db.tables import MemoryRow

    async with session_factory() as session:
        rows = (
            await session.execute(
                select(MemoryRow).where(MemoryRow.subject_id == subject_id)
            )
        ).scalars().all()
        return list(rows)


def _rows_per_episode(rows) -> dict[str, int]:
    """How many memory rows each episode backs.

    Deliberately not keyed on content. A compiler whose wording varies, which
    is the production and benchmark configuration, produces DIFFERENT text on
    each run, so a content-keyed duplicate check would report nothing for
    exactly the race this file exists to catch.
    """
    counts: dict[str, int] = {}
    for r in rows:
        for ep in r.source_episode_ids or []:
            counts[str(ep)] = counts.get(str(ep), 0) + 1
    return counts


async def test_two_sync_compiles_do_not_duplicate(client, session_factory, monkeypatch):
    from server.api import memories as api_memories

    subject_id = f"race-sync-{uuid.uuid4().hex[:8]}"
    await _seed(client, subject_id)
    compiler = _slow_paraphrasing_compiler()
    monkeypatch.setattr(api_memories, "get_compiler", lambda: compiler)

    a, b = await asyncio.gather(
        client.post("/v1/memories/compile", json={"subject_id": subject_id}),
        client.post("/v1/memories/compile", json={"subject_id": subject_id}),
        return_exceptions=False,
    )
    # One does the work. The other either waited it out and found nothing,
    # or was told the subject was busy. Never both compiling.
    assert {a.status_code, b.status_code} <= {200, 409}
    assert 200 in (a.status_code, b.status_code)

    rows = await _memory_rows(session_factory, subject_id)
    assert rows, "the winning compile must still produce memories"
    per_episode = _rows_per_episode(rows)
    assert max(per_episode.values()) == 1, f"an episode was compiled twice: {per_episode}"


async def test_sync_racing_async_does_not_duplicate(client, session_factory, monkeypatch):
    from server.api import memories as api_memories

    subject_id = f"race-mixed-{uuid.uuid4().hex[:8]}"
    await _seed(client, subject_id)
    compiler = _slow_paraphrasing_compiler()
    monkeypatch.setattr(api_memories, "get_compiler", lambda: compiler)

    async_resp, sync_resp = await asyncio.gather(
        client.post("/v1/memories/compile", json={"subject_id": subject_id, "async": True}),
        client.post("/v1/memories/compile", json={"subject_id": subject_id}),
    )
    assert async_resp.status_code == 202
    assert sync_resp.status_code in (200, 409)
    for _ in range(40):
        await asyncio.sleep(0.25)
        rows = await _memory_rows(session_factory, subject_id)
        if rows:
            break
    per_episode = _rows_per_episode(rows)
    assert max(per_episode.values()) == 1, f"an episode was compiled twice: {per_episode}"


async def test_two_async_submits_share_one_job(client, session_factory):
    subject_id = f"race-async-{uuid.uuid4().hex[:8]}"
    await _seed(client, subject_id)

    a, b = await asyncio.gather(
        client.post("/v1/memories/compile", json={"subject_id": subject_id, "async": True}),
        client.post("/v1/memories/compile", json={"subject_id": subject_id, "async": True}),
    )
    assert a.status_code == 202 and b.status_code == 202
    # The find-then-insert is serialised, so the loser attaches to the
    # winner's job instead of creating a second one.
    assert a.json()["job_id"] == b.json()["job_id"], (a.json(), b.json())


    for _ in range(40):
        await asyncio.sleep(0.25)
        rows = await _memory_rows(session_factory, subject_id)
        if rows:
            break
    per_episode = _rows_per_episode(rows)
    assert max(per_episode.values()) == 1, f"an episode was compiled twice: {per_episode}"


async def test_paraphrased_output_leaves_one_active(client, session_factory, monkeypatch):
    """The retrieval-visible failure.

    With varying wording the duplicate is not retired by lexical
    supersession, so without the guard BOTH copies stay active and a reader
    sees the same fact twice. This is the one that fails for the reason a
    user would actually notice.
    """
    from server.api import memories as api_memories

    compiler = _slow_paraphrasing_compiler()
    monkeypatch.setattr(api_memories, "get_compiler", lambda: compiler)

    subject_id = f"race-para-{uuid.uuid4().hex[:8]}"
    await _seed(client, subject_id, n=3)

    await asyncio.gather(
        client.post("/v1/memories/compile", json={"subject_id": subject_id}),
        client.post("/v1/memories/compile", json={"subject_id": subject_id}),
    )

    rows = await _memory_rows(session_factory, subject_id)
    active = [r for r in rows if r.status == "active"]
    per_episode: dict[str, int] = {}
    for r in active:
        for ep in r.source_episode_ids or []:
            per_episode[str(ep)] = per_episode.get(str(ep), 0) + 1
    assert per_episode, "expected active memories"
    assert max(per_episode.values()) == 1, f"a reader would see a fact twice: {per_episode}"


async def test_failed_compile_releases_the_lock_and_keeps_episodes(client, session_factory):
    """Issue #201 under the lock: a failed compile must consume nothing and
    hold nothing. The lock is transaction-scoped, so rollback releases it."""
    from sqlalchemy import select, text

    from server.api import memories as api_memories
    from server.db.tables import EpisodeRow
    from server.services.compilers.errors import CompilationError

    subject_id = f"race-fail-{uuid.uuid4().hex[:8]}"
    await _seed(client, subject_id, n=3)

    class _Raising:
        def compile(self, episodes, *, claim_keys=None):
            raise CompilationError("no reachable provider")

    original = api_memories.get_compiler
    api_memories.get_compiler = lambda: _Raising()
    try:
        resp = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
        assert resp.status_code == 502
    finally:
        api_memories.get_compiler = original

    async with session_factory() as session:
        uncompiled = (
            await session.execute(
                select(EpisodeRow).where(
                    EpisodeRow.subject_id == subject_id,
                    EpisodeRow.last_compiled_at.is_(None),
                )
            )
        ).scalars().all()
        assert len(uncompiled) == 3, "a failed compile must not consume episodes"
        held = (
            await session.execute(
                text("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory'")
            )
        ).scalar()
        assert held == 0, "the subject lock must not survive a failed compile"

    # And the subject is immediately compilable again.
    retry = await client.post("/v1/memories/compile", json={"subject_id": subject_id})
    assert retry.status_code == 200
    assert retry.json()["memories_created"] > 0
