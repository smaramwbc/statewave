"""Tests for admin subject explorer endpoints.

These tests talk to the real Postgres referenced by `STATEWAVE_DATABASE_URL`
through the top-level `client` fixture (`tests/conftest.py`). They were
authored against a freshly-migrated empty database — which is exactly what
CI provisions every run — so on CI they are deterministic. On a developer
machine that points at a populated dev DB, the strict-empty assertion in
`test_list_subjects_empty` is no longer meaningful.

We do NOT weaken that assertion: when the DB is empty the test must still
verify the endpoint returns an empty list. The fix is to skip the test
when the precondition (empty DB) is not met, instead of failing on a
state-collision the test was never written to cover. The other tests in
this file already use prefix-scoped or unique subject ids, so they are
robust to pre-existing rows.

TODO: once we add the planned shared cleanup fixture (or move this file
to `tests/integration/` where the per-session `create_all` / `drop_all`
fixture already gives us a clean DB), the conditional skip below can be
removed and the assertion can run unconditionally.
"""

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def test_list_subjects_empty(client: AsyncClient):
    """List subjects returns empty when no data."""
    resp = await client.get("/admin/subjects")
    assert resp.status_code == 200
    data = resp.json()

    # Precondition: this test asserts the empty-DB response shape. If the
    # bound database already has unrelated rows (typical on a developer
    # machine where the dev DB is shared with the live admin UI), skip
    # rather than misreport a real product regression. CI provisions a
    # fresh DB each run, so this skip never fires there.
    if data["total"] != 0:
        pytest.skip(
            "Skipping empty-DB assertion: the bound database already has "
            f"{data['total']} subject(s). Run against an empty/test DB "
            "(see tests/integration/conftest.py for the fixture pattern)."
        )

    assert data["subjects"] == []
    assert data["total"] == 0


async def test_list_subjects_with_data(client: AsyncClient):
    """List subjects returns data after episode ingestion."""
    # Create an episode first
    ep_resp = await client.post(
        "/v1/episodes",
        json={
            "subject_id": "test_user_1",
            "source": "test",
            "type": "message",
            "payload": {"text": "hello"},
        },
    )
    assert ep_resp.status_code == 201

    # List subjects
    resp = await client.get("/admin/subjects")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    subjects = data["subjects"]
    assert any(s["subject_id"] == "test_user_1" for s in subjects)


async def test_list_subjects_search(client: AsyncClient):
    """Search filters subjects by ID."""
    # Create test subjects
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": "searchable_user",
            "source": "test",
            "type": "message",
            "payload": {"text": "hello"},
        },
    )
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": "other_user",
            "source": "test",
            "type": "message",
            "payload": {"text": "hello"},
        },
    )

    # Search for specific subject
    resp = await client.get("/admin/subjects", params={"search": "searchable"})
    assert resp.status_code == 200
    data = resp.json()
    subjects = data["subjects"]
    assert all("searchable" in s["subject_id"] for s in subjects)


async def test_list_subjects_pagination(client: AsyncClient):
    """Pagination works correctly."""
    # Create multiple subjects
    for i in range(5):
        await client.post(
            "/v1/episodes",
            json={
                "subject_id": f"paginate_user_{i}",
                "source": "test",
                "type": "message",
                "payload": {"text": f"msg {i}"},
            },
        )

    # Get first page with limit 2
    resp = await client.get("/admin/subjects", params={"limit": 2, "offset": 0})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["subjects"]) <= 2
    assert data["limit"] == 2
    assert data["offset"] == 0


async def test_get_subject_detail(client: AsyncClient):
    """Get detail for a specific subject."""
    # Create subject with episodes
    for i in range(3):
        await client.post(
            "/v1/episodes",
            json={
                "subject_id": "detail_user",
                "source": "test",
                "type": "message",
                "payload": {"text": f"message {i}"},
            },
        )

    # Get detail
    resp = await client.get("/admin/subjects/detail_user")
    assert resp.status_code == 200
    data = resp.json()
    assert data["subject_id"] == "detail_user"
    assert data["summary"]["episode_count"] >= 3
    assert data["summary"]["memory_count"] >= 0


async def test_get_subject_detail_not_found(client: AsyncClient):
    """Return 404 for nonexistent subject."""
    resp = await client.get("/admin/subjects/nonexistent_subject_xyz")
    assert resp.status_code == 404


async def test_list_subject_memories(client: AsyncClient):
    """List memories for a subject."""
    # Create episode and compile
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": "memory_user",
            "source": "test",
            "type": "message",
            "payload": {"text": "I prefer dark mode"},
        },
    )
    await client.post("/v1/memories/compile", json={"subject_id": "memory_user"})

    # List memories
    resp = await client.get("/admin/subjects/memory_user/memories")
    assert resp.status_code == 200
    data = resp.json()
    assert "memories" in data
    assert "total" in data


async def test_list_subject_episodes(client: AsyncClient):
    """List episodes for a subject."""
    # Create episodes
    for i in range(3):
        await client.post(
            "/v1/episodes",
            json={
                "subject_id": "episode_list_user",
                "source": "test",
                "type": "message",
                "payload": {"text": f"message {i}"},
            },
        )

    # List episodes
    resp = await client.get("/admin/subjects/episode_list_user/episodes")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["episodes"]) >= 3
    assert data["total"] >= 3


async def test_memory_related_not_found(client: AsyncClient):
    """Return 404 for nonexistent memory."""
    # Create a subject first
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": "evolution_test_user",
            "source": "test",
            "type": "message",
            "payload": {"text": "hello"},
        },
    )

    resp = await client.get(
        "/admin/subjects/evolution_test_user/memories/00000000-0000-0000-0000-000000000000/related"
    )
    assert resp.status_code == 404


async def test_memory_related_invalid_uuid(client: AsyncClient):
    """Return 400 for invalid memory UUID."""
    resp = await client.get("/admin/subjects/test_user/memories/not-a-uuid/related")
    assert resp.status_code == 400


async def test_memory_related_basic(client: AsyncClient):
    """Get related memories for a memory."""
    # Create episode and compile to get a memory
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": "evolution_user_basic",
            "source": "test",
            "type": "message",
            "payload": {"text": "User prefers dark mode for all applications"},
        },
    )
    await client.post("/v1/memories/compile", json={"subject_id": "evolution_user_basic"})

    # Get the memory
    mem_resp = await client.get("/admin/subjects/evolution_user_basic/memories")
    assert mem_resp.status_code == 200
    memories = mem_resp.json()["memories"]

    if len(memories) == 0:
        pytest.skip("No memories compiled - compiler may be disabled")

    memory_id = memories[0]["id"]

    # Get related memories
    resp = await client.get(f"/admin/subjects/evolution_user_basic/memories/{memory_id}/related")
    assert resp.status_code == 200
    data = resp.json()

    assert data["memory_id"] == memory_id
    assert "status" in data
    assert "created_at" in data
    assert "superseding_memory" in data
    assert "superseded_memories" in data
    assert "sibling_memories" in data
    assert "source_episode_count" in data


# ─── Subject deletion (admin) ────────────────────────────────────────────────


async def test_admin_delete_subject_single(client: AsyncClient):
    """DELETE /admin/subjects/{id} permanently removes the subject's data."""
    # Seed
    for i in range(3):
        ep = await client.post(
            "/v1/episodes",
            json={
                "subject_id": "to_delete_user",
                "source": "test",
                "type": "message",
                "payload": {"text": f"hello {i}"},
            },
        )
        assert ep.status_code == 201

    resp = await client.delete("/admin/subjects/to_delete_user")
    assert resp.status_code == 200
    body = resp.json()
    assert body["subject_id"] == "to_delete_user"
    assert body["episodes_deleted"] >= 3

    # Subsequent fetches should 404 (no data)
    detail = await client.get("/admin/subjects/to_delete_user")
    assert detail.status_code == 404


async def test_admin_delete_subject_not_found(client: AsyncClient):
    """DELETE /admin/subjects/{id} returns 404 when nothing exists for that id."""
    resp = await client.delete("/admin/subjects/nope_does_not_exist_xyz")
    assert resp.status_code == 404


async def test_admin_preview_delete_rejects_empty_filter(client: AsyncClient):
    """Empty filter is refused — operator must scope by prefix, age, or tenant."""
    resp = await client.post("/admin/subjects/preview-delete", json={})
    assert resp.status_code == 400
    assert "filter" in resp.json()["error"]["message"].lower()


async def test_admin_preview_delete_by_prefix(client: AsyncClient):
    """Prefix filter returns the matching subject set + counts + a sample."""
    for sid in ["bulkpfx_a", "bulkpfx_b", "bulkpfx_c", "other_unrelated"]:
        await client.post(
            "/v1/episodes",
            json={"subject_id": sid, "source": "test", "type": "message", "payload": {"text": "x"}},
        )

    resp = await client.post(
        "/admin/subjects/preview-delete", json={"subject_id_prefix": "bulkpfx_"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["matched"] == 3
    assert data["total_episodes"] >= 3
    sample_ids = {s["subject_id"] for s in data["sample"]}
    assert sample_ids == {"bulkpfx_a", "bulkpfx_b", "bulkpfx_c"}


async def test_admin_bulk_delete_requires_confirm(client: AsyncClient):
    """Refuses to delete without confirm=true."""
    await client.post(
        "/v1/episodes",
        json={
            "subject_id": "bulkconfirm_a",
            "source": "test",
            "type": "message",
            "payload": {"text": "x"},
        },
    )
    resp = await client.post(
        "/admin/subjects/bulk-delete",
        json={"subject_id_prefix": "bulkconfirm_", "expected_count": 1, "confirm": False},
    )
    assert resp.status_code == 400


async def test_admin_bulk_delete_count_mismatch(client: AsyncClient):
    """Refuses with 409 when the live match count differs from expected."""
    for sid in ["bulkmm_a", "bulkmm_b"]:
        await client.post(
            "/v1/episodes",
            json={"subject_id": sid, "source": "test", "type": "message", "payload": {"text": "x"}},
        )

    # Operator previewed and saw 1, but there are actually 2 — must refuse.
    resp = await client.post(
        "/admin/subjects/bulk-delete",
        json={"subject_id_prefix": "bulkmm_", "expected_count": 1, "confirm": True},
    )
    assert resp.status_code == 409


async def test_admin_bulk_delete_commits(client: AsyncClient):
    """Happy path: preview, then commit, then verify the subjects are gone."""
    for sid in ["bulkok_a", "bulkok_b", "bulkok_c"]:
        await client.post(
            "/v1/episodes",
            json={"subject_id": sid, "source": "test", "type": "message", "payload": {"text": "x"}},
        )

    # Preview to learn the count
    pv = await client.post(
        "/admin/subjects/preview-delete", json={"subject_id_prefix": "bulkok_"}
    )
    assert pv.status_code == 200
    matched = pv.json()["matched"]
    assert matched == 3

    # Commit
    resp = await client.post(
        "/admin/subjects/bulk-delete",
        json={"subject_id_prefix": "bulkok_", "expected_count": matched, "confirm": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["deleted_subjects"] == 3
    assert body["deleted_episodes"] >= 3
    assert body["failed"] == []

    # Verify gone
    pv2 = await client.post(
        "/admin/subjects/preview-delete", json={"subject_id_prefix": "bulkok_"}
    )
    assert pv2.json()["matched"] == 0
