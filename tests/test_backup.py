"""Tests for subject backup/restore service.

Unit tests use mocked data; verifies export format, checksum, import logic.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone

import pytest

from server.services.backup import _FORMAT_VERSION


def _make_export_doc(
    subject_id: str = "user-1",
    tenant_id: str | None = None,
    episodes: list | None = None,
    memories: list | None = None,
) -> dict:
    """Build a valid export document for testing."""
    eps = episodes or [
        {
            "id": str(uuid.uuid4()),
            "subject_id": subject_id,
            "tenant_id": tenant_id,
            "source": "chat",
            "type": "message",
            "payload": {"role": "user", "content": "Hello"},
            "metadata": {},
            "provenance": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_compiled_at": None,
        }
    ]
    mems = memories or [
        {
            "id": str(uuid.uuid4()),
            "subject_id": subject_id,
            "tenant_id": tenant_id,
            "kind": "profile_fact",
            "content": "User prefers dark mode",
            "summary": "dark mode preference",
            "confidence": 0.9,
            "valid_from": datetime.now(timezone.utc).isoformat(),
            "valid_to": None,
            "source_episode_ids": [eps[0]["id"]],
            "metadata": {},
            "status": "active",
            "embedding": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ]
    content_str = json.dumps({"episodes": eps, "memories": mems}, sort_keys=True)
    checksum = hashlib.sha256(content_str.encode()).hexdigest()

    return {
        "format_version": _FORMAT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "subject_id": subject_id,
        "tenant_id": tenant_id,
        "counts": {"episodes": len(eps), "memories": len(mems)},
        "episodes": eps,
        "memories": mems,
        "checksum": checksum,
    }


class TestExportDocFormat:
    """Tests for export document structure and validation."""

    def test_valid_doc_has_required_fields(self):
        doc = _make_export_doc()
        assert doc["format_version"] == _FORMAT_VERSION
        assert "checksum" in doc
        assert "exported_at" in doc
        assert doc["counts"]["episodes"] == 1
        assert doc["counts"]["memories"] == 1

    def test_checksum_validates(self):
        doc = _make_export_doc()
        content_str = json.dumps(
            {"episodes": doc["episodes"], "memories": doc["memories"]}, sort_keys=True
        )
        expected = hashlib.sha256(content_str.encode()).hexdigest()
        assert doc["checksum"] == expected

    def test_tampered_checksum_detectable(self):
        doc = _make_export_doc()
        doc["episodes"][0]["payload"]["content"] = "TAMPERED"
        content_str = json.dumps(
            {"episodes": doc["episodes"], "memories": doc["memories"]}, sort_keys=True
        )
        new_checksum = hashlib.sha256(content_str.encode()).hexdigest()
        assert doc["checksum"] != new_checksum


class TestImportValidation:
    """Tests for import safety checks (no DB needed)."""

    @pytest.mark.anyio
    async def test_rejects_wrong_format_version(self):
        from server.services.backup import import_subject

        doc = _make_export_doc()
        doc["format_version"] = "99.0"
        with pytest.raises(ValueError, match="Unsupported format version"):
            await import_subject(doc)

    @pytest.mark.anyio
    async def test_rejects_corrupted_checksum(self):
        from server.services.backup import import_subject

        doc = _make_export_doc()
        doc["checksum"] = "bad_checksum_value"
        with pytest.raises(ValueError, match="Checksum mismatch"):
            await import_subject(doc)

    @pytest.mark.anyio
    async def test_rejects_missing_subject_id(self):
        from server.services.backup import import_subject

        doc = _make_export_doc()
        doc["subject_id"] = None
        # Recalculate checksum with modified subject_id in episodes
        for ep in doc["episodes"]:
            ep["subject_id"] = None
        for mem in doc["memories"]:
            mem["subject_id"] = None
        content_str = json.dumps(
            {"episodes": doc["episodes"], "memories": doc["memories"]}, sort_keys=True
        )
        doc["checksum"] = hashlib.sha256(content_str.encode()).hexdigest()

        with pytest.raises(ValueError, match="No subject_id"):
            await import_subject(doc)


class TestExportPreservesData:
    """Verify exported data preserves all fields."""

    def test_episode_fields_complete(self):
        doc = _make_export_doc()
        ep = doc["episodes"][0]
        required_fields = [
            "id",
            "subject_id",
            "source",
            "type",
            "payload",
            "metadata",
            "provenance",
            "created_at",
        ]
        for field in required_fields:
            assert field in ep, f"Missing field: {field}"

    def test_memory_fields_complete(self):
        doc = _make_export_doc()
        mem = doc["memories"][0]
        required_fields = [
            "id",
            "subject_id",
            "kind",
            "content",
            "summary",
            "confidence",
            "valid_from",
            "source_episode_ids",
            "metadata",
            "status",
            "created_at",
            "updated_at",
        ]
        for field in required_fields:
            assert field in mem, f"Missing field: {field}"

    def test_provenance_preserved_in_export(self):
        doc = _make_export_doc()
        mem = doc["memories"][0]
        assert mem["source_episode_ids"] == [doc["episodes"][0]["id"]]
