"""Tests for memory templates — loader, render, and the API surface."""

from __future__ import annotations

import pytest

from server.services import templates

# The five templates bundled in server/templates/.
BUNDLED_IDS = {
    "account-onboarding",
    "customer-support-handoff",
    "incident-summary",
    "project-decision",
    "user-preference",
}


def _write(directory, name: str, body: str) -> None:
    (directory / name).write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Bundled registry
# ---------------------------------------------------------------------------

def test_bundled_templates_load():
    """Every bundled template loads, validates, and has the expected shape."""
    loaded = {t.id for t in templates.list_templates()}
    assert loaded == BUNDLED_IDS
    for template in templates.list_templates():
        assert template.version >= 1
        assert template.title
        assert template.episode_type
        assert template.fields
        assert template.content_template


def test_get_template_known_and_unknown():
    assert templates.get_template("user-preference") is not None
    assert templates.get_template("does-not-exist") is None


def test_bundled_templates_are_internally_consistent():
    """Every {placeholder} in a bundled template is a declared field."""
    import re

    for template in templates.list_templates():
        referenced = set(re.findall(r"{([a-zA-Z0-9_]+)}", template.content_template))
        assert referenced <= template.field_names(), template.id


# ---------------------------------------------------------------------------
# load_templates — validation
# ---------------------------------------------------------------------------

_VALID = """\
id: temp-test
version: 1
title: Temp test
episode_type: temp.test
fields:
  - name: a
    required: true
content_template: "value is {a}"
"""


def test_load_templates_from_directory(tmp_path):
    _write(tmp_path, "temp-test.yaml", _VALID)
    registry = templates.load_templates(tmp_path)
    assert set(registry) == {"temp-test"}


def test_load_rejects_undeclared_placeholder(tmp_path):
    _write(
        tmp_path,
        "bad.yaml",
        "id: bad-x\nversion: 1\ntitle: Bad\nepisode_type: bad.t\n"
        "fields:\n  - name: a\ncontent_template: \"{a} and {b}\"\n",
    )
    with pytest.raises(templates.TemplateError, match="undeclared field"):
        templates.load_templates(tmp_path)


def test_load_rejects_duplicate_field_names(tmp_path):
    _write(
        tmp_path,
        "bad.yaml",
        "id: bad-y\nversion: 1\ntitle: Bad\nepisode_type: bad.t\n"
        "fields:\n  - name: a\n  - name: a\ncontent_template: \"{a}\"\n",
    )
    with pytest.raises(templates.TemplateError, match="duplicate field name"):
        templates.load_templates(tmp_path)


def test_load_rejects_duplicate_template_ids(tmp_path):
    _write(tmp_path, "one.yaml", _VALID)
    _write(tmp_path, "two.yaml", _VALID)  # same id: temp-test
    with pytest.raises(templates.TemplateError, match="duplicate template id"):
        templates.load_templates(tmp_path)


def test_load_rejects_non_mapping_file(tmp_path):
    _write(tmp_path, "bad.yaml", "- just\n- a list\n")
    with pytest.raises(templates.TemplateError, match="must be a YAML mapping"):
        templates.load_templates(tmp_path)


def test_load_rejects_missing_required_key(tmp_path):
    # No content_template — pydantic validation fails.
    _write(
        tmp_path,
        "bad.yaml",
        "id: bad-z\nversion: 1\ntitle: Bad\nepisode_type: bad.t\nfields:\n  - name: a\n",
    )
    with pytest.raises(templates.TemplateError):
        templates.load_templates(tmp_path)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def test_render_happy_path():
    template = templates.get_template("customer-support-handoff")
    payload = templates.render(
        template,
        {"customer": "Globex", "issue": "duplicate charge", "next_owner": "Tier 2"},
    )
    assert payload["template_id"] == "customer-support-handoff"
    assert payload["template_version"] == 1
    assert payload["fields"] == {
        "customer": "Globex",
        "issue": "duplicate charge",
        "next_owner": "Tier 2",
    }
    assert "Globex" in payload["content"]
    assert "duplicate charge" in payload["content"]
    assert "Tier 2" in payload["content"]


def test_render_is_deterministic():
    template = templates.get_template("user-preference")
    values = {"subject": "notifications", "preference": "email only"}
    assert templates.render(template, values) == templates.render(template, values)


def test_render_omitted_optional_is_empty_in_content_and_absent_from_fields():
    template = templates.get_template("user-preference")
    payload = templates.render(
        template, {"subject": "tone", "preference": "concise"}
    )
    # rationale is optional and was not supplied.
    assert "rationale" not in payload["fields"]
    # The {rationale} placeholder rendered to an empty string — no leftover
    # placeholder token, and the content ends at the bare label.
    assert "{rationale}" not in payload["content"]
    assert payload["content"].endswith("Rationale:")


def test_render_rejects_missing_required():
    template = templates.get_template("customer-support-handoff")
    with pytest.raises(templates.TemplateError, match="missing required field"):
        templates.render(template, {"customer": "Globex"})  # issue is required


def test_render_rejects_unknown_field():
    template = templates.get_template("user-preference")
    with pytest.raises(templates.TemplateError, match="unknown field"):
        templates.render(
            template, {"subject": "x", "preference": "y", "bogus": "z"}
        )


def test_render_rejects_non_string_value():
    template = templates.get_template("user-preference")
    with pytest.raises(templates.TemplateError, match="must be a string"):
        templates.render(template, {"subject": "x", "preference": 42})


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

async def test_list_endpoint(client):
    resp = await client.get("/v1/memory-templates")
    assert resp.status_code == 200
    body = resp.json()
    assert {t["id"] for t in body["templates"]} == BUNDLED_IDS
    # The full field schema is exposed — templates are inspectable.
    handoff = next(t for t in body["templates"] if t["id"] == "customer-support-handoff")
    assert any(f["name"] == "issue" and f["required"] for f in handoff["fields"])


async def test_get_endpoint(client):
    resp = await client.get("/v1/memory-templates/incident-summary")
    assert resp.status_code == 200
    assert resp.json()["episode_type"] == "incident.summary"


async def test_get_endpoint_unknown_is_404(client):
    resp = await client.get("/v1/memory-templates/nope")
    assert resp.status_code == 404


async def test_apply_unknown_template_is_404(client):
    resp = await client.post(
        "/v1/memory-templates/nope/apply",
        json={"subject_id": "u1", "values": {}},
    )
    assert resp.status_code == 404


async def test_apply_missing_required_field_is_422(client):
    resp = await client.post(
        "/v1/memory-templates/customer-support-handoff/apply",
        json={"subject_id": "u1", "values": {"customer": "Globex"}},
    )
    assert resp.status_code == 422
    assert "issue" in resp.json()["error"]["message"]


async def test_apply_creates_episode_with_provenance(client):
    resp = await client.post(
        "/v1/memory-templates/customer-support-handoff/apply",
        json={
            "subject_id": "template-test-subject",
            "values": {"customer": "Globex", "issue": "duplicate charge"},
        },
    )
    assert resp.status_code == 201
    episode = resp.json()
    assert episode["source"] == "memory-template"
    assert episode["type"] == "support.handoff_note"
    assert episode["payload"]["template_id"] == "customer-support-handoff"
    assert episode["payload"]["template_version"] == 1
    assert episode["payload"]["fields"]["customer"] == "Globex"
    assert "Globex" in episode["payload"]["content"]
    # Provenance: metadata records the template id + version.
    assert episode["metadata"]["template"] == {
        "id": "customer-support-handoff",
        "version": 1,
    }
