"""Unit tests for entity_extraction._normalize and _parse_entities.

Pure, hermetic tests — no DB, no LLM, no network. Both functions accept
plain strings and return deterministic results.
"""

from __future__ import annotations

import json

from server.services.entity_extraction import _normalize, _parse_entities


# ── _normalize ──────────────────────────────────────────────────────────

class TestNormalize:
    def test_collapses_whitespace(self):
        assert _normalize("  Acme   Corp \n") == "acme corp"

    def test_tabs_and_newlines(self):
        assert _normalize("foo\t\tbar\n\nbaz") == "foo bar baz"

    def test_lowercases(self):
        assert _normalize("ACME Corp") == "acme corp"

    def test_strips_leading_trailing(self):
        assert _normalize("  hello  ") == "hello"

    def test_empty_string(self):
        assert _normalize("") == ""

    def test_whitespace_only(self):
        assert _normalize("   \t\n  ") == ""


# ── _parse_entities ─────────────────────────────────────────────────────

def _entity(text: str, kind: str | None = "ORG") -> dict:
    e: dict = {"text": text}
    if kind is not None:
        e["kind"] = kind
    return e


def _wrap(entities: list[dict]) -> str:
    return json.dumps({"entities": entities})


class TestParseEntitiesValid:
    def test_valid_json(self):
        raw = _wrap([_entity("Acme", "ORG"), _entity("Alice", "PERSON")])
        result = _parse_entities(raw)
        assert len(result) == 2
        assert result[0].text == "Acme"
        assert result[0].kind == "ORG"
        assert result[1].text == "Alice"
        assert result[1].kind == "PERSON"

    def test_code_fence_wrapped(self):
        inner = _wrap([_entity("Acme", "ORG")])
        raw = f"```json\n{inner}\n```"
        result = _parse_entities(raw)
        assert len(result) == 1
        assert result[0].text == "Acme"

    def test_code_fence_without_json_label(self):
        inner = _wrap([_entity("Acme", "ORG")])
        raw = f"```\n{inner}\n```"
        result = _parse_entities(raw)
        assert len(result) == 1

    def test_kind_uppercased(self):
        raw = _wrap([_entity("Acme", "org")])
        result = _parse_entities(raw)
        assert result[0].kind == "ORG"

    def test_normalized_field_populated(self):
        raw = _wrap([_entity("  Acme  Corp  ", "ORG")])
        result = _parse_entities(raw)
        assert result[0].normalized == "acme corp"


class TestParseEntitiesDedup:
    def test_duplicate_entities_deduped(self):
        raw = _wrap([_entity("Acme", "ORG"), _entity("acme", "ORG")])
        result = _parse_entities(raw)
        assert len(result) == 1
        assert result[0].text == "Acme"

    def test_first_occurrence_kept(self):
        raw = _wrap([_entity("Acme Corp", "ORG"), _entity("Acme  Corp", "ORG")])
        result = _parse_entities(raw)
        assert len(result) == 1
        assert result[0].text == "Acme Corp"


class TestParseEntitiesEdgeCases:
    def test_malformed_json_returns_empty(self):
        assert _parse_entities("not json at all") == []

    def test_empty_string_returns_empty(self):
        assert _parse_entities("") == []

    def test_none_returns_empty(self):
        assert _parse_entities(None) == []

    def test_entity_missing_text_skipped(self):
        raw = _wrap([{"kind": "ORG"}])
        assert _parse_entities(raw) == []

    def test_entity_empty_text_skipped(self):
        raw = _wrap([{"text": "", "kind": "ORG"}])
        assert _parse_entities(raw) == []

    def test_entity_whitespace_text_skipped(self):
        raw = _wrap([{"text": "   ", "kind": "ORG"}])
        assert _parse_entities(raw) == []

    def test_entity_non_string_text_skipped(self):
        raw = _wrap([{"text": 123, "kind": "ORG"}])
        assert _parse_entities(raw) == []

    def test_entity_missing_kind_defaults_none(self):
        raw = _wrap([{"text": "Acme"}])
        result = _parse_entities(raw)
        assert len(result) == 1
        assert result[0].kind is None

    def test_entity_empty_kind_defaults_none(self):
        raw = _wrap([{"text": "Acme", "kind": ""}])
        result = _parse_entities(raw)
        assert result[0].kind is None

    def test_non_dict_items_skipped(self):
        raw = json.dumps({"entities": ["Acme", 42, None]})
        assert _parse_entities(raw) == []

    def test_entities_key_missing(self):
        assert _parse_entities(json.dumps({"other": []})) == []

    def test_entities_not_a_list(self):
        assert _parse_entities(json.dumps({"entities": "Acme"})) == []

    def test_root_not_a_dict(self):
        assert _parse_entities(json.dumps(["Acme"])) == []

    def test_partial_json_with_entities_block(self):
        raw = 'blah blah {"entities": [{"text": "Acme", "kind": "ORG"}]} trailing'
        result = _parse_entities(raw)
        assert len(result) == 1
        assert result[0].text == "Acme"

    def test_truncated_json_returns_empty(self):
        assert _parse_entities('{"entities": [{"text": "Acme"') == []
