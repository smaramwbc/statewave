"""Memory templates — declarative scaffolds for common episode patterns.

A *memory template* is a named, versioned, inspectable definition of a
recurring information pattern: a customer-support handoff, a project
decision, an incident summary, and so on. Applying a template validates
caller-supplied field values against the template's schema and ingests a
normal episode whose payload carries both the structured field values
and a deterministically-rendered content string.

Design constraints (deliberately un-magical):

- **Pure data.** Templates are YAML files. No code runs inside a
  template; nothing is inferred or guessed.
- **Deterministic.** Rendering is plain string substitution — the same
  values always produce the same bytes.
- **Provenance is explicit.** Every episode an `apply` produces records
  ``template_id`` / ``template_version`` in its payload, and
  ``metadata.template`` carries the same pair. You can always see
  exactly which template, at which version, produced a memory.
- **Composes, doesn't replace.** A template produces an ordinary
  episode. The compiler is untouched; templated episodes flow through
  the existing ingest → compile → context pipeline like any other.

Bundled templates live in ``server/templates/*.yaml``. Adding one is
dropping a new YAML file there — see ``docs/memory-templates.md``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

# Directory holding the bundled template YAML files.
TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

# Placeholder tokens in a content_template: {field_name}.
_PLACEHOLDER_RE = re.compile(r"{([a-zA-Z0-9_]+)}")


class TemplateError(ValueError):
    """Raised when a template file is malformed, or when caller-supplied
    values do not satisfy a template's field schema."""


class TemplateField(BaseModel):
    """One field a template expects.

    ``type`` is advisory metadata for callers and UIs — ``string`` hints
    a short single-line value, ``text`` a longer multi-line one. Both are
    carried and rendered as plain strings; the distinction is a
    documented extension point, not behaviour.
    """

    name: str = Field(..., pattern=r"^[a-zA-Z0-9_]+$")
    type: Literal["string", "text"] = "string"
    required: bool = False
    description: str = ""


class MemoryTemplate(BaseModel):
    """A declarative pattern for a common kind of episode."""

    id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9-]*$")
    version: int = Field(..., ge=1)
    title: str = Field(..., min_length=1)
    description: str = ""
    # The episode `type` stamped on every episode this template produces.
    episode_type: str = Field(..., min_length=1, max_length=128)
    fields: list[TemplateField] = Field(..., min_length=1)
    # Content scaffold; {field_name} placeholders are substituted at apply.
    content_template: str = Field(..., min_length=1)

    def field_names(self) -> set[str]:
        return {f.name for f in self.fields}


class MemoryTemplateList(BaseModel):
    """Wrapper for the list endpoint."""

    templates: list[MemoryTemplate]


def _validate_template(template: MemoryTemplate, source: str) -> None:
    """Check a loaded template is internally consistent.

    Field names must be unique, and every ``{placeholder}`` in
    ``content_template`` must correspond to a declared field — so a
    template author's typo fails at startup, not at apply time.
    """
    names = [f.name for f in template.fields]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise TemplateError(
            f"template {source}: duplicate field name(s): {', '.join(sorted(duplicates))}"
        )
    referenced = set(_PLACEHOLDER_RE.findall(template.content_template))
    undeclared = referenced - template.field_names()
    if undeclared:
        raise TemplateError(
            f"template {source}: content_template references undeclared "
            f"field(s): {', '.join(sorted(undeclared))}"
        )


def load_templates(directory: Path = TEMPLATES_DIR) -> dict[str, MemoryTemplate]:
    """Load and validate every ``*.yaml`` template in ``directory``.

    Raises ``TemplateError`` on a malformed file, an internally
    inconsistent template, or a duplicate template id — so a bad bundled
    template fails the server at startup rather than silently.
    """
    registry: dict[str, MemoryTemplate] = {}
    for path in sorted(directory.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise TemplateError(f"template {path.name}: invalid YAML: {exc}") from exc
        if not isinstance(raw, dict):
            raise TemplateError(f"template {path.name}: file must be a YAML mapping")
        try:
            template = MemoryTemplate.model_validate(raw)
        except ValidationError as exc:
            raise TemplateError(f"template {path.name}: {exc}") from exc
        _validate_template(template, path.name)
        if template.id in registry:
            raise TemplateError(
                f"template {path.name}: duplicate template id {template.id!r}"
            )
        registry[template.id] = template
    return registry


# Loaded once at import time — a malformed bundled template fails fast.
_REGISTRY: dict[str, MemoryTemplate] = load_templates()


def list_templates() -> list[MemoryTemplate]:
    """Every registered template, ordered by id."""
    return [_REGISTRY[tid] for tid in sorted(_REGISTRY)]


def get_template(template_id: str) -> MemoryTemplate | None:
    """Look up a single template by id, or None if there is no such template."""
    return _REGISTRY.get(template_id)


def render(template: MemoryTemplate, values: dict[str, Any]) -> dict[str, Any]:
    """Validate ``values`` against ``template`` and build the episode payload.

    Validation is strict: an unknown field, a missing required field, or
    a non-string value all raise ``TemplateError``. Rendering is
    deterministic — every ``{field}`` placeholder is replaced with the
    supplied value, or with an empty string when an optional field is
    omitted.

    The returned payload records the template id/version, the
    caller-supplied fields verbatim, and the rendered ``content``.
    """
    declared = template.field_names()
    unknown = set(values) - declared
    if unknown:
        raise TemplateError(
            f"template {template.id}: unknown field(s): {', '.join(sorted(unknown))}"
        )

    supplied: dict[str, str] = {}
    for name, value in values.items():
        if not isinstance(value, str):
            raise TemplateError(
                f"template {template.id}: field {name!r} must be a string, "
                f"got {type(value).__name__}"
            )
        supplied[name] = value

    missing = sorted(
        f.name for f in template.fields if f.required and f.name not in supplied
    )
    if missing:
        raise TemplateError(
            f"template {template.id}: missing required field(s): {', '.join(missing)}"
        )

    content = template.content_template
    for field in template.fields:
        content = content.replace("{" + field.name + "}", supplied.get(field.name, ""))

    return {
        "template_id": template.id,
        "template_version": template.version,
        "fields": supplied,
        "content": content.strip(),
    }
