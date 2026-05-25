"""Unit tests for the auto-labeling detector registry + pipeline (#158).

Pure tests: no DB, no app. Exercises each detector in isolation so
regressions are pinpointable, plus the pipeline-level dedup / merge
contract.
"""

from __future__ import annotations

import pytest

from server.services.auto_labeling import (
    DETECTORS,
    apply_suggestions,
    label_suggestions,
)
from server.services.auto_labeling.detectors import label_catalogue


# ---------------------------------------------------------------------------
# Registry contract — guard against accidental detector deletion
# ---------------------------------------------------------------------------


def test_detector_registry_has_v0_9_first_wave():
    """The v0.9 first wave is `pii.email`, `pii.phone`, `financial.card`,
    `secret.token`. A regression that silently drops one of these is a
    governance bug — assert directly on the labels."""
    labels = {d.label for d in DETECTORS}
    assert labels == {
        "pii.email",
        "pii.phone",
        "financial.card",
        "secret.token",
    }


def test_detector_label_schema():
    """All detector labels must follow `<category>.<specific>` so the
    admin UI groups them sanely."""
    for d in DETECTORS:
        assert "." in d.label, f"{d.label} violates `<category>.<specific>`"
        category, specific = d.label.split(".", 1)
        assert category and specific


def test_label_catalogue_matches_registry():
    """The public catalogue helper must mirror DETECTORS one-to-one."""
    cat = label_catalogue()
    assert len(cat) == len(DETECTORS)
    assert {entry["label"] for entry in cat} == {d.label for d in DETECTORS}
    # Each entry has a non-empty description.
    for entry in cat:
        assert entry["description"]


# ---------------------------------------------------------------------------
# pii.email
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Contact me at alice@example.com please.",
        "alice.brown+tag@sub.example.co.uk",
        "support@statewave.dev wrote in.",
    ],
)
def test_email_positive(text):
    assert "pii.email" in label_suggestions(text)


@pytest.mark.parametrize(
    "text",
    [
        "no email here, only @handle and @another",
        "the price is @5usd",
        "user@",  # no domain
        "",
    ],
)
def test_email_negative(text):
    assert "pii.email" not in label_suggestions(text)


# ---------------------------------------------------------------------------
# pii.phone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Call me at +1 415 555 0199.",
        "+442071234567",
        "Cell: (415) 555-0199",
        "415-555-0199",
        "415.555.0199",
    ],
)
def test_phone_positive(text):
    assert "pii.phone" in label_suggestions(text)


@pytest.mark.parametrize(
    "text",
    [
        # Raw 10-digit run with no grouping / no leading +
        "transaction id 4155550199",
        "no phone at all",
        "version 1.2.3",
    ],
)
def test_phone_negative(text):
    assert "pii.phone" not in label_suggestions(text)


# ---------------------------------------------------------------------------
# financial.card — Luhn-validated
# ---------------------------------------------------------------------------


def test_card_positive_valid_luhn():
    """4111-1111-1111-1111 is the canonical test card (passes Luhn)."""
    assert "financial.card" in label_suggestions("card 4111-1111-1111-1111 charge")


def test_card_positive_with_spaces():
    assert "financial.card" in label_suggestions("4111 1111 1111 1111")


def test_card_positive_amex_15_digits():
    """378282246310005 — AmEx test number, passes Luhn at 15 digits."""
    assert "financial.card" in label_suggestions("amex 378282246310005")


def test_card_negative_fails_luhn():
    """A 16-digit run that does NOT pass Luhn must not trigger. The
    length-only path would false-positive every order-id; the Luhn
    gate is what keeps the detector usable."""
    # 4111-1111-1111-1112 fails the checksum (off by one).
    assert "financial.card" not in label_suggestions("id 4111-1111-1111-1112")


def test_card_negative_short_run():
    """12 digits is below the 13-digit floor — too short to be a card."""
    assert "financial.card" not in label_suggestions("ref 123456789012")


# ---------------------------------------------------------------------------
# secret.token
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # AWS access key id
        "key=AKIAIOSFODNN7EXAMPLE",
        # GitHub PAT
        "token: ghp_" + "a" * 40,
        # OpenAI key
        "OPENAI_API_KEY=sk-" + "a" * 30,
        # Google API key — must be exactly AIza + 35 chars
        "GOOGLE_API_KEY=AIza" + "x" * 35,
        # Slack OAuth
        "xoxb-1234567890-abcdefg",
        # JWT (three base64-ish segments)
        "Authorization: eyJabcdefgh.abcdefghijklmnop.abcdefghijklmnop",
    ],
)
def test_token_positive(text):
    assert "secret.token" in label_suggestions(text)


@pytest.mark.parametrize(
    "text",
    [
        "no token here",
        # A bare UUID is NOT a token in our v0.9 precision-first policy.
        "id 550e8400-e29b-41d4-a716-446655440000",
        # A git SHA is NOT a token.
        "commit deadbeefcafebabe1234567890abcdef12345678",
    ],
)
def test_token_negative(text):
    assert "secret.token" not in label_suggestions(text)


# ---------------------------------------------------------------------------
# label_suggestions — composition + ordering
# ---------------------------------------------------------------------------


def test_label_suggestions_dedupes_and_sorts():
    text = "Email alice@example.com and call +1 415 555 0199."
    out = label_suggestions(text)
    assert out == sorted(out)  # sorted
    assert len(out) == len(set(out))  # de-duplicated
    assert "pii.email" in out
    assert "pii.phone" in out


def test_label_suggestions_empty_text():
    assert label_suggestions("") == []
    assert label_suggestions(None) == []  # type: ignore[arg-type]


def test_label_suggestions_multiple_categories():
    """A single memory can carry multiple categories simultaneously."""
    text = (
        "alice@example.com paid with 4111-1111-1111-1111. "
        "Bearer eyJabcdefgh.abcdefghijklmnop.abcdefghijklmnop"
    )
    out = set(label_suggestions(text))
    assert {"pii.email", "financial.card", "secret.token"}.issubset(out)


# ---------------------------------------------------------------------------
# apply_suggestions — pipeline-level idempotency + merge
# ---------------------------------------------------------------------------


class _StubMemory:
    """Minimal stand-in for MemoryRow — keeps tests free of SQLAlchemy."""

    def __init__(self, content: str, suggested_labels: list[str] | None = None):
        self.id = None
        self.content = content
        self.suggested_labels = suggested_labels or []


def test_apply_suggestions_stamps_labels():
    mem = _StubMemory("Reach me at alice@example.com")
    apply_suggestions([mem])
    assert "pii.email" in mem.suggested_labels


def test_apply_suggestions_merges_with_existing():
    """If suggested_labels already carries values (e.g. from the LLM
    compiler that ran first), the heuristic apply path must MERGE
    rather than overwrite."""
    mem = _StubMemory("alice@example.com", suggested_labels=["pii.custom"])
    apply_suggestions([mem])
    assert "pii.email" in mem.suggested_labels
    assert "pii.custom" in mem.suggested_labels
    assert mem.suggested_labels == sorted(mem.suggested_labels)


def test_apply_suggestions_idempotent():
    """Running the pipeline twice must not duplicate labels."""
    mem = _StubMemory("alice@example.com")
    apply_suggestions([mem])
    first = list(mem.suggested_labels)
    apply_suggestions([mem])
    assert mem.suggested_labels == first


def test_apply_suggestions_skips_empty_content():
    mem = _StubMemory("")
    apply_suggestions([mem])
    assert mem.suggested_labels == []


def test_apply_suggestions_detector_failure_isolated(monkeypatch):
    """A detector raising at runtime must NOT prevent the rest of the
    pipeline from labelling the memory. We monkey-patch one detector
    to raise and assert the others still fire."""
    from server.services.auto_labeling import detectors

    # Find pii.email and force it to raise. The text below also
    # contains a phone number, so we expect pii.phone to still land.
    original = detectors._detect_email  # type: ignore[attr-defined]

    def _raises(_text):
        raise RuntimeError("boom")

    monkeypatch.setattr(detectors, "_detect_email", _raises)
    # The Detector dataclass is frozen so we rebuild the tuple in-place.
    new_detectors = tuple(
        detectors.Detector(label=d.label, description=d.description, detect=_raises)
        if d.label == "pii.email"
        else d
        for d in detectors.DETECTORS
    )
    monkeypatch.setattr(detectors, "DETECTORS", new_detectors)
    # Re-import the pipeline's view of DETECTORS via attribute lookup.
    from server.services import auto_labeling

    monkeypatch.setattr(auto_labeling, "DETECTORS", new_detectors)

    text = "alice@example.com or +1 415 555 0199"
    mem = _StubMemory(text)
    apply_suggestions([mem])

    # pii.email suppressed, pii.phone still landed.
    assert "pii.email" not in mem.suggested_labels
    assert "pii.phone" in mem.suggested_labels

    # Restore for sanity.
    monkeypatch.setattr(detectors, "_detect_email", original)
