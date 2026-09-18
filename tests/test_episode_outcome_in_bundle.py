"""An attempt's outcome reaches the bundle line (issue #415).

A harness usually knows the moment an attempt failed: a non-zero exit, a
rejected call, a test that turned red. Before this it had nowhere to put
that where the answering model would see it. `source` and `type` reach
the bundle; `metadata` never did.

The envelope is required rather than a bare string, and that is the
load-bearing decision: `outcome` is an ordinary word, so callers already
keep their own bookkeeping under it. A bare
`metadata["outcome"] = "escalated to tier 2"` must never start reaching
the model on upgrade day, because episodes are immutable (no way to take
it back) and the receipts for those subjects would change hash with no
action by their owner.
"""

from __future__ import annotations

import pytest

from server.services.context import (
    OUTCOME_METADATA_KEY,
    _episode_outcome_suffix,
    _short_episode_text,
)

_PAYLOAD = {"text": "ran the deploy pipeline against staging"}


def _line(metadata=None) -> str:
    return _short_episode_text(_PAYLOAD, "agent", "tool_call", metadata)


def test_outcome_reaches_the_line():
    line = _line({OUTCOME_METADATA_KEY: {"status": "failed", "reason": "migration 0031 timed out"}})
    assert line.endswith(" [failed: migration 0031 timed out]")
    assert "ran the deploy pipeline" in line


def test_status_alone_renders_without_a_reason():
    assert _line({OUTCOME_METADATA_KEY: {"status": "succeeded"}}).endswith(" [succeeded]")


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"outcome": "escalated to tier 2"},          # pre-existing bare string
        {"outcome": ["failed"]},                      # wrong container
        {"unrelated": {"status": "failed"}},          # different key entirely
    ],
)
def test_no_envelope_renders_exactly_as_before(metadata):
    """The no-outcome path must be byte-for-byte unchanged. This is what
    keeps the change off the benchmark's path: fixture episodes carry no
    envelope, so their bundles are identical."""
    assert _line(metadata) == "[agent/tool_call] ran the deploy pipeline against staging"


def test_a_pre_existing_bare_outcome_string_is_never_rendered():
    # The retro-activation guard, stated on its own so it cannot be
    # weakened by accident: someone else's bookkeeping stays invisible.
    assert _episode_outcome_suffix({"outcome": "escalated to tier 2"}) == ""


@pytest.mark.parametrize("status", ["fail", "FAILED", "partial", "", None, 3])
def test_unknown_status_is_rejected(status):
    assert _episode_outcome_suffix({OUTCOME_METADATA_KEY: {"status": status}}) == ""


def test_reason_newlines_cannot_forge_a_second_bullet():
    # The bundle renders episodes as "- <line>"; an embedded newline in a
    # caller-supplied reason would otherwise inject a list item.
    suffix = _episode_outcome_suffix(
        {OUTCOME_METADATA_KEY: {"status": "failed", "reason": "boom\n- injected bullet"}}
    )
    assert "\n" not in suffix
    assert suffix == " [failed: boom - injected bullet]"


def test_reason_is_capped():
    suffix = _episode_outcome_suffix(
        {OUTCOME_METADATA_KEY: {"status": "failed", "reason": "x" * 500}}
    )
    assert len(suffix) < 100


def test_label_survives_a_long_body():
    # The suffix is appended after the body slice, so a long episode can
    # never truncate the outcome away.
    line = _short_episode_text(
        {"text": "y" * 4000}, "agent", "tool_call", {OUTCOME_METADATA_KEY: {"status": "failed"}}
    )
    assert line.endswith(" [failed]")


def test_empty_payload_still_carries_the_outcome():
    line = _short_episode_text({}, "agent", "tool_call", {OUTCOME_METADATA_KEY: {"status": "failed"}})
    assert line == "[agent/tool_call] (no text content) [failed]"


def test_malformed_metadata_never_raises():
    for junk in ["a string", 42, [1, 2], {"outcome": None}, {"outcome": {"status": {}}}]:
        assert _episode_outcome_suffix(junk) == ""
