"""Unit tests for ``aembed_texts`` response-ordering.

The OpenAI-compatible embeddings API does not guarantee that the response
``data`` list is in the same order as the input ``texts`` — each item carries
an ``index`` precisely so clients can re-order. ``aembed_texts`` must return
vectors in INPUT order so that callers (e.g. the background backfill, which
zips the result against the memory ids) write the right vector to the right
memory.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.services import llm as llm_mod


@pytest.mark.anyio
async def test_aembed_texts_reorders_by_index(monkeypatch):
    """A provider that returns the batch out of order must be re-sorted by
    ``index`` so result[i] corresponds to texts[i]."""
    monkeypatch.setattr(llm_mod.settings, "litellm_embedding_model", "text-embedding-3-small")

    # Response shuffled relative to input order (index 2, 0, 1); each vector
    # is distinct so a mis-mapping is detectable.
    shuffled = [
        {"index": 2, "embedding": [2.0]},
        {"index": 0, "embedding": [0.0]},
        {"index": 1, "embedding": [1.0]},
    ]
    fake_litellm = MagicMock()
    fake_litellm.aembedding = AsyncMock(return_value=SimpleNamespace(data=shuffled, usage=None))

    with patch.object(llm_mod, "_ensure_litellm", return_value=fake_litellm):
        result = await llm_mod.aembed_texts(["t0", "t1", "t2"])

    assert result == [[0.0], [1.0], [2.0]]


@pytest.mark.anyio
async def test_aembed_texts_preserves_order_when_index_absent(monkeypatch):
    """If a provider omits per-item ``index``, keep response order rather
    than raising."""
    monkeypatch.setattr(llm_mod.settings, "litellm_embedding_model", "text-embedding-3-small")

    no_index = [{"embedding": [0.0]}, {"embedding": [1.0]}]
    fake_litellm = MagicMock()
    fake_litellm.aembedding = AsyncMock(return_value=SimpleNamespace(data=no_index, usage=None))

    with patch.object(llm_mod, "_ensure_litellm", return_value=fake_litellm):
        result = await llm_mod.aembed_texts(["a", "b"])

    assert result == [[0.0], [1.0]]
