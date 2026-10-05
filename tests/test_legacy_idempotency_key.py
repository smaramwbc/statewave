"""Legacy metadata keys receive the same bounded-string input validation."""

import pytest
from pydantic import ValidationError

from server.schemas.requests import BatchCreateEpisodesRequest, CreateEpisodeRequest


def episode(key):
    return {
        "subject_id": "test-user", "source": "test", "type": "message",
        "payload": {}, "metadata": {"idempotency_key": key},
    }


@pytest.mark.parametrize("key", [123, True, {}, [], "x" * 513])
def test_schema_rejects_invalid_legacy_key(key):
    with pytest.raises(ValidationError, match=r"metadata\.idempotency_key"):
        CreateEpisodeRequest.model_validate(episode(key))


@pytest.mark.parametrize("key", [None, "", "legacy", "x" * 512])
def test_schema_resolves_valid_legacy_key_without_mutating_metadata(key):
    request = CreateEpisodeRequest.model_validate(episode(key))
    assert request.idempotency_key == key
    assert request.metadata == {"idempotency_key": key}


def test_explicit_key_wins_over_unused_metadata():
    body = episode({"unrelated": "data"})
    body["idempotency_key"] = "explicit"
    assert CreateEpisodeRequest.model_validate(body).idempotency_key == "explicit"


def test_empty_explicit_key_retains_legacy_fallback():
    body = episode("legacy")
    body["idempotency_key"] = ""
    assert CreateEpisodeRequest.model_validate(body).idempotency_key == "legacy"


def test_batch_uses_the_same_validation():
    with pytest.raises(ValidationError, match=r"metadata\.idempotency_key"):
        BatchCreateEpisodesRequest.model_validate({"episodes": [episode("valid"), episode(123)]})


@pytest.mark.parametrize("path", ["/v1/episodes", "/v1/episodes/batch"])
@pytest.mark.parametrize("key", [123, {}, "x" * 513])
async def test_invalid_http_input_is_rejected_before_database_work(client, path, key):
    body = episode(key)
    response = await client.post(path, json={"episodes": [body]} if path.endswith("batch") else body)
    assert response.status_code == 422
    assert "metadata.idempotency_key" in str(response.json())
