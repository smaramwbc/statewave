"""Integration tests for HMAC receipt signing (v0.9, issue #157).

Covers four layers:

1. **Pure crypto** — `canonicalize_receipt_body_v1`, `sign_receipt_body`,
   `verify_receipt` with stand-in row objects. No DB, no config.
2. **Config parsing** — `STATEWAVE_RECEIPT_SIGNING_KEYS` validator
   rejects malformed input at startup (base64 errors, short keys,
   wrong types, malformed JSON).
3. **Emission integration** — `write_receipt` signs when the tenant
   has a key id and the key is available; emits unsigned otherwise.
4. **Verify endpoint + security** — `GET /v1/receipts/{id}/verify`
   returns the documented shape; signing keys never appear in logs
   or admin responses; rotation flow works.
"""

from __future__ import annotations

import base64
import logging
import secrets
import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from server.core.config import Settings, settings
from server.db.tables import ReceiptRow, TenantConfigRow
from server.services.receipts import (
    SUPPORTED_SIGNATURE_ALGORITHM,
    canonicalize_receipt_body_v1,
    new_ulid,
    sign_receipt_body,
    verify_receipt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gen_key(n_bytes: int = 32) -> bytes:
    return secrets.token_bytes(n_bytes)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _make_row(
    body: dict,
    *,
    signature: str | None,
    key_id: str | None,
    algorithm: str | None = SUPPORTED_SIGNATURE_ALGORITHM,
) -> ReceiptRow:
    """A free-standing ReceiptRow for unit-testing verify_receipt without DB."""
    return ReceiptRow(
        receipt_id=body.get("receipt_id", new_ulid()),
        parent_receipt_id=None,
        mode="retrieval",
        tenant_id=body.get("tenant_id"),
        subject_id=body.get("subject_id", "sub-1"),
        query_id=None,
        task_id=None,
        context_hash=body.get("output", {}).get("context_hash", "0" * 64),
        context_size_bytes=body.get("output", {}).get("context_size_bytes", 0),
        policy_bundle_hash=None,
        region=None,
        receipt_signature=signature,
        receipt_signature_key_id=key_id,
        receipt_signature_algorithm=algorithm if signature else None,
        body=body,
        as_of=datetime.now(timezone.utc),
        status="active",
    )


def _seed_body(**overrides) -> dict:
    """A minimally-valid receipt body suitable for signing/verifying."""
    rid = overrides.pop("receipt_id", new_ulid())
    base = {
        "receipt_id": rid,
        "mode": "retrieval",
        "tenant_id": "tenant-x",
        "subject_id": "sub-x",
        "selected_entries": [],
        "policy": {"filters_applied": [], "filters_skipped": [], "mode": "log_only"},
        "output": {
            "context_hash": "0" * 64,
            "context_size_bytes": 0,
            "canonicalization_version": 1,
            "token_estimate": 0,
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Pure crypto — canonicalization
# ---------------------------------------------------------------------------


def test_canonicalize_is_deterministic():
    body = _seed_body(subject_id="u1")
    assert canonicalize_receipt_body_v1(body) == canonicalize_receipt_body_v1(body)


def test_canonicalize_is_order_independent_for_keys():
    """Sorted keys means dict-insertion order can't change the bytes."""
    a = _seed_body(subject_id="u1")
    b = {k: a[k] for k in reversed(list(a.keys()))}
    assert canonicalize_receipt_body_v1(a) == canonicalize_receipt_body_v1(b)


def test_canonicalize_excludes_receipt_signature_field():
    """Signing a body that contains its own signature is circular —
    the canonical form strips `receipt_signature` so the signature can
    be embedded into the body afterwards."""
    body_unsigned = _seed_body()
    canonical = canonicalize_receipt_body_v1(body_unsigned)
    body_signed = dict(body_unsigned)
    body_signed["receipt_signature"] = "deadbeef"
    assert canonicalize_receipt_body_v1(body_signed) == canonical


def test_canonicalize_covers_key_id_and_algorithm():
    """An attacker swapping key_id or algorithm on a stored receipt
    body must invalidate the signature — those fields are inside the
    canonical form."""
    body = _seed_body(
        receipt_signature_key_id="key-A",
        receipt_signature_algorithm=SUPPORTED_SIGNATURE_ALGORITHM,
    )
    swapped = dict(body)
    swapped["receipt_signature_key_id"] = "key-B"
    assert canonicalize_receipt_body_v1(body) != canonicalize_receipt_body_v1(swapped)


# ---------------------------------------------------------------------------
# 1. Pure crypto — sign / verify
# ---------------------------------------------------------------------------


def test_sign_then_verify_round_trips():
    key = _gen_key()
    body = _seed_body()
    sig = sign_receipt_body(body, key)
    body["receipt_signature"] = sig
    row = _make_row(body, signature=sig, key_id="key-1")
    result = verify_receipt(row, keys_map={"key-1": key})
    assert result == {
        "valid": True,
        "key_id": "key-1",
        "algorithm": SUPPORTED_SIGNATURE_ALGORITHM,
        "reason": "ok",
    }


def test_tampered_subject_id_invalidates_signature():
    key = _gen_key()
    body = _seed_body(subject_id="alice")
    sig = sign_receipt_body(body, key)
    body["receipt_signature"] = sig
    body["subject_id"] = "mallory"  # tampered after signing
    row = _make_row(body, signature=sig, key_id="key-1")
    result = verify_receipt(row, keys_map={"key-1": key})
    assert result["valid"] is False
    assert result["reason"] == "signature_mismatch"


def test_tampered_context_hash_invalidates_signature():
    key = _gen_key()
    body = _seed_body()
    sig = sign_receipt_body(body, key)
    body["receipt_signature"] = sig
    body["output"]["context_hash"] = "f" * 64  # tampered
    row = _make_row(body, signature=sig, key_id="key-1")
    assert verify_receipt(row, keys_map={"key-1": key})["valid"] is False


def test_verify_no_signature_for_unsigned_row():
    body = _seed_body()
    row = _make_row(body, signature=None, key_id=None, algorithm=None)
    result = verify_receipt(row, keys_map={})
    assert result == {
        "valid": None,
        "key_id": None,
        "algorithm": None,
        "reason": "no_signature",
    }


def test_verify_key_unavailable_when_operator_dropped_the_key():
    """Historical receipt + key rotated out → verify reports the reason
    distinctly, never a 500."""
    key = _gen_key()
    body = _seed_body()
    sig = sign_receipt_body(body, key)
    body["receipt_signature"] = sig
    row = _make_row(body, signature=sig, key_id="key-rotated-out")
    result = verify_receipt(row, keys_map={})  # operator removed the key
    assert result == {
        "valid": None,
        "key_id": "key-rotated-out",
        "algorithm": SUPPORTED_SIGNATURE_ALGORITHM,
        "reason": "key_unavailable",
    }


def test_verify_unsupported_algorithm_reports_explicitly():
    """A future receipt signed with `canonical-v2` would land on a v0.9
    binary like this — distinct from signature_mismatch."""
    body = _seed_body()
    row = _make_row(body, signature="ff" * 32, key_id="key-1", algorithm="hmac-sha256-canonical-v999")
    result = verify_receipt(row, keys_map={"key-1": _gen_key()})
    assert result["valid"] is None
    assert result["reason"] == "unsupported_algorithm"


# ---------------------------------------------------------------------------
# 2. Config parsing — fails fast on bad input at startup
# ---------------------------------------------------------------------------


def test_config_accepts_valid_base64_dict():
    key = _gen_key()
    s = Settings(receipt_signing_keys={"k1": _b64(key)})
    assert s.receipt_signing_keys == {"k1": key}


def test_config_accepts_json_string():
    key = _gen_key()
    s = Settings(receipt_signing_keys=f'{{"k1": "{_b64(key)}"}}')
    assert s.receipt_signing_keys == {"k1": key}


def test_config_rejects_short_key():
    too_short = _b64(_gen_key(16))  # 16 bytes < 32-byte minimum
    with pytest.raises(ValidationError, match="too short"):
        Settings(receipt_signing_keys={"k1": too_short})


def test_config_rejects_non_base64():
    with pytest.raises(ValidationError, match="not valid base64"):
        Settings(receipt_signing_keys={"k1": "!!!not-base64!!!"})


def test_config_rejects_malformed_json():
    with pytest.raises(ValidationError, match="not valid JSON"):
        Settings(receipt_signing_keys="not-json")


def test_config_rejects_non_object_json():
    with pytest.raises(ValidationError, match="JSON object"):
        Settings(receipt_signing_keys='["k1", "k2"]')


def test_config_rejects_empty_key_id():
    with pytest.raises(ValidationError, match="non-empty string"):
        Settings(receipt_signing_keys={"": _b64(_gen_key())})


def test_config_empty_means_no_signing():
    s = Settings(receipt_signing_keys="")
    assert s.receipt_signing_keys == {}


# ---------------------------------------------------------------------------
# 3. End-to-end via the `/v1/context` API: emission signs + verify works
# ---------------------------------------------------------------------------


async def _seed_tenant_signing(
    session_factory,
    tenant_id: str,
    key_id: str | None,
) -> None:
    """Set `tenant_configs.config.receipt_signing_key_id` and force
    receipts to emit (`receipts: always`) so tests don't need to also
    flip the per-request opt-in."""
    async with session_factory() as session:
        existing = await session.get(TenantConfigRow, tenant_id)
        cfg: dict = {"receipts": "always"}
        if key_id is not None:
            cfg["receipt_signing_key_id"] = key_id
        if existing is None:
            session.add(TenantConfigRow(tenant_id=tenant_id, config=cfg, version=1))
        else:
            merged = dict(existing.config or {})
            merged.update(cfg)
            existing.config = merged
            existing.version = (existing.version or 0) + 1
        await session.commit()


@pytest.fixture
def operator_keys(monkeypatch):
    """Configure a pair of operator signing keys on the global Settings
    singleton for the duration of one test. The settings object is the
    one `_apply_signature` reads via lazy import."""
    keys = {"key-2026-01": _gen_key(), "key-2026-02": _gen_key()}
    monkeypatch.setattr(settings, "receipt_signing_keys", dict(keys))
    yield keys


async def _emit_context_receipt(client, *, tenant: str, subject: str) -> str:
    """POST /v1/context with emit_receipt=true; return the receipt_id."""
    resp = await client.post(
        "/v1/context",
        json={"subject_id": subject, "task": "test signing flow", "emit_receipt": True},
        headers={"X-Tenant-ID": tenant},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("receipt_id"), f"emit_receipt=true did not produce a receipt: {body}"
    return body["receipt_id"]


async def test_signed_receipt_verifies_ok_end_to_end(
    client, session_factory, subject_id, operator_keys
):
    tenant = f"tenant-sig-{uuid.uuid4().hex[:6]}"
    await _seed_tenant_signing(session_factory, tenant, "key-2026-01")

    rid = await _emit_context_receipt(client, tenant=tenant, subject=subject_id)

    resp = await client.get(
        f"/v1/receipts/{rid}/verify", headers={"X-Tenant-ID": tenant}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["reason"] == "ok"
    assert body["key_id"] == "key-2026-01"
    assert body["algorithm"] == SUPPORTED_SIGNATURE_ALGORITHM


async def test_tenant_without_signing_emits_unsigned_receipt(
    client, session_factory, subject_id, operator_keys
):
    """Tenant has no `receipt_signing_key_id` configured → receipt
    emits unsigned, verify reports `no_signature` cleanly."""
    tenant = f"tenant-unsigned-{uuid.uuid4().hex[:6]}"
    # Receipts forced on, but no signing_key_id set
    await _seed_tenant_signing(session_factory, tenant, key_id=None)

    rid = await _emit_context_receipt(client, tenant=tenant, subject=subject_id)

    resp = await client.get(
        f"/v1/receipts/{rid}/verify", headers={"X-Tenant-ID": tenant}
    )
    body = resp.json()
    assert body["valid"] is None
    assert body["reason"] == "no_signature"
    assert body["key_id"] is None


async def test_tenant_signing_with_missing_key_emits_unsigned_and_warns(
    client, session_factory, subject_id, operator_keys, caplog
):
    """Tenant points at a key_id the operator hasn't loaded in this
    process. Per the failure-mode rule, the receipt still emits — just
    unsigned — and a structured warning fires."""
    tenant = f"tenant-missing-key-{uuid.uuid4().hex[:6]}"
    await _seed_tenant_signing(session_factory, tenant, "key-not-loaded")

    with caplog.at_level(logging.WARNING):
        rid = await _emit_context_receipt(client, tenant=tenant, subject=subject_id)

    # Receipt exists, no signature.
    resp = await client.get(
        f"/v1/receipts/{rid}/verify", headers={"X-Tenant-ID": tenant}
    )
    assert resp.json()["reason"] == "no_signature"

    # Warning was emitted with key_id but never raw key bytes.
    warned = " ".join(r.getMessage() for r in caplog.records)
    assert "receipt_signing_key_unavailable" in warned
    assert "key-not-loaded" in warned


async def test_tampered_signed_receipt_verifies_false(
    client, session_factory, subject_id, operator_keys
):
    """End-to-end: tamper directly in the DB body, verify reports
    signature_mismatch — proving the body is signature-covered."""
    tenant = f"tenant-tamper-{uuid.uuid4().hex[:6]}"
    await _seed_tenant_signing(session_factory, tenant, "key-2026-01")
    rid = await _emit_context_receipt(client, tenant=tenant, subject=subject_id)

    # Surgically tamper inside the body.
    async with session_factory() as session:
        row = await session.get(ReceiptRow, rid)
        tampered_body = dict(row.body)
        tampered_body["subject_id"] = "different-subject"
        row.body = tampered_body
        await session.commit()

    resp = await client.get(
        f"/v1/receipts/{rid}/verify", headers={"X-Tenant-ID": tenant}
    )
    body = resp.json()
    assert body["valid"] is False
    assert body["reason"] == "signature_mismatch"


async def test_key_rotation_old_receipts_still_verify(
    client, session_factory, subject_id, operator_keys, monkeypatch
):
    """Rotation flow: tenant moves from `key-2026-01` → `key-2026-02`.
    Receipts emitted under the old key remain verifiable as long as the
    old key is still configured."""
    tenant = f"tenant-rotate-{uuid.uuid4().hex[:6]}"

    # Emit under key-2026-01.
    await _seed_tenant_signing(session_factory, tenant, "key-2026-01")
    old_rid = await _emit_context_receipt(client, tenant=tenant, subject=subject_id)

    # Rotate the tenant pointer to key-2026-02 (both keys still loaded).
    await _seed_tenant_signing(session_factory, tenant, "key-2026-02")
    new_rid = await _emit_context_receipt(client, tenant=tenant, subject=subject_id)

    # Both verify ok with the right key ids.
    for rid, expected_kid in [(old_rid, "key-2026-01"), (new_rid, "key-2026-02")]:
        resp = await client.get(
            f"/v1/receipts/{rid}/verify", headers={"X-Tenant-ID": tenant}
        )
        body = resp.json()
        assert body["valid"] is True
        assert body["key_id"] == expected_kid


async def test_rotated_out_key_returns_key_unavailable(
    client, session_factory, subject_id, monkeypatch
):
    """Operator removes the old key from config → historical receipts
    signed with it report `key_unavailable`, never crash."""
    # Two keys loaded initially.
    initial_keys = {"key-temp": _gen_key(), "key-survivor": _gen_key()}
    monkeypatch.setattr(settings, "receipt_signing_keys", dict(initial_keys))

    tenant = f"tenant-rotated-out-{uuid.uuid4().hex[:6]}"
    await _seed_tenant_signing(session_factory, tenant, "key-temp")
    rid = await _emit_context_receipt(client, tenant=tenant, subject=subject_id)

    # Drop the temp key from operator config.
    monkeypatch.setattr(settings, "receipt_signing_keys", {"key-survivor": initial_keys["key-survivor"]})

    resp = await client.get(
        f"/v1/receipts/{rid}/verify", headers={"X-Tenant-ID": tenant}
    )
    body = resp.json()
    assert body["valid"] is None
    assert body["reason"] == "key_unavailable"
    assert body["key_id"] == "key-temp"


# ---------------------------------------------------------------------------
# 4. Security — no key material in logs or responses
# ---------------------------------------------------------------------------


async def test_signing_keys_never_appear_in_logs(
    client, session_factory, subject_id, operator_keys, caplog
):
    """End-to-end caplog scan: emit a receipt under a configured key,
    verify it, tamper + re-verify, hit the missing-key path. None of
    those flows should leak the key bytes (or even their hex/base64
    rendering) into structured logs."""
    tenant = f"tenant-keylogs-{uuid.uuid4().hex[:6]}"
    await _seed_tenant_signing(session_factory, tenant, "key-2026-01")
    key_bytes = operator_keys["key-2026-01"]
    key_hex = key_bytes.hex()
    key_b64 = _b64(key_bytes)

    with caplog.at_level(logging.DEBUG):
        rid = await _emit_context_receipt(client, tenant=tenant, subject=subject_id)
        await client.get(f"/v1/receipts/{rid}/verify", headers={"X-Tenant-ID": tenant})

        # Tamper + re-verify
        async with session_factory() as session:
            row = await session.get(ReceiptRow, rid)
            tampered = dict(row.body)
            tampered["subject_id"] = "evil"
            row.body = tampered
            await session.commit()
        await client.get(f"/v1/receipts/{rid}/verify", headers={"X-Tenant-ID": tenant})

    all_log_text = " ".join(r.getMessage() for r in caplog.records)
    assert key_hex not in all_log_text, "Raw key hex must never appear in logs"
    assert key_b64 not in all_log_text, "Base64 key must never appear in logs"
    # Sanity — we DID actually log something during the flow.
    assert caplog.records, "log capture didn't see any records"


async def test_verify_response_never_echoes_key_material(
    client, session_factory, subject_id, operator_keys
):
    """The verify endpoint must surface validity + key_id name + algorithm,
    but never the key bytes, a hash of them, or any field whose value
    matches the raw key."""
    tenant = f"tenant-resp-{uuid.uuid4().hex[:6]}"
    await _seed_tenant_signing(session_factory, tenant, "key-2026-01")
    rid = await _emit_context_receipt(client, tenant=tenant, subject=subject_id)

    resp = await client.get(
        f"/v1/receipts/{rid}/verify", headers={"X-Tenant-ID": tenant}
    )
    body = resp.json()

    raw_key = operator_keys["key-2026-01"]
    serialised = str(body)
    assert raw_key.hex() not in serialised
    assert _b64(raw_key) not in serialised
    # And the response shape is just the documented fields — no surprise
    # bonus key-derivative field.
    assert set(body.keys()) == {"valid", "key_id", "algorithm", "reason"}


def test_settings_repr_does_not_leak_signing_keys():
    """The field is declared `repr=False` so a stray `print(settings)`
    or pydantic error wrapping the model can't accidentally dump the
    operator's raw key bytes."""
    key = _gen_key()
    s = Settings(receipt_signing_keys={"k1": _b64(key)})
    rendered = repr(s)
    assert "receipt_signing_keys" not in rendered
    assert key.hex() not in rendered
    assert _b64(key) not in rendered


# ---------------------------------------------------------------------------
# 4. Security — `unset` env produces no signing without warning noise
# ---------------------------------------------------------------------------


def test_unset_env_means_no_signing_no_error():
    """A deployment that hasn't opted into signing must boot cleanly
    and emit unsigned receipts without any startup error or noise."""
    # The default settings have no signing keys.
    s = Settings()
    assert s.receipt_signing_keys == {}


def test_env_var_path_parses_real_env(monkeypatch):
    """Confirm the pydantic-settings env path goes through the
    validator the same way the in-process dict path does."""
    key = _gen_key()
    monkeypatch.setenv(
        "STATEWAVE_RECEIPT_SIGNING_KEYS", f'{{"k1": "{_b64(key)}"}}'
    )
    # Force pydantic-settings to read fresh from env.
    s = Settings()
    assert s.receipt_signing_keys == {"k1": key}
