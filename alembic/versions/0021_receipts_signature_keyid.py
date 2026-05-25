"""receipts: add signature_key_id + signature_algorithm columns (v0.9 #157)

Revision ID: 0021_receipts_signature_keyid
Revises: 0020_receipts_retention
Create Date: 2026-05-25

v0.8 reserved `receipts.receipt_signature` for HMAC tamper-evidence.
v0.9 (issue #157) lights it up. This migration adds the two columns
the verifier needs alongside the signature itself:

  * ``receipt_signature_key_id`` — the operator's key identifier that
    signed this receipt. Public information (it's just a name). The
    actual key bytes live in operator config / secret manager and are
    never persisted in the database.

  * ``receipt_signature_algorithm`` — the signing algorithm + canonical
    form version, baked into a single string (``hmac-sha256-canonical-v1``
    in v0.9). Future migrations to JCS canonicalization or asymmetric
    signing land as new algorithm strings without a schema break.

All three fields (signature + key_id + algorithm) are nullable: pre-v0.9
receipts and v0.9 receipts emitted on tenants without signing configured
remain unsigned, and the verifier reports `valid: null, reason: "no_signature"`
for those. The migration is reversible — downgrade drops the two columns
and leaves the existing `receipt_signature` column from v0.8.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "0021_receipts_signature_keyid"
down_revision: Union[str, None] = "0020_receipts_retention"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "receipts",
        sa.Column(
            "receipt_signature_key_id",
            sa.String(length=64),
            nullable=True,
        ),
    )
    op.add_column(
        "receipts",
        sa.Column(
            "receipt_signature_algorithm",
            sa.String(length=64),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("receipts", "receipt_signature_algorithm")
    op.drop_column("receipts", "receipt_signature_key_id")
