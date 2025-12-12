"""Add connector_secrets table for encrypted credential storage.

This migration adds a table to store encrypted connector credentials
(IMAP passwords, Git tokens, SSH keys) using Fernet symmetric encryption.

Secrets are:
- Encrypted at rest with Fernet (AES-128-CBC + HMAC)
- Never returned via API responses
- Associated with a key_id for key rotation support

Changes:
- Add connector_secrets table with:
  - collection_source_id (FK with cascade delete)
  - secret_type ('password', 'token', 'ssh_key', 'ssh_passphrase')
  - ciphertext (BYTEA for encrypted data)
  - key_id (identifies which encryption key was used)
  - timestamps

Indexes:
- ix_connector_secrets_source: Fast lookup by source

Constraints:
- uq_source_secret_type: One secret per type per source

Revision ID: 202512130200
Revises: 202512130100
Create Date: 2025-12-13 02:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


revision: str = "202512130200"
down_revision: str | Sequence[str] | None = "202512130100"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add connector_secrets table."""

    op.create_table(
        "connector_secrets",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "collection_source_id",
            sa.Integer(),
            sa.ForeignKey("collection_sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("secret_type", sa.String(50), nullable=False),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("key_id", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # Create index for fast lookup by source
    op.create_index(
        "ix_connector_secrets_source",
        "connector_secrets",
        ["collection_source_id"],
    )

    # Create unique constraint (one secret per type per source)
    op.create_unique_constraint(
        "uq_source_secret_type",
        "connector_secrets",
        ["collection_source_id", "secret_type"],
    )


def downgrade() -> None:
    """Remove connector_secrets table."""

    op.drop_table("connector_secrets")
