"""Ensure chunking infrastructure matches production schema.

Revision ID: 202510170001
Revises: f1a2b3c4d5e6
Create Date: 2025-10-17 09:15:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

from packages.shared.database.schema_setup import drop_chunking_infrastructure, ensure_chunking_infrastructure

# revision identifiers, used by Alembic.
revision: str = "202510170001"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the consolidated chunking schema."""
    bind = op.get_bind()
    ensure_chunking_infrastructure(bind)


def downgrade() -> None:
    """Best-effort rollback of chunking schema objects."""
    bind = op.get_bind()
    drop_chunking_infrastructure(bind)
