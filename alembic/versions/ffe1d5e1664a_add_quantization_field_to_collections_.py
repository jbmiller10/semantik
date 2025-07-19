"""Add quantization field to collections table

Revision ID: ffe1d5e1664a
Revises: 91784cc819aa
Create Date: 2025-07-18 08:50:23.050916

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ffe1d5e1664a"
down_revision: str | Sequence[str] | None = "91784cc819aa"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add quantization column with default value for existing rows
    with op.batch_alter_table("collections") as batch_op:
        batch_op.add_column(sa.Column("quantization", sa.String(), nullable=False, server_default="float16"))
    # Remove server default after adding column
    with op.batch_alter_table("collections") as batch_op:
        batch_op.alter_column("quantization", server_default=None)


def downgrade() -> None:
    """Downgrade schema."""
    # Remove quantization column
    op.drop_column("collections", "quantization")
