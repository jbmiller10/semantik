"""Add chunking_strategy and chunking_config columns to collections

Revision ID: add_chunking_strategy_cols
Revises: a1b2c3d4e5f6
Create Date: 2025-08-08 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "add_chunking_strategy_cols"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add chunking_strategy and chunking_config columns to collections table
    op.add_column("collections", sa.Column("chunking_strategy", sa.String(), nullable=True))
    op.add_column("collections", sa.Column("chunking_config", postgresql.JSON(astext_type=sa.Text()), nullable=True))

    # Update existing collections to use 'character' strategy with their current chunk_size/overlap
    op.execute(
        """
        UPDATE collections
        SET chunking_strategy = 'character',
            chunking_config = jsonb_build_object(
                'chunk_size', chunk_size,
                'chunk_overlap', chunk_overlap
            )
        WHERE chunking_strategy IS NULL
    """
    )


def downgrade() -> None:
    # Remove the new columns
    op.drop_column("collections", "chunking_config")
    op.drop_column("collections", "chunking_strategy")
