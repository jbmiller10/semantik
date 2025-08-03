"""Add chunking strategy fields to collections table

Revision ID: 20250803150601
Revises: 20250727151108
Create Date: 2025-08-03 15:06:01

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "20250803150601"
down_revision: str | Sequence[str] | None = "20250727151108"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add chunking_strategy and chunking_params columns to collections table."""
    # Add chunking_strategy column with default value
    op.add_column(
        "collections",
        sa.Column(
            "chunking_strategy",
            sa.String(),
            nullable=False,
            server_default="recursive",
        ),
    )
    
    # Add chunking_params column (nullable JSON)
    op.add_column(
        "collections",
        sa.Column(
            "chunking_params",
            sa.JSON(),
            nullable=True,
        ),
    )
    
    # Create index on chunking_strategy for better query performance
    op.create_index(
        "ix_collections_chunking_strategy",
        "collections",
        ["chunking_strategy"],
        unique=False,
    )


def downgrade() -> None:
    """Remove chunking_strategy and chunking_params columns from collections table."""
    # Drop the index first
    op.drop_index("ix_collections_chunking_strategy", table_name="collections")
    
    # Drop the columns
    op.drop_column("collections", "chunking_params")
    op.drop_column("collections", "chunking_strategy")