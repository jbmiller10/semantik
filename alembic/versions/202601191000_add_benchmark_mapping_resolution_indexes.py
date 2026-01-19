"""Add benchmark mapping resolution indexes.

Revision ID: 202601191000
Revises: 202601181100
Create Date: 2026-01-19

Adds a composite index to support efficient batch resolution of benchmark relevance
judgments by (mapping_id, id).
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "202601191000"
down_revision: str | None = "202601181100"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_benchmark_relevance_mapping_id_id",
        "benchmark_relevance",
        ["mapping_id", "id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_benchmark_relevance_mapping_id_id",
        table_name="benchmark_relevance",
    )

