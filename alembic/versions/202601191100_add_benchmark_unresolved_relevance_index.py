"""Add partial index for unresolved benchmark relevance queries.

Revision ID: 202601191100
Revises: 202601191000
Create Date: 2026-01-19

Adds a partial index to optimize queries that filter for unresolved relevance
judgments (resolved_document_id IS NULL). This significantly improves performance
of the list_unresolved_relevance_for_mapping repository method.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "202601191100"
down_revision: str | None = "202601191000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_benchmark_relevance_mapping_unresolved",
        "benchmark_relevance",
        ["mapping_id", "id"],
        postgresql_where="resolved_document_id IS NULL",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_benchmark_relevance_mapping_unresolved",
        table_name="benchmark_relevance",
    )
