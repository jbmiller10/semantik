"""Add CHECK constraints to MCP profiles table.

Adds database-level validation for MCP profile fields:
- search_type: Must be one of semantic, hybrid, keyword, question, code
- result_count: Must be between 1 and 100
- score_threshold: Must be NULL or between 0 and 1
- hybrid_alpha: Must be NULL or between 0 and 1

Revision ID: 202601061001
Revises: 202601061000
Create Date: 2026-01-06 10:01:00.000000

"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "202601061001"
down_revision: str = "202601061000"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    """Add CHECK constraints for MCP profile validation."""
    op.create_check_constraint(
        "ck_mcp_profiles_search_type",
        "mcp_profiles",
        "search_type IN ('semantic', 'hybrid', 'keyword', 'question', 'code')",
    )
    op.create_check_constraint(
        "ck_mcp_profiles_result_count",
        "mcp_profiles",
        "result_count >= 1 AND result_count <= 100",
    )
    op.create_check_constraint(
        "ck_mcp_profiles_score_threshold",
        "mcp_profiles",
        "score_threshold IS NULL OR (score_threshold >= 0 AND score_threshold <= 1)",
    )
    op.create_check_constraint(
        "ck_mcp_profiles_hybrid_alpha",
        "mcp_profiles",
        "hybrid_alpha IS NULL OR (hybrid_alpha >= 0 AND hybrid_alpha <= 1)",
    )


def downgrade() -> None:
    """Remove CHECK constraints."""
    op.drop_constraint("ck_mcp_profiles_hybrid_alpha", "mcp_profiles", type_="check")
    op.drop_constraint("ck_mcp_profiles_score_threshold", "mcp_profiles", type_="check")
    op.drop_constraint("ck_mcp_profiles_result_count", "mcp_profiles", type_="check")
    op.drop_constraint("ck_mcp_profiles_search_type", "mcp_profiles", type_="check")
