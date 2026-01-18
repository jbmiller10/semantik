"""Add user_preferences table for settings page expansion.

Creates user_preferences table for per-user search and collection defaults:
- Search preferences: top_k, mode, reranker, RRF, similarity threshold
- Collection defaults: embedding model, quantization, chunking settings, sparse/hybrid

Revision ID: 202601140001
Revises: 202601131000
Create Date: 2026-01-14 00:01:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


revision: str = "202601140001"
down_revision: str | Sequence[str] | None = "202601131000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create user_preferences table."""
    op.create_table(
        "user_preferences",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        # Search preferences
        sa.Column("search_top_k", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("search_mode", sa.String(16), nullable=False, server_default="dense"),
        sa.Column("search_use_reranker", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("search_rrf_k", sa.Integer(), nullable=False, server_default="60"),
        sa.Column("search_similarity_threshold", sa.Float(), nullable=True),
        # Collection defaults
        sa.Column("default_embedding_model", sa.String(128), nullable=True),
        sa.Column("default_quantization", sa.String(16), nullable=False, server_default="float16"),
        sa.Column("default_chunking_strategy", sa.String(32), nullable=False, server_default="recursive"),
        sa.Column("default_chunk_size", sa.Integer(), nullable=False, server_default="1024"),
        sa.Column("default_chunk_overlap", sa.Integer(), nullable=False, server_default="200"),
        sa.Column("default_enable_sparse", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("default_sparse_type", sa.String(16), nullable=False, server_default="bm25"),
        sa.Column("default_enable_hybrid", sa.Boolean(), nullable=False, server_default="false"),
        # Timestamps
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

    # Unique index on user_id for fast lookups
    op.create_index(
        "ix_user_preferences_user_id",
        "user_preferences",
        ["user_id"],
        unique=True,
    )

    # CHECK constraints for valid ranges
    op.create_check_constraint(
        "ck_user_preferences_search_top_k",
        "user_preferences",
        "search_top_k >= 5 AND search_top_k <= 50",
    )

    op.create_check_constraint(
        "ck_user_preferences_search_mode",
        "user_preferences",
        "search_mode IN ('dense', 'sparse', 'hybrid')",
    )

    op.create_check_constraint(
        "ck_user_preferences_search_rrf_k",
        "user_preferences",
        "search_rrf_k >= 1 AND search_rrf_k <= 100",
    )

    op.create_check_constraint(
        "ck_user_preferences_search_similarity_threshold",
        "user_preferences",
        "search_similarity_threshold IS NULL OR (search_similarity_threshold >= 0.0 AND search_similarity_threshold <= 1.0)",
    )

    op.create_check_constraint(
        "ck_user_preferences_default_quantization",
        "user_preferences",
        "default_quantization IN ('float32', 'float16', 'int8')",
    )

    op.create_check_constraint(
        "ck_user_preferences_default_chunking_strategy",
        "user_preferences",
        "default_chunking_strategy IN ('character', 'recursive', 'markdown', 'semantic')",
    )

    op.create_check_constraint(
        "ck_user_preferences_default_chunk_size",
        "user_preferences",
        "default_chunk_size >= 256 AND default_chunk_size <= 4096",
    )

    op.create_check_constraint(
        "ck_user_preferences_default_chunk_overlap",
        "user_preferences",
        "default_chunk_overlap >= 0 AND default_chunk_overlap <= 512",
    )

    op.create_check_constraint(
        "ck_user_preferences_default_sparse_type",
        "user_preferences",
        "default_sparse_type IN ('bm25', 'splade')",
    )

    # Hybrid requires sparse constraint
    op.create_check_constraint(
        "ck_user_preferences_hybrid_requires_sparse",
        "user_preferences",
        "default_enable_hybrid = false OR default_enable_sparse = true",
    )


def downgrade() -> None:
    """Drop user_preferences table."""
    # Drop constraints in reverse order
    op.drop_constraint("ck_user_preferences_hybrid_requires_sparse", "user_preferences", type_="check")
    op.drop_constraint("ck_user_preferences_default_sparse_type", "user_preferences", type_="check")
    op.drop_constraint("ck_user_preferences_default_chunk_overlap", "user_preferences", type_="check")
    op.drop_constraint("ck_user_preferences_default_chunk_size", "user_preferences", type_="check")
    op.drop_constraint("ck_user_preferences_default_chunking_strategy", "user_preferences", type_="check")
    op.drop_constraint("ck_user_preferences_default_quantization", "user_preferences", type_="check")
    op.drop_constraint("ck_user_preferences_search_similarity_threshold", "user_preferences", type_="check")
    op.drop_constraint("ck_user_preferences_search_rrf_k", "user_preferences", type_="check")
    op.drop_constraint("ck_user_preferences_search_mode", "user_preferences", type_="check")
    op.drop_constraint("ck_user_preferences_search_top_k", "user_preferences", type_="check")

    op.drop_index("ix_user_preferences_user_id", table_name="user_preferences")
    op.drop_table("user_preferences")
