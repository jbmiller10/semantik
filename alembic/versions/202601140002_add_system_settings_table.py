"""Add system_settings table for admin-configurable system parameters.

Creates key-value store for system-wide admin settings:
- Resource limits (max collections, storage, document size)
- Performance tuning (cache TTL, model timeout, search multiplier)
- GPU/memory configuration (reserves, offload, eviction)
- Search/rerank settings (candidates, weights)

Values are JSON; null means "use environment variable fallback".

Revision ID: 202601140002
Revises: 202601140001
Create Date: 2026-01-14 00:02:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


revision: str = "202601140002"
down_revision: str | Sequence[str] | None = "202601140001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Default settings keys with null values (use env var fallback)
DEFAULT_SETTINGS = [
    # Resource Limits
    "max_collections_per_user",
    "max_storage_gb_per_user",
    "max_document_size_mb",
    "max_artifact_size_mb",
    # Performance
    "cache_ttl_seconds",
    "model_unload_timeout_seconds",
    # GPU & Memory
    "gpu_memory_reserve_percent",
    "gpu_memory_max_percent",
    "cpu_memory_reserve_percent",
    "cpu_memory_max_percent",
    "enable_cpu_offload",
    "eviction_idle_threshold_seconds",
    # Search & Reranking
    "rerank_candidate_multiplier",
    "rerank_min_candidates",
    "rerank_max_candidates",
    "rerank_hybrid_weight",
]


def upgrade() -> None:
    """Create system_settings table and insert default keys."""
    # Create the table
    op.create_table(
        "system_settings",
        sa.Column("key", sa.String(64), primary_key=True),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_by",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )

    # Create index on updated_by for faster lookups
    op.create_index(
        "ix_system_settings_updated_by",
        "system_settings",
        ["updated_by"],
    )

    # Insert default settings with null values (means "use env var fallback")
    conn = op.get_bind()
    for key in DEFAULT_SETTINGS:
        conn.execute(
            sa.text("INSERT INTO system_settings (key, value) VALUES (:key, 'null'::json)"),
            {"key": key},
        )


def downgrade() -> None:
    """Drop system_settings table."""
    op.drop_index("ix_system_settings_updated_by", table_name="system_settings")
    op.drop_table("system_settings")
