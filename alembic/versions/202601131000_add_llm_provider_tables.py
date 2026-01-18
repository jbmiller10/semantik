"""Add LLM provider configuration tables.

Creates three tables for LLM provider integration:
- llm_provider_configs: Per-user LLM settings (one-to-one with users)
- llm_provider_api_keys: Encrypted API keys per provider
- llm_usage_events: Token usage tracking

Revision ID: 202601131000
Revises: 202601120001
Create Date: 2026-01-13 10:00:00.000000

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


revision: str = "202601131000"
down_revision: str | Sequence[str] | None = "202601120001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create LLM provider tables."""
    # 1. Create llm_provider_configs table (one-to-one with users)
    op.create_table(
        "llm_provider_configs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        # High-quality tier settings
        sa.Column("high_quality_provider", sa.String(32), nullable=True),
        sa.Column("high_quality_model", sa.String(128), nullable=True),
        # Low-quality tier settings
        sa.Column("low_quality_provider", sa.String(32), nullable=True),
        sa.Column("low_quality_model", sa.String(128), nullable=True),
        # Optional defaults
        sa.Column("default_temperature", sa.Float(), nullable=True),
        sa.Column("default_max_tokens", sa.Integer(), nullable=True),
        sa.Column("provider_config", sa.JSON(), nullable=True),
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

    # Index on user_id for fast lookups (unique constraint creates index)
    op.create_index(
        "ix_llm_provider_configs_user_id",
        "llm_provider_configs",
        ["user_id"],
        unique=True,
    )

    # CHECK constraint for temperature range
    op.create_check_constraint(
        "ck_llm_provider_configs_temperature",
        "llm_provider_configs",
        "default_temperature IS NULL OR (default_temperature >= 0 AND default_temperature <= 2)",
    )

    # 2. Create llm_provider_api_keys table (encrypted keys per provider)
    op.create_table(
        "llm_provider_api_keys",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "config_id",
            sa.Integer(),
            sa.ForeignKey("llm_provider_configs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(32), nullable=False),
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
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Index on config_id for fast lookups
    op.create_index(
        "ix_llm_provider_api_keys_config_id",
        "llm_provider_api_keys",
        ["config_id"],
    )

    # Unique constraint: one key per provider per config
    op.create_unique_constraint(
        "uq_llm_api_keys_config_provider",
        "llm_provider_api_keys",
        ["config_id", "provider"],
    )

    # 3. Create llm_usage_events table (token tracking)
    op.create_table(
        "llm_usage_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # What was called
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("quality_tier", sa.String(16), nullable=False),
        sa.Column("feature", sa.String(50), nullable=False),
        # Token counts
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        # Optional context
        sa.Column("operation_id", sa.Integer(), nullable=True),
        sa.Column("collection_id", sa.String(36), nullable=True),
        sa.Column("request_metadata", sa.JSON(), nullable=True),
        # Timestamp
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # Indexes for common query patterns
    op.create_index(
        "idx_llm_usage_user_created",
        "llm_usage_events",
        ["user_id", "created_at"],
    )

    op.create_index(
        "idx_llm_usage_feature",
        "llm_usage_events",
        ["user_id", "feature"],
    )


def downgrade() -> None:
    """Drop LLM provider tables."""
    # Drop indexes first
    op.drop_index("idx_llm_usage_feature", table_name="llm_usage_events")
    op.drop_index("idx_llm_usage_user_created", table_name="llm_usage_events")

    # Drop tables in reverse order (respects FK dependencies)
    op.drop_table("llm_usage_events")

    op.drop_index("ix_llm_provider_api_keys_config_id", table_name="llm_provider_api_keys")
    op.drop_constraint("uq_llm_api_keys_config_provider", "llm_provider_api_keys", type_="unique")
    op.drop_table("llm_provider_api_keys")

    op.drop_constraint("ck_llm_provider_configs_temperature", "llm_provider_configs", type_="check")
    op.drop_index("ix_llm_provider_configs_user_id", table_name="llm_provider_configs")
    op.drop_table("llm_provider_configs")
