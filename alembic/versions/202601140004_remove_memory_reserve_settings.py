"""Remove redundant memory reserve settings.

Revision ID: 202601140004
Revises: 202601140003
Create Date: 2026-01-13

The gpu_memory_reserve_percent and cpu_memory_reserve_percent settings
were redundant with gpu_memory_max_percent and cpu_memory_max_percent.
The reserve settings have been removed in favor of the simpler max settings.
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "202601140004"
down_revision = "202601140003"
branch_labels = None
depends_on = None

# Settings to remove
REMOVED_KEYS = [
    "gpu_memory_reserve_percent",
    "cpu_memory_reserve_percent",
]


def upgrade() -> None:
    """Remove redundant memory reserve settings from system_settings table."""
    conn = op.get_bind()
    for key in REMOVED_KEYS:
        conn.execute(
            sa.text("DELETE FROM system_settings WHERE key = :key"),
            {"key": key},
        )


def downgrade() -> None:
    """Re-add memory reserve settings with default values."""
    conn = op.get_bind()
    defaults = {
        "gpu_memory_reserve_percent": 0.10,
        "cpu_memory_reserve_percent": 0.20,
    }
    for key, value in defaults.items():
        # Use INSERT ... ON CONFLICT to handle case where key might exist
        conn.execute(
            sa.text(
                """
            INSERT INTO system_settings (key, value, updated_at)
            VALUES (:key, :value, NOW())
            ON CONFLICT (key) DO NOTHING
            """
            ),
            {"key": key, "value": value},
        )
