"""Add HyDE settings to user preferences and MCP profiles

Revision ID: 8b6768bb8360
Revises: 202601140004
Create Date: 2026-01-14 20:34:07.903950

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '8b6768bb8360'
down_revision: Union[str, Sequence[str], None] = '202601140004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add columns to user_preferences
    op.add_column('user_preferences', sa.Column('hyde_enabled_default', sa.Boolean(), nullable=False, server_default=sa.text('false')))
    op.add_column('user_preferences', sa.Column('hyde_llm_tier', sa.String(length=16), nullable=False, server_default='low'))
    
    # Add column to mcp_profiles
    op.add_column('mcp_profiles', sa.Column('hyde_enabled', sa.Boolean(), nullable=False, server_default=sa.text('false')))


def downgrade() -> None:
    op.drop_column('mcp_profiles', 'hyde_enabled')
    op.drop_column('user_preferences', 'hyde_llm_tier')
    op.drop_column('user_preferences', 'hyde_enabled_default')