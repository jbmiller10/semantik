"""Add default chunking strategies data migration

Revision ID: a1b2c3d4e5f6
Revises: 8f67aa430c5d
Create Date: 2025-08-04 10:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "8f67aa430c5d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Insert default chunking strategies as data migration."""

    # Create a temporary table to check if strategies already exist
    # This makes the migration idempotent
    connection = op.get_bind()

    # Check if strategies already exist
    result = connection.execute(sa.text("SELECT COUNT(*) FROM chunking_strategies"))
    count = result.scalar()

    if count == 0:
        # Insert default chunking strategies
        op.execute(
            """
            INSERT INTO chunking_strategies (name, description, is_active, meta) VALUES
            ('character', 'Simple fixed-size character-based chunking using TokenTextSplitter', true, '{"supports_streaming": true}'),
            ('recursive', 'Smart sentence-aware splitting using SentenceSplitter', true, '{"supports_streaming": true, "recommended_default": true}'),
            ('markdown', 'Respects markdown structure using MarkdownNodeParser', true, '{"supports_streaming": true, "file_types": [".md", ".mdx"]}'),
            ('semantic', 'Uses AI embeddings to find natural boundaries using SemanticSplitterNodeParser', false, '{"supports_streaming": false, "requires_embeddings": true}'),
            ('hierarchical', 'Creates parent-child chunks using HierarchicalNodeParser', false, '{"supports_streaming": false}'),
            ('hybrid', 'Automatically selects strategy based on content', false, '{"supports_streaming": false}')
            ON CONFLICT (name) DO NOTHING;
            """
        )
        print("Default chunking strategies inserted successfully")
    else:
        print(f"Skipping insertion - {count} chunking strategies already exist")


def downgrade() -> None:
    """Remove default chunking strategies.

    Note: This only removes the default strategies by name, not all strategies.
    """
    op.execute(
        """
        DELETE FROM chunking_strategies
        WHERE name IN ('character', 'recursive', 'markdown', 'semantic', 'hierarchical', 'hybrid')
        AND id NOT IN (
            SELECT DISTINCT chunking_strategy_id
            FROM chunking_configs
            WHERE chunking_strategy_id IS NOT NULL
        );
        """
    )
