"""backfill_chunking_strategy

Phase 2 Backfill: Populate null chunking_strategy values in collections table.

This migration performs an idempotent backfill operation that:
1. Sets chunking_strategy to "recursive" as the general default (recommended strategy)
2. Sets chunking_strategy to "character" for collections with custom chunk_size/chunk_overlap
   (non-default values indicate they were using token-based chunking with custom settings)
3. Leaves existing non-null values unchanged (idempotent)

Revision ID: p2_backfill_001
Revises: f1a2b3c4d5e6
Create Date: 2025-08-12 15:30:00.000000

"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.sql import text

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = "p2_backfill_001"
down_revision: Union[str, None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Backfill chunking_strategy column with appropriate defaults."""
    bind = op.get_bind()

    # First, check if the chunking_strategy column exists (backward compatibility)
    column_exists = bind.execute(
        text(
            """
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.columns 
                WHERE table_name = 'collections' 
                AND column_name = 'chunking_strategy'
            )
        """
        )
    ).scalar()

    if not column_exists:
        logger.info("chunking_strategy column does not exist. Skipping backfill.")
        return

    # Count collections that need backfilling
    null_count = bind.execute(
        text(
            """
            SELECT COUNT(*) 
            FROM collections 
            WHERE chunking_strategy IS NULL
        """
        )
    ).scalar()

    if null_count == 0:
        logger.info("No collections with null chunking_strategy found. Nothing to backfill.")
        return

    logger.info(f"Found {null_count} collections with null chunking_strategy to backfill.")

    # Step 1: Set chunking_strategy to "character" for collections with custom chunk settings
    # Default values are chunk_size=1000 and chunk_overlap=200
    # If either differs from defaults, the collection was using custom token-based chunking
    custom_updated = bind.execute(
        text(
            """
            UPDATE collections
            SET chunking_strategy = 'character'
            WHERE chunking_strategy IS NULL
            AND (
                chunk_size != 1000 
                OR chunk_overlap != 200
            )
            RETURNING id
        """
        )
    )
    custom_count = custom_updated.rowcount

    if custom_count > 0:
        logger.info(f"Set chunking_strategy to 'character' for {custom_count} collections with custom chunk settings.")

    # Step 2: Set remaining null values to "recursive" (the recommended default)
    recursive_updated = bind.execute(
        text(
            """
            UPDATE collections
            SET chunking_strategy = 'recursive'
            WHERE chunking_strategy IS NULL
            RETURNING id
        """
        )
    )
    recursive_count = recursive_updated.rowcount

    if recursive_count > 0:
        logger.info(f"Set chunking_strategy to 'recursive' for {recursive_count} collections (default).")

    # Verify all nulls have been filled
    remaining_nulls = bind.execute(
        text(
            """
            SELECT COUNT(*) 
            FROM collections 
            WHERE chunking_strategy IS NULL
        """
        )
    ).scalar()

    if remaining_nulls > 0:
        logger.info(f"WARNING: {remaining_nulls} collections still have null chunking_strategy.")
    else:
        logger.info(f"Successfully backfilled {custom_count + recursive_count} collections.")

    # Log final distribution for visibility
    distribution = bind.execute(
        text(
            """
            SELECT chunking_strategy, COUNT(*) as count
            FROM collections
            GROUP BY chunking_strategy
            ORDER BY count DESC
        """
        )
    ).fetchall()

    logger.info("\nFinal chunking_strategy distribution:")
    for strategy, count in distribution:
        logger.info(f"  {strategy}: {count}")


def downgrade() -> None:
    """Revert backfilled chunking_strategy values to NULL.

    Note: This only reverts values that match the backfill logic.
    Any manually set values or values set through the application
    will be preserved.
    """
    bind = op.get_bind()

    # Check if the column exists
    column_exists = bind.execute(
        text(
            """
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.columns 
                WHERE table_name = 'collections' 
                AND column_name = 'chunking_strategy'
            )
        """
        )
    ).scalar()

    if not column_exists:
        logger.info("chunking_strategy column does not exist. Nothing to downgrade.")
        return

    # Revert "character" strategy that matches our backfill criteria
    character_reverted = bind.execute(
        text(
            """
            UPDATE collections
            SET chunking_strategy = NULL
            WHERE chunking_strategy = 'character'
            AND (chunk_size != 1000 OR chunk_overlap != 200)
            RETURNING id
        """
        )
    )
    character_count = character_reverted.rowcount

    if character_count > 0:
        logger.info(f"Reverted {character_count} collections from 'character' strategy to NULL.")

    # Revert "recursive" strategy that matches our backfill criteria
    # Only revert if using default chunk settings (as that's what we backfilled)
    recursive_reverted = bind.execute(
        text(
            """
            UPDATE collections
            SET chunking_strategy = NULL
            WHERE chunking_strategy = 'recursive'
            AND chunk_size = 1000
            AND chunk_overlap = 200
            RETURNING id
        """
        )
    )
    recursive_count = recursive_reverted.rowcount

    if recursive_count > 0:
        logger.info(f"Reverted {recursive_count} collections from 'recursive' strategy to NULL.")

    logger.info(f"Total collections reverted: {character_count + recursive_count}")
