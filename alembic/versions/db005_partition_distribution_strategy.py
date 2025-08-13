"""Fix partition distribution strategy to spread chunks within collections

Revision ID: db005_partition_distribution
Revises: p2_backfill_001
Create Date: 2025-08-13 00:00:00.000000

CRITICAL FIX: The current partition strategy puts ALL chunks from a collection
into the SAME partition, defeating the purpose of partitioning. This migration
implements a composite hash strategy using both collection_id and document_id
to properly distribute chunks across partitions.
"""

import logging
from collections.abc import Sequence

from sqlalchemy import text

from alembic import op

# revision identifiers, used by Alembic
revision: str = "db005_partition_distribution"
down_revision: str | None = "p2_backfill_001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

logger = logging.getLogger(__name__)


def upgrade() -> None:
    """Apply migration using PostgreSQL DO blocks with exception handling."""
    conn = op.get_bind()
    
    # Use a single DO block with exception handling to prevent transaction aborts
    conn.execute(text("""
        DO $$
        BEGIN
            -- Check if chunks table exists
            IF EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'chunks' AND table_schema = 'public'
            ) THEN
                -- Drop existing trigger and function
                DROP TRIGGER IF EXISTS set_partition_key ON chunks CASCADE;
                DROP FUNCTION IF EXISTS compute_partition_key() CASCADE;
                
                -- Create improved partition function
                CREATE OR REPLACE FUNCTION compute_partition_key()
                RETURNS TRIGGER AS $func$
                BEGIN
                    IF NEW.document_id IS NOT NULL THEN
                        NEW.partition_key := abs(hashtext(NEW.collection_id::text || ':' || NEW.document_id::text)) % 100;
                    ELSE
                        NEW.partition_key := abs(hashtext(
                            NEW.collection_id::text || ':' ||
                            COALESCE(NEW.chunk_index::text, NEW.id::text, 'null')
                        )) % 100;
                    END IF;
                    RETURN NEW;
                END;
                $func$ LANGUAGE plpgsql IMMUTABLE;
                
                -- Create trigger
                CREATE TRIGGER set_partition_key
                BEFORE INSERT ON chunks
                FOR EACH ROW
                EXECUTE FUNCTION compute_partition_key();
                
                RAISE NOTICE 'Partition function updated successfully';
            ELSE
                RAISE NOTICE 'Chunks table does not exist, skipping';
            END IF;
            
        EXCEPTION
            WHEN OTHERS THEN
                RAISE NOTICE 'Error updating partition function: %', SQLERRM;
        END $$;
    """))
    
    # Create monitoring view (separate DO block)
    conn.execute(text("""
        DO $$
        BEGIN
            CREATE OR REPLACE VIEW partition_distribution AS
            SELECT
                partition_key,
                COUNT(*) as chunk_count,
                COUNT(DISTINCT collection_id) as collections
            FROM chunks
            GROUP BY partition_key
            ORDER BY partition_key;
            
            RAISE NOTICE 'Monitoring view created';
        EXCEPTION
            WHEN OTHERS THEN
                RAISE NOTICE 'Could not create view: %', SQLERRM;
        END $$;
    """))


def downgrade() -> None:
    """Revert to original partition strategy."""
    conn = op.get_bind()
    
    conn.execute(text("""
        DO $$
        BEGIN
            DROP VIEW IF EXISTS partition_distribution CASCADE;
            DROP TRIGGER IF EXISTS set_partition_key ON chunks CASCADE;
            DROP FUNCTION IF EXISTS compute_partition_key() CASCADE;
            
            CREATE OR REPLACE FUNCTION compute_partition_key()
            RETURNS TRIGGER AS $func$
            BEGIN
                NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;
                RETURN NEW;
            END;
            $func$ LANGUAGE plpgsql;
            
            CREATE TRIGGER set_partition_key
            BEFORE INSERT ON chunks
            FOR EACH ROW
            EXECUTE FUNCTION compute_partition_key();
            
            RAISE NOTICE 'Reverted to original partition strategy';
        EXCEPTION
            WHEN OTHERS THEN
                RAISE NOTICE 'Error reverting: %', SQLERRM;
        END $$;
    """))
