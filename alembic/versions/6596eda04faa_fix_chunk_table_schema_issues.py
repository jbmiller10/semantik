"""fix_chunk_table_schema_issues

Revision ID: 6596eda04faa
Revises: 52db15bd2686
Create Date: 2025-08-04 16:27:40.142728

This migration fixes several issues identified in the code review:
1. Standardizes UUID columns to use String type for consistency
2. Adds missing index on chunks.embedding_vector_id
3. Adds unique constraint on (collection_id, document_id, chunk_index)
"""

import logging
import os
from collections.abc import Sequence

from migrations_utils.migration_safety import (
    safe_drop_table,
)
from sqlalchemy.engine import Connection

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6596eda04faa"
down_revision: str | Sequence[str] | None = "52db15bd2686"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Set up logging
logger = logging.getLogger(__name__)


def _index_exists(conn: Connection, index_name: str) -> bool:
    """Check if an index exists in the database using pg_indexes."""
    from sqlalchemy import text

    result = conn.execute(
        text(
            """
        SELECT EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE indexname = :index_name
        )
        """
        ),
        {"index_name": index_name},
    ).scalar()
    return bool(result)


def _create_index_if_not_exists(
    conn: Connection, index_name: str, table_name: str, columns: list[str], unique: bool = False
) -> None:
    """Create an index only if it doesn't already exist."""
    if not _index_exists(conn, index_name):
        op.create_index(index_name, table_name, columns, unique=unique)


def _create_indexes_if_not_exist(conn: Connection) -> None:
    """Create all required indexes if they don't already exist."""
    # Standard indexes
    _create_index_if_not_exists(conn, "ix_chunks_collection_id_document_id", "chunks", ["collection_id", "document_id"])
    _create_index_if_not_exists(conn, "ix_chunks_document_id", "chunks", ["document_id"])
    _create_index_if_not_exists(conn, "ix_chunks_chunking_config_id", "chunks", ["chunking_config_id"])
    _create_index_if_not_exists(conn, "ix_chunks_collection_id_chunk_index", "chunks", ["collection_id", "chunk_index"])
    _create_index_if_not_exists(conn, "ix_chunks_created_at", "chunks", ["created_at"])

    # New indexes from the review
    _create_index_if_not_exists(conn, "ix_chunks_embedding_vector_id", "chunks", ["embedding_vector_id"])

    # Unique constraint as an index
    _create_index_if_not_exists(
        conn,
        "uq_chunks_collection_document_index",
        "chunks",
        ["collection_id", "document_id", "chunk_index"],
        unique=True,
    )


def _create_or_replace_materialized_view() -> None:
    """Create or replace the materialized view and its index."""
    op.execute("DROP MATERIALIZED VIEW IF EXISTS collection_chunking_stats;")
    op.execute(
        """
        CREATE MATERIALIZED VIEW collection_chunking_stats AS
        SELECT
            c.id,
            c.name,
            COUNT(DISTINCT ch.document_id) as chunked_documents,
            COUNT(ch.id) as total_chunks,
            AVG(ch.token_count)::NUMERIC(10,2) as avg_tokens_per_chunk,
            MAX(ch.created_at) as last_chunk_created
        FROM collections c
        LEFT JOIN chunks ch ON c.id = ch.collection_id
        GROUP BY c.id, c.name
        WITH DATA;
    """
    )

    # Create index on the materialized view if it doesn't exist
    conn = op.get_bind()
    if not _index_exists(conn, "ix_collection_chunking_stats_id"):
        op.create_index("ix_collection_chunking_stats_id", "collection_chunking_stats", ["id"], unique=True)


def upgrade() -> None:
    """
    Fix chunk table schema issues:
    1. Convert UUID column from UUID(as_uuid=True) to String for consistency
    2. Add index on embedding_vector_id
    3. Add unique constraint on (collection_id, document_id, chunk_index)
    """

    # Get partition count from environment variable, default to 16
    partition_count = int(os.environ.get("CHUNK_PARTITION_COUNT", "16"))

    # Check if the migration has already been applied
    conn = op.get_bind()
    from sqlalchemy import text

    result = conn.execute(
        text(
            """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = 'chunks_old'
        )
        """
        )
    ).scalar()

    if result:
        # Migration was partially applied, clean up old tables first
        logger.info("Dropping chunks_old table if it exists")
        safe_drop_table(conn, "chunks_old", revision, cascade=True, backup=False)

    # Check if chunks table exists and what type its id column is
    chunks_exists = conn.execute(
        text(
            """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = 'chunks'
        )
        """
        )
    ).scalar()

    if chunks_exists:
        # Check if id column is already VARCHAR (migration already complete)
        id_type = conn.execute(
            text(
                """
            SELECT data_type
            FROM information_schema.columns
            WHERE table_name = 'chunks'
            AND column_name = 'id'
            """
            )
        ).scalar()

        if id_type and id_type.upper() == "CHARACTER VARYING":
            # Migration already complete, just ensure indexes exist
            _create_indexes_if_not_exist(conn)
            _create_or_replace_materialized_view()
            return

    # Step 1: Create a new chunks table with the corrected schema
    # We need to recreate the table because changing the primary key type is complex
    op.execute(
        """
        -- First, rename the existing table and its partitions
        ALTER TABLE chunks RENAME TO chunks_old;
    """
    )

    # Rename all partitions
    for i in range(partition_count):
        # Check if partition exists before renaming
        partition_exists = conn.execute(
            text(
                """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = :table_name
            )
            """
            ),
            {"table_name": f"chunks_p{i}"},
        ).scalar()

        if partition_exists:
            # Use PL/pgSQL with format for safe identifier quoting
            conn.execute(
                text(
                    """
                    DO $$
                    BEGIN
                        EXECUTE format('ALTER TABLE %I RENAME TO %I', :old_name, :new_name);
                    END $$;
                    """
                ),
                {"old_name": f"chunks_p{i}", "new_name": f"chunks_old_p{i}"},
            )

    # Step 2: Create new chunks table with String ID instead of UUID
    op.execute(
        """
        CREATE TABLE chunks (
            id VARCHAR NOT NULL,
            collection_id VARCHAR NOT NULL,
            document_id VARCHAR NOT NULL,
            chunking_config_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            token_count INTEGER,
            embedding_vector_id VARCHAR,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            meta JSON,
            PRIMARY KEY (id, collection_id),
            FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY (chunking_config_id) REFERENCES chunking_configs(id)
        ) PARTITION BY HASH (collection_id);
    """
    )

    # Create partitions using safe PL/pgSQL
    # Validate partition_count first
    if not 1 <= partition_count <= 1000:
        raise ValueError(f"Partition count must be between 1 and 1000, got {partition_count}")

    conn.execute(
        text(
            """
            DO $$
            DECLARE
                i INT;
                partition_name TEXT;
            BEGIN
                FOR i IN 0..:partition_count - 1 LOOP
                    partition_name := 'chunks_p' || i;
                    EXECUTE format('CREATE TABLE %I PARTITION OF chunks FOR VALUES WITH (MODULUS %s, REMAINDER %s)',
                        partition_name, :partition_count, i);
                END LOOP;
            END $$;
            """
        ),
        {"partition_count": partition_count},
    )

    # Step 3: Copy data from old table to new table, converting UUID to string
    op.execute(
        """
        INSERT INTO chunks (
            id, collection_id, document_id, chunking_config_id, chunk_index,
            content, start_offset, end_offset, token_count, embedding_vector_id,
            created_at, meta
        )
        SELECT
            id::text, collection_id, document_id, chunking_config_id, chunk_index,
            content, start_offset, end_offset, token_count, embedding_vector_id,
            created_at, meta
        FROM chunks_old;
    """
    )

    # Step 4: Create all indexes including the new ones
    _create_indexes_if_not_exist(conn)

    # Step 5: Drop the old table and its partitions
    logger.info("Dropping chunks_old table after migration")
    safe_drop_table(conn, "chunks_old", revision, cascade=True, backup=False)

    # Step 6: Update the materialized view to handle the new string ID type
    _create_or_replace_materialized_view()


def downgrade() -> None:
    """
    Revert chunk table schema changes:
    1. Convert String ID back to UUID
    2. Remove added indexes
    3. Remove unique constraint
    """

    # Get partition count from environment variable, default to 16
    partition_count = int(os.environ.get("CHUNK_PARTITION_COUNT", "16"))

    conn = op.get_bind()
    from sqlalchemy import text

    # Check if chunks table exists
    chunks_exists = conn.execute(
        text(
            """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = 'chunks'
        )
        """
        )
    ).scalar()

    if not chunks_exists:
        # Nothing to downgrade
        return

    # Check if id column is UUID (already downgraded)
    id_type = conn.execute(
        text(
            """
        SELECT data_type
        FROM information_schema.columns
        WHERE table_name = 'chunks'
        AND column_name = 'id'
        """
        )
    ).scalar()

    if id_type and id_type.upper() == "UUID":
        # Already downgraded
        return

    # Clean up any leftover chunks_new table from failed downgrade
    logger.info("Dropping chunks_new table if it exists")
    safe_drop_table(conn, "chunks_new", revision, cascade=True, backup=False)

    # Step 1: Rename current table
    op.execute("ALTER TABLE chunks RENAME TO chunks_new;")

    # Rename all partitions
    for i in range(partition_count):
        partition_exists = conn.execute(
            text(
                """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = :table_name
            )
            """
            ),
            {"table_name": f"chunks_p{i}"},
        ).scalar()

        if partition_exists:
            # Use PL/pgSQL with format for safe identifier quoting
            conn.execute(
                text(
                    """
                    DO $$
                    BEGIN
                        EXECUTE format('ALTER TABLE %I RENAME TO %I', :old_name, :new_name);
                    END $$;
                    """
                ),
                {"old_name": f"chunks_p{i}", "new_name": f"chunks_new_p{i}"},
            )

    # Step 2: Recreate original table with UUID type
    op.execute(
        """
        CREATE TABLE chunks (
            id UUID DEFAULT gen_random_uuid() NOT NULL,
            collection_id VARCHAR NOT NULL,
            document_id VARCHAR NOT NULL,
            chunking_config_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            token_count INTEGER,
            embedding_vector_id VARCHAR,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            meta JSON,
            PRIMARY KEY (id, collection_id),
            FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY (chunking_config_id) REFERENCES chunking_configs(id)
        ) PARTITION BY HASH (collection_id);
    """
    )

    # Create partitions using safe PL/pgSQL
    # Validate partition_count first
    if not 1 <= partition_count <= 1000:
        raise ValueError(f"Partition count must be between 1 and 1000, got {partition_count}")

    conn.execute(
        text(
            """
            DO $$
            DECLARE
                i INT;
                partition_name TEXT;
            BEGIN
                FOR i IN 0..:partition_count - 1 LOOP
                    partition_name := 'chunks_p' || i;
                    EXECUTE format('CREATE TABLE %I PARTITION OF chunks FOR VALUES WITH (MODULUS %s, REMAINDER %s)',
                        partition_name, :partition_count, i);
                END LOOP;
            END $$;
            """
        ),
        {"partition_count": partition_count},
    )

    # Step 3: Copy data back, converting string to UUID
    op.execute(
        """
        INSERT INTO chunks (
            id, collection_id, document_id, chunking_config_id, chunk_index,
            content, start_offset, end_offset, token_count, embedding_vector_id,
            created_at, meta
        )
        SELECT
            id::uuid, collection_id, document_id, chunking_config_id, chunk_index,
            content, start_offset, end_offset, token_count, embedding_vector_id,
            created_at, meta
        FROM chunks_new;
    """
    )

    # Step 4: Recreate original indexes (without the new ones)
    # Only create the original indexes that existed before the migration
    _create_index_if_not_exists(conn, "ix_chunks_collection_id_document_id", "chunks", ["collection_id", "document_id"])
    _create_index_if_not_exists(conn, "ix_chunks_document_id", "chunks", ["document_id"])
    _create_index_if_not_exists(conn, "ix_chunks_chunking_config_id", "chunks", ["chunking_config_id"])
    _create_index_if_not_exists(conn, "ix_chunks_collection_id_chunk_index", "chunks", ["collection_id", "chunk_index"])
    _create_index_if_not_exists(conn, "ix_chunks_created_at", "chunks", ["created_at"])

    # Note: We don't recreate ix_chunks_embedding_vector_id and uq_chunks_collection_document_index
    # as they were added by this migration

    # Step 5: Drop the new table
    logger.info("Dropping chunks_new table after downgrade")
    safe_drop_table(conn, "chunks_new", revision, cascade=True, backup=False)

    # Step 6: Restore original materialized view
    _create_or_replace_materialized_view()
