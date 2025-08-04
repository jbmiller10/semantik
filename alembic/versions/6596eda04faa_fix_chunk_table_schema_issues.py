"""fix_chunk_table_schema_issues

Revision ID: 6596eda04faa
Revises: 52db15bd2686
Create Date: 2025-08-04 16:27:40.142728

This migration fixes several issues identified in the code review:
1. Standardizes UUID columns to use String type for consistency
2. Adds missing index on chunks.embedding_vector_id
3. Adds unique constraint on (collection_id, document_id, chunk_index)
"""

import os
from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6596eda04faa"
down_revision: str | Sequence[str] | None = "52db15bd2686"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """
    Fix chunk table schema issues:
    1. Convert UUID column from UUID(as_uuid=True) to String for consistency
    2. Add index on embedding_vector_id
    3. Add unique constraint on (collection_id, document_id, chunk_index)
    """

    # Get partition count from environment variable, default to 16
    partition_count = int(os.environ.get("CHUNK_PARTITION_COUNT", "16"))

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
        op.execute(
            f"""
            ALTER TABLE chunks_p{i} RENAME TO chunks_old_p{i};
        """
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

    # Create partitions
    for i in range(partition_count):
        op.execute(
            f"""
            CREATE TABLE chunks_p{i} PARTITION OF chunks
            FOR VALUES WITH (MODULUS {partition_count}, REMAINDER {i});
        """
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
    op.create_index("ix_chunks_collection_id_document_id", "chunks", ["collection_id", "document_id"], unique=False)
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"], unique=False)
    op.create_index("ix_chunks_chunking_config_id", "chunks", ["chunking_config_id"], unique=False)
    op.create_index("ix_chunks_collection_id_chunk_index", "chunks", ["collection_id", "chunk_index"], unique=False)
    op.create_index("ix_chunks_created_at", "chunks", ["created_at"], unique=False)

    # New indexes from the review
    op.create_index("ix_chunks_embedding_vector_id", "chunks", ["embedding_vector_id"], unique=False)

    # Add unique constraint on (collection_id, document_id, chunk_index)
    # Note: We can't use a regular unique constraint because collection_id is part of the partition key
    # Instead, we create a unique index which serves the same purpose
    op.create_index(
        "uq_chunks_collection_document_index", "chunks", ["collection_id", "document_id", "chunk_index"], unique=True
    )

    # Step 5: Drop the old table and its partitions
    op.execute("DROP TABLE chunks_old CASCADE;")

    # Step 6: Update the materialized view to handle the new string ID type
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
    op.create_index("ix_collection_chunking_stats_id", "collection_chunking_stats", ["id"], unique=True)


def downgrade() -> None:
    """
    Revert chunk table schema changes:
    1. Convert String ID back to UUID
    2. Remove added indexes
    3. Remove unique constraint
    """

    # Get partition count from environment variable, default to 16
    partition_count = int(os.environ.get("CHUNK_PARTITION_COUNT", "16"))

    # Step 1: Rename current table
    op.execute("ALTER TABLE chunks RENAME TO chunks_new;")

    # Rename all partitions
    for i in range(partition_count):
        op.execute(f"ALTER TABLE chunks_p{i} RENAME TO chunks_new_p{i};")

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

    # Create partitions
    for i in range(partition_count):
        op.execute(
            f"""
            CREATE TABLE chunks_p{i} PARTITION OF chunks
            FOR VALUES WITH (MODULUS {partition_count}, REMAINDER {i});
        """
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
    op.create_index("ix_chunks_collection_id_document_id", "chunks", ["collection_id", "document_id"], unique=False)
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"], unique=False)
    op.create_index("ix_chunks_chunking_config_id", "chunks", ["chunking_config_id"], unique=False)
    op.create_index("ix_chunks_collection_id_chunk_index", "chunks", ["collection_id", "chunk_index"], unique=False)
    op.create_index("ix_chunks_created_at", "chunks", ["created_at"], unique=False)

    # Step 5: Drop the new table
    op.execute("DROP TABLE chunks_new CASCADE;")

    # Step 6: Restore original materialized view
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
    op.create_index("ix_collection_chunking_stats_id", "collection_chunking_stats", ["id"], unique=True)
