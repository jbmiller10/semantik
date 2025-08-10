"""Database test fixtures and configuration."""

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

# Get database URL from environment or use a test database
# First check if DATABASE_URL is directly provided
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    # Convert to async URL if needed
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif not DATABASE_URL:
    # Otherwise construct it from individual components
    POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")  # Default password for testing
    POSTGRES_DB = os.environ.get("POSTGRES_DB", "semantik_test")
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")

    # Construct the database URL
    if POSTGRES_PASSWORD:
        DATABASE_URL = (
            f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        )
    else:
        DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


@pytest_asyncio.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Create a database session for testing.

    This fixture creates a real database connection for integration tests.
    It ensures the database has the required migrations applied.

    Requirements:
    - PostgreSQL must be running (use `make docker-postgres-up`)
    - Database migrations must be applied (use `poetry run alembic upgrade head`)
    """
    import sys

    # Print database connection info for debugging
    print(
        f"\nConnecting to database: {DATABASE_URL.replace(':' + DATABASE_URL.split(':')[2].split('@')[0] + '@', ':****@')}",
        file=sys.stderr,
    )

    # Create engine
    try:
        engine = create_async_engine(
            DATABASE_URL,
            echo=False,
            pool_pre_ping=True,
        )
    except Exception as e:
        pytest.skip(f"Database connection failed: {e}\nMake sure PostgreSQL is running with: make docker-postgres-up")

    try:
        async with engine.begin() as conn:
            # Ensure the partition views exist by running the migration SQL
            # This is a simplified version that creates the views if they don't exist
            await conn.execute(
                text(
                    """
            -- Create partition_health view if it doesn't exist
            CREATE OR REPLACE VIEW partition_health AS
            WITH partition_stats AS (
                SELECT
                    c.relname AS partition_name,
                    pg_relation_size(c.oid) AS size_bytes,
                    (SELECT COUNT(*) FROM pg_class WHERE oid = c.oid) AS estimated_rows
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relispartition
                AND c.relkind = 'r'
                AND n.nspname = 'public'
                AND c.relname LIKE 'chunks_part_%'
            )
            SELECT
                CAST(SUBSTRING(partition_name FROM 'chunks_part_([0-9]+)') AS INTEGER) AS partition_id,
                partition_name,
                COALESCE(size_bytes, 0) AS size_bytes,
                COALESCE(estimated_rows, 0) AS row_count,
                CASE
                    WHEN size_bytes > 1073741824 THEN 'HOT'
                    WHEN size_bytes < 1048576 THEN 'COLD'
                    ELSE 'NORMAL'
                END AS partition_status
            FROM partition_stats
            UNION ALL
            SELECT
                s.partition_id,
                'chunks_part_' || LPAD(s.partition_id::TEXT, 2, '0') AS partition_name,
                0 AS size_bytes,
                0 AS row_count,
                'COLD' AS partition_status
            FROM generate_series(0, 99) AS s(partition_id)
            WHERE NOT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relispartition
                AND c.relkind = 'r'
                AND n.nspname = 'public'
                AND c.relname = 'chunks_part_' || LPAD(s.partition_id::TEXT, 2, '0')
            )
            ORDER BY partition_id;
        """
                )
            )

        await conn.execute(
            text(
                """
            -- Create partition_distribution view if it doesn't exist
            CREATE OR REPLACE VIEW partition_distribution AS
            WITH partition_counts AS (
                SELECT
                    CAST(SUBSTRING(c.relname FROM 'chunks_part_([0-9]+)') AS INTEGER) AS partition_id,
                    c.relname AS partition_name,
                    (SELECT COUNT(*) FROM pg_class WHERE oid = c.oid) AS row_count
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relispartition
                AND c.relkind = 'r'
                AND n.nspname = 'public'
                AND c.relname LIKE 'chunks_part_%'
            ),
            stats AS (
                SELECT
                    COUNT(*) AS partitions_used,
                    100 - COUNT(*) AS empty_partitions,
                    COALESCE(AVG(row_count), 0) AS avg_rows_per_partition,
                    COALESCE(STDDEV(row_count), 0) AS stddev_rows,
                    COALESCE(MAX(row_count), 0) AS max_rows,
                    COALESCE(MIN(row_count), 0) AS min_rows
                FROM partition_counts
            )
            SELECT
                partitions_used,
                empty_partitions,
                avg_rows_per_partition,
                stddev_rows,
                CASE
                    WHEN avg_rows_per_partition = 0 THEN 0
                    ELSE stddev_rows / avg_rows_per_partition
                END AS coefficient_of_variation,
                max_rows,
                min_rows,
                CASE
                    WHEN partitions_used = 0 THEN 'NO_DATA'
                    WHEN empty_partitions > 50 THEN 'WARNING'
                    WHEN stddev_rows / NULLIF(avg_rows_per_partition, 0) > 2 THEN 'REBALANCE NEEDED'
                    ELSE 'HEALTHY'
                END AS distribution_status
            FROM stats;
        """
            )
        )

        # Create compute_partition_key function for trigger
        await conn.execute(
            text(
                """
            CREATE OR REPLACE FUNCTION compute_partition_key()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """
            )
        )

        # Create get_partition_key helper function
        await conn.execute(
            text(
                """
            CREATE OR REPLACE FUNCTION get_partition_key(collection_id VARCHAR)
            RETURNS INTEGER AS $$
            BEGIN
                RETURN abs(hashtext(collection_id::text)) % 100;
            END;
            $$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
        """
            )
        )

        # Create the chunks table if it doesn't exist (LIST partitioned by partition_key)
        await conn.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS chunks (
                id BIGSERIAL,
                collection_id VARCHAR NOT NULL,
                partition_key INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB DEFAULT '{}',
                document_id VARCHAR,
                chunking_config_id INTEGER,
                start_offset INTEGER,
                end_offset INTEGER,
                token_count INTEGER,
                embedding_vector_id VARCHAR,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (id, collection_id, partition_key)
            ) PARTITION BY LIST (partition_key);
        """
            )
        )

        # Create trigger to auto-compute partition_key
        await conn.execute(
            text(
                """
            DROP TRIGGER IF EXISTS set_partition_key ON chunks;
            CREATE TRIGGER set_partition_key
            BEFORE INSERT ON chunks
            FOR EACH ROW
            EXECUTE FUNCTION compute_partition_key();
        """
            )
        )

        # Create partitions if they don't exist
        for i in range(100):
            partition_name = f"chunks_part_{i:02d}"
            await conn.execute(
                text(
                    f"""
                CREATE TABLE IF NOT EXISTS {partition_name}
                PARTITION OF chunks
                FOR VALUES IN ({i});
            """
                )
            )
    except Exception as e:
        await engine.dispose()
        pytest.skip(
            f"Database setup failed: {e}\nMake sure migrations are applied with: poetry run alembic upgrade head"
        )

    # Create a new session for the test
    try:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            yield session
            await session.rollback()
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def test_db() -> AsyncIterator[AsyncSession]:
    """Alias for db_session for compatibility."""
    async for session in db_session():
        yield session
