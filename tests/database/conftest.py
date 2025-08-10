"""Database test fixtures and configuration."""

import os
import sys
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
            # Check if the migration has been applied correctly
            # The migration should have created these views and functions
            # We'll just verify they exist, not recreate them

            # Verify partition_health view exists
            result = await conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM pg_views
                    WHERE viewname = 'partition_health'
                    """
                )
            )
            if result.scalar() == 0:
                # Only create if it doesn't exist (for backward compatibility)
                await conn.execute(
                    text(
                        """
                        CREATE OR REPLACE VIEW partition_health AS
                        WITH partition_stats AS (
                            SELECT
                                schemaname,
                                relname as partition_name,
                                SUBSTRING(relname FROM 'chunks_part_([0-9]+)')::INT as partition_id,
                                pg_total_relation_size(schemaname||'.'||relname) as size_bytes,
                                n_live_tup as row_count,
                                n_dead_tup as dead_rows,
                                last_vacuum,
                                last_autovacuum,
                                n_tup_ins as inserts_since_vacuum,
                                n_tup_upd as updates_since_vacuum,
                                n_tup_del as deletes_since_vacuum
                            FROM pg_stat_user_tables
                            WHERE relname LIKE 'chunks_part_%'
                        ),
                        stats_summary AS (
                            SELECT
                                AVG(row_count) as avg_rows,
                                MAX(row_count) as max_rows,
                                MIN(row_count) as min_rows,
                                STDDEV(row_count) as stddev_rows,
                                AVG(size_bytes) as avg_size,
                                SUM(row_count) as total_rows,
                                SUM(size_bytes) as total_size
                            FROM partition_stats
                        )
                        SELECT
                            ps.*,
                            pg_size_pretty(ps.size_bytes) as size_pretty,
                            ROUND((ps.row_count::NUMERIC / NULLIF(ss.avg_rows, 0) - 1) * 100, 2) as pct_deviation_from_avg,
                            CASE
                                WHEN ss.avg_rows > 0 AND ps.row_count > ss.avg_rows * 1.2 THEN 'HOT'
                                WHEN ss.avg_rows > 0 AND ps.row_count < ss.avg_rows * 0.8 THEN 'COLD'
                                ELSE 'NORMAL'
                            END as partition_status,
                            ps.dead_rows > ps.row_count * 0.1 as needs_vacuum
                        FROM partition_stats ps
                        CROSS JOIN stats_summary ss
                        ORDER BY partition_id;
                        """
                    )
                )

            # Verify partition_distribution view exists
            result = await conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM pg_views
                    WHERE viewname = 'partition_distribution'
                    """
                )
            )
            if result.scalar() == 0:
                await conn.execute(
                    text(
                        """
                        CREATE OR REPLACE VIEW partition_distribution AS
                        WITH partition_counts AS (
                            SELECT
                                abs(hashtext(collection_id::text)) % 100 as partition_id,
                                COUNT(DISTINCT collection_id) as collection_count,
                                COUNT(*) as chunk_count
                            FROM chunks
                            GROUP BY abs(hashtext(collection_id::text)) % 100
                        ),
                        distribution_stats AS (
                            SELECT
                                COUNT(*) as partitions_used,
                                AVG(chunk_count) as avg_chunks_per_partition,
                                STDDEV(chunk_count) as stddev_chunks,
                                MAX(chunk_count) as max_chunks,
                                MIN(chunk_count) as min_chunks,
                                MAX(chunk_count)::FLOAT / NULLIF(AVG(chunk_count), 0) as max_skew_ratio
                            FROM partition_counts
                        )
                        SELECT
                            ds.*,
                            CASE
                                WHEN max_skew_ratio > 1.2 THEN 'REBALANCE NEEDED'
                                WHEN max_skew_ratio > 1.1 THEN 'WARNING'
                                ELSE 'HEALTHY'
                            END as distribution_status,
                            100 - partitions_used as empty_partitions
                        FROM distribution_stats ds;
                        """
                    )
                )

            # Check if get_partition_for_collection function exists
            result = await conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM pg_proc
                    WHERE proname = 'get_partition_for_collection'
                    """
                )
            )
            if result.scalar() == 0:
                await conn.execute(
                    text(
                        """
                        CREATE OR REPLACE FUNCTION get_partition_for_collection(collection_id VARCHAR)
                        RETURNS TEXT AS $$
                        BEGIN
                            -- Ensure partition_key is always positive (0-99)
                            RETURN 'chunks_part_' || LPAD((abs(hashtext(collection_id::text)) % 100)::text, 2, '0');
                        END;
                        $$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
                        """
                    )
                )

            # Check if get_partition_key function exists
            result = await conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM pg_proc
                    WHERE proname = 'get_partition_key'
                    """
                )
            )
            if result.scalar() == 0:
                await conn.execute(
                    text(
                        """
                        CREATE OR REPLACE FUNCTION get_partition_key(collection_id VARCHAR)
                        RETURNS INTEGER AS $$
                        BEGIN
                            -- Ensure partition_key is always positive (0-99)
                            RETURN abs(hashtext(collection_id::text)) % 100;
                        END;
                        $$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
                        """
                    )
                )
            
            # Check if compute_partition_key function exists (for trigger)
            result = await conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM pg_proc
                    WHERE proname = 'compute_partition_key'
                    """
                )
            )
            if result.scalar() == 0:
                await conn.execute(
                    text(
                        """
                        CREATE OR REPLACE FUNCTION compute_partition_key()
                        RETURNS TRIGGER AS $$
                        BEGIN
                            -- Ensure partition_key is always positive (0-99)
                            NEW.partition_key := abs(hashtext(NEW.collection_id::text)) % 100;
                            RETURN NEW;
                        END;
                        $$ LANGUAGE plpgsql;
                        """
                    )
                )
            
            # Check if trigger exists on chunks table
            result = await conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM pg_trigger
                    WHERE tgname = 'set_partition_key'
                    AND tgrelid = 'chunks'::regclass
                    """
                )
            )
            if result.scalar() == 0:
                await conn.execute(
                    text(
                        """
                        CREATE TRIGGER set_partition_key
                        BEFORE INSERT ON chunks
                        FOR EACH ROW
                        EXECUTE FUNCTION compute_partition_key();
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
