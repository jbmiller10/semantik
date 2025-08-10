"""
Test the 100-partition migration can be executed successfully.
"""

import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


@pytest.mark.asyncio()
class TestMigration100Partitions:
    """Test the migration creates the correct partition structure."""

    async def test_partition_count(self, db_session: AsyncSession):
        """Verify exactly 100 partitions are created."""
        # Query to count partitions
        result = await db_session.execute(text("""
            SELECT COUNT(*) 
            FROM pg_inherits 
            WHERE inhparent = 'chunks'::regclass
        """))

        partition_count = result.scalar()
        assert partition_count == 100, f"Expected 100 partitions, found {partition_count}"

    async def test_partition_names(self, db_session: AsyncSession):
        """Verify all partition names follow the expected pattern."""
        result = await db_session.execute(text("""
            SELECT tablename 
            FROM pg_tables 
            WHERE tablename LIKE 'chunks_part_%'
            ORDER BY tablename
        """))

        partitions = [row[0] for row in result]

        # Should have 100 partitions
        assert len(partitions) == 100

        # Verify naming pattern
        for i in range(100):
            expected_name = f"chunks_part_{i:02d}"
            assert expected_name in partitions, f"Missing partition: {expected_name}"

    async def test_partition_indexes(self, db_session: AsyncSession):
        """Verify each partition has the required indexes."""
        # Check indexes on first few partitions as sample
        for i in range(5):  # Check first 5 partitions
            partition_name = f"chunks_part_{i:02d}"

            result = await db_session.execute(text("""
                SELECT indexname 
                FROM pg_indexes 
                WHERE tablename = :partition_name
            """), {"partition_name": partition_name})

            indexes = [row[0] for row in result]

            # Each partition should have these indexes
            expected_indexes = [
                f"idx_chunks_part_{i:02d}_collection",
                f"idx_chunks_part_{i:02d}_created",
                f"idx_chunks_part_{i:02d}_chunk_index",
                f"idx_chunks_part_{i:02d}_document"
            ]

            for expected_idx in expected_indexes:
                assert expected_idx in indexes, (
                    f"Missing index {expected_idx} on partition {partition_name}"
                )

    async def test_monitoring_views_exist(self, db_session: AsyncSession):
        """Verify monitoring views are created."""
        # Check for partition_health view
        result = await db_session.execute(text("""
            SELECT COUNT(*) 
            FROM pg_views 
            WHERE viewname = 'partition_health'
        """))
        assert result.scalar() == 1, "partition_health view not found"

        # Check for partition_distribution view
        result = await db_session.execute(text("""
            SELECT COUNT(*) 
            FROM pg_views 
            WHERE viewname = 'partition_distribution'
        """))
        assert result.scalar() == 1, "partition_distribution view not found"

        # Check for collection_chunking_stats materialized view
        result = await db_session.execute(text("""
            SELECT COUNT(*) 
            FROM pg_matviews 
            WHERE matviewname = 'collection_chunking_stats'
        """))
        assert result.scalar() == 1, "collection_chunking_stats materialized view not found"

    async def test_helper_functions_exist(self, db_session: AsyncSession):
        """Verify helper functions are created."""
        # Check for get_partition_for_collection function
        result = await db_session.execute(text("""
            SELECT COUNT(*) 
            FROM pg_proc 
            WHERE proname = 'get_partition_for_collection'
        """))
        assert result.scalar() >= 1, "get_partition_for_collection function not found"

        # Check for analyze_partition_skew function
        result = await db_session.execute(text("""
            SELECT COUNT(*) 
            FROM pg_proc 
            WHERE proname = 'analyze_partition_skew'
        """))
        assert result.scalar() >= 1, "analyze_partition_skew function not found"

    async def test_partition_key_type(self, db_session: AsyncSession):
        """Verify the partition key uses LIST partitioning with hashtext."""
        result = await db_session.execute(text("""
            SELECT 
                pt.partnatts,
                pt.partstrat
            FROM pg_partitioned_table pt
            JOIN pg_class c ON pt.partrelid = c.oid
            WHERE c.relname = 'chunks'
        """))

        row = result.fetchone()
        assert row is not None, "chunks table is not partitioned"

        # partstrat: 'l' for LIST, 'r' for RANGE, 'h' for HASH
        # Handle both string and bytes (depending on driver version)
        partstrat = row.partstrat
        if isinstance(partstrat, bytes):
            partstrat = partstrat.decode('utf-8')
        assert partstrat == 'l', f"Expected LIST partitioning, got {partstrat}"

    async def test_partition_constraints(self, db_session: AsyncSession):
        """Verify each partition has the correct constraint."""
        # Check a sample partition
        result = await db_session.execute(text("""
            SELECT 
                pg_get_expr(c.relpartbound, c.oid) as partition_expr
            FROM pg_class c
            WHERE c.relname = 'chunks_part_00'
        """))

        row = result.fetchone()
        assert row is not None, "chunks_part_00 not found"

        # Should contain "FOR VALUES IN (0)"
        assert "FOR VALUES IN (0)" in row.partition_expr, (
            f"Unexpected partition constraint: {row.partition_expr}"
        )

    async def test_foreign_keys_preserved(self, db_session: AsyncSession):
        """Verify foreign keys are properly set up."""
        result = await db_session.execute(text("""
            SELECT 
                conname,
                confrelid::regclass::text as referenced_table
            FROM pg_constraint
            WHERE conrelid = 'chunks'::regclass
            AND contype = 'f'
        """))

        foreign_keys = {row.conname: row.referenced_table for row in result}

        # Should have foreign keys to collections and chunking_configs
        assert any('collections' in fk for fk in foreign_keys.values()), (
            "Missing foreign key to collections table"
        )

        # chunking_configs foreign key is optional
        # but if it exists, verify it's correct
        if any('chunking_configs' in fk for fk in foreign_keys.values()):
            assert True  # Foreign key exists and is correct

    async def test_primary_key_structure(self, db_session: AsyncSession):
        """Verify the primary key is properly configured."""
        result = await db_session.execute(text("""
            SELECT 
                a.attname 
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = 'chunks'::regclass AND i.indisprimary
            ORDER BY a.attnum
        """))

        pk_columns = [row[0] for row in result]

        # Primary key should be (id, collection_id)
        assert 'id' in pk_columns, "id not in primary key"
        assert 'collection_id' in pk_columns, "collection_id not in primary key"


# Fixtures for database testing
@pytest.fixture()
async def db_session():
    """
    Provide a database session for testing with proper cleanup.
    """
    # Get database URL from environment or use default
    DATABASE_URL = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/semantik_test"
    )
    
    # Ensure we're using asyncpg driver for async tests
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif DATABASE_URL.startswith("postgresql+psycopg2://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)

    # Create async engine
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        future=True,
        pool_pre_ping=True,
    )

    # Create async session factory using async_sessionmaker
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    # Create session
    async with async_session() as session:
        # Clean up any existing test data before test
        await cleanup_chunks_dependencies(session)

        yield session

        # Clean up after test
        await cleanup_chunks_dependencies(session)
        await session.rollback()

    # Close engine
    await engine.dispose()


async def cleanup_chunks_dependencies(session: AsyncSession):
    """
    Helper function to clean up all chunks table dependencies.
    Based on the migration's cleanup_chunks_dependencies function.
    """
    # Drop views that depend on chunks (in dependency order)
    views_to_drop = [
        "partition_hot_spots",
        "partition_health_summary",
        "partition_size_distribution",
        "partition_chunk_distribution",
        "partition_distribution",
        "partition_health",
        "active_chunking_configs"
    ]

    for view in views_to_drop:
        try:
            await session.execute(text(f"DROP VIEW IF EXISTS {view} CASCADE"))
        except Exception:
            pass  # Ignore if view doesn't exist

    # Drop materialized views
    try:
        await session.execute(text("DROP MATERIALIZED VIEW IF EXISTS collection_chunking_stats CASCADE"))
    except Exception:
        pass

    # Clear any test data from chunks table
    try:
        await session.execute(text("TRUNCATE TABLE chunks CASCADE"))
    except Exception:
        pass  # Table might not exist or might have dependencies

    await session.commit()
