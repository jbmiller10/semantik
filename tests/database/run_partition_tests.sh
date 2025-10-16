#!/bin/bash
# Script to run partition tests when database is available

echo "=================================="
echo "Running Partition Tests"
echo "=================================="

# Set required environment variables if not set
export POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-"postgres"}
export DATABASE_URL=${DATABASE_URL:-"postgresql+asyncpg://postgres:postgres@localhost:5432/semantik"}

echo "Using DATABASE_URL: ${DATABASE_URL}"
echo ""

# Run the migration
echo "Applying migration..."
uv run alembic upgrade head

if [ $? -ne 0 ]; then
    echo "❌ Migration failed!"
    exit 1
fi

echo "✅ Migration applied successfully"
echo ""

# Run the tests
echo "Running partition tests..."
uv run pytest tests/database/test_partitioning.py -v

if [ $? -ne 0 ]; then
    echo "❌ Some tests failed!"
    exit 1
fi

echo ""
echo "Running migration tests..."
uv run pytest tests/database/test_migration_100_partitions.py -v

if [ $? -ne 0 ]; then
    echo "❌ Some migration tests failed!"
    exit 1
fi

echo ""
echo "=================================="
echo "✅ All partition tests passed!"
echo "=================================="

# Query partition health
echo ""
echo "Checking partition health..."
uv run python -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from packages.shared.chunking.infrastructure.repositories.partition_manager import PartitionManager

async def check_health():
    engine = create_async_engine('${DATABASE_URL}')
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        manager = PartitionManager()
        
        # Get distribution stats
        stats = await manager.get_distribution_stats(session)
        print(f'Partitions used: {stats.partitions_used}/{manager.PARTITION_COUNT}')
        print(f'Distribution status: {stats.distribution_status}')
        print(f'Max skew ratio: {stats.max_skew_ratio:.2f}')
        
        # Get efficiency report
        report = await manager.get_efficiency_report(session)
        print(f'Efficiency score: {report[\"efficiency_score\"]}/100')
        
    await engine.dispose()

asyncio.run(check_health())
"
