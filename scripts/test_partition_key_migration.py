#!/usr/bin/env python3
"""
Test script for partition key migration (DB-003).

This script tests the migration from trigger-based to GENERATED column
implementation for partition_key computation.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config import Settings
from shared.database.partition_utils import PartitionImplementationDetector
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_partition_implementation():
    """Test the partition key implementation and generate a health report."""

    # Get database URL from settings
    settings = Settings()
    database_url = settings.database_url

    # Convert to async URL if needed
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
    elif database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+asyncpg://")

    logger.info("Connecting to database...")

    # Create async engine
    engine = create_async_engine(
        database_url,
        echo=False,
        pool_pre_ping=True,
    )

    # Create session factory
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session() as session:
            logger.info("=" * 60)
            logger.info("PARTITION KEY MIGRATION TEST")
            logger.info("=" * 60)

            # 1. Detect current implementation
            logger.info("\n1. Detecting current implementation...")
            impl = await PartitionImplementationDetector.detect_implementation(session)

            logger.info(f"   PostgreSQL Version: {impl['postgres_version']}")
            logger.info(f"   Current Method: {impl['method'].upper()}")
            logger.info(f"   Has Trigger: {impl['has_trigger']}")
            logger.info(f"   Has Generated Column: {impl['has_generated_column']}")
            logger.info(f"   Supports Generated: {impl['supports_generated']}")
            logger.info(f"   Is Optimal: {impl['is_optimal']}")

            if impl["performance_impact"]:
                logger.info(f"   Performance Impact: {impl['performance_impact']}")

            # 2. Verify partition keys
            logger.info("\n2. Verifying partition keys...")
            verification = await PartitionImplementationDetector.verify_partition_keys(session, sample_size=500)

            logger.info(f"   Checked: {verification['checked']} records")
            logger.info(f"   Correct: {verification['correct']}")
            logger.info(f"   Incorrect: {verification['incorrect']}")
            logger.info(f"   Valid: {verification['is_valid']}")

            if verification["incorrect"] > 0:
                logger.warning("   Sample errors:")
                for err in verification["errors"][:5]:
                    if "error" not in err:
                        logger.warning(
                            f"     - Collection {err['collection_id']}: "
                            f"stored={err['stored']}, expected={err['expected']}"
                        )

            # 3. Get performance metrics
            logger.info("\n3. Analyzing performance metrics...")
            metrics = await PartitionImplementationDetector.get_performance_metrics(session)

            logger.info(f"   Total Chunks: {metrics['total_chunks']:,}")
            logger.info(f"   Active Partitions: {metrics['partition_count']}/100")
            logger.info(f"   Empty Partitions: {metrics['empty_partitions']}")

            if metrics["partition_count"] > 0:
                logger.info(f"   Avg Chunks/Partition: {metrics['avg_chunks_per_partition']:.1f}")
                logger.info(f"   Max Chunks: {metrics['max_chunks_in_partition']:,}")
                logger.info(f"   Min Chunks: {metrics['min_chunks_in_partition']:,}")

            if metrics["hot_partitions"]:
                logger.info(f"   Hot Partitions: {len(metrics['hot_partitions'])}")
                for hot in metrics["hot_partitions"][:3]:
                    logger.info(
                        f"     - Partition {hot['partition_key']}: "
                        f"{hot['chunk_count']:,} chunks ({hot['ratio_to_avg']:.1f}x avg)"
                    )

            # 4. Test insert performance (if optimal)
            if impl["is_optimal"] and impl["has_generated_column"]:
                logger.info("\n4. Testing GENERATED column performance...")

                # Create test collection
                await session.execute(
                    text(
                        """
                    INSERT INTO collections (id, name, description, created_at, updated_at)
                    VALUES ('perf-test-00000000-0000-0000-0000-000000000001',
                            'Performance Test', 'Testing GENERATED column performance',
                            NOW(), NOW())
                    ON CONFLICT (id) DO NOTHING
                """
                    )
                )
                await session.commit()

                import time

                # Test batch insert
                start_time = time.time()

                await session.execute(
                    text(
                        """
                    INSERT INTO chunks (
                        collection_id, chunk_index, content,
                        metadata, created_at, updated_at
                    )
                    SELECT
                        'perf-test-00000000-0000-0000-0000-000000000001',
                        generate_series,
                        'Performance test content ' || generate_series,
                        '{}',
                        NOW(),
                        NOW()
                    FROM generate_series(1, 1000)
                """
                    )
                )
                await session.commit()

                elapsed = (time.time() - start_time) * 1000
                per_row = elapsed / 1000

                logger.info(f"   Insert Performance: {elapsed:.2f}ms for 1000 rows")
                logger.info(f"   Average: {per_row:.3f}ms per row")

                # Clean up
                await session.execute(
                    text(
                        """
                    DELETE FROM chunks
                    WHERE collection_id = 'perf-test-00000000-0000-0000-0000-000000000001'
                """
                    )
                )
                await session.commit()

            # 5. Generate full health report
            logger.info("\n5. Generating comprehensive health report...")
            report = await PartitionImplementationDetector.generate_health_report(session)

            print("\n" + report)

            # 6. Recommendations
            if impl["recommendation"]:
                logger.info("\n" + "=" * 60)
                logger.info("ACTION REQUIRED:")
                logger.info("=" * 60)
                for line in impl["recommendation"].split(". "):
                    if line:
                        logger.info(f"  • {line.strip()}{'.' if not line.endswith('.') else ''}")
                logger.info("=" * 60)
            else:
                logger.info("\n✅ Partition key implementation is optimal!")

    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        return 1
    finally:
        await engine.dispose()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(test_partition_implementation())
    sys.exit(exit_code)
