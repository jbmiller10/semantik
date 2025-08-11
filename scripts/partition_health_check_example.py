#!/usr/bin/env python3
"""
Example of integrating partition health checks into a service or API endpoint.

This demonstrates how to use the PartitionImplementationDetector to monitor
the health of the partition key implementation in production.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database import get_db
from packages.shared.database.partition_utils import PartitionImplementationDetector

# Example API router for health checks
router = APIRouter(prefix="/admin/partition", tags=["admin", "monitoring"])


@router.get("/health")
async def get_partition_health(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Get partition implementation health status.

    This endpoint provides detailed information about the partition key
    implementation, including whether it's using the optimal method for
    the current PostgreSQL version.

    Returns:
        Health status including implementation method, performance impact,
        and recommendations.
    """
    try:
        # Get implementation details
        impl = await PartitionImplementationDetector.detect_implementation(db)

        # Get verification results
        verification = await PartitionImplementationDetector.verify_partition_keys(db, sample_size=100)

        # Get performance metrics
        metrics = await PartitionImplementationDetector.get_performance_metrics(db)

        # Build response
        response = {
            "status": "healthy" if impl["is_optimal"] and verification["is_valid"] else "degraded",
            "postgres_version": impl["postgres_version"],
            "implementation": {
                "method": impl["method"],
                "is_optimal": impl["is_optimal"],
                "has_trigger": impl["has_trigger"],
                "has_generated_column": impl["has_generated_column"],
                "performance_impact": impl["performance_impact"],
            },
            "data_integrity": {
                "is_valid": verification["is_valid"],
                "checked": verification["checked"],
                "correct": verification["correct"],
                "incorrect": verification["incorrect"],
            },
            "performance": {
                "total_chunks": metrics["total_chunks"],
                "active_partitions": metrics["partition_count"],
                "empty_partitions": metrics["empty_partitions"],
                "avg_chunks_per_partition": (
                    round(metrics["avg_chunks_per_partition"], 1) if metrics["partition_count"] > 0 else 0
                ),
                "hot_partitions_count": len(metrics["hot_partitions"]),
            },
            "recommendation": impl["recommendation"],
        }

        # Add warnings if there are issues
        warnings = []

        if not impl["is_optimal"]:
            warnings.append(f"Suboptimal implementation: {impl['performance_impact']}")

        if not verification["is_valid"]:
            warnings.append(f"Data integrity issues: {verification['incorrect']} incorrect partition keys")

        if metrics["hot_partitions"]:
            warnings.append(f"{len(metrics['hot_partitions'])} hot partitions detected (uneven distribution)")

        if warnings:
            response["warnings"] = warnings

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check partition health: {str(e)}") from e


@router.get("/health/report")
async def get_partition_health_report(
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """
    Get a comprehensive partition health report.

    This endpoint generates a detailed text report suitable for logging
    or monitoring systems.

    Returns:
        Formatted health report as text.
    """
    try:
        report = await PartitionImplementationDetector.generate_health_report(db)

        return {
            "report": report,
            "generated_at": "now",  # You could add proper timestamp here
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate health report: {str(e)}") from e


@router.post("/verify")
async def verify_partition_keys(
    sample_size: int = 1000,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Verify partition key correctness.

    This endpoint checks a sample of partition keys to ensure they're
    correctly computed.

    Args:
        sample_size: Number of records to check (default: 1000, max: 10000)

    Returns:
        Verification results including any errors found.
    """
    # Limit sample size to prevent excessive load
    sample_size = min(sample_size, 10000)

    try:
        verification = await PartitionImplementationDetector.verify_partition_keys(db, sample_size=sample_size)

        return {
            "is_valid": verification["is_valid"],
            "sample_size": verification["checked"],
            "correct": verification["correct"],
            "incorrect": verification["incorrect"],
            "error_samples": verification["errors"][:10] if verification["errors"] else [],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify partition keys: {str(e)}") from e


# Example of a background task that could run periodically
async def periodic_partition_health_check(db: AsyncSession) -> None:
    """
    Background task to periodically check partition health.

    This could be scheduled to run daily or weekly to monitor the
    partition implementation and alert if issues are detected.
    """
    import logging

    logger = logging.getLogger(__name__)

    try:
        # Check implementation
        impl = await PartitionImplementationDetector.detect_implementation(db)

        if not impl["is_optimal"]:
            logger.warning(
                f"Suboptimal partition implementation detected: {impl['method']}. "
                f"Performance impact: {impl['performance_impact']}"
            )

            if impl["recommendation"]:
                logger.info(f"Recommendation: {impl['recommendation']}")

        # Verify data integrity
        verification = await PartitionImplementationDetector.verify_partition_keys(db, sample_size=500)

        if not verification["is_valid"]:
            logger.error(
                f"Partition key integrity check failed! "
                f"{verification['incorrect']} incorrect keys out of {verification['checked']}"
            )

            # Log some examples
            for err in verification["errors"][:5]:
                if "error" not in err:
                    logger.error(
                        f"  Collection {err['collection_id']}: stored={err['stored']}, expected={err['expected']}"
                    )

        # Check for hot partitions
        metrics = await PartitionImplementationDetector.get_performance_metrics(db)

        if metrics["hot_partitions"]:
            logger.warning(f"Detected {len(metrics['hot_partitions'])} hot partitions with uneven data distribution")

            for hot in metrics["hot_partitions"][:3]:
                logger.info(
                    f"  Partition {hot['partition_key']}: "
                    f"{hot['chunk_count']} chunks ({hot['ratio_to_avg']:.1f}x average)"
                )

        # Log success if everything is optimal
        if impl["is_optimal"] and verification["is_valid"] and not metrics["hot_partitions"]:
            logger.info("Partition health check passed: All systems optimal")

    except Exception as e:
        logger.error(f"Partition health check failed: {e}", exc_info=True)


# Example Celery task (if using Celery for background tasks)
def create_celery_task(celery_app):
    """
    Create a Celery task for periodic partition health checks.

    Usage:
        from celery import Celery
        app = Celery('tasks')
        check_partition_health = create_celery_task(app)

        # Schedule to run daily
        app.conf.beat_schedule = {
            'check-partition-health': {
                'task': 'check_partition_health',
                'schedule': crontab(hour=2, minute=0),  # Run at 2 AM daily
            },
        }
    """
    from packages.shared.database import get_async_session

    @celery_app.task(name="check_partition_health")
    def check_partition_health():
        """Celery task to check partition health."""
        import asyncio

        async def run_check():
            async with get_async_session() as db:
                await periodic_partition_health_check(db)

        asyncio.run(run_check())

    return check_partition_health


# Example CLI command (if using Click or similar)
def create_cli_command():
    """
    Create a CLI command for partition health checks.

    Usage:
        import click

        @click.command()
        @click.option('--sample-size', default=1000, help='Number of records to verify')
        @click.option('--report', is_flag=True, help='Generate full report')
        def check_partitions(sample_size, report):
            '''Check partition key implementation health.'''
            command = create_cli_command()
            command(sample_size, report)
    """
    import asyncio

    from packages.shared.database import get_async_session

    def command(sample_size: int, report: bool):
        async def run():
            async with get_async_session() as db:
                if report:
                    health_report = await PartitionImplementationDetector.generate_health_report(db)
                    print(health_report)
                else:
                    impl = await PartitionImplementationDetector.detect_implementation(db)
                    verification = await PartitionImplementationDetector.verify_partition_keys(
                        db, sample_size=sample_size
                    )

                    print(f"Implementation: {impl['method']}")
                    print(f"Is Optimal: {impl['is_optimal']}")
                    print(f"Data Valid: {verification['is_valid']}")

                    if impl["recommendation"]:
                        print(f"\nRecommendation: {impl['recommendation']}")

        asyncio.run(run())

    return command


if __name__ == "__main__":
    # Example: Run the health check as a standalone script
    import asyncio
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from packages.shared.database import get_async_session

    async def main():
        async with get_async_session() as db:
            report = await PartitionImplementationDetector.generate_health_report(db)
            print(report)

    asyncio.run(main())
