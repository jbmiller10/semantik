"""API endpoints for partition monitoring.

This module provides endpoints to monitor partition health,
view statistics, and get rebalancing recommendations.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.database import get_db
from packages.webui.dependencies import require_admin_or_internal_key
from packages.webui.services.partition_monitoring_service import PartitionMonitoringService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2/partitions",
    tags=["partition-monitoring"],
)


@router.get("/health", summary="Get partition health status")
async def get_partition_health(
    _: None = Depends(require_admin_or_internal_key),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get current partition health status.

    Returns comprehensive health information including:
    - Health status for each partition
    - Skew metrics
    - Alerts for any issues
    - Recommendations for optimization
    """
    try:
        service = PartitionMonitoringService(db)
        result = await service.check_partition_health()

        return {
            "status": result.status,
            "timestamp": result.timestamp,
            "alerts": result.alerts,
            "metrics": result.metrics,
            "error": result.error,
        }

    except Exception as e:
        logger.error(f"Failed to get partition health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve partition health") from e


@router.get("/statistics", summary="Get partition statistics")
async def get_partition_statistics(
    partition_num: int | None = None,
    _: None = Depends(require_admin_or_internal_key),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get detailed statistics for partitions.

    Args:
        partition_num: Specific partition number, or None for aggregate statistics

    Returns:
        Partition statistics including chunk counts, sizes, and distribution metrics
    """
    try:
        service = PartitionMonitoringService(db)
        stats = await service.get_partition_statistics(partition_num)

        if partition_num is not None and not stats:
            raise HTTPException(status_code=404, detail=f"Partition {partition_num} not found")

        return {"partition_num": partition_num, "statistics": stats}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get partition statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve partition statistics") from e


@router.get("/recommendations", summary="Get rebalancing recommendations")
async def get_rebalancing_recommendations(
    _: None = Depends(require_admin_or_internal_key),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get recommendations for partition rebalancing.

    Returns recommendations based on current partition distribution and skew metrics.
    """
    try:
        service = PartitionMonitoringService(db)
        recommendations = await service.get_rebalancing_recommendations()

        return {
            "count": len(recommendations),
            "recommendations": recommendations,
        }

    except Exception as e:
        logger.error(f"Failed to get rebalancing recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve rebalancing recommendations") from e


@router.get("/health-summary", summary="Get partition health summary")
async def get_partition_health_summary(
    _: None = Depends(require_admin_or_internal_key),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get a summary of partition health across all partitions.

    Returns a simplified view of partition health suitable for dashboards.
    """
    try:
        service = PartitionMonitoringService(db)
        health_data = await service.get_partition_health_summary()

        # Calculate summary metrics
        total_partitions = len(health_data)
        healthy_count = sum(1 for h in health_data if h.health_status.value == "HEALTHY")
        warning_count = sum(1 for h in health_data if h.health_status.value == "WARNING")
        unbalanced_count = sum(1 for h in health_data if h.health_status.value == "UNBALANCED")

        return {
            "total_partitions": total_partitions,
            "healthy_count": healthy_count,
            "warning_count": warning_count,
            "unbalanced_count": unbalanced_count,
            "health_percentage": (healthy_count / total_partitions * 100) if total_partitions > 0 else 0,
            "partitions": [
                {
                    "partition_num": h.partition_num,
                    "health_status": h.health_status.value,
                    "chunk_percentage": h.chunk_percentage,
                    "chunk_skew": h.chunk_skew,
                }
                for h in health_data
            ],
        }

    except Exception as e:
        logger.error(f"Failed to get partition health summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve partition health summary") from e
