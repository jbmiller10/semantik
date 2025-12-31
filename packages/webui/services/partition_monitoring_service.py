"""Service for monitoring partition health and performance.

This service provides methods to analyze partition distribution,
detect imbalances, and generate recommendations for optimization.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

UTC = UTC


class PartitionHealthStatus(Enum):
    """Health status for partitions."""

    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    UNBALANCED = "UNBALANCED"
    CRITICAL = "CRITICAL"


class SkewStatus(Enum):
    """Status for skew metrics."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class PartitionHealth:
    """Health information for a single partition."""

    partition_num: int
    chunk_count: int
    total_chunks: int
    chunk_percentage: float
    size_percentage: float
    health_status: PartitionHealthStatus
    chunk_skew: float
    size_skew: float
    recommendation: str | None = None


@dataclass
class SkewMetric:
    """Skew analysis metric."""

    metric: str
    value: float
    status: SkewStatus
    details: str


@dataclass
class MonitoringResult:
    """Result of partition monitoring."""

    status: str
    timestamp: str
    alerts: list[dict[str, Any]]
    metrics: dict[str, Any]
    error: str | None = None


class PartitionMonitoringService:
    """Service for monitoring partition health and performance."""

    # Thresholds for health status
    SKEW_WARNING_THRESHOLD = 0.3  # 30% deviation
    SKEW_CRITICAL_THRESHOLD = 0.5  # 50% deviation
    REBALANCE_THRESHOLD = 0.4  # 40% deviation triggers rebalance recommendation

    def __init__(self, session: AsyncSession):
        """Initialize the service with a database session.

        Args:
            session: AsyncSession for database operations
        """
        self.session = session

    async def get_partition_health_summary(self) -> list[PartitionHealth]:
        """Get health summary for all partitions.

        Returns:
            List of PartitionHealth objects

        Raises:
            Exception: If health summary cannot be retrieved
        """
        try:
            result = await self.session.execute(
                text(
                    """
                    SELECT
                        partition_num,
                        chunk_count,
                        total_chunks,
                        chunk_percentage,
                        size_percentage,
                        health_status,
                        chunk_skew,
                        size_skew,
                        recommendation
                    FROM partition_health_summary
                    ORDER BY chunk_skew DESC
                    """
                )
            )

            health_data = []
            for row in result:
                health_data.append(
                    PartitionHealth(
                        partition_num=row.partition_num,
                        chunk_count=row.chunk_count,
                        total_chunks=row.total_chunks,
                        chunk_percentage=float(row.chunk_percentage),
                        size_percentage=float(row.size_percentage),
                        health_status=PartitionHealthStatus(row.health_status),
                        chunk_skew=float(row.chunk_skew),
                        size_skew=float(row.size_skew),
                        recommendation=row.recommendation,
                    )
                )

            return health_data

        except Exception as e:
            logger.error("Failed to get partition health summary: %s", e, exc_info=True)
            raise

    async def analyze_partition_skew(self) -> list[SkewMetric]:
        """Analyze partition skew metrics.

        Returns:
            List of SkewMetric objects
        """
        try:
            result = await self.session.execute(text("SELECT * FROM analyze_partition_skew()"))

            metrics = []
            for row in result:
                metrics.append(
                    SkewMetric(
                        metric=row.metric,
                        value=float(row.value),
                        status=SkewStatus(row.status),
                        details=row.details,
                    )
                )

            return metrics

        except Exception as e:
            logger.error("Failed to analyze partition skew: %s", e, exc_info=True)
            raise

    async def check_partition_health(self) -> MonitoringResult:
        """Perform comprehensive partition health check.

        Returns:
            MonitoringResult with health status and any alerts
        """
        monitoring_result = MonitoringResult(
            status="success",
            timestamp=datetime.now(UTC).isoformat(),
            alerts=[],
            metrics={},
        )

        try:
            # Get health summary
            health_data = await self.get_partition_health_summary()

            # Get skew analysis
            skew_metrics = await self.analyze_partition_skew()

            # Process health data
            unbalanced_count = 0
            warning_count = 0
            critical_partitions = []

            for health in health_data:
                if health.health_status == PartitionHealthStatus.UNBALANCED:
                    unbalanced_count += 1
                    critical_partitions.append(
                        {
                            "partition": health.partition_num,
                            "chunk_percentage": health.chunk_percentage,
                            "size_percentage": health.size_percentage,
                            "recommendation": health.recommendation,
                        }
                    )
                elif health.health_status == PartitionHealthStatus.WARNING:
                    warning_count += 1

            # Process skew metrics
            skew_metrics_dict = {}
            for metric in skew_metrics:
                skew_metrics_dict[metric.metric] = {
                    "value": metric.value,
                    "status": metric.status.value,
                    "details": metric.details,
                }

            # Generate alerts
            if unbalanced_count > 0:
                monitoring_result.alerts.append(
                    {
                        "level": "ERROR",
                        "message": f"{unbalanced_count} partitions are severely unbalanced",
                        "details": critical_partitions,
                        "action": "Consider rebalancing or adjusting partition strategy",
                    }
                )

            if warning_count > 0:
                monitoring_result.alerts.append(
                    {
                        "level": "WARNING",
                        "message": f"{warning_count} partitions showing early signs of imbalance",
                        "action": "Monitor closely and plan preventive maintenance",
                    }
                )

            # Update metrics
            monitoring_result.metrics = {
                "total_partitions": len(health_data),
                "unbalanced_count": unbalanced_count,
                "warning_count": warning_count,
                "healthy_count": len(health_data) - unbalanced_count - warning_count,
                "skew_metrics": skew_metrics_dict,
            }

            # Log summary
            if unbalanced_count > 0:
                logger.error("Partition health check: %d unbalanced partitions detected", unbalanced_count)
            elif warning_count > 0:
                logger.warning("Partition health check: %d partitions with warnings", warning_count)
            else:
                logger.info("Partition health check: All partitions healthy")

        except Exception as e:
            monitoring_result.status = "failed"
            monitoring_result.error = str(e)
            logger.error("Partition health check failed: %s", e, exc_info=True)

        return monitoring_result

    async def get_partition_statistics(self, partition_num: int | None = None) -> dict[str, Any]:
        """Get detailed statistics for partitions.

        Args:
            partition_num: Specific partition number, or None for all partitions

        Returns:
            Dictionary with partition statistics
        """
        try:
            if partition_num is not None:
                # Single partition stats
                query = text(
                    """
                    SELECT
                        partition_num,
                        chunk_count,
                        total_size_mb,
                        avg_chunk_size_kb,
                        created_at
                    FROM partition_stats
                    WHERE partition_num = :partition_num
                    """
                )
                result = await self.session.execute(query, {"partition_num": partition_num})
                row = result.fetchone()

                if row:
                    return {
                        "partition_num": row.partition_num,
                        "chunk_count": row.chunk_count,
                        "total_size_mb": float(row.total_size_mb),
                        "avg_chunk_size_kb": float(row.avg_chunk_size_kb),
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                    }
                return {}

            # All partitions stats
            query = text(
                """
                SELECT
                    COUNT(DISTINCT partition_num) as partition_count,
                    SUM(chunk_count) as total_chunks,
                    SUM(total_size_mb) as total_size_mb,
                    AVG(chunk_count) as avg_chunks_per_partition,
                    STDDEV(chunk_count) as chunk_count_stddev
                FROM partition_stats
                """
            )
            result = await self.session.execute(query)
            row = result.fetchone()

            if not row:
                # This should never happen with aggregate queries, but satisfy mypy
                return {
                    "partition_count": 0,
                    "total_chunks": 0,
                    "total_size_mb": 0.0,
                    "avg_chunks_per_partition": 0.0,
                    "chunk_count_stddev": 0.0,
                }

            return {
                "partition_count": row.partition_count or 0,
                "total_chunks": row.total_chunks or 0,
                "total_size_mb": float(row.total_size_mb or 0),
                "avg_chunks_per_partition": float(row.avg_chunks_per_partition or 0),
                "chunk_count_stddev": float(row.chunk_count_stddev or 0),
            }

        except Exception as e:
            logger.error("Failed to get partition statistics: %s", e, exc_info=True)
            raise

    async def get_rebalancing_recommendations(self) -> list[dict[str, Any]]:
        """Get recommendations for partition rebalancing.

        Returns:
            List of rebalancing recommendations
        """
        try:
            health_data = await self.get_partition_health_summary()

            recommendations = []
            for health in health_data:
                if health.chunk_skew > self.REBALANCE_THRESHOLD:
                    recommendations.append(
                        {
                            "partition": health.partition_num,
                            "reason": "High chunk skew",
                            "current_skew": health.chunk_skew,
                            "action": health.recommendation or "Consider data redistribution",
                            "priority": ("HIGH" if health.chunk_skew > self.SKEW_CRITICAL_THRESHOLD else "MEDIUM"),
                        }
                    )

            return recommendations

        except Exception as e:
            logger.error("Failed to get rebalancing recommendations: %s", e, exc_info=True)
            raise
