"""
Partition manager for handling chunk distribution across 100 direct LIST partitions.

This module provides utilities for:
- Determining partition assignment for collections
- Monitoring partition health and distribution
- Analyzing data skew across partitions
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class PartitionHealth:
    """Health metrics for a single partition."""

    partition_id: int
    partition_name: str
    row_count: int
    size_bytes: int
    size_pretty: str
    pct_deviation_from_avg: float
    partition_status: str  # 'HOT', 'COLD', or 'NORMAL'
    needs_vacuum: bool
    dead_rows: int
    last_vacuum: datetime | None
    last_autovacuum: datetime | None


@dataclass
class DistributionStats:
    """Overall distribution statistics across all partitions."""

    partitions_used: int
    empty_partitions: int
    total_rows: int
    avg_chunks_per_partition: float
    max_chunks: int
    min_chunks: int
    max_skew_ratio: float
    distribution_status: str  # 'HEALTHY', 'WARNING', or 'REBALANCE NEEDED'
    recommendations: list[str]


class PartitionManager:
    """
    Manages partition assignment and health monitoring for chunk storage.

    Uses 100 direct LIST partitions with PostgreSQL's hashtext() function
    for even distribution of collections across partitions.

    The chunks table uses a trigger to automatically compute the 'partition_key'
    column as mod(hashtext(collection_id::text), 100) on INSERT. This approach
    works around PostgreSQL's limitations with PRIMARY KEY constraints and
    expression-based partitioning.
    """

    PARTITION_COUNT = 100
    SKEW_WARNING_THRESHOLD = 1.1  # 10% deviation triggers warning
    SKEW_CRITICAL_THRESHOLD = 1.2  # 20% deviation is critical

    @staticmethod
    def get_partition_id(collection_id: str) -> int:
        """
        Calculate partition ID for a collection using consistent hashing.

        NOTE: This is for reference/monitoring only. PostgreSQL uses its own
        hashtext() function for actual partition assignment during INSERT operations.
        The partition assignment is handled automatically by PostgreSQL's
        PARTITION BY LIST (mod(hashtext(collection_id::text), 100)) clause.

        Args:
            collection_id: Collection ID string (VARCHAR in database)

        Returns:
            Partition ID (0-99) - approximate, for monitoring purposes
        """
        # Use MD5 for consistent hashing in Python
        # IMPORTANT: PostgreSQL's hashtext() uses a different algorithm,
        # so this calculation is only for monitoring/statistics purposes
        hash_val = hashlib.md5(collection_id.encode()).hexdigest()
        # Convert first 8 hex chars to int and mod by partition count
        hash_int = int(hash_val[:8], 16)
        return hash_int % PartitionManager.PARTITION_COUNT

    @staticmethod
    def get_partition_name(collection_id: str) -> str:
        """
        Get the partition table name for a collection.

        Args:
            collection_id: Collection ID string (VARCHAR in database)

        Returns:
            Partition table name (e.g., 'chunks_part_42')
        """
        partition_id = PartitionManager.get_partition_id(collection_id)
        return f"chunks_part_{partition_id:02d}"

    @staticmethod
    def get_all_partition_names() -> list[str]:
        """
        Get list of all partition table names.

        Returns:
            List of all 100 partition table names
        """
        return [f"chunks_part_{i:02d}" for i in range(PartitionManager.PARTITION_COUNT)]

    async def get_partition_health(self, db: AsyncSession) -> list[PartitionHealth]:
        """
        Get health metrics for all partitions.

        Args:
            db: Database session

        Returns:
            List of PartitionHealth objects for all partitions
        """
        query = text(
            """
            SELECT
                partition_id,
                partition_name,
                row_count,
                size_bytes,
                size_pretty,
                pct_deviation_from_avg,
                partition_status,
                needs_vacuum,
                dead_rows,
                last_vacuum,
                last_autovacuum
            FROM partition_health
            ORDER BY partition_id
        """
        )

        result = await db.execute(query)
        rows = result.fetchall()

        return [
            PartitionHealth(
                partition_id=row.partition_id,
                partition_name=row.partition_name,
                row_count=row.row_count,
                size_bytes=row.size_bytes,
                size_pretty=row.size_pretty,
                pct_deviation_from_avg=float(row.pct_deviation_from_avg or 0),
                partition_status=row.partition_status,
                needs_vacuum=row.needs_vacuum,
                dead_rows=row.dead_rows,
                last_vacuum=row.last_vacuum,
                last_autovacuum=row.last_autovacuum,
            )
            for row in rows
        ]

    async def get_distribution_stats(self, db: AsyncSession) -> DistributionStats:
        """
        Get overall distribution statistics across all partitions.

        Args:
            db: Database session

        Returns:
            DistributionStats object with analysis and recommendations
        """
        # Get basic distribution metrics
        query = text(
            """
            SELECT
                partitions_used,
                empty_partitions,
                avg_chunks_per_partition,
                max_chunks,
                min_chunks,
                max_skew_ratio,
                distribution_status
            FROM partition_distribution
        """
        )

        result = await db.execute(query)
        row = result.fetchone()

        if not row:
            # No data yet
            return DistributionStats(
                partitions_used=0,
                empty_partitions=self.PARTITION_COUNT,
                total_rows=0,
                avg_chunks_per_partition=0,
                max_chunks=0,
                min_chunks=0,
                max_skew_ratio=0,
                distribution_status="HEALTHY",
                recommendations=["No data in partitions yet"],
            )

        # Get total row count
        count_query = text("SELECT COUNT(*) as total FROM chunks")
        count_result = await db.execute(count_query)
        total_rows = count_result.scalar() or 0

        # Generate recommendations based on metrics
        recommendations = []

        if row.max_skew_ratio > self.SKEW_CRITICAL_THRESHOLD:
            recommendations.append(
                f"Critical skew detected ({row.max_skew_ratio:.2f}x). "
                "Some partitions have significantly more data than others."
            )
            recommendations.append("Consider reviewing collection distribution patterns.")
        elif row.max_skew_ratio > self.SKEW_WARNING_THRESHOLD:
            recommendations.append(
                f"Moderate skew detected ({row.max_skew_ratio:.2f}x). Monitor partition growth closely."
            )

        if row.empty_partitions > self.PARTITION_COUNT * 0.5:
            recommendations.append(f"{row.empty_partitions} partitions are empty. This is normal for small datasets.")

        if not recommendations:
            recommendations.append("Distribution is healthy and balanced.")

        return DistributionStats(
            partitions_used=row.partitions_used,
            empty_partitions=row.empty_partitions,
            total_rows=total_rows,
            avg_chunks_per_partition=float(row.avg_chunks_per_partition or 0),
            max_chunks=row.max_chunks or 0,
            min_chunks=row.min_chunks or 0,
            max_skew_ratio=float(row.max_skew_ratio or 0),
            distribution_status=row.distribution_status,
            recommendations=recommendations,
        )

    async def analyze_partition_skew(self, db: AsyncSession) -> dict[str, Any]:
        """
        Analyze partition skew and provide detailed metrics.

        Args:
            db: Database session

        Returns:
            Dictionary with skew analysis results
        """
        query = text("SELECT * FROM analyze_partition_skew()")
        result = await db.execute(query)
        row = result.fetchone()

        if not row:
            return {
                "status": "NO_DATA",
                "avg_rows": 0,
                "max_rows": 0,
                "min_rows": 0,
                "max_skew_ratio": 0,
                "partitions_over_threshold": 0,
                "recommendation": "No data available for analysis",
            }

        return {
            "status": row.status,
            "avg_rows": float(row.avg_rows or 0),
            "max_rows": row.max_rows or 0,
            "min_rows": row.min_rows or 0,
            "max_skew_ratio": float(row.max_skew_ratio or 0),
            "partitions_over_threshold": row.partitions_over_threshold or 0,
            "recommendation": row.recommendation,
        }

    async def get_hot_partitions(self, db: AsyncSession, threshold: float | None = None) -> list[PartitionHealth]:
        """
        Get list of hot partitions (those with above-average load).

        Args:
            db: Database session
            threshold: Custom threshold multiplier (default: SKEW_WARNING_THRESHOLD)

        Returns:
            List of PartitionHealth objects for hot partitions
        """
        if threshold is None:
            threshold = self.SKEW_WARNING_THRESHOLD

        all_partitions = await self.get_partition_health(db)

        # Filter for hot partitions
        hot_partitions = [
            p for p in all_partitions if p.partition_status == "HOT" or p.pct_deviation_from_avg > (threshold - 1) * 100
        ]

        # Sort by deviation (most loaded first)
        hot_partitions.sort(key=lambda p: p.pct_deviation_from_avg, reverse=True)

        return hot_partitions

    async def verify_partition_for_collection(self, db: AsyncSession, collection_id: str) -> dict[str, Any]:
        """
        Verify partition assignment for a specific collection.

        Useful for debugging and ensuring partition routing works correctly.

        Args:
            db: Database session
            collection_id: Collection ID string (VARCHAR in database)

        Returns:
            Dictionary with partition assignment details
        """
        # Get Python-calculated partition
        python_partition_id = self.get_partition_id(collection_id)
        python_partition_name = self.get_partition_name(collection_id)

        # Get PostgreSQL-calculated partition
        query = text(
            """
            SELECT
                mod(hashtext(:collection_id), 100) as db_partition_id,
                get_partition_for_collection(:collection_id) as db_partition_name
        """
        )

        result = await db.execute(query, {"collection_id": collection_id})
        row = result.fetchone()

        # Check if there's actual data for this collection
        data_query = text(
            """
            SELECT
                COUNT(*) as chunk_count,
                MIN(created_at) as first_chunk,
                MAX(created_at) as last_chunk
            FROM chunks
            WHERE collection_id = :collection_id
        """
        )

        data_result = await db.execute(data_query, {"collection_id": collection_id})
        data_row = data_result.fetchone()

        return {
            "collection_id": collection_id,
            "python_partition_id": python_partition_id,
            "python_partition_name": python_partition_name,
            "db_partition_id": row.db_partition_id if row else None,
            "db_partition_name": row.db_partition_name if row else None,
            "partition_match": (python_partition_id == row.db_partition_id if row else False),
            "chunk_count": data_row.chunk_count if data_row else 0,
            "first_chunk": data_row.first_chunk if data_row else None,
            "last_chunk": data_row.last_chunk if data_row else None,
        }

    async def get_efficiency_report(self, db: AsyncSession) -> dict[str, Any]:
        """
        Generate a comprehensive efficiency report for the partitioning system.

        Args:
            db: Database session

        Returns:
            Dictionary with efficiency metrics and analysis
        """
        # Get distribution stats
        dist_stats = await self.get_distribution_stats(db)

        # Get skew analysis
        skew_analysis = await self.analyze_partition_skew(db)

        # Get hot partitions
        hot_partitions = await self.get_hot_partitions(db)

        # Calculate efficiency score (0-100)
        efficiency_score = 100

        # Deduct points for skew
        if dist_stats.max_skew_ratio > self.SKEW_CRITICAL_THRESHOLD:
            efficiency_score -= 30
        elif dist_stats.max_skew_ratio > self.SKEW_WARNING_THRESHOLD:
            efficiency_score -= 15

        # Deduct points for empty partitions (only if we have data)
        if dist_stats.total_rows > 0:
            empty_ratio = dist_stats.empty_partitions / self.PARTITION_COUNT
            if empty_ratio > 0.7:
                efficiency_score -= 20
            elif empty_ratio > 0.5:
                efficiency_score -= 10

        # Deduct points for hot partitions
        hot_partition_ratio = len(hot_partitions) / self.PARTITION_COUNT
        if hot_partition_ratio > 0.2:
            efficiency_score -= 15
        elif hot_partition_ratio > 0.1:
            efficiency_score -= 5

        efficiency_score = max(0, efficiency_score)  # Ensure non-negative

        return {
            "efficiency_score": efficiency_score,
            "total_partitions": self.PARTITION_COUNT,
            "partitions_used": dist_stats.partitions_used,
            "empty_partitions": dist_stats.empty_partitions,
            "total_rows": dist_stats.total_rows,
            "avg_rows_per_partition": dist_stats.avg_chunks_per_partition,
            "max_skew_ratio": dist_stats.max_skew_ratio,
            "hot_partitions_count": len(hot_partitions),
            "hot_partition_ids": [p.partition_id for p in hot_partitions[:10]],  # Top 10
            "distribution_status": dist_stats.distribution_status,
            "skew_status": skew_analysis["status"],
            "recommendations": dist_stats.recommendations + [skew_analysis["recommendation"]],
            "partition_efficiency": {
                "excellent": efficiency_score >= 90,
                "good": 70 <= efficiency_score < 90,
                "fair": 50 <= efficiency_score < 70,
                "poor": efficiency_score < 50,
            },
        }

    async def get_partition_efficiency_report(self, db: AsyncSession) -> dict[str, Any]:
        """
        Alias for get_efficiency_report for backward compatibility.

        Args:
            db: Database session

        Returns:
            Dictionary with efficiency metrics and analysis
        """
        return await self.get_efficiency_report(db)
