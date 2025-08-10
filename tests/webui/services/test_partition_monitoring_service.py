"""Unit tests for PartitionMonitoringService."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from packages.webui.services.partition_monitoring_service import (
    MonitoringResult,
    PartitionHealth,
    PartitionHealthStatus,
    PartitionMonitoringService,
    SkewMetric,
    SkewStatus,
)


class TestPartitionMonitoringService:
    """Test cases for PartitionMonitoringService."""

    @pytest.fixture()
    def mock_session(self) -> None:
        """Create a mock async session."""
        session = AsyncMock(spec=AsyncSession)
        session.execute = AsyncMock()
        return session

    @pytest.fixture()
    def service(self, mock_session) -> None:
        """Create a PartitionMonitoringService instance."""
        return PartitionMonitoringService(mock_session)

    @pytest.fixture()
    def sample_health_data(self) -> None:
        """Create sample partition health data."""
        return [
            {
                "partition_num": 0,
                "chunk_count": 1000,
                "total_chunks": 3000,
                "chunk_percentage": 33.33,
                "size_percentage": 33.0,
                "health_status": "HEALTHY",
                "chunk_skew": 0.01,
                "size_skew": 0.02,
                "recommendation": None,
            },
            {
                "partition_num": 1,
                "chunk_count": 1500,
                "total_chunks": 3000,
                "chunk_percentage": 50.0,
                "size_percentage": 48.0,
                "health_status": "WARNING",
                "chunk_skew": 0.35,
                "size_skew": 0.30,
                "recommendation": "Monitor partition growth",
            },
            {
                "partition_num": 2,
                "chunk_count": 500,
                "total_chunks": 3000,
                "chunk_percentage": 16.67,
                "size_percentage": 19.0,
                "health_status": "UNBALANCED",
                "chunk_skew": 0.55,
                "size_skew": 0.45,
                "recommendation": "Consider rebalancing data",
            },
        ]

    @pytest.fixture()
    def sample_skew_metrics(self) -> None:
        """Create sample skew metrics."""
        return [
            {
                "metric": "chunk_distribution",
                "value": 0.25,
                "status": "WARNING",
                "details": "Moderate imbalance detected",
            },
            {
                "metric": "size_distribution",
                "value": 0.15,
                "status": "NORMAL",
                "details": "Within acceptable range",
            },
        ]

    @pytest.mark.asyncio()
    async def test_get_partition_health_summary_success(self, service, mock_session, sample_health_data) -> None:
        """Test getting partition health summary successfully."""
        # Mock the database result
        mock_rows = []
        for data in sample_health_data:
            mock_row = Mock()
            for key, value in data.items():
                setattr(mock_row, key, value)
            mock_rows.append(mock_row)

        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter(mock_rows))
        mock_session.execute.return_value = mock_result

        result = await service.get_partition_health_summary()

        assert len(result) == 3
        assert isinstance(result[0], PartitionHealth)
        assert result[0].partition_num == 0
        assert result[0].health_status == PartitionHealthStatus.HEALTHY
        assert result[1].health_status == PartitionHealthStatus.WARNING
        assert result[2].health_status == PartitionHealthStatus.UNBALANCED

    @pytest.mark.asyncio()
    async def test_get_partition_health_summary_error(self, service, mock_session) -> None:
        """Test error handling in get_partition_health_summary."""
        mock_session.execute.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            await service.get_partition_health_summary()

    @pytest.mark.asyncio()
    async def test_analyze_partition_skew_success(self, service, mock_session, sample_skew_metrics) -> None:
        """Test analyzing partition skew successfully."""
        # Mock the database result
        mock_rows = []
        for data in sample_skew_metrics:
            mock_row = Mock()
            for key, value in data.items():
                setattr(mock_row, key, value)
            mock_rows.append(mock_row)

        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter(mock_rows))
        mock_session.execute.return_value = mock_result

        result = await service.analyze_partition_skew()

        assert len(result) == 2
        assert isinstance(result[0], SkewMetric)
        assert result[0].metric == "chunk_distribution"
        assert result[0].status == SkewStatus.WARNING
        assert result[1].status == SkewStatus.NORMAL

    @pytest.mark.asyncio()
    async def test_analyze_partition_skew_error(self, service, mock_session) -> None:
        """Test error handling in analyze_partition_skew."""
        mock_session.execute.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error"):
            await service.analyze_partition_skew()

    @pytest.mark.asyncio()
    async def test_check_partition_health_all_healthy(self, service, mock_session) -> None:
        """Test health check when all partitions are healthy."""
        # Mock healthy partition data
        healthy_data = [
            {
                "partition_num": i,
                "chunk_count": 1000,
                "total_chunks": 3000,
                "chunk_percentage": 33.33,
                "size_percentage": 33.33,
                "health_status": "HEALTHY",
                "chunk_skew": 0.01,
                "size_skew": 0.01,
                "recommendation": None,
            }
            for i in range(3)
        ]

        # Mock health summary
        with patch.object(service, "get_partition_health_summary") as mock_health_summary:
            mock_health_summary.return_value = [PartitionHealth(**data) for data in healthy_data]

            # Mock skew analysis
            with patch.object(service, "analyze_partition_skew") as mock_skew:
                mock_skew.return_value = [
                    SkewMetric(
                        metric="chunk_distribution",
                        value=0.01,
                        status=SkewStatus.NORMAL,
                        details="All partitions balanced",
                    )
                ]

                result = await service.check_partition_health()

                assert isinstance(result, MonitoringResult)
                assert result.status == "success"
                assert len(result.alerts) == 0
                assert result.metrics["unbalanced_count"] == 0
                assert result.metrics["warning_count"] == 0
                assert result.metrics["healthy_count"] == 3

    @pytest.mark.asyncio()
    async def test_check_partition_health_with_warnings(self, service, mock_session, sample_health_data) -> None:
        """Test health check with warnings and unbalanced partitions."""
        # Mock partition data with issues
        with patch.object(service, "get_partition_health_summary") as mock_health_summary:
            # Create PartitionHealth objects with proper enums
            health_objects = []
            for data in sample_health_data:
                health_objects.append(
                    PartitionHealth(
                        partition_num=data["partition_num"],
                        chunk_count=data["chunk_count"],
                        total_chunks=data["total_chunks"],
                        chunk_percentage=data["chunk_percentage"],
                        size_percentage=data["size_percentage"],
                        health_status=PartitionHealthStatus[data["health_status"]],
                        chunk_skew=data["chunk_skew"],
                        size_skew=data["size_skew"],
                        recommendation=data["recommendation"],
                    )
                )
            mock_health_summary.return_value = health_objects

            with patch.object(service, "analyze_partition_skew") as mock_skew:
                mock_skew.return_value = []

                result = await service.check_partition_health()

                assert result.status == "success"
                assert len(result.alerts) == 2  # One for unbalanced, one for warnings

                # Check error alert
                error_alert = next(a for a in result.alerts if a["level"] == "ERROR")
                assert "1 partitions are severely unbalanced" in error_alert["message"]
                assert len(error_alert["details"]) == 1

                # Check warning alert
                warning_alert = next(a for a in result.alerts if a["level"] == "WARNING")
                assert "1 partitions showing early signs" in warning_alert["message"]

                # Check metrics
                assert result.metrics["unbalanced_count"] == 1
                assert result.metrics["warning_count"] == 1
                assert result.metrics["healthy_count"] == 1

    @pytest.mark.asyncio()
    async def test_check_partition_health_error(self, service, mock_session) -> None:
        """Test health check error handling."""
        with patch.object(service, "get_partition_health_summary") as mock_health_summary:
            mock_health_summary.side_effect = Exception("Database connection failed")

            result = await service.check_partition_health()

            assert result.status == "failed"
            assert result.error == "Database connection failed"
            assert len(result.alerts) == 0

    @pytest.mark.asyncio()
    async def test_get_partition_statistics_single_partition(self, service, mock_session) -> None:
        """Test getting statistics for a single partition."""
        partition_num = 1

        mock_row = Mock(
            partition_num=partition_num,
            chunk_count=1000,
            total_size_mb=500.5,
            avg_chunk_size_kb=0.5,
            created_at=datetime.now(UTC),
        )

        mock_result = Mock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        stats = await service.get_partition_statistics(partition_num)

        assert stats["partition_num"] == partition_num
        assert stats["chunk_count"] == 1000
        assert stats["total_size_mb"] == 500.5
        assert stats["avg_chunk_size_kb"] == 0.5
        assert "created_at" in stats

    @pytest.mark.asyncio()
    async def test_get_partition_statistics_single_partition_not_found(self, service, mock_session) -> None:
        """Test getting statistics for non-existent partition."""
        mock_result = Mock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        stats = await service.get_partition_statistics(999)

        assert stats == {}

    @pytest.mark.asyncio()
    async def test_get_partition_statistics_all_partitions(self, service, mock_session) -> None:
        """Test getting statistics for all partitions."""
        mock_row = Mock(
            partition_count=10,
            total_chunks=50000,
            total_size_mb=25000.0,
            avg_chunks_per_partition=5000.0,
            chunk_count_stddev=250.5,
        )

        mock_result = Mock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        stats = await service.get_partition_statistics()

        assert stats["partition_count"] == 10
        assert stats["total_chunks"] == 50000
        assert stats["total_size_mb"] == 25000.0
        assert stats["avg_chunks_per_partition"] == 5000.0
        assert stats["chunk_count_stddev"] == 250.5

    @pytest.mark.asyncio()
    async def test_get_partition_statistics_all_partitions_none_values(self, service, mock_session) -> None:
        """Test getting statistics with None values returns defaults."""
        mock_row = Mock(
            partition_count=None,
            total_chunks=None,
            total_size_mb=None,
            avg_chunks_per_partition=None,
            chunk_count_stddev=None,
        )

        mock_result = Mock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        stats = await service.get_partition_statistics()

        assert stats["partition_count"] == 0
        assert stats["total_chunks"] == 0
        assert stats["total_size_mb"] == 0.0
        assert stats["avg_chunks_per_partition"] == 0.0
        assert stats["chunk_count_stddev"] == 0.0

    @pytest.mark.asyncio()
    async def test_get_partition_statistics_error(self, service, mock_session) -> None:
        """Test error handling in get_partition_statistics."""
        mock_session.execute.side_effect = Exception("Query failed")

        with pytest.raises(Exception, match="Query failed"):
            await service.get_partition_statistics()

    @pytest.mark.asyncio()
    async def test_get_rebalancing_recommendations_no_issues(self, service, mock_session) -> None:
        """Test getting recommendations when no rebalancing needed."""
        # Mock healthy partitions
        healthy_data = [
            {
                "partition_num": i,
                "chunk_count": 1000,
                "total_chunks": 3000,
                "chunk_percentage": 33.33,
                "size_percentage": 33.33,
                "health_status": "HEALTHY",
                "chunk_skew": 0.1,  # Below threshold
                "size_skew": 0.1,
                "recommendation": None,
            }
            for i in range(3)
        ]

        with patch.object(service, "get_partition_health_summary") as mock_health_summary:
            mock_health_summary.return_value = [PartitionHealth(**data) for data in healthy_data]

            recommendations = await service.get_rebalancing_recommendations()

            assert len(recommendations) == 0

    @pytest.mark.asyncio()
    async def test_get_rebalancing_recommendations_with_issues(self, service, mock_session) -> None:
        """Test getting recommendations for unbalanced partitions."""
        # Mock unbalanced partitions
        unbalanced_data = [
            {
                "partition_num": 0,
                "chunk_count": 2000,
                "total_chunks": 3000,
                "chunk_percentage": 66.67,
                "size_percentage": 65.0,
                "health_status": "UNBALANCED",
                "chunk_skew": 0.6,  # Above critical threshold
                "size_skew": 0.55,
                "recommendation": "Redistribute data from partition 0",
            },
            {
                "partition_num": 1,
                "chunk_count": 800,
                "total_chunks": 3000,
                "chunk_percentage": 26.67,
                "size_percentage": 25.0,
                "health_status": "WARNING",
                "chunk_skew": 0.45,  # Above rebalance threshold
                "size_skew": 0.4,
                "recommendation": "Monitor partition growth",
            },
        ]

        with patch.object(service, "get_partition_health_summary") as mock_health_summary:
            mock_health_summary.return_value = [PartitionHealth(**data) for data in unbalanced_data]

            recommendations = await service.get_rebalancing_recommendations()

            assert len(recommendations) == 2

            # Check first recommendation (critical)
            assert recommendations[0]["partition"] == 0
            assert recommendations[0]["reason"] == "High chunk skew"
            assert recommendations[0]["priority"] == "HIGH"
            assert recommendations[0]["current_skew"] == 0.6

            # Check second recommendation (medium)
            assert recommendations[1]["partition"] == 1
            assert recommendations[1]["priority"] == "MEDIUM"
            assert recommendations[1]["current_skew"] == 0.45

    @pytest.mark.asyncio()
    async def test_get_rebalancing_recommendations_error(self, service, mock_session) -> None:
        """Test error handling in get_rebalancing_recommendations."""
        with patch.object(service, "get_partition_health_summary") as mock_health_summary:
            mock_health_summary.side_effect = Exception("Health check failed")

            with pytest.raises(Exception, match="Health check failed"):
                await service.get_rebalancing_recommendations()

    def test_threshold_constants(self, service) -> None:
        """Test that threshold constants are properly set."""
        assert service.SKEW_WARNING_THRESHOLD == 0.3
        assert service.SKEW_CRITICAL_THRESHOLD == 0.5
        assert service.REBALANCE_THRESHOLD == 0.4
