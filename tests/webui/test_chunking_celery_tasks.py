"""Unit tests for chunking-related Celery tasks."""

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from packages.webui.tasks import monitor_partition_health, refresh_collection_chunking_stats


class TestChunkingCeleryTasks:
    """Test cases for chunking-related Celery tasks."""

    @patch("shared.database.database.AsyncSessionLocal")
    @patch("packages.webui.tasks.logger")
    def test_refresh_collection_chunking_stats_success(self, mock_logger, mock_session_local):
        """Test successful refresh of collection chunking stats."""
        # Mock the async context manager
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session
        mock_session_local.return_value.__aexit__.return_value = None
        
        # Run the task
        with patch("asyncio.run") as mock_asyncio_run:
            # Simulate successful execution
            mock_asyncio_run.return_value = None
            
            result = refresh_collection_chunking_stats()
        
        # Verify result
        assert result["status"] == "success"
        assert "duration_seconds" in result
        assert result["duration_seconds"] >= 0
        assert result["error"] is None
        
        # Verify logging
        assert mock_logger.info.call_count == 2
        mock_logger.info.assert_any_call("Starting refresh of collection_chunking_stats materialized view")

    @patch("shared.database.database.AsyncSessionLocal")
    @patch("packages.webui.tasks.logger")
    def test_refresh_collection_chunking_stats_failure(self, mock_logger, mock_session_local):
        """Test failed refresh of collection chunking stats."""
        # Mock the async context manager to raise an error
        error_msg = "Database connection failed"
        
        with patch("asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = Exception(error_msg)
            
            # The task should re-raise the exception
            with pytest.raises(Exception, match=error_msg):
                refresh_collection_chunking_stats()
        
        # Verify error logging
        mock_logger.error.assert_called_once()
        error_log = mock_logger.error.call_args[0][0]
        assert "Failed to refresh collection_chunking_stats" in error_log
        assert error_msg in error_log

    @patch("shared.database.database.AsyncSessionLocal")
    @patch("packages.webui.services.partition_monitoring_service.PartitionMonitoringService")
    @patch("packages.webui.tasks.logger")
    def test_monitor_partition_health_success_no_alerts(self, mock_logger, mock_service_class, mock_session_local):
        """Test successful partition health monitoring with no alerts."""
        # Mock the monitoring service
        mock_service = AsyncMock()
        mock_monitoring_result = Mock(
            status="success",
            timestamp=datetime.now(UTC).isoformat(),
            alerts=[],
            metrics={
                "total_partitions": 3,
                "unbalanced_count": 0,
                "warning_count": 0,
                "healthy_count": 3,
            },
            error=None,
        )
        mock_service.check_partition_health = AsyncMock(return_value=mock_monitoring_result)
        mock_service_class.return_value = mock_service
        
        # Mock the session
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session
        mock_session_local.return_value.__aexit__.return_value = None
        
        # Run the task
        with patch("asyncio.run") as mock_asyncio_run:
            # Return the expected dictionary structure
            mock_asyncio_run.return_value = {
                "status": mock_monitoring_result.status,
                "timestamp": mock_monitoring_result.timestamp,
                "alerts": mock_monitoring_result.alerts,
                "metrics": mock_monitoring_result.metrics,
                "error": mock_monitoring_result.error,
            }
            
            result = monitor_partition_health()
        
        # Verify result
        assert result["status"] == "success"
        assert result["alerts"] == []
        assert result["metrics"]["healthy_count"] == 3
        assert result["error"] is None
        
        # Verify logging
        mock_logger.info.assert_any_call("Starting partition health monitoring")

    @patch("shared.database.database.AsyncSessionLocal")
    @patch("packages.webui.services.partition_monitoring_service.PartitionMonitoringService")
    @patch("packages.webui.tasks.logger")
    def test_monitor_partition_health_with_warnings(self, mock_logger, mock_service_class, mock_session_local):
        """Test partition health monitoring with warnings."""
        # Mock the monitoring service with warnings
        mock_service = AsyncMock()
        mock_monitoring_result = Mock(
            status="success",
            timestamp=datetime.now(UTC).isoformat(),
            alerts=[
                {
                    "level": "WARNING",
                    "message": "2 partitions showing early signs of imbalance",
                    "action": "Monitor closely",
                }
            ],
            metrics={
                "total_partitions": 5,
                "unbalanced_count": 0,
                "warning_count": 2,
                "healthy_count": 3,
            },
            error=None,
        )
        mock_service.check_partition_health = AsyncMock(return_value=mock_monitoring_result)
        mock_service_class.return_value = mock_service
        
        # Mock the session
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session
        mock_session_local.return_value.__aexit__.return_value = None
        
        # Run the task
        with patch("asyncio.run") as mock_asyncio_run:
            # Return the expected dictionary structure
            mock_asyncio_run.return_value = {
                "status": mock_monitoring_result.status,
                "timestamp": mock_monitoring_result.timestamp,
                "alerts": mock_monitoring_result.alerts,
                "metrics": mock_monitoring_result.metrics,
                "error": mock_monitoring_result.error,
            }
            
            result = monitor_partition_health()
        
        # Verify result
        assert result["status"] == "success"
        assert len(result["alerts"]) == 1
        assert result["alerts"][0]["level"] == "WARNING"
        assert result["metrics"]["warning_count"] == 2
        
        # Verify logging
        mock_logger.info.assert_any_call("Starting partition health monitoring")

    @patch("shared.database.database.AsyncSessionLocal")
    @patch("packages.webui.services.partition_monitoring_service.PartitionMonitoringService")
    @patch("packages.webui.tasks.logger")
    def test_monitor_partition_health_with_errors(self, mock_logger, mock_service_class, mock_session_local):
        """Test partition health monitoring with critical errors."""
        # Mock the monitoring service with errors
        mock_service = AsyncMock()
        mock_monitoring_result = Mock(
            status="success",
            timestamp=datetime.now(UTC).isoformat(),
            alerts=[
                {
                    "level": "ERROR",
                    "message": "3 partitions are severely unbalanced",
                    "details": [
                        {"partition": 1, "chunk_percentage": 60.0},
                        {"partition": 2, "chunk_percentage": 5.0},
                        {"partition": 3, "chunk_percentage": 3.0},
                    ],
                    "action": "Consider rebalancing",
                }
            ],
            metrics={
                "total_partitions": 5,
                "unbalanced_count": 3,
                "warning_count": 0,
                "healthy_count": 2,
            },
            error=None,
        )
        mock_service.check_partition_health = AsyncMock(return_value=mock_monitoring_result)
        mock_service_class.return_value = mock_service
        
        # Mock the session
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session
        mock_session_local.return_value.__aexit__.return_value = None
        
        # Run the task
        with patch("asyncio.run") as mock_asyncio_run:
            # Return the expected dictionary structure
            mock_asyncio_run.return_value = {
                "status": mock_monitoring_result.status,
                "timestamp": mock_monitoring_result.timestamp,
                "alerts": mock_monitoring_result.alerts,
                "metrics": mock_monitoring_result.metrics,
                "error": mock_monitoring_result.error,
            }
            
            result = monitor_partition_health()
        
        # Verify result
        assert result["status"] == "success"
        assert len(result["alerts"]) == 1
        assert result["alerts"][0]["level"] == "ERROR"
        assert result["metrics"]["unbalanced_count"] == 3
        
        # Verify logging
        mock_logger.info.assert_any_call("Starting partition health monitoring")

    @patch("shared.database.database.AsyncSessionLocal")
    @patch("packages.webui.services.partition_monitoring_service.PartitionMonitoringService")
    @patch("packages.webui.tasks.logger")
    def test_monitor_partition_health_service_failure(self, mock_logger, mock_service_class, mock_session_local):
        """Test partition health monitoring when service fails."""
        # Mock the monitoring service to fail
        mock_service = AsyncMock()
        mock_monitoring_result = Mock(
            status="failed",
            timestamp=datetime.now(UTC).isoformat(),
            alerts=[],
            metrics={},
            error="Failed to connect to database",
        )
        mock_service.check_partition_health = AsyncMock(return_value=mock_monitoring_result)
        mock_service_class.return_value = mock_service
        
        # Mock the session
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session
        mock_session_local.return_value.__aexit__.return_value = None
        
        # Run the task
        with patch("asyncio.run") as mock_asyncio_run:
            # Return the expected dictionary structure
            mock_asyncio_run.return_value = {
                "status": mock_monitoring_result.status,
                "timestamp": mock_monitoring_result.timestamp,
                "alerts": mock_monitoring_result.alerts,
                "metrics": mock_monitoring_result.metrics,
                "error": mock_monitoring_result.error,
            }
            
            result = monitor_partition_health()
        
        # Verify result
        assert result["status"] == "failed"
        assert result["error"] == "Failed to connect to database"
        
        # Verify logging
        mock_logger.info.assert_any_call("Starting partition health monitoring")

    @patch("shared.database.database.AsyncSessionLocal")
    @patch("packages.webui.tasks.logger")
    def test_monitor_partition_health_exception(self, mock_logger, mock_session_local):
        """Test partition health monitoring with unexpected exception."""
        error_msg = "Unexpected error occurred"
        
        with patch("asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.side_effect = Exception(error_msg)
            
            # The task should re-raise the exception
            with pytest.raises(Exception, match=error_msg):
                monitor_partition_health()
        
        # Verify error logging
        mock_logger.error.assert_called_once()
        error_log = mock_logger.error.call_args[0][0]
        assert "Partition health monitoring failed" in error_log
        assert error_msg in error_log