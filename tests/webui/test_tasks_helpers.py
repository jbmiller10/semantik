"""Additional tests for webui.tasks helper functions and edge cases.

This test suite covers additional functionality not covered in the main test file:
- Audit logging functions
- Metrics recording
- WebSocket update edge cases
- Resource cleanup scenarios
- Concurrent operation handling
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from packages.webui.tasks import (
    CeleryTaskWithOperationUpdates,
    _audit_collection_deletion,
    _audit_collection_deletions_batch,
    _audit_log_operation,
    _cleanup_staging_resources,
    _get_active_collections,
    _record_operation_metrics,
    _sanitize_audit_details,
    _update_collection_metrics,
    cleanup_old_results,
    extract_and_serialize_thread_safe,
    test_task,
)


class TestTaskHelperFunctions:
    """Test various helper functions used in tasks."""

    def test_test_task(self):
        """Test the test_task for Celery verification."""
        # Mock self parameter
        mock_self = Mock()
        
        result = test_task(mock_self)
        
        assert result["status"] == "success"
        assert result["message"] == "Celery is working!"

    def test_sanitize_audit_details(self):
        """Test audit details sanitization."""
        # Test with sensitive keys
        details = {
            "user": "john",
            "password": "secret123",
            "api_token": "token-xyz",
            "secret_key": "key-123",
            "normal_data": "this is fine"
        }
        
        sanitized = _sanitize_audit_details(details)
        
        # Sensitive keys should be removed
        assert "password" not in sanitized
        assert "api_token" not in sanitized
        assert "secret_key" not in sanitized
        assert sanitized["normal_data"] == "this is fine"
        
    def test_sanitize_audit_details_nested(self):
        """Test sanitization of nested structures."""
        details = {
            "config": {
                "database_password": "db-pass",
                "host": "localhost",
                "nested": {
                    "api_secret": "secret",
                    "public_key": "pk-123"
                }
            },
            "paths": [
                "/home/username/file.txt",
                "/Users/john/Documents/data.csv"
            ]
        }
        
        sanitized = _sanitize_audit_details(details)
        
        # Check nested sanitization
        assert "database_password" not in sanitized["config"]
        assert sanitized["config"]["host"] == "localhost"
        assert "api_secret" not in sanitized["config"]["nested"]
        assert sanitized["config"]["nested"]["public_key"] == "pk-123"
        
        # Check path sanitization
        assert sanitized["paths"][0] == "/home/~/file.txt"
        assert sanitized["paths"][1] == "/Users/~/Documents/data.csv"

    def test_sanitize_audit_details_none(self):
        """Test sanitization with None input."""
        assert _sanitize_audit_details(None) is None

    @patch("packages.webui.tasks.executor")
    def test_extract_and_serialize_thread_safe(self, mock_executor):
        """Test thread-safe text extraction wrapper."""
        # Mock the extraction function
        with patch("shared.text_processing.extraction.extract_and_serialize") as mock_extract:
            mock_extract.return_value = [("Test content", {"page": 1})]
            
            result = extract_and_serialize_thread_safe("/test/file.pdf")
            
            assert result == [("Test content", {"page": 1})]
            mock_extract.assert_called_once_with("/test/file.pdf")


class TestAuditLogging:
    """Test audit logging functionality."""

    @pytest.mark.asyncio
    @patch("shared.database.database.AsyncSessionLocal")
    @patch("shared.database.models.CollectionAuditLog")
    async def test_audit_log_operation_success(self, mock_audit_log_class, mock_session_local):
        """Test successful audit log creation."""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session_local.return_value = mock_session
        
        mock_audit_log = Mock()
        mock_audit_log_class.return_value = mock_audit_log
        
        # Create audit log
        await _audit_log_operation(
            collection_id="col-123",
            operation_id=456,
            user_id=1,
            action="test_action",
            details={"key": "value", "password": "secret"}
        )
        
        # Verify audit log was created
        mock_audit_log_class.assert_called_once()
        call_args = mock_audit_log_class.call_args[1]
        assert call_args["collection_id"] == "col-123"
        assert call_args["operation_id"] == 456
        assert call_args["user_id"] == 1
        assert call_args["action"] == "test_action"
        
        # Details should be sanitized
        assert "password" not in call_args["details"]
        assert call_args["details"]["key"] == "value"
        
        # Verify session operations
        mock_session.add.assert_called_once_with(mock_audit_log)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    @patch("shared.database.database.AsyncSessionLocal")
    async def test_audit_log_operation_failure(self, mock_session_local):
        """Test audit log creation handles failures gracefully."""
        # Setup session to fail
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock(side_effect=Exception("Database error"))
        mock_session_local.return_value = mock_session
        
        # Should not raise exception
        await _audit_log_operation(
            collection_id="col-123",
            operation_id=456,
            user_id=1,
            action="test_action"
        )
        
        # Function should complete without raising

    @pytest.mark.asyncio
    @patch("shared.database.database.AsyncSessionLocal")
    @patch("shared.database.models.CollectionAuditLog")
    async def test_audit_collection_deletion(self, mock_audit_log_class, mock_session_local):
        """Test audit logging for collection deletion."""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session_local.return_value = mock_session
        
        # Create deletion audit log
        await _audit_collection_deletion("test_collection", 5000)
        
        # Verify audit log creation
        mock_audit_log_class.assert_called_once()
        call_args = mock_audit_log_class.call_args[1]
        assert call_args["collection_id"] is None  # System operation
        assert call_args["operation_id"] is None
        assert call_args["user_id"] is None
        assert call_args["action"] == "qdrant_collection_deleted"
        assert call_args["details"]["collection_name"] == "test_collection"
        assert call_args["details"]["vector_count"] == 5000
        assert "deleted_at" in call_args["details"]

    @pytest.mark.asyncio
    @patch("shared.database.database.AsyncSessionLocal")
    @patch("shared.database.models.CollectionAuditLog")
    async def test_audit_collection_deletions_batch(self, mock_audit_log_class, mock_session_local):
        """Test batch audit logging for multiple collection deletions."""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session_local.return_value = mock_session
        
        # Create batch deletion audit logs
        deletions = [
            ("collection_1", 1000),
            ("collection_2", 2000),
            ("collection_3", 3000)
        ]
        
        await _audit_collection_deletions_batch(deletions)
        
        # Verify multiple audit logs created
        assert mock_audit_log_class.call_count == 3
        assert mock_session.add.call_count == 3
        assert mock_session.commit.call_count == 1  # Single commit for batch

    @pytest.mark.asyncio
    async def test_audit_collection_deletions_batch_empty(self):
        """Test batch audit with empty list."""
        # Should handle empty list gracefully
        await _audit_collection_deletions_batch([])
        # No exception should be raised


class TestMetricsRecording:
    """Test metrics recording functionality."""

    @pytest.mark.asyncio
    @patch("shared.database.database.AsyncSessionLocal")
    @patch("shared.database.models.OperationMetrics")
    async def test_record_operation_metrics_success(self, mock_metrics_class, mock_session_local):
        """Test successful operation metrics recording."""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session_local.return_value = mock_session
        
        # Mock operation repository
        operation_repo = AsyncMock()
        operation = Mock()
        operation.id = 123
        operation_repo.get_by_uuid.return_value = operation
        
        # Record metrics
        metrics = {
            "duration_seconds": 45.5,
            "cpu_seconds": 40.2,
            "memory_peak_bytes": 1024000,
            "documents_processed": 100,
            "success": True
        }
        
        await _record_operation_metrics(operation_repo, "op-123", metrics)
        
        # Verify metrics were created
        assert mock_metrics_class.call_count == 4  # Numeric metrics only
        
        # Verify session operations
        assert mock_session.add.call_count == 4
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    @patch("packages.webui.tasks.update_collection_stats")
    async def test_update_collection_metrics(self, mock_update_stats):
        """Test collection metrics update."""
        await _update_collection_metrics("col-123", 100, 1000, 10240000)
        
        mock_update_stats.assert_called_once_with("col-123", 100, 1000, 10240000)

    @pytest.mark.asyncio
    @patch("packages.webui.tasks.update_collection_stats")
    async def test_update_collection_metrics_failure(self, mock_update_stats):
        """Test collection metrics update handles failures."""
        mock_update_stats.side_effect = Exception("Metrics error")
        
        # Should not raise exception
        await _update_collection_metrics("col-123", 100, 1000, 10240000)


class TestActiveCollections:
    """Test active collections retrieval."""

    @pytest.mark.asyncio
    @patch("shared.database.database.AsyncSessionLocal")
    @patch("shared.database.repositories.collection_repository.CollectionRepository")
    async def test_get_active_collections(self, mock_repo_class, mock_session_local):
        """Test getting active collections from database."""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_local.return_value = mock_session
        
        mock_repo = AsyncMock()
        mock_collections = [
            {
                "id": "col1",
                "vector_store_name": "vec_col_1",
                "qdrant_collections": ["col_1_v1", "col_1_v2"],
                "qdrant_staging": None
            },
            {
                "id": "col2",
                "vector_store_name": "vec_col_2",
                "qdrant_collections": None,
                "qdrant_staging": {"collection_name": "staging_col_2"}
            },
            {
                "id": "col3",
                "vector_store_name": None,
                "qdrant_collections": [],
                "qdrant_staging": None
            }
        ]
        mock_repo.list_all.return_value = mock_collections
        mock_repo_class.return_value = mock_repo
        
        # Get active collections
        active = await _get_active_collections()
        
        # Verify results
        assert isinstance(active, set)
        assert "vec_col_1" in active
        assert "col_1_v1" in active
        assert "col_1_v2" in active
        assert "vec_col_2" in active
        assert "staging_col_2" in active
        # col3 has no vector store name, so nothing from it

    @pytest.mark.asyncio
    @patch("shared.database.database.AsyncSessionLocal")
    @patch("shared.database.repositories.collection_repository.CollectionRepository")
    async def test_get_active_collections_with_string_staging(self, mock_repo_class, mock_session_local):
        """Test handling of staging info as JSON string."""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_local.return_value = mock_session
        
        mock_repo = AsyncMock()
        mock_collections = [
            {
                "id": "col1",
                "vector_store_name": "vec_col_1",
                "qdrant_collections": None,
                "qdrant_staging": '{"collection_name": "staging_from_json"}'
            }
        ]
        mock_repo.list_all.return_value = mock_collections
        mock_repo_class.return_value = mock_repo
        
        # Get active collections
        active = await _get_active_collections()
        
        # Verify JSON string was parsed
        assert "staging_from_json" in active


class TestStagingCleanup:
    """Test staging resource cleanup."""

    @pytest.mark.asyncio
    @patch("packages.webui.tasks.qdrant_manager")
    @patch("shared.database.database.AsyncSessionLocal")
    @patch("shared.database.repositories.collection_repository.CollectionRepository")
    async def test_cleanup_staging_resources_success(
        self,
        mock_repo_class,
        mock_session_local,
        mock_qdrant_manager
    ):
        """Test successful staging cleanup."""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_local.return_value = mock_session
        
        mock_repo = AsyncMock()
        collection = Mock()
        collection.qdrant_staging = {
            "collection_name": "staging_test_123"
        }
        mock_repo.get_by_uuid.return_value = collection
        mock_repo.update = AsyncMock()
        mock_repo_class.return_value = mock_repo
        
        # Mock Qdrant
        client = Mock()
        collections_response = Mock()
        CollectionInfo = type("CollectionInfo", (), {"name": "staging_test_123"})
        collections_response.collections = [CollectionInfo]
        client.get_collections.return_value = collections_response
        client.delete_collection = Mock()
        mock_qdrant_manager.get_client.return_value = client
        
        # Clean up staging
        operation = {"type": "REINDEX"}
        await _cleanup_staging_resources("col-123", operation)
        
        # Verify deletion
        client.delete_collection.assert_called_once_with("staging_test_123")
        
        # Verify database update
        mock_repo.update.assert_called_once_with("col-123", {"qdrant_staging": None})

    @pytest.mark.asyncio
    @patch("shared.database.database.AsyncSessionLocal")
    @patch("shared.database.repositories.collection_repository.CollectionRepository")
    async def test_cleanup_staging_resources_no_staging(
        self,
        mock_repo_class,
        mock_session_local
    ):
        """Test cleanup when no staging exists."""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_local.return_value = mock_session
        
        mock_repo = AsyncMock()
        collection = Mock()
        collection.qdrant_staging = None
        mock_repo.get_by_uuid.return_value = collection
        mock_repo_class.return_value = mock_repo
        
        # Clean up staging (should handle gracefully)
        operation = {"type": "REINDEX"}
        await _cleanup_staging_resources("col-123", operation)
        
        # No update should be called
        mock_repo.update.assert_not_called()

    @pytest.mark.asyncio
    @patch("packages.webui.tasks.qdrant_manager")
    @patch("shared.database.database.AsyncSessionLocal")
    @patch("shared.database.repositories.collection_repository.CollectionRepository")
    async def test_cleanup_staging_resources_qdrant_failure(
        self,
        mock_repo_class,
        mock_session_local,
        mock_qdrant_manager
    ):
        """Test cleanup continues despite Qdrant failures."""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_local.return_value = mock_session
        
        mock_repo = AsyncMock()
        collection = Mock()
        collection.qdrant_staging = {
            "collection_name": "staging_test_123"
        }
        mock_repo.get_by_uuid.return_value = collection
        mock_repo.update = AsyncMock()
        mock_repo_class.return_value = mock_repo
        
        # Mock Qdrant to fail
        client = Mock()
        client.get_collections.side_effect = Exception("Qdrant error")
        mock_qdrant_manager.get_client.return_value = client
        
        # Clean up staging (should not raise)
        operation = {"type": "REINDEX"}
        await _cleanup_staging_resources("col-123", operation)
        
        # Database should still be updated
        mock_repo.update.assert_called_once_with("col-123", {"qdrant_staging": None})


class TestCleanupOldResults:
    """Test cleanup of old Celery results."""

    def test_cleanup_old_results_default(self):
        """Test cleanup with default parameters."""
        result = cleanup_old_results()
        
        assert "celery_results_deleted" in result
        assert "old_operations_marked" in result
        assert "errors" in result
        assert isinstance(result["errors"], list)

    def test_cleanup_old_results_custom_days(self):
        """Test cleanup with custom retention period."""
        result = cleanup_old_results(days_to_keep=30)
        
        assert "celery_results_deleted" in result
        assert result["celery_results_deleted"] >= 0

    @patch("packages.webui.tasks.logger")
    def test_cleanup_old_results_with_error(self, mock_logger):
        """Test cleanup handles errors gracefully."""
        with patch("packages.webui.tasks.datetime") as mock_datetime:
            mock_datetime.now.side_effect = Exception("Time error")
            
            result = cleanup_old_results()
            
            assert len(result["errors"]) > 0
            assert "Time error" in result["errors"][0]


class TestConcurrentOperations:
    """Test handling of concurrent operations."""

    @pytest.mark.asyncio
    async def test_multiple_updaters_same_operation(self):
        """Test multiple updaters for same operation don't conflict."""
        operation_id = "shared-op-123"
        
        # Create multiple updaters
        updaters = [
            CeleryTaskWithOperationUpdates(operation_id)
            for _ in range(3)
        ]
        
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.xadd = AsyncMock()
            mock_redis.expire = AsyncMock()
            mock_redis.close = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_from_url.return_value = mock_redis
            
            # Send updates concurrently
            tasks = []
            for i, updater in enumerate(updaters):
                async with updater:
                    tasks.append(updater.send_update(f"update_{i}", {"index": i}))
            
            await asyncio.gather(*tasks)
            
            # All updates should be sent
            assert mock_redis.xadd.call_count >= len(updaters)

    @pytest.mark.asyncio
    async def test_concurrent_operation_processing(self):
        """Test concurrent processing doesn't cause issues."""
        # This is more of a design validation test
        # In practice, Celery handles concurrency at the task level
        
        operation_ids = ["op-1", "op-2", "op-3"]
        
        async def process_mock_operation(op_id):
            """Mock operation processing."""
            await asyncio.sleep(0.1)  # Simulate work
            return {"operation_id": op_id, "success": True}
        
        # Process operations concurrently
        results = await asyncio.gather(*[
            process_mock_operation(op_id) for op_id in operation_ids
        ])
        
        # Verify all completed
        assert len(results) == 3
        assert all(r["success"] for r in results)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_updater_with_invalid_redis_url(self):
        """Test updater handles invalid Redis URL."""
        updater = CeleryTaskWithOperationUpdates("test-op")
        
        # Override Redis URL to invalid value
        updater.redis_url = "invalid://url"
        
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_from_url.side_effect = Exception("Invalid URL")
            
            # Context manager should raise on entry
            with pytest.raises(Exception, match="Invalid URL"):
                async with updater:
                    pass

    def test_cleanup_delay_calculation_edge_cases(self):
        """Test cleanup delay calculation edge cases."""
        from packages.webui.tasks import (
            CLEANUP_DELAY_MAX_SECONDS,
            CLEANUP_DELAY_MIN_SECONDS,
            calculate_cleanup_delay,
        )
        
        # Negative vector count should use minimum
        assert calculate_cleanup_delay(-100) == CLEANUP_DELAY_MIN_SECONDS
        
        # Very large number should cap at maximum
        assert calculate_cleanup_delay(10**9) == CLEANUP_DELAY_MAX_SECONDS

    @pytest.mark.asyncio
    @patch("shared.database.database.AsyncSessionLocal")
    async def test_audit_log_with_circular_reference(self, mock_session_local):
        """Test audit logging handles circular references in details."""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session_local.return_value = mock_session
        
        # Create circular reference
        details = {"a": {"b": None}}
        details["a"]["b"] = details["a"]  # Circular ref
        
        # Should handle without infinite recursion
        await _audit_log_operation(
            collection_id="col-123",
            operation_id=456,
            user_id=1,
            action="test_circular",
            details=details
        )
        
        # Should complete without error
        mock_session.commit.assert_called_once()


# Performance and stress tests
@pytest.mark.performance
class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.asyncio
    async def test_large_batch_updates(self):
        """Test sending many updates in quick succession."""
        updater = CeleryTaskWithOperationUpdates("perf-test-op")
        
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.xadd = AsyncMock()
            mock_redis.expire = AsyncMock()
            mock_redis.close = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_from_url.return_value = mock_redis
            
            async with updater:
                # Send many updates
                for i in range(100):
                    await updater.send_update(
                        "progress",
                        {"index": i, "total": 100, "percent": i}
                    )
            
            # All updates should be sent
            assert mock_redis.xadd.call_count == 100

    def test_sanitize_large_audit_details(self):
        """Test sanitization performance with large nested structure."""
        # Create large nested structure
        large_details = {
            f"level1_{i}": {
                f"level2_{j}": {
                    "data": f"value_{i}_{j}",
                    "password": "should_be_removed",
                    "path": f"/home/user{i}/data{j}.txt"
                }
                for j in range(10)
            }
            for i in range(10)
        }
        
        # Should complete in reasonable time
        import time
        start = time.time()
        sanitized = _sanitize_audit_details(large_details)
        duration = time.time() - start
        
        # Should be fast even with large structure
        assert duration < 1.0  # Less than 1 second
        
        # Verify sanitization worked
        for i in range(10):
            for j in range(10):
                level2 = sanitized[f"level1_{i}"][f"level2_{j}"]
                assert "password" not in level2
                assert "/home/~/" in level2["path"]