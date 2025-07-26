"""Comprehensive test suite for webui.tasks module - covering Celery task processing.

This test suite focuses on testing the core collection operation tasks:
- process_collection_operation (the main task entry point)
- INDEX operations (initial collection creation)
- APPEND operations (adding documents to collections)
- REINDEX operations (blue-green reindexing)
- REMOVE_SOURCE operations (removing documents from a source)

The tests cover success scenarios, failure scenarios, WebSocket notifications,
and error handling with proper mocking of external dependencies.
"""

import asyncio
import json
import time
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from celery import states
from celery.exceptions import SoftTimeLimitExceeded
from qdrant_client.models import PointStruct

from packages.webui.tasks import (
    CeleryTaskWithOperationUpdates,
    _handle_task_failure,
    _process_append_operation,
    _process_collection_operation_async,
    _process_index_operation,
    _process_reindex_operation,
    _process_remove_source_operation,
    _sanitize_error_message,
    _validate_reindex,
    calculate_cleanup_delay,
    cleanup_old_collections,
    cleanup_qdrant_collections,
    process_collection_operation,
)
from shared.database.models import (
    CollectionStatus,
    DocumentStatus,
    OperationStatus,
    OperationType,
)


class TestCeleryTaskHelpers:
    """Test helper functions used by Celery tasks."""

    def test_calculate_cleanup_delay(self):
        """Test cleanup delay calculation based on vector count."""
        # Test minimum delay
        assert calculate_cleanup_delay(0) == 300  # 5 minutes minimum
        assert calculate_cleanup_delay(5000) == 300  # Still minimum

        # Test scaling
        assert calculate_cleanup_delay(10000) == 360  # 5 min + 1 min
        assert calculate_cleanup_delay(50000) == 600  # 5 min + 5 min
        assert calculate_cleanup_delay(100000) == 900  # 5 min + 10 min

        # Test maximum cap
        assert calculate_cleanup_delay(500000) == 1800  # Capped at 30 min
        assert calculate_cleanup_delay(1000000) == 1800  # Still capped

    def test_sanitize_error_message(self):
        """Test error message sanitization to remove PII."""
        # Test home directory sanitization
        assert _sanitize_error_message("/home/username/file.txt") == "/home/~/file.txt"
        assert _sanitize_error_message("/Users/johndoe/Documents") == "/Users/~/Documents"
        assert _sanitize_error_message("C:\\Users\\JaneDoe\\Desktop") == "C:\\Users\\~\\Desktop"

        # Test email sanitization
        assert _sanitize_error_message("Error from user@example.com") == "Error from [email]"
        assert _sanitize_error_message("Contact admin@company.org for help") == "Contact [email] for help"

        # Test combined sanitization
        msg = "User john@example.com at /home/john/projects failed"
        expected = "User [email] at /home/~/projects failed"
        assert _sanitize_error_message(msg) == expected


@pytest.mark.asyncio
class TestCeleryTaskWithOperationUpdates:
    """Test the operation updates helper class."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.xadd = AsyncMock()
        mock.expire = AsyncMock()
        mock.close = AsyncMock()
        mock.ping = AsyncMock()
        return mock

    @pytest.fixture
    def updater(self):
        """Create an updater instance."""
        return CeleryTaskWithOperationUpdates("test-op-123")

    async def test_context_manager_lifecycle(self, updater, mock_redis):
        """Test context manager properly manages Redis connection."""
        # Create an async function that returns the mock_redis
        async def async_from_url(*args, **kwargs):
            return mock_redis
        
        with patch("redis.asyncio.from_url", side_effect=async_from_url):
            async with updater as u:
                assert u == updater
                mock_redis.ping.assert_called_once()

            # After exit, connection should be closed
            mock_redis.close.assert_called_once()

    async def test_send_update_formats_message(self, updater, mock_redis):
        """Test update message formatting."""
        # Create an async function that returns the mock_redis
        async def async_from_url(*args, **kwargs):
            return mock_redis
        
        with patch("redis.asyncio.from_url", side_effect=async_from_url):
            await updater.send_update("test_type", {"key": "value"})

            # Verify xadd was called with correct format
            mock_redis.xadd.assert_called_once()
            call_args = mock_redis.xadd.call_args
            stream_key = call_args[0][0]
            message_data = call_args[0][1]

            assert stream_key == "operation-progress:test-op-123"
            message = json.loads(message_data["message"])
            assert message["type"] == "test_type"
            assert message["data"] == {"key": "value"}
            assert "timestamp" in message

    async def test_send_update_handles_errors(self, updater, mock_redis):
        """Test graceful error handling in send_update."""
        mock_redis.xadd.side_effect = Exception("Redis error")
        
        # Create an async function that returns the mock_redis
        async def async_from_url(*args, **kwargs):
            return mock_redis
        
        with patch("redis.asyncio.from_url", side_effect=async_from_url):
            # Should not raise exception
            await updater.send_update("error_test", {})
            
            # Error was attempted
            mock_redis.xadd.assert_called_once()


class TestProcessCollectionOperation:
    """Test the main process_collection_operation task."""

    @pytest.fixture
    def mock_celery_task(self):
        """Create a mock Celery task instance."""
        task = Mock()
        task.request.id = "celery-task-123"
        task.retry = Mock(side_effect=Exception("Retry called"))
        return task

    @pytest.fixture
    def mock_repositories(self):
        """Create mock repository instances."""
        # Operation repository
        operation_repo = AsyncMock()
        operation_obj = Mock()
        operation_obj.uuid = "op-123"
        operation_obj.collection_id = "col-123"
        operation_obj.type = OperationType.INDEX
        operation_obj.config = {}
        operation_obj.user_id = 1
        operation_repo.get_by_uuid.return_value = operation_obj
        operation_repo.set_task_id = AsyncMock()
        operation_repo.update_status = AsyncMock()

        # Collection repository
        collection_repo = AsyncMock()
        collection_obj = Mock()
        collection_obj.id = "col-123"
        collection_obj.name = "Test Collection"
        collection_obj.vector_store_name = "test_collection_vec"
        collection_obj.config = {"vector_dim": 1024}
        collection_repo.get_by_uuid.return_value = collection_obj
        collection_repo.update = AsyncMock()
        collection_repo.update_status = AsyncMock()
        collection_repo.update_stats = AsyncMock()

        # Document repository
        document_repo = AsyncMock()
        document_repo.get_stats_by_collection.return_value = {
            "total_documents": 10,
            "total_chunks": 100,
            "total_size_bytes": 1024000
        }

        return {
            "operation": operation_repo,
            "collection": collection_repo,
            "document": document_repo
        }

    @patch("packages.webui.tasks.pg_connection_manager")
    @patch("packages.webui.tasks.AsyncSessionLocal")
    @patch("packages.webui.tasks.OperationRepository")
    @patch("packages.webui.tasks.CollectionRepository")
    @patch("packages.webui.tasks.DocumentRepository")
    async def test_process_collection_operation_index_success(
        self,
        mock_doc_repo_class,
        mock_col_repo_class,
        mock_op_repo_class,
        mock_session_local,
        mock_pg_manager,
        mock_repositories,
        mock_celery_task
    ):
        """Test successful INDEX operation processing."""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session_local.return_value = mock_session

        mock_op_repo_class.return_value = mock_repositories["operation"]
        mock_col_repo_class.return_value = mock_repositories["collection"]
        mock_doc_repo_class.return_value = mock_repositories["document"]

        # Mock the index operation handler
        with patch("packages.webui.tasks._process_index_operation") as mock_process_index:
            mock_process_index.return_value = {
                "success": True,
                "qdrant_collection": "test_collection_vec",
                "vector_dim": 1024
            }

            # Run the task
            result = await _process_collection_operation_async("op-123", mock_celery_task)

            # Verify task ID was set immediately
            mock_repositories["operation"].set_task_id.assert_called_once_with(
                "op-123", "celery-task-123"
            )

            # Verify operation status updates
            status_calls = mock_repositories["operation"].update_status.call_args_list
            assert len(status_calls) >= 2
            assert status_calls[0][0] == ("op-123", OperationStatus.PROCESSING)
            assert status_calls[-1][0] == ("op-123", OperationStatus.COMPLETED)

            # Verify collection status was updated
            mock_repositories["collection"].update_status.assert_called_with(
                "col-123", CollectionStatus.READY
            )

            # Verify result
            assert result["success"] is True
            assert result["qdrant_collection"] == "test_collection_vec"

    @patch("packages.webui.tasks.pg_connection_manager")
    @patch("packages.webui.tasks.AsyncSessionLocal")
    @patch("packages.webui.tasks.OperationRepository")
    @patch("packages.webui.tasks.CollectionRepository")
    @patch("packages.webui.tasks.DocumentRepository")
    async def test_process_collection_operation_failure_handling(
        self,
        mock_doc_repo_class,
        mock_col_repo_class,
        mock_op_repo_class,
        mock_session_local,
        mock_pg_manager,
        mock_repositories,
        mock_celery_task
    ):
        """Test failure handling in process_collection_operation."""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session_local.return_value = mock_session

        mock_op_repo_class.return_value = mock_repositories["operation"]
        mock_col_repo_class.return_value = mock_repositories["collection"]
        mock_doc_repo_class.return_value = mock_repositories["document"]

        # Mock the index operation to fail
        with patch("packages.webui.tasks._process_index_operation") as mock_process_index:
            mock_process_index.side_effect = Exception("Qdrant connection failed")

            # Run the task and expect exception
            with pytest.raises(Exception, match="Qdrant connection failed"):
                await _process_collection_operation_async("op-123", mock_celery_task)

            # Verify rollback was called
            mock_session.rollback.assert_called_once()

            # Verify operation status was updated to failed
            final_status_call = mock_repositories["operation"].update_status.call_args_list[-1]
            assert final_status_call[0][1] == OperationStatus.FAILED
            assert "Qdrant connection failed" in str(final_status_call[1].get("error_message", ""))

    def test_process_collection_operation_sync_wrapper(self, mock_celery_task):
        """Test the synchronous wrapper handles async execution."""
        with patch("packages.webui.tasks._process_collection_operation_async") as mock_async:
            mock_async.return_value = {"success": True}
            
            # Call sync wrapper - self is the first parameter for bound tasks
            result = process_collection_operation(mock_celery_task, "op-123")
            
            # Verify async function was called with correct parameters
            mock_async.assert_called_once_with("op-123", mock_celery_task)
            assert result == {"success": True}

    def test_process_collection_operation_retry_on_network_error(self, mock_celery_task):
        """Test that network errors trigger retry."""
        with patch("packages.webui.tasks._process_collection_operation_async") as mock_async:
            mock_async.side_effect = Exception("Network error")
            
            # Call sync wrapper - should retry (self is the first parameter for bound tasks)
            with pytest.raises(Exception, match="Retry called"):
                process_collection_operation(mock_celery_task, "op-123")
            
            # Verify retry was called with the exception
            mock_celery_task.retry.assert_called_once()
            retry_call = mock_celery_task.retry.call_args
            assert "exc" in retry_call[1]
            assert "countdown" in retry_call[1]
            assert retry_call[1]["countdown"] == 60


class TestIndexOperation:
    """Test INDEX operation processing."""

    @pytest.fixture
    def mock_qdrant_manager(self):
        """Create mock Qdrant manager."""
        manager = Mock()
        client = Mock()
        manager.get_client.return_value = client
        
        # Mock successful collection creation
        client.create_collection = Mock()
        collection_info = Mock()
        collection_info.vectors_count = 0
        client.get_collection.return_value = collection_info
        
        return manager

    @pytest.fixture
    def mock_updater(self):
        """Create mock operation updater."""
        updater = AsyncMock()
        updater.send_update = AsyncMock()
        return updater

    @patch("packages.webui.tasks.qdrant_manager")
    @patch("packages.webui.tasks.get_model_config")
    async def test_process_index_operation_success(
        self,
        mock_get_model_config,
        mock_qdrant_global,
        mock_qdrant_manager,
        mock_updater
    ):
        """Test successful INDEX operation."""
        # Setup mocks
        mock_qdrant_global.get_client.return_value = mock_qdrant_manager.get_client()
        mock_get_model_config.return_value = Mock(dimension=1536)
        
        operation = {
            "id": "op-123",
            "collection_id": "col-123",
            "type": OperationType.INDEX,
            "config": {},
            "user_id": 1
        }
        
        collection = {
            "id": "col-123",
            "uuid": "col-123",
            "name": "Test Collection",
            "vector_store_name": "col_test_123",
            "config": {"vector_dim": 1536},
            "embedding_model": "test-model"
        }
        
        collection_repo = AsyncMock()
        collection_repo.update = AsyncMock()
        
        document_repo = AsyncMock()
        
        # Run operation
        result = await _process_index_operation(
            operation, collection, collection_repo, document_repo, mock_updater
        )
        
        # Verify collection was created in Qdrant
        client = mock_qdrant_manager.get_client()
        client.create_collection.assert_called_once()
        call_args = client.create_collection.call_args
        assert call_args[1]["collection_name"] == "col_test_123"
        
        # Verify collection was updated in database
        collection_repo.update.assert_called_once_with(
            "col-123", {"vector_store_name": "col_test_123"}
        )
        
        # Verify success response
        assert result["success"] is True
        assert result["qdrant_collection"] == "col_test_123"
        assert result["vector_dim"] == 1536
        
        # Verify updates were sent
        mock_updater.send_update.assert_called()

    @patch("packages.webui.tasks.qdrant_manager")
    async def test_process_index_operation_qdrant_failure(
        self,
        mock_qdrant_global,
        mock_qdrant_manager,
        mock_updater
    ):
        """Test INDEX operation when Qdrant fails."""
        # Setup mocks
        client = mock_qdrant_manager.get_client()
        client.create_collection.side_effect = Exception("Qdrant unavailable")
        mock_qdrant_global.get_client.return_value = client
        
        operation = {
            "id": "op-123",
            "collection_id": "col-123",
            "type": OperationType.INDEX,
            "config": {},
            "user_id": 1
        }
        
        collection = {
            "id": "col-123",
            "uuid": "col-123",
            "name": "Test Collection",
            "vector_store_name": None,
            "config": {}
        }
        
        collection_repo = AsyncMock()
        document_repo = AsyncMock()
        
        # Run operation and expect failure
        with pytest.raises(Exception, match="Qdrant unavailable"):
            await _process_index_operation(
                operation, collection, collection_repo, document_repo, mock_updater
            )


class TestAppendOperation:
    """Test APPEND operation processing."""

    @pytest.fixture
    def mock_document_scanner(self):
        """Create mock document scanner."""
        scanner = AsyncMock()
        scanner.scan_directory_and_register_documents.return_value = {
            "total_documents_found": 10,
            "new_documents_registered": 8,
            "duplicate_documents_skipped": 2,
            "total_size_bytes": 1024000,
            "errors": []
        }
        return scanner

    @pytest.fixture
    def mock_documents(self):
        """Create mock documents."""
        docs = []
        for i in range(3):
            doc = Mock()
            doc.id = f"doc-{i}"
            doc.file_path = f"/test/doc{i}.txt"
            doc.chunk_count = 0
            doc.status = DocumentStatus.PENDING
            docs.append(doc)
        return docs

    @patch("packages.webui.tasks.DocumentScanningService")
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("packages.webui.tasks.httpx.AsyncClient")
    @patch("packages.webui.tasks.qdrant_manager")
    async def test_process_append_operation_success(
        self,
        mock_qdrant_global,
        mock_httpx,
        mock_extract,
        mock_scanner_class,
        mock_document_scanner,
        mock_documents,
        mock_updater
    ):
        """Test successful APPEND operation."""
        # Setup mocks
        mock_scanner_class.return_value = mock_document_scanner
        mock_extract.return_value = [("This is test content", {"page": 1})]
        
        # Mock httpx client for vecpipe API
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embeddings": [[0.1] * 1024]}
        mock_client.post.return_value = mock_response
        mock_httpx.return_value.__aenter__.return_value = mock_client
        
        # Mock Qdrant
        qdrant_client = Mock()
        collection_info = Mock()
        collection_info.points_count = 100
        qdrant_client.get_collection.return_value = collection_info
        mock_qdrant_global.get_client.return_value = qdrant_client
        
        operation = {
            "id": "op-123",
            "collection_id": "col-123",
            "type": OperationType.APPEND,
            "config": {"source_path": "/test/documents"},
            "user_id": 1
        }
        
        collection = {
            "id": "col-123",
            "uuid": "col-123",
            "name": "Test Collection",
            "vector_store_name": "col_test_123",
            "embedding_model": "test-model",
            "quantization": "float32",
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
        
        collection_repo = AsyncMock()
        collection_repo.update_stats = AsyncMock()
        
        document_repo = AsyncMock()
        document_repo.session = AsyncMock()
        document_repo.list_by_collection.return_value = (mock_documents, 3)
        document_repo.update_status = AsyncMock()
        document_repo.get_stats_by_collection.return_value = {
            "total_documents": 10,
            "total_chunks": 100,
            "total_size_bytes": 1024000
        }
        
        # Run operation
        result = await _process_append_operation(
            operation, collection, collection_repo, document_repo, mock_updater
        )
        
        # Verify document scanning
        mock_document_scanner.scan_directory_and_register_documents.assert_called_once()
        scan_args = mock_document_scanner.scan_directory_and_register_documents.call_args
        assert scan_args[1]["collection_id"] == "col-123"
        assert scan_args[1]["source_path"] == "/test/documents"
        
        # Verify embeddings were generated
        assert mock_client.post.call_count >= 1
        embed_call = mock_client.post.call_args_list[0]
        assert "vecpipe:8000/embed" in embed_call[0][0]
        
        # Verify success response
        assert result["success"] is True
        assert result["documents_added"] == 8
        assert result["source_path"] == "/test/documents"
        
        # Verify progress updates
        mock_updater.send_update.assert_called()

    @patch("packages.webui.tasks.DocumentScanningService")
    async def test_process_append_operation_no_source_path(
        self,
        mock_scanner_class,
        mock_updater
    ):
        """Test APPEND operation without source_path."""
        operation = {
            "id": "op-123",
            "collection_id": "col-123",
            "type": OperationType.APPEND,
            "config": {},  # Missing source_path
            "user_id": 1
        }
        
        collection = {"id": "col-123"}
        collection_repo = AsyncMock()
        document_repo = AsyncMock()
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="source_path is required"):
            await _process_append_operation(
                operation, collection, collection_repo, document_repo, mock_updater
            )


class TestReindexOperation:
    """Test REINDEX operation processing."""

    @pytest.fixture
    def mock_qdrant_manager_instance(self):
        """Create mock QdrantManager instance."""
        manager = Mock()
        manager.create_staging_collection.return_value = "staging_col_123_20240115_120000"
        manager.collection_exists.return_value = True
        manager.get_collection_info.return_value = Mock(vectors_count=1000)
        return manager

    @patch("packages.webui.tasks.QdrantManager")
    @patch("packages.webui.tasks.qdrant_manager")
    @patch("packages.webui.tasks.reindex_handler")
    @patch("packages.webui.tasks._validate_reindex")
    @patch("packages.webui.tasks.httpx.AsyncClient")
    @patch("packages.webui.tasks.extract_and_serialize_thread_safe")
    @patch("packages.webui.tasks.cleanup_old_collections")
    async def test_process_reindex_operation_success(
        self,
        mock_cleanup_task,
        mock_extract,
        mock_httpx,
        mock_validate,
        mock_reindex_handler,
        mock_qdrant_global,
        mock_qdrant_manager_class,
        mock_qdrant_manager_instance,
        mock_updater
    ):
        """Test successful REINDEX operation."""
        # Setup mocks
        mock_qdrant_manager_class.return_value = mock_qdrant_manager_instance
        qdrant_client = Mock()
        mock_qdrant_global.get_client.return_value = qdrant_client
        
        # Mock reindex handler
        mock_reindex_handler.return_value = {
            "collection_name": "staging_col_123_20240115_120000",
            "created_at": datetime.now(UTC).isoformat(),
            "vector_dim": 1536,
            "base_collection": "col_test_123"
        }
        
        # Mock validation
        mock_validate.return_value = {
            "passed": True,
            "issues": [],
            "warnings": [],
            "sample_size": 10,
            "old_count": 1000,
            "new_count": 1050
        }
        
        # Mock internal API call for atomic switch
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"old_collection_names": ["col_test_123_old"]}
        mock_client.post.return_value = mock_response
        mock_httpx.return_value.__aenter__.return_value = mock_client
        
        # Mock text extraction
        mock_extract.return_value = [("Test content", {"page": 1})]
        
        # Mock cleanup task
        mock_cleanup_task.apply_async.return_value = Mock(id="cleanup-task-123")
        
        operation = {
            "id": "op-123",
            "collection_id": "col-123",
            "type": OperationType.REINDEX,
            "config": {"new_config": {"chunk_size": 800}},
            "user_id": 1
        }
        
        collection = {
            "id": "col-123",
            "uuid": "col-123",
            "name": "Test Collection",
            "vector_store_name": "col_test_123",
            "status": CollectionStatus.READY,
            "config": {"chunk_size": 1000},
            "vector_count": 1000
        }
        
        collection_repo = AsyncMock()
        collection_repo.update = AsyncMock()
        
        document_repo = AsyncMock()
        document_repo.get_stats_by_collection.return_value = {"total_documents": 10}
        document_repo.list_by_collection.return_value = [
            {"id": "doc1", "file_path": "/test/doc1.txt", "status": DocumentStatus.COMPLETED}
        ]
        
        # Run operation
        result = await _process_reindex_operation(
            operation, collection, collection_repo, document_repo, mock_updater
        )
        
        # Verify staging collection was created
        mock_reindex_handler.assert_called_once()
        
        # Verify validation was performed
        mock_validate.assert_called_once()
        validate_args = mock_validate.call_args[0]
        assert validate_args[1] == "col_test_123"
        assert validate_args[2] == "staging_col_123_20240115_120000"
        
        # Verify atomic switch was called
        api_calls = [call for call in mock_client.post.call_args_list 
                     if "internal/complete-reindex" in str(call)]
        assert len(api_calls) == 1
        
        # Verify cleanup was scheduled
        mock_cleanup_task.apply_async.assert_called_once()
        
        # Verify success response
        assert result["success"] is True
        assert result["new_collection"] == "staging_col_123_20240115_120000"
        assert "cleanup_task_id" in result

    @patch("packages.webui.tasks.QdrantManager")
    @patch("packages.webui.tasks.qdrant_manager")
    async def test_process_reindex_operation_validation_failure(
        self,
        mock_qdrant_global,
        mock_qdrant_manager_class,
        mock_qdrant_manager_instance,
        mock_updater
    ):
        """Test REINDEX operation with validation failure."""
        # Setup mocks
        mock_qdrant_manager_class.return_value = mock_qdrant_manager_instance
        qdrant_client = Mock()
        qdrant_client.delete_collection = Mock()
        mock_qdrant_global.get_client.return_value = qdrant_client
        
        with patch("packages.webui.tasks.reindex_handler") as mock_handler:
            mock_handler.return_value = {
                "collection_name": "staging_col_123",
                "vector_dim": 1536
            }
            
            with patch("packages.webui.tasks._validate_reindex") as mock_validate:
                mock_validate.return_value = {
                    "passed": False,
                    "issues": ["Vector count mismatch: 1000 -> 500"],
                    "sample_size": 10
                }
                
                operation = {
                    "id": "op-123",
                    "collection_id": "col-123",
                    "type": OperationType.REINDEX,
                    "config": {},
                    "user_id": 1
                }
                
                collection = {
                    "id": "col-123",
                    "uuid": "col-123",
                    "vector_store_name": "col_test_123",
                    "status": CollectionStatus.READY
                }
                
                collection_repo = AsyncMock()
                collection_repo.update = AsyncMock()
                
                document_repo = AsyncMock()
                document_repo.get_stats_by_collection.return_value = {"total_documents": 10}
                document_repo.list_by_collection.return_value = []
                
                # Should raise validation error
                with pytest.raises(ValueError, match="Reindex validation failed"):
                    await _process_reindex_operation(
                        operation, collection, collection_repo, document_repo, mock_updater
                    )
                
                # Verify staging collection was cleaned up
                qdrant_client.delete_collection.assert_called_with("staging_col_123")
                collection_repo.update.assert_called_with("col-123", {"qdrant_staging": None})


class TestRemoveSourceOperation:
    """Test REMOVE_SOURCE operation processing."""

    @patch("packages.webui.tasks.AsyncSessionLocal")
    async def test_process_remove_source_operation_success(
        self,
        mock_session_local,
        mock_updater
    ):
        """Test successful REMOVE_SOURCE operation."""
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = AsyncMock()
        mock_session.begin.return_value.__aenter__ = AsyncMock()
        mock_session.begin.return_value.__aexit__ = AsyncMock()
        mock_session_local.return_value = mock_session
        
        operation = {
            "id": "op-123",
            "collection_id": "col-123",
            "type": OperationType.REMOVE_SOURCE,
            "config": {"source_path": "/test/old_docs"},
            "user_id": 1
        }
        
        collection = {
            "id": "col-123",
            "uuid": "col-123",
            "name": "Test Collection",
            "vector_store_name": "col_test_123",
            "vector_count": 1000
        }
        
        collection_repo = AsyncMock()
        
        document_repo = AsyncMock()
        documents = [
            {"id": "doc1", "file_path": "/test/old_docs/doc1.txt"},
            {"id": "doc2", "file_path": "/test/old_docs/doc2.txt"}
        ]
        document_repo.list_by_collection_and_source.return_value = documents
        document_repo.bulk_update_status = AsyncMock()
        document_repo.get_stats_by_collection.return_value = {
            "total_documents": 8,
            "total_chunks": 80,
            "total_size_bytes": 800000
        }
        
        # Mock the document and collection repos created in transaction
        with patch("packages.webui.tasks.DocumentRepository") as mock_doc_repo_class:
            with patch("packages.webui.tasks.CollectionRepository") as mock_col_repo_class:
                mock_doc_repo_tx = AsyncMock()
                mock_doc_repo_tx.bulk_update_status = AsyncMock()
                mock_doc_repo_tx.get_stats_by_collection.return_value = {
                    "total_documents": 8,
                    "total_chunks": 80,
                    "total_size_bytes": 800000
                }
                mock_doc_repo_class.return_value = mock_doc_repo_tx
                
                mock_col_repo_tx = AsyncMock()
                mock_col_repo_tx.update_stats = AsyncMock()
                mock_col_repo_class.return_value = mock_col_repo_tx
                
                # Run operation
                result = await _process_remove_source_operation(
                    operation, collection, collection_repo, document_repo, mock_updater
                )
        
        # Verify documents were marked as deleted
        mock_doc_repo_tx.bulk_update_status.assert_called_once_with(
            ["doc1", "doc2"], DocumentStatus.DELETED
        )
        
        # Verify collection stats were updated
        mock_col_repo_tx.update_stats.assert_called_once()
        
        # Verify success response
        assert result["success"] is True
        assert result["documents_removed"] == 2
        assert result["source_path"] == "/test/old_docs"

    async def test_process_remove_source_operation_no_documents(
        self,
        mock_updater
    ):
        """Test REMOVE_SOURCE operation with no documents found."""
        operation = {
            "id": "op-123",
            "collection_id": "col-123",
            "type": OperationType.REMOVE_SOURCE,
            "config": {"source_path": "/test/empty"},
            "user_id": 1
        }
        
        collection = {"id": "col-123"}
        collection_repo = AsyncMock()
        
        document_repo = AsyncMock()
        document_repo.list_by_collection_and_source.return_value = []
        
        # Run operation
        result = await _process_remove_source_operation(
            operation, collection, collection_repo, document_repo, mock_updater
        )
        
        # Verify response for empty source
        assert result["success"] is True
        assert result["documents_removed"] == 0
        assert result["source_path"] == "/test/empty"


class TestReindexValidation:
    """Test reindex validation logic."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        client = Mock()
        
        # Mock collection info
        old_info = Mock()
        old_info.points_count = 1000
        old_info.config.params.vectors.size = 1536
        
        new_info = Mock()
        new_info.points_count = 1050
        new_info.config.params.vectors.size = 1536
        
        client.get_collection.side_effect = lambda name: old_info if name == "old" else new_info
        
        # Mock scroll for sampling
        sample_points = [Mock(id=f"p{i}", vector=[0.1] * 1536) for i in range(10)]
        for point in sample_points:
            point.payload = {"doc_id": f"doc{i}" for i in range(10)}
        client.scroll.return_value = (sample_points, None)
        
        # Mock search results
        search_results = [Mock(score=0.95, payload={"doc_id": "doc1"}) for _ in range(5)]
        client.search.return_value = search_results
        
        return client

    async def test_validate_reindex_success(self, mock_qdrant_client):
        """Test successful reindex validation."""
        result = await _validate_reindex(mock_qdrant_client, "old", "new", sample_size=10)
        
        assert result["passed"] is True
        assert result["issues"] == []
        assert result["old_count"] == 1000
        assert result["new_count"] == 1050
        assert result["sample_size"] == 10

    async def test_validate_reindex_vector_count_mismatch(self, mock_qdrant_client):
        """Test validation failure due to vector count mismatch."""
        # Mock significant vector count difference
        new_info = Mock()
        new_info.points_count = 500  # 50% loss
        new_info.config.params.vectors.size = 1536
        
        mock_qdrant_client.get_collection.side_effect = lambda name: (
            Mock(points_count=1000, config=Mock(params=Mock(vectors=Mock(size=1536)))) 
            if name == "old" else new_info
        )
        
        result = await _validate_reindex(mock_qdrant_client, "old", "new")
        
        assert result["passed"] is False
        assert any("Vector count mismatch" in issue for issue in result["issues"])

    async def test_validate_reindex_empty_new_collection(self, mock_qdrant_client):
        """Test validation failure when new collection is empty."""
        # Mock empty new collection
        new_info = Mock()
        new_info.points_count = 0
        
        mock_qdrant_client.get_collection.side_effect = lambda name: (
            Mock(points_count=1000) if name == "old" else new_info
        )
        
        result = await _validate_reindex(mock_qdrant_client, "old", "new")
        
        assert result["passed"] is False
        assert any("no vectors" in issue for issue in result["issues"])


class TestTaskFailureHandling:
    """Test task failure handling and cleanup."""

    @patch("packages.webui.tasks.asyncio.run")
    def test_handle_task_failure_index_operation(self, mock_asyncio_run):
        """Test failure handling for INDEX operation."""
        # Mock the async handler
        async def mock_handler(op_id, exc, task_id):
            pass
        
        mock_asyncio_run.side_effect = lambda coro: None
        
        # Create mock Celery context
        self_mock = Mock()
        exc = Exception("Database connection failed")
        task_id = "task-123"
        args = (self_mock, "op-123")
        kwargs = {}
        einfo = None
        
        # Call failure handler
        _handle_task_failure(self_mock, exc, task_id, args, kwargs, einfo)
        
        # Verify async handler was called
        mock_asyncio_run.assert_called_once()

    @patch("packages.webui.tasks.AsyncSessionLocal")
    @patch("packages.webui.tasks.OperationRepository")
    @patch("packages.webui.tasks.CollectionRepository")
    async def test_handle_task_failure_async_index(
        self,
        mock_col_repo_class,
        mock_op_repo_class,
        mock_session_local
    ):
        """Test async failure handling for INDEX operation."""
        from packages.webui.tasks import _handle_task_failure_async
        
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()
        mock_session_local.return_value = mock_session
        
        # Mock repositories
        operation_repo = AsyncMock()
        operation_obj = Mock()
        operation_obj.uuid = "op-123"
        operation_obj.collection_id = "col-123"
        operation_obj.type = OperationType.INDEX
        operation_repo.get_by_uuid.return_value = operation_obj
        operation_repo.update_status = AsyncMock()
        mock_op_repo_class.return_value = operation_repo
        
        collection_repo = AsyncMock()
        collection_obj = Mock()
        collection_obj.id = "col-123"
        collection_obj.uuid = "col-123"
        collection_obj.status = CollectionStatus.PENDING
        collection_repo.get_by_uuid.return_value = collection_obj
        collection_repo.update_status = AsyncMock()
        mock_col_repo_class.return_value = collection_repo
        
        # Run failure handler
        exc = Exception("Qdrant initialization failed")
        await _handle_task_failure_async("op-123", exc, "task-123")
        
        # Verify operation status updated to failed
        operation_repo.update_status.assert_called_with(
            "op-123",
            OperationStatus.FAILED,
            error_message=pytest.StringContaining("Qdrant initialization failed")
        )
        
        # Verify collection status updated to error (for INDEX operation)
        collection_repo.update_status.assert_called_with(
            "col-123",
            CollectionStatus.ERROR,
            status_message=pytest.StringContaining("Initial indexing failed")
        )

    @patch("packages.webui.tasks.AsyncSessionLocal")
    @patch("packages.webui.tasks.OperationRepository")
    @patch("packages.webui.tasks.CollectionRepository")
    @patch("packages.webui.tasks._cleanup_staging_resources")
    async def test_handle_task_failure_async_reindex(
        self,
        mock_cleanup_staging,
        mock_col_repo_class,
        mock_op_repo_class,
        mock_session_local
    ):
        """Test async failure handling for REINDEX operation."""
        from packages.webui.tasks import _handle_task_failure_async
        
        # Setup mocks similar to above but for REINDEX
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.commit = AsyncMock()
        mock_session_local.return_value = mock_session
        
        operation_repo = AsyncMock()
        operation_obj = Mock()
        operation_obj.uuid = "op-123"
        operation_obj.collection_id = "col-123"
        operation_obj.type = OperationType.REINDEX
        operation_repo.get_by_uuid.return_value = operation_obj
        operation_repo.update_status = AsyncMock()
        mock_op_repo_class.return_value = operation_repo
        
        collection_repo = AsyncMock()
        collection_obj = Mock()
        collection_obj.id = "col-123"
        collection_obj.uuid = "col-123"
        collection_obj.status = CollectionStatus.READY
        collection_repo.get_by_uuid.return_value = collection_obj
        collection_repo.update_status = AsyncMock()
        mock_col_repo_class.return_value = collection_repo
        
        # Run failure handler
        exc = Exception("Validation failed")
        await _handle_task_failure_async("op-123", exc, "task-123")
        
        # Verify collection status updated to degraded (for REINDEX)
        collection_repo.update_status.assert_called_with(
            "col-123",
            CollectionStatus.DEGRADED,
            status_message=pytest.StringContaining("Re-indexing failed")
        )
        
        # Verify staging cleanup was called
        mock_cleanup_staging.assert_called_once()


class TestCleanupTasks:
    """Test cleanup tasks."""

    def test_cleanup_old_collections_success(self):
        """Test successful cleanup of old collections."""
        with patch("packages.webui.tasks.qdrant_manager") as mock_qdrant_manager:
            # Setup mocks
            client = Mock()
            mock_qdrant_manager.get_client.return_value = client
            
            # Mock collections exist
            from collections import namedtuple
            CollectionInfo = namedtuple("CollectionInfo", ["name"])
            collections_response = Mock()
            collections_response.collections = [
                CollectionInfo(name="col_old_1"),
                CollectionInfo(name="col_old_2")
            ]
            client.get_collections.return_value = collections_response
            client.delete_collection = Mock()
            
            # Run cleanup
            result = cleanup_old_collections(["col_old_1", "col_old_2"], "col-123")
            
            # Verify results
            assert result["collections_deleted"] == 2
            assert result["collections_failed"] == 0
            assert client.delete_collection.call_count == 2

    def test_cleanup_old_collections_partial_failure(self):
        """Test cleanup with some failures."""
        with patch("packages.webui.tasks.qdrant_manager") as mock_qdrant_manager:
            # Setup mocks
            client = Mock()
            mock_qdrant_manager.get_client.return_value = client
            
            # Mock collections exist
            from collections import namedtuple
            CollectionInfo = namedtuple("CollectionInfo", ["name"])
            collections_response = Mock()
            collections_response.collections = [
                CollectionInfo(name="col_old_1"),
                CollectionInfo(name="col_old_2")
            ]
            client.get_collections.return_value = collections_response
            
            # First deletion succeeds, second fails
            client.delete_collection.side_effect = [None, Exception("Permission denied")]
            
            # Run cleanup
            result = cleanup_old_collections(["col_old_1", "col_old_2"], "col-123")
            
            # Verify partial success
            assert result["collections_deleted"] == 1
            assert result["collections_failed"] == 1
            assert len(result["errors"]) == 1
            assert "Permission denied" in result["errors"][0]

    @patch("packages.webui.tasks.asyncio.run")
    @patch("packages.webui.tasks.qdrant_manager")
    @patch("packages.webui.tasks.QdrantManager")
    def test_cleanup_qdrant_collections_with_safety_checks(
        self,
        mock_qdrant_manager_class,
        mock_qdrant_global,
        mock_asyncio_run
    ):
        """Test enhanced cleanup with safety checks."""
        # Setup mocks
        client = Mock()
        mock_qdrant_global.get_client.return_value = client
        
        manager = Mock()
        manager.collection_exists.side_effect = [True, True, False]  # Third doesn't exist
        manager.get_collection_info.return_value = Mock(vectors_count=1000)
        manager._is_staging_collection_old.return_value = True
        mock_qdrant_manager_class.return_value = manager
        
        # Mock active collections check
        mock_asyncio_run.side_effect = lambda coro: (
            {"active_col"} if "_get_active_collections" in str(coro) else None
        )
        
        # Run cleanup
        result = cleanup_qdrant_collections([
            "_system",  # System collection
            "active_col",  # Active collection
            "old_col",  # Should be deleted
            "missing_col"  # Doesn't exist
        ])
        
        # Verify safety checks
        assert result["collections_skipped"] >= 3  # system, active, missing
        assert result["collections_deleted"] <= 1  # only old_col
        assert "_system" in result["safety_checks"]
        assert result["safety_checks"]["_system"] == "system_collection"
        assert result["safety_checks"]["active_col"] == "active_collection"


# Integration test for full task flow
@pytest.mark.integration
class TestTaskIntegration:
    """Integration tests for complete task flows."""

    @patch("packages.webui.tasks.process_collection_operation.apply_async")
    def test_task_scheduling(self, mock_apply_async):
        """Test that tasks can be scheduled properly."""
        # Mock task result
        mock_result = Mock()
        mock_result.id = "task-123"
        mock_apply_async.return_value = mock_result
        
        # Schedule a task
        result = process_collection_operation.apply_async(args=["op-123"])
        
        # Verify scheduling
        assert result.id == "task-123"
        mock_apply_async.assert_called_once_with(args=["op-123"])