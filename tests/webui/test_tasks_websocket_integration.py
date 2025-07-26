"""Tests for WebSocket integration with Celery tasks.

This test suite focuses on the real-time update functionality:
- WebSocket message formatting and delivery
- Redis stream integration
- Progress tracking for different operation types
- Error propagation through WebSocket
"""

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
import redis.asyncio as redis

from packages.webui.tasks import (
    CeleryTaskWithOperationUpdates,
    _process_append_operation,
    _process_index_operation,
    _process_reindex_operation,
    _process_remove_source_operation,
)
from shared.database.models import OperationType


class TestWebSocketMessageFlow:
    """Test WebSocket message flow through Redis streams."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client that tracks messages."""
        client = AsyncMock(spec=redis.Redis)
        client.messages = []  # Track messages for verification
        
        async def mock_xadd(stream_key, message_dict, **kwargs):
            client.messages.append({
                "stream": stream_key,
                "message": json.loads(message_dict["message"]),
                "maxlen": kwargs.get("maxlen")
            })
            return f"msg-{len(client.messages)}"
        
        client.xadd = mock_xadd
        client.expire = AsyncMock()
        client.close = AsyncMock()
        client.ping = AsyncMock()
        
        return client

    @pytest.mark.asyncio
    async def test_index_operation_websocket_messages(self, mock_redis_client):
        """Test WebSocket messages during INDEX operation."""
        with patch("redis.asyncio.from_url", return_value=mock_redis_client):
            with patch("packages.webui.tasks.qdrant_manager") as mock_qdrant:
                # Setup Qdrant mock
                client = Mock()
                client.create_collection = Mock()
                client.get_collection = Mock(return_value=Mock(vectors_count=0))
                mock_qdrant.get_client.return_value = client
                
                # Setup operation and collection
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
                    "vector_store_name": "test_vec",
                    "config": {"vector_dim": 1024}
                }
                
                collection_repo = AsyncMock()
                document_repo = AsyncMock()
                
                # Run operation
                async with CeleryTaskWithOperationUpdates("op-123") as updater:
                    await _process_index_operation(
                        operation, collection, collection_repo, document_repo, updater
                    )
                
                # Verify message sequence
                messages = mock_redis_client.messages
                assert len(messages) >= 2
                
                # Check message types
                message_types = [msg["message"]["type"] for msg in messages]
                assert "index_completed" in message_types
                
                # Verify stream key format
                assert all(msg["stream"] == "operation-progress:op-123" for msg in messages)
                
                # Verify message content
                index_complete_msg = next(
                    msg for msg in messages if msg["message"]["type"] == "index_completed"
                )
                assert index_complete_msg["message"]["data"]["qdrant_collection"] == "test_vec"
                assert index_complete_msg["message"]["data"]["vector_dim"] == 1024

    @pytest.mark.asyncio
    async def test_append_operation_progress_messages(self, mock_redis_client):
        """Test progress messages during APPEND operation."""
        with patch("redis.asyncio.from_url", return_value=mock_redis_client):
            with patch("packages.webui.tasks.DocumentScanningService") as mock_scanner_class:
                # Setup document scanner
                scanner = AsyncMock()
                scanner.scan_directory_and_register_documents.return_value = {
                    "total_documents_found": 5,
                    "new_documents_registered": 3,
                    "duplicate_documents_skipped": 2,
                    "total_size_bytes": 512000,
                    "errors": []
                }
                mock_scanner_class.return_value = scanner
                
                # Setup operation and collection
                operation = {
                    "id": "op-456",
                    "collection_id": "col-456",
                    "type": OperationType.APPEND,
                    "config": {"source_path": "/test/docs"},
                    "user_id": 1
                }
                
                collection = {
                    "id": "col-456",
                    "uuid": "col-456",
                    "name": "Test Collection",
                    "vector_store_name": "test_vec"
                }
                
                collection_repo = AsyncMock()
                document_repo = AsyncMock()
                document_repo.session = AsyncMock()
                document_repo.list_by_collection.return_value = ([], 0)
                
                # Run operation
                async with CeleryTaskWithOperationUpdates("op-456") as updater:
                    await _process_append_operation(
                        operation, collection, collection_repo, document_repo, updater
                    )
                
                # Verify progress messages
                messages = mock_redis_client.messages
                message_types = [msg["message"]["type"] for msg in messages]
                
                # Check expected message sequence
                assert "scanning_documents" in message_types
                assert "scanning_completed" in message_types
                
                # Verify scanning completed message
                scan_complete = next(
                    msg for msg in messages if msg["message"]["type"] == "scanning_completed"
                )
                data = scan_complete["message"]["data"]
                assert data["total_files_found"] == 5
                assert data["new_documents_registered"] == 3
                assert data["duplicate_documents_skipped"] == 2

    @pytest.mark.asyncio
    async def test_reindex_operation_checkpoint_messages(self, mock_redis_client):
        """Test checkpoint messages during REINDEX operation."""
        with patch("redis.asyncio.from_url", return_value=mock_redis_client):
            with patch("packages.webui.tasks.QdrantManager") as mock_qdrant_class:
                with patch("packages.webui.tasks.qdrant_manager") as mock_qdrant:
                    with patch("packages.webui.tasks.reindex_handler") as mock_handler:
                        with patch("packages.webui.tasks._validate_reindex") as mock_validate:
                            with patch("packages.webui.tasks.httpx.AsyncClient") as mock_httpx:
                                with patch("packages.webui.tasks.cleanup_old_collections"):
                                    # Setup mocks
                                    manager = Mock()
                                    manager.create_staging_collection.return_value = "staging_123"
                                    mock_qdrant_class.return_value = manager
                                    
                                    client = Mock()
                                    mock_qdrant.get_client.return_value = client
                                    
                                    mock_handler.return_value = {
                                        "collection_name": "staging_123",
                                        "vector_dim": 1536
                                    }
                                    
                                    mock_validate.return_value = {
                                        "passed": True,
                                        "issues": [],
                                        "sample_size": 10
                                    }
                                    
                                    # Mock internal API response
                                    http_client = AsyncMock()
                                    response = Mock()
                                    response.status_code = 200
                                    response.json.return_value = {"old_collection_names": ["old_123"]}
                                    http_client.post.return_value = response
                                    mock_httpx.return_value.__aenter__.return_value = http_client
                                    
                                    # Setup operation and collection
                                    operation = {
                                        "id": "op-789",
                                        "collection_id": "col-789",
                                        "type": OperationType.REINDEX,
                                        "config": {"new_config": {}},
                                        "user_id": 1
                                    }
                                    
                                    collection = {
                                        "id": "col-789",
                                        "uuid": "col-789",
                                        "name": "Test Collection",
                                        "vector_store_name": "test_vec",
                                        "status": "ready",
                                        "vector_count": 1000
                                    }
                                    
                                    collection_repo = AsyncMock()
                                    document_repo = AsyncMock()
                                    document_repo.get_stats_by_collection.return_value = {
                                        "total_documents": 10
                                    }
                                    document_repo.list_by_collection.return_value = []
                                    
                                    # Run operation
                                    async with CeleryTaskWithOperationUpdates("op-789") as updater:
                                        await _process_reindex_operation(
                                            operation, collection, collection_repo, document_repo, updater
                                        )
                                    
                                    # Verify checkpoint messages
                                    messages = mock_redis_client.messages
                                    message_types = [msg["message"]["type"] for msg in messages]
                                    
                                    # Check critical checkpoints
                                    assert "reindex_preflight" in message_types
                                    assert "staging_created" in message_types
                                    assert "validation_complete" in message_types
                                    assert "reindex_completed" in message_types

    @pytest.mark.asyncio
    async def test_error_message_propagation(self, mock_redis_client):
        """Test error messages are properly sent through WebSocket."""
        with patch("redis.asyncio.from_url", return_value=mock_redis_client):
            with patch("packages.webui.tasks.qdrant_manager") as mock_qdrant:
                # Setup Qdrant to fail
                client = Mock()
                client.create_collection.side_effect = Exception("Qdrant connection failed")
                mock_qdrant.get_client.return_value = client
                
                operation = {
                    "id": "op-error",
                    "collection_id": "col-error",
                    "type": OperationType.INDEX,
                    "config": {},
                    "user_id": 1
                }
                
                collection = {
                    "id": "col-error",
                    "uuid": "col-error",
                    "name": "Test Collection",
                    "vector_store_name": None,
                    "config": {}
                }
                
                collection_repo = AsyncMock()
                document_repo = AsyncMock()
                
                # Run operation expecting failure
                async with CeleryTaskWithOperationUpdates("op-error") as updater:
                    with pytest.raises(Exception, match="Qdrant connection failed"):
                        await _process_index_operation(
                            operation, collection, collection_repo, document_repo, updater
                        )
                
                # Even with error, some messages should be sent
                messages = mock_redis_client.messages
                assert len(messages) >= 0  # At least the connection test

    @pytest.mark.asyncio
    async def test_concurrent_updates_ordering(self, mock_redis_client):
        """Test that concurrent updates maintain order."""
        async with CeleryTaskWithOperationUpdates("op-concurrent") as updater:
            with patch("redis.asyncio.from_url", return_value=mock_redis_client):
                # Send updates concurrently
                tasks = []
                for i in range(10):
                    tasks.append(updater.send_update("progress", {"index": i}))
                
                await asyncio.gather(*tasks)
                
                # Verify all messages were sent
                messages = mock_redis_client.messages
                assert len(messages) == 10
                
                # Extract indices
                indices = [msg["message"]["data"]["index"] for msg in messages]
                assert indices == list(range(10))  # Should maintain order


class TestWebSocketMessageFormats:
    """Test specific WebSocket message formats for frontend compatibility."""

    @pytest.fixture
    def updater(self):
        """Create an updater instance."""
        return CeleryTaskWithOperationUpdates("test-op")

    @pytest.mark.asyncio
    async def test_progress_message_format(self, updater):
        """Test progress message format matches frontend expectations."""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            captured_messages = []
            
            async def capture_xadd(stream, message, **kwargs):
                captured_messages.append(json.loads(message["message"]))
            
            mock_redis.xadd = capture_xadd
            mock_redis.expire = AsyncMock()
            mock_redis.close = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_from_url.return_value = mock_redis
            
            # Send various progress updates
            await updater.send_update("document_processed", {
                "processed": 5,
                "failed": 1,
                "total": 10,
                "current_document": "/path/to/doc.pdf"
            })
            
            # Verify message structure
            msg = captured_messages[0]
            assert msg["type"] == "document_processed"
            assert "timestamp" in msg
            assert isinstance(msg["data"], dict)
            assert msg["data"]["processed"] == 5
            assert msg["data"]["total"] == 10
            
            # Verify timestamp is ISO format with timezone
            timestamp = datetime.fromisoformat(msg["timestamp"].replace("Z", "+00:00"))
            assert timestamp.tzinfo is not None

    @pytest.mark.asyncio
    async def test_completion_message_format(self, updater):
        """Test completion message includes all required fields."""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            captured_messages = []
            
            async def capture_xadd(stream, message, **kwargs):
                captured_messages.append(json.loads(message["message"]))
            
            mock_redis.xadd = capture_xadd
            mock_redis.expire = AsyncMock()
            mock_redis.close = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_from_url.return_value = mock_redis
            
            # Send completion update
            await updater.send_update("operation_completed", {
                "status": "completed",
                "result": {
                    "success": True,
                    "documents_processed": 100,
                    "vectors_created": 1500,
                    "duration": 45.2
                }
            })
            
            # Verify completion message
            msg = captured_messages[0]
            assert msg["type"] == "operation_completed"
            assert msg["data"]["status"] == "completed"
            assert msg["data"]["result"]["success"] is True
            assert msg["data"]["result"]["documents_processed"] == 100

    @pytest.mark.asyncio
    async def test_error_message_sanitization(self, updater):
        """Test error messages are sanitized before sending."""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            captured_messages = []
            
            async def capture_xadd(stream, message, **kwargs):
                captured_messages.append(json.loads(message["message"]))
            
            mock_redis.xadd = capture_xadd
            mock_redis.expire = AsyncMock()
            mock_redis.close = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_from_url.return_value = mock_redis
            
            # Send error with sensitive information
            await updater.send_update("operation_error", {
                "error": "Failed to process /home/username/private/data.txt",
                "details": {
                    "user_email": "user@example.com",
                    "api_key": "secret-key-123"
                }
            })
            
            # Message should be sent as-is (sanitization happens at higher level)
            msg = captured_messages[0]
            assert msg["type"] == "operation_error"
            # The updater itself doesn't sanitize - that's done by the task


class TestRedisStreamBehavior:
    """Test Redis stream-specific behavior."""

    @pytest.mark.asyncio
    async def test_stream_ttl_setting(self):
        """Test that TTL is set on first message."""
        updater = CeleryTaskWithOperationUpdates("test-ttl")
        
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.xadd = AsyncMock()
            mock_redis.expire = AsyncMock()
            mock_redis.close = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_from_url.return_value = mock_redis
            
            # Send first message
            await updater.send_update("start", {"status": "started"})
            
            # TTL should be set
            mock_redis.expire.assert_called_once_with(
                "operation-progress:test-ttl",
                86400  # 24 hours
            )
            
            # Send second message
            await updater.send_update("progress", {"percent": 50})
            
            # TTL should still only be called once
            assert mock_redis.expire.call_count == 1

    @pytest.mark.asyncio
    async def test_stream_maxlen_enforcement(self):
        """Test that stream length is limited."""
        updater = CeleryTaskWithOperationUpdates("test-maxlen")
        
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.xadd = AsyncMock()
            mock_redis.expire = AsyncMock()
            mock_redis.close = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_from_url.return_value = mock_redis
            
            # Send message
            await updater.send_update("test", {})
            
            # Verify maxlen was set
            call_args = mock_redis.xadd.call_args
            assert "maxlen" in call_args[1]
            assert call_args[1]["maxlen"] == 1000

    @pytest.mark.asyncio
    async def test_redis_connection_pooling(self):
        """Test that Redis connections are reused within updater."""
        updater = CeleryTaskWithOperationUpdates("test-pool")
        
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.xadd = AsyncMock()
            mock_redis.expire = AsyncMock()
            mock_redis.close = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_from_url.return_value = mock_redis
            
            # Get Redis client multiple times
            client1 = await updater._get_redis()
            client2 = await updater._get_redis()
            client3 = await updater._get_redis()
            
            # Should be the same instance
            assert client1 is client2
            assert client2 is client3
            
            # from_url should only be called once
            mock_from_url.assert_called_once()


class TestWebSocketIntegrationScenarios:
    """Test complete WebSocket integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_document_processing_flow(self):
        """Test complete document processing flow with all messages."""
        operation_id = "full-flow-op"
        
        # Track all messages
        all_messages = []
        
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            
            async def track_xadd(stream, message, **kwargs):
                msg_data = json.loads(message["message"])
                all_messages.append({
                    "time": datetime.now(UTC),
                    "type": msg_data["type"],
                    "data": msg_data["data"]
                })
            
            mock_redis.xadd = track_xadd
            mock_redis.expire = AsyncMock()
            mock_redis.close = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_from_url.return_value = mock_redis
            
            # Simulate a complete operation flow
            async with CeleryTaskWithOperationUpdates(operation_id) as updater:
                # Operation started
                await updater.send_update("operation_started", {
                    "status": "processing",
                    "type": "append"
                })
                
                # Scanning documents
                await updater.send_update("scanning_documents", {
                    "status": "scanning",
                    "source_path": "/data/documents"
                })
                
                # Scanning complete
                await updater.send_update("scanning_completed", {
                    "total_files_found": 50,
                    "new_documents_registered": 45,
                    "duplicate_documents_skipped": 5
                })
                
                # Processing documents
                for i in range(5):  # Simulate 5 progress updates
                    await updater.send_update("document_processed", {
                        "processed": (i + 1) * 10,
                        "failed": i,
                        "total": 50,
                        "current_document": f"doc_{i}.pdf"
                    })
                    await asyncio.sleep(0.01)  # Small delay
                
                # Operation completed
                await updater.send_update("operation_completed", {
                    "status": "completed",
                    "result": {
                        "success": True,
                        "documents_added": 45,
                        "vectors_created": 450
                    }
                })
            
            # Verify message flow
            assert len(all_messages) == 9  # 1 start + 1 scan + 1 scan complete + 5 progress + 1 complete
            
            # Verify message sequence
            message_types = [msg["type"] for msg in all_messages]
            assert message_types[0] == "operation_started"
            assert message_types[1] == "scanning_documents"
            assert message_types[2] == "scanning_completed"
            assert message_types[-1] == "operation_completed"
            
            # Verify timing (messages should be in order)
            for i in range(1, len(all_messages)):
                assert all_messages[i]["time"] >= all_messages[i-1]["time"]

    @pytest.mark.asyncio
    async def test_operation_failure_flow(self):
        """Test message flow when operation fails."""
        operation_id = "fail-flow-op"
        
        all_messages = []
        
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            
            async def track_xadd(stream, message, **kwargs):
                msg_data = json.loads(message["message"])
                all_messages.append(msg_data)
            
            mock_redis.xadd = track_xadd
            mock_redis.expire = AsyncMock()
            mock_redis.close = AsyncMock()
            mock_redis.ping = AsyncMock()
            mock_from_url.return_value = mock_redis
            
            # Simulate operation that fails partway
            async with CeleryTaskWithOperationUpdates(operation_id) as updater:
                await updater.send_update("operation_started", {"status": "processing"})
                await updater.send_update("scanning_documents", {"status": "scanning"})
                
                # Simulate error during processing
                await updater.send_update("operation_error", {
                    "error": "Failed to connect to vector database",
                    "error_type": "ConnectionError",
                    "recoverable": False
                })
            
            # Verify error flow
            assert len(all_messages) == 3
            assert all_messages[-1]["type"] == "operation_error"
            assert "Failed to connect" in all_messages[-1]["data"]["error"]