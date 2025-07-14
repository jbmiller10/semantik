"""Test suite for CeleryTaskWithUpdates class."""

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
import redis.asyncio as redis
from webui.tasks import CeleryTaskWithUpdates


class TestCeleryTaskWithUpdates:
    """Test suite for CeleryTaskWithUpdates class."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock(spec=redis.Redis)
        mock.xadd = AsyncMock()
        mock.expire = AsyncMock()
        mock.close = AsyncMock()
        return mock

    @pytest.fixture
    def task_updater(self):
        """Create a CeleryTaskWithUpdates instance."""
        return CeleryTaskWithUpdates("test-job-123")

    @pytest.mark.asyncio
    async def test_initialization(self, task_updater):
        """Test proper initialization of CeleryTaskWithUpdates."""
        assert task_updater.job_id == "test-job-123"
        assert task_updater.stream_key == "job:updates:test-job-123"
        assert task_updater._redis_client is None

    @pytest.mark.asyncio
    async def test_get_redis_creates_client(self, task_updater, mock_redis):
        """Test that _get_redis creates and caches Redis client."""
        with patch("webui.tasks.redis.from_url", return_value=mock_redis):
            # First call should create client
            client1 = await task_updater._get_redis()
            assert client1 == mock_redis
            assert task_updater._redis_client == mock_redis
            
            # Second call should return cached client
            client2 = await task_updater._get_redis()
            assert client2 == mock_redis
            
            # Verify from_url was called only once
            redis.asyncio.from_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_update_success(self, task_updater, mock_redis):
        """Test successful sending of update to Redis stream."""
        with patch("webui.tasks.redis.from_url", return_value=mock_redis):
            update_data = {
                "progress": 50,
                "current_file": "document.pdf",
                "status": "processing"
            }
            
            await task_updater.send_update("progress", update_data)
            
            # Verify xadd was called
            mock_redis.xadd.assert_called_once()
            
            # Verify stream key and maxlen
            call_args = mock_redis.xadd.call_args[0]
            assert call_args[0] == "job:updates:test-job-123"
            assert call_args[2] == 1000  # maxlen
            
            # Verify message format
            message_data = call_args[1]
            message = json.loads(message_data["message"])
            assert message["type"] == "progress"
            assert message["data"] == update_data
            assert "timestamp" in message
            
            # Verify timestamp is ISO format
            timestamp = datetime.fromisoformat(message["timestamp"].replace("Z", "+00:00"))
            assert timestamp.tzinfo is not None

    @pytest.mark.asyncio
    async def test_send_update_sets_ttl(self, task_updater, mock_redis):
        """Test that TTL is set on Redis stream."""
        with patch("webui.tasks.redis.from_url", return_value=mock_redis):
            await task_updater.send_update("start", {"status": "started"})
            
            # Verify expire was called with 24 hours (86400 seconds)
            mock_redis.expire.assert_called_once_with(
                "job:updates:test-job-123",
                86400
            )

    @pytest.mark.asyncio
    async def test_send_update_handles_redis_error(self, task_updater, mock_redis):
        """Test graceful handling of Redis errors."""
        mock_redis.xadd.side_effect = Exception("Redis connection lost")
        
        with patch("webui.tasks.redis.from_url", return_value=mock_redis):
            # Should not raise exception
            await task_updater.send_update("error", {"error": "test"})
            
            # Verify error was attempted
            mock_redis.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_multiple_updates(self, task_updater, mock_redis):
        """Test sending multiple updates in sequence."""
        with patch("webui.tasks.redis.from_url", return_value=mock_redis):
            # Send multiple updates
            await task_updater.send_update("start", {"status": "started"})
            await task_updater.send_update("progress", {"progress": 25})
            await task_updater.send_update("progress", {"progress": 50})
            await task_updater.send_update("complete", {"status": "completed"})
            
            # Verify all updates sent
            assert mock_redis.xadd.call_count == 4
            
            # Verify different update types
            sent_messages = []
            for call in mock_redis.xadd.call_args_list:
                message_data = call[0][1]
                message = json.loads(message_data["message"])
                sent_messages.append(message)
            
            assert sent_messages[0]["type"] == "start"
            assert sent_messages[1]["type"] == "progress"
            assert sent_messages[2]["type"] == "progress"
            assert sent_messages[3]["type"] == "complete"
            
            # Verify timestamps are increasing
            timestamps = [msg["timestamp"] for msg in sent_messages]
            assert timestamps == sorted(timestamps)

    @pytest.mark.asyncio
    async def test_close(self, task_updater, mock_redis):
        """Test proper cleanup of Redis connection."""
        with patch("webui.tasks.redis.from_url", return_value=mock_redis):
            # Create connection
            await task_updater._get_redis()
            
            # Close it
            await task_updater.close()
            
            # Verify Redis client was closed
            mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_connection(self, task_updater):
        """Test close when no Redis connection exists."""
        # Should not raise exception
        await task_updater.close()

    @pytest.mark.asyncio
    async def test_update_types(self, task_updater, mock_redis):
        """Test various update types that might be sent."""
        with patch("webui.tasks.redis.from_url", return_value=mock_redis):
            # Test different update scenarios
            test_updates = [
                ("file_start", {"file": "doc1.pdf", "index": 1, "total": 10}),
                ("file_progress", {"file": "doc1.pdf", "chunks_processed": 5, "total_chunks": 20}),
                ("file_complete", {"file": "doc1.pdf", "vectors_created": 15}),
                ("file_error", {"file": "doc2.txt", "error": "Extraction failed"}),
                ("job_error", {"error": "Model loading failed", "traceback": "..."}),
                ("status", {"status": "completed", "summary": {"processed": 10, "failed": 1}}),
            ]
            
            for update_type, data in test_updates:
                await task_updater.send_update(update_type, data)
            
            # Verify all updates sent
            assert mock_redis.xadd.call_count == len(test_updates)
            
            # Verify each update has correct structure
            for i, (update_type, data) in enumerate(test_updates):
                call_args = mock_redis.xadd.call_args_list[i][0]
                message = json.loads(call_args[1]["message"])
                assert message["type"] == update_type
                assert message["data"] == data

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, task_updater, mock_redis):
        """Test sending updates concurrently."""
        with patch("webui.tasks.redis.from_url", return_value=mock_redis):
            # Send multiple updates concurrently
            update_tasks = [
                task_updater.send_update("progress", {"progress": i})
                for i in range(10)
            ]
            
            await asyncio.gather(*update_tasks)
            
            # Verify all updates were sent
            assert mock_redis.xadd.call_count == 10

    @pytest.mark.asyncio
    async def test_stream_key_format(self, mock_redis):
        """Test that stream keys are properly formatted for different job IDs."""
        with patch("webui.tasks.redis.from_url", return_value=mock_redis):
            # Test various job ID formats
            job_ids = [
                "simple-job",
                "job_with_underscore",
                "job-with-uuid-123e4567-e89b-12d3-a456-426614174000",
                "JOB_UPPERCASE",
            ]
            
            for job_id in job_ids:
                updater = CeleryTaskWithUpdates(job_id)
                await updater.send_update("test", {})
                
                # Verify correct stream key used
                call_args = mock_redis.xadd.call_args[0]
                expected_key = f"job:updates:{job_id}"
                assert call_args[0] == expected_key
                
                # Reset mock for next iteration
                mock_redis.xadd.reset_mock()

    @pytest.mark.asyncio
    async def test_message_size_limits(self, task_updater, mock_redis):
        """Test handling of large messages."""
        with patch("webui.tasks.redis.from_url", return_value=mock_redis):
            # Create a large data payload
            large_data = {
                "file_list": [f"file_{i}.pdf" for i in range(1000)],
                "large_text": "x" * 10000,
                "nested_data": {
                    f"key_{i}": {"value": i, "description": f"Description {i}"}
                    for i in range(100)
                }
            }
            
            await task_updater.send_update("large_update", large_data)
            
            # Verify update was sent successfully
            mock_redis.xadd.assert_called_once()
            
            # Verify message can be deserialized
            call_args = mock_redis.xadd.call_args[0]
            message = json.loads(call_args[1]["message"])
            assert message["data"] == large_data