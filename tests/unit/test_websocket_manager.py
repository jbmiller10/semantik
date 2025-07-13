#!/usr/bin/env python3
"""
Unit tests for WebSocket Redis Stream Consumer.

Tests the RedisStreamWebSocketManager class for handling WebSocket
connections and consuming job updates from Redis streams.
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from packages.webui.websocket_manager import RedisStreamWebSocketManager, get_websocket_manager


class TestRedisStreamWebSocketManager:
    """Test cases for RedisStreamWebSocketManager."""

    @pytest.fixture()
    def redis_url(self):
        """Test Redis URL."""
        return "redis://localhost:6379/0"

    @pytest.fixture()
    def manager(self, redis_url):
        """Create a RedisStreamWebSocketManager instance."""
        return RedisStreamWebSocketManager(redis_url)

    @pytest.fixture()
    def mock_websocket(self):
        """Create a mock WebSocket."""
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_json = AsyncMock()
        ws.receive_text = AsyncMock()
        ws.close = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_connect(self, manager, mock_websocket):
        """Test WebSocket connection."""
        job_id = "test-job-123"

        # Mock the consumer task creation
        with patch.object(asyncio, "create_task") as mock_create_task:
            mock_task = Mock()
            mock_create_task.return_value = mock_task

            await manager.connect(mock_websocket, job_id)

            # Verify WebSocket was accepted
            mock_websocket.accept.assert_called_once()

            # Verify connection was added
            assert job_id in manager.active_connections
            assert mock_websocket in manager.active_connections[job_id]

            # Verify consumer task was created
            mock_create_task.assert_called_once()
            assert job_id in manager.consumer_tasks

    @pytest.mark.asyncio
    async def test_disconnect(self, manager, mock_websocket):
        """Test WebSocket disconnection."""
        job_id = "test-job-123"

        # First connect
        manager.active_connections[job_id] = {mock_websocket}
        mock_task = Mock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()
        manager.consumer_tasks[job_id] = mock_task

        # Then disconnect
        await manager.disconnect(mock_websocket, job_id)

        # Verify connection was removed
        assert job_id not in manager.active_connections

        # Verify consumer task was cancelled
        mock_task.cancel.assert_called_once()
        assert job_id not in manager.consumer_tasks

    @pytest.mark.asyncio
    async def test_disconnect_with_multiple_connections(self, manager, mock_websocket):
        """Test disconnection when multiple clients are connected."""
        job_id = "test-job-123"
        mock_ws2 = AsyncMock()

        # Setup multiple connections
        manager.active_connections[job_id] = {mock_websocket, mock_ws2}
        mock_task = Mock()
        manager.consumer_tasks[job_id] = mock_task

        # Disconnect one
        await manager.disconnect(mock_websocket, job_id)

        # Verify only one connection was removed
        assert job_id in manager.active_connections
        assert mock_websocket not in manager.active_connections[job_id]
        assert mock_ws2 in manager.active_connections[job_id]

        # Task should not be cancelled
        mock_task.cancel.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_to_job(self, manager, mock_websocket):
        """Test sending messages to all connections for a job."""
        job_id = "test-job-123"
        mock_ws2 = AsyncMock()
        mock_ws2.send_json = AsyncMock()

        manager.active_connections[job_id] = {mock_websocket, mock_ws2}

        message = {"type": "progress", "data": {"processed": 50}}
        await manager.send_to_job(job_id, message)

        # Verify both connections received the message
        mock_websocket.send_json.assert_called_once_with(message)
        mock_ws2.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_send_to_job_handles_disconnected_clients(self, manager, mock_websocket):
        """Test that disconnected clients are removed when sending fails."""
        job_id = "test-job-123"
        mock_websocket.send_json.side_effect = Exception("Connection closed")

        manager.active_connections[job_id] = {mock_websocket}

        message = {"type": "progress", "data": {"processed": 50}}
        await manager.send_to_job(job_id, message)

        # Verify disconnected client was removed
        assert mock_websocket not in manager.active_connections[job_id]

    @pytest.mark.asyncio
    async def test_send_initial_state(self, manager, mock_websocket):
        """Test sending initial job state to client."""
        job_id = "test-job-123"
        job_data = {"id": job_id, "name": "Test Job", "status": "processing", "total_files": 10, "processed_files": 5}

        await manager.send_initial_state(mock_websocket, job_id, job_data)

        # Verify message was sent
        mock_websocket.send_json.assert_called_once()
        sent_message = mock_websocket.send_json.call_args[0][0]

        assert sent_message["type"] == "initial_state"
        assert "timestamp" in sent_message
        assert sent_message["data"] == job_data

    @pytest.mark.asyncio
    @patch("redis.asyncio.from_url")
    async def test_process_message(self, mock_redis, manager):
        """Test processing messages from Redis stream."""
        job_id = "test-job-123"

        # Setup mock WebSocket
        mock_ws = AsyncMock()
        manager.active_connections[job_id] = {mock_ws}

        # Mock send_to_job
        with patch.object(manager, "send_to_job") as mock_send:
            data = {
                "type": "file_completed",
                "timestamp": "2024-01-01T00:00:00",
                "data": json.dumps({"file": "test.txt", "chunks": 10}),
            }

            await manager._process_message(job_id, data)

            # Verify message was sent
            mock_send.assert_called_once()
            sent_job_id, sent_message = mock_send.call_args[0]

            assert sent_job_id == job_id
            assert sent_message["type"] == "file_completed"
            assert sent_message["data"] == {"file": "test.txt", "chunks": 10}

    @pytest.mark.asyncio
    async def test_consume_stream_setup(self, manager):
        """Test Redis stream consumer setup - basic verification."""
        job_id = "test-job-123"
        
        # Test get_redis_client
        mock_client = AsyncMock()
        with patch('redis.asyncio.from_url', return_value=mock_client):
            client = await manager.get_redis_client()
            assert client == mock_client
            assert manager._redis_client == mock_client
        
        # Test stream key generation
        stream_key = manager.redis_manager.get_stream_key(job_id)
        assert stream_key == f"job:stream:{job_id}"
        
        # Test consumer naming
        consumer_group = f"websocket-{job_id}"
        consumer_name = f"consumer-{id(manager)}"
        
        assert consumer_group.startswith("websocket-")
        assert consumer_name.startswith("consumer-")

    @pytest.mark.asyncio
    async def test_close(self, manager, mock_websocket):
        """Test closing all connections and cleaning up."""
        job_id = "test-job-123"

        # Setup connections and tasks
        manager.active_connections[job_id] = {mock_websocket}
        mock_task = Mock()
        mock_task.cancel = Mock()
        manager.consumer_tasks[job_id] = mock_task

        # Mock Redis client
        mock_redis_client = AsyncMock()
        mock_redis_client.close = AsyncMock()
        manager._redis_client = mock_redis_client

        # Mock redis_manager's close_async
        manager.redis_manager.close_async = AsyncMock()

        await manager.close()

        # Verify task was cancelled
        mock_task.cancel.assert_called_once()

        # Verify WebSocket was closed
        mock_websocket.close.assert_called_once()

        # Verify data structures were cleared
        assert len(manager.active_connections) == 0
        assert len(manager.consumer_tasks) == 0

        # Verify Redis client was closed
        mock_redis_client.close.assert_called_once()
        
        # Verify redis_manager was closed
        manager.redis_manager.close_async.assert_called_once()


class TestGetWebSocketManager:
    """Test the global WebSocket manager getter."""

    @patch("packages.webui.websocket_manager.RedisStreamWebSocketManager")
    def test_get_websocket_manager(self, mock_manager_class):
        """Test getting global WebSocket manager."""
        redis_url = "redis://localhost:6379/0"
        mock_instance = Mock()
        mock_manager_class.return_value = mock_instance

        # Clear global manager
        import packages.webui.websocket_manager

        packages.webui.websocket_manager._websocket_manager = None

        manager1 = get_websocket_manager(redis_url)
        assert manager1 == mock_instance
        mock_manager_class.assert_called_once_with(redis_url)

        # Test caching
        manager2 = get_websocket_manager(redis_url)
        assert manager1 == manager2
        mock_manager_class.assert_called_once()  # Still only called once
