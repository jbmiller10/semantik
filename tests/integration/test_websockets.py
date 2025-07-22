"""Integration tests for operations WebSocket endpoint.

This file contains tests specifically for the /ws/operations/{operation_id} endpoint
as required by Ticket-003.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import WebSocket


class TestOperationsWebSocket:
    """Integration tests for the operations WebSocket endpoint."""

    @pytest.fixture()
    def mock_websocket_client(self):
        """Create a mock WebSocket client for testing."""
        mock = AsyncMock(spec=WebSocket)
        mock.accept = AsyncMock()
        mock.send_json = AsyncMock()
        mock.close = AsyncMock()
        mock.query_params = {"token": "valid-test-token"}
        mock.received_messages = []

        # Store messages when send_json is called
        async def track_send_json(data):
            mock.received_messages.append(data)

        mock.send_json.side_effect = track_send_json
        return mock

    @pytest.fixture()
    def mock_redis(self):
        """Create a mock Redis client for testing."""

        class MockRedis:
            def __init__(self):
                self.published_messages = []
                self.subscriptions = {}

            async def publish(self, channel, message):
                self.published_messages.append({"channel": channel, "message": json.loads(message)})
                # Simulate message delivery to subscribers
                if channel in self.subscriptions:
                    for callback in self.subscriptions[channel]:
                        await callback(message)

            async def subscribe(self, channel):
                if channel not in self.subscriptions:
                    self.subscriptions[channel] = []

            def add_subscriber(self, channel, callback):
                if channel not in self.subscriptions:
                    self.subscriptions[channel] = []
                self.subscriptions[channel].append(callback)

        return MockRedis()

    @pytest.mark.asyncio()
    async def test_websocket_authentication_success(self, mock_websocket_client):
        """Test successful WebSocket authentication and connection."""
        from packages.webui.api.v2.operations import operation_websocket

        # Mock authentication
        mock_user = {"id": "1", "username": "testuser"}
        with patch("packages.webui.auth.get_current_user_websocket", return_value=mock_user):
            # Mock operation repository
            with patch("packages.webui.api.v2.operations.OperationRepository") as mock_repo_class:
                mock_repo = AsyncMock()
                mock_operation = AsyncMock()
                mock_operation.uuid = "test-operation-id"
                mock_operation.user_id = 1
                mock_repo.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_operation)
                mock_repo_class.return_value = mock_repo

                # Mock database session
                with patch("packages.webui.api.v2.operations.get_db") as mock_get_db:
                    mock_db = AsyncMock()
                    mock_get_db.return_value.__aenter__.return_value = mock_db

                    # Mock WebSocket manager
                    with patch("packages.webui.api.v2.operations.ws_manager") as mock_ws_manager:
                        mock_ws_manager.connect = AsyncMock()
                        mock_ws_manager.disconnect = AsyncMock()

                        # Mock receive_json to simulate client disconnect
                        mock_websocket_client.receive_json = AsyncMock(side_effect=Exception("Client disconnected"))

                        # Test the WebSocket connection
                        await operation_websocket(mock_websocket_client, "test-operation-id")

                        # Verify authentication was checked
                        assert mock_websocket_client.query_params.get("token") == "valid-test-token"

                        # Verify connection was established
                        mock_ws_manager.connect.assert_called_once_with(
                            mock_websocket_client, "operation:test-operation-id", "1"
                        )

                        # Verify disconnection was called
                        mock_ws_manager.disconnect.assert_called_once_with(
                            mock_websocket_client, "operation:test-operation-id", "1"
                        )

    @pytest.mark.asyncio()
    async def test_websocket_authentication_failure(self, mock_websocket_client):
        """Test WebSocket connection failure due to invalid authentication."""
        from packages.webui.api.v2.operations import operation_websocket

        # Mock authentication failure
        with patch("packages.webui.auth.get_current_user_websocket", side_effect=ValueError("Invalid token")):
            await operation_websocket(mock_websocket_client, "test-operation-id")

            # Verify connection was closed with proper error
            mock_websocket_client.close.assert_called_once_with(code=1008, reason="Invalid token")

    @pytest.mark.asyncio()
    async def test_websocket_operation_not_found(self, mock_websocket_client):
        """Test WebSocket connection failure when operation doesn't exist."""
        from packages.shared.database.exceptions import EntityNotFoundError
        from packages.webui.api.v2.operations import operation_websocket

        # Mock successful authentication
        mock_user = {"id": "1", "username": "testuser"}
        with patch("packages.webui.auth.get_current_user_websocket", return_value=mock_user):
            # Mock operation not found
            with patch("packages.webui.api.v2.operations.OperationRepository") as mock_repo_class:
                mock_repo = AsyncMock()
                mock_repo.get_by_uuid_with_permission_check = AsyncMock(
                    side_effect=EntityNotFoundError("Operation not found")
                )
                mock_repo_class.return_value = mock_repo

                with patch("packages.webui.api.v2.operations.get_db") as mock_get_db:
                    mock_db = AsyncMock()
                    mock_get_db.return_value.__aenter__.return_value = mock_db

                    await operation_websocket(mock_websocket_client, "non-existent-id")

                    # Verify connection was closed with proper error
                    mock_websocket_client.close.assert_called_once_with(
                        code=1008, reason="Operation 'non-existent-id' not found"
                    )

    @pytest.mark.asyncio()
    async def test_websocket_access_denied(self, mock_websocket_client):
        """Test WebSocket connection failure when user lacks permission."""
        from packages.shared.database.exceptions import AccessDeniedError
        from packages.webui.api.v2.operations import operation_websocket

        # Mock successful authentication
        mock_user = {"id": "1", "username": "testuser"}
        with patch("packages.webui.auth.get_current_user_websocket", return_value=mock_user):
            # Mock access denied
            with patch("packages.webui.api.v2.operations.OperationRepository") as mock_repo_class:
                mock_repo = AsyncMock()
                mock_repo.get_by_uuid_with_permission_check = AsyncMock(side_effect=AccessDeniedError("Access denied"))
                mock_repo_class.return_value = mock_repo

                with patch("packages.webui.api.v2.operations.get_db") as mock_get_db:
                    mock_db = AsyncMock()
                    mock_get_db.return_value.__aenter__.return_value = mock_db

                    await operation_websocket(mock_websocket_client, "test-operation-id")

                    # Verify connection was closed with proper error
                    mock_websocket_client.close.assert_called_once_with(
                        code=1008, reason="You don't have access to this operation"
                    )

    @pytest.mark.asyncio()
    async def test_websocket_receives_redis_updates(self, mock_websocket_client, mock_redis):
        """Test that WebSocket receives updates published to Redis channel."""
        from packages.webui.api.v2.operations import operation_websocket

        # Mock successful authentication
        mock_user = {"id": "1", "username": "testuser"}
        with patch("packages.webui.auth.get_current_user_websocket", return_value=mock_user):
            # Mock operation repository
            with patch("packages.webui.api.v2.operations.OperationRepository") as mock_repo_class:
                mock_repo = AsyncMock()
                mock_operation = AsyncMock()
                mock_operation.uuid = "test-operation-id"
                mock_operation.user_id = 1
                mock_repo.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_operation)
                mock_repo_class.return_value = mock_repo

                with patch("packages.webui.api.v2.operations.get_db") as mock_get_db:
                    mock_db = AsyncMock()
                    mock_get_db.return_value.__aenter__.return_value = mock_db

                    # Mock WebSocket manager with Redis integration
                    with patch("packages.webui.api.v2.operations.ws_manager") as mock_ws_manager:
                        connected_websocket = None
                        connected_channel = None

                        async def mock_connect(websocket, channel_id, user_id):
                            nonlocal connected_websocket, connected_channel
                            connected_websocket = websocket
                            connected_channel = channel_id
                            # Simulate initial connection message
                            await websocket.send_json({"type": "connected", "channel": channel_id})

                        mock_ws_manager.connect = AsyncMock(side_effect=mock_connect)
                        mock_ws_manager.disconnect = AsyncMock()

                        # Start WebSocket connection in background
                        task = asyncio.create_task(operation_websocket(mock_websocket_client, "test-operation-id"))

                        # Wait for connection
                        await asyncio.sleep(0.1)

                        # Simulate Redis message via WebSocket manager
                        if connected_websocket:
                            await connected_websocket.send_json(
                                {"type": "progress", "data": {"progress": 50, "current_file": "test.pdf"}}
                            )

                        # Allow time for message delivery
                        await asyncio.sleep(0.1)

                        # Cancel the WebSocket task
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                        # Verify messages were received
                        assert len(mock_websocket_client.received_messages) >= 2
                        assert mock_websocket_client.received_messages[0]["type"] == "connected"
                        assert mock_websocket_client.received_messages[1]["type"] == "progress"
                        assert mock_websocket_client.received_messages[1]["data"]["progress"] == 50

    @pytest.mark.asyncio()
    async def test_websocket_cleanup_on_disconnect(self, mock_websocket_client):
        """Test that resources are properly cleaned up when client disconnects."""
        from fastapi import WebSocketDisconnect

        from packages.webui.api.v2.operations import operation_websocket

        # Mock successful authentication
        mock_user = {"id": "1", "username": "testuser"}
        with patch("packages.webui.auth.get_current_user_websocket", return_value=mock_user):
            # Mock operation repository
            with patch("packages.webui.api.v2.operations.OperationRepository") as mock_repo_class:
                mock_repo = AsyncMock()
                mock_operation = AsyncMock()
                mock_operation.uuid = "test-operation-id"
                mock_operation.user_id = 1
                mock_repo.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_operation)
                mock_repo_class.return_value = mock_repo

                with patch("packages.webui.api.v2.operations.get_db") as mock_get_db:
                    mock_db = AsyncMock()
                    mock_get_db.return_value.__aenter__.return_value = mock_db

                    # Mock WebSocket manager
                    with patch("packages.webui.api.v2.operations.ws_manager") as mock_ws_manager:
                        mock_ws_manager.connect = AsyncMock()
                        mock_ws_manager.disconnect = AsyncMock()

                        # Mock receive_json to simulate WebSocketDisconnect
                        mock_websocket_client.receive_json = AsyncMock(side_effect=WebSocketDisconnect())

                        # Test the WebSocket connection
                        await operation_websocket(mock_websocket_client, "test-operation-id")

                        # Verify cleanup was performed
                        mock_ws_manager.disconnect.assert_called_once_with(
                            mock_websocket_client, "operation:test-operation-id", "1"
                        )

    @pytest.mark.asyncio()
    async def test_websocket_ping_pong_handling(self, mock_websocket_client):
        """Test that WebSocket handles ping/pong messages for keepalive."""
        from packages.webui.api.v2.operations import operation_websocket

        # Mock successful authentication
        mock_user = {"id": "1", "username": "testuser"}
        with patch("packages.webui.auth.get_current_user_websocket", return_value=mock_user):
            # Mock operation repository
            with patch("packages.webui.api.v2.operations.OperationRepository") as mock_repo_class:
                mock_repo = AsyncMock()
                mock_operation = AsyncMock()
                mock_operation.uuid = "test-operation-id"
                mock_operation.user_id = 1
                mock_repo.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_operation)
                mock_repo_class.return_value = mock_repo

                with patch("packages.webui.api.v2.operations.get_db") as mock_get_db:
                    mock_db = AsyncMock()
                    mock_get_db.return_value.__aenter__.return_value = mock_db

                    # Mock WebSocket manager
                    with patch("packages.webui.api.v2.operations.ws_manager") as mock_ws_manager:
                        mock_ws_manager.connect = AsyncMock()
                        mock_ws_manager.disconnect = AsyncMock()

                        # Mock receive_json to return ping message, then disconnect
                        ping_message = {"type": "ping"}
                        mock_websocket_client.receive_json = AsyncMock(
                            side_effect=[ping_message, Exception("Client disconnected")]
                        )

                        # Test the WebSocket connection
                        await operation_websocket(mock_websocket_client, "test-operation-id")

                        # Verify pong was sent
                        assert any(msg == {"type": "pong"} for msg in mock_websocket_client.received_messages)

    @pytest.mark.asyncio()
    async def test_full_integration_with_celery_updates(self, mock_redis):
        """Test full integration: Celery task publishes update, WebSocket receives it."""
        # This test simulates the complete flow from Celery task to WebSocket client

        # Create a mock operation
        operation_id = "test-operation-123"

        # Mock WebSocket client
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_json = AsyncMock()
        mock_websocket.close = AsyncMock()
        mock_websocket.query_params = {"token": "valid-test-token"}
        received_messages = []

        async def track_messages(data):
            received_messages.append(data)

        mock_websocket.send_json.side_effect = track_messages

        # Mock authentication
        mock_user = {"id": "1", "username": "testuser"}

        # Simulate the complete flow
        from packages.webui.api.v2.operations import operation_websocket
        from packages.webui.tasks import CeleryTaskWithOperationUpdates

        with patch("packages.webui.auth.get_current_user_websocket", return_value=mock_user):
            with patch("packages.webui.api.v2.operations.OperationRepository") as mock_repo_class:
                mock_repo = AsyncMock()
                mock_operation = AsyncMock()
                mock_operation.uuid = operation_id
                mock_operation.user_id = 1
                mock_repo.get_by_uuid_with_permission_check = AsyncMock(return_value=mock_operation)
                mock_repo_class.return_value = mock_repo

                with patch("packages.webui.api.v2.operations.get_db") as mock_get_db:
                    mock_db = AsyncMock()
                    mock_get_db.return_value.__aenter__.return_value = mock_db

                    # Mock the WebSocket manager to simulate Redis pub/sub
                    with patch("packages.webui.api.v2.operations.ws_manager") as mock_ws_manager:
                        # Track connected WebSocket
                        connected_ws = None

                        async def mock_connect(ws, channel_id, user_id):
                            nonlocal connected_ws
                            connected_ws = ws

                        mock_ws_manager.connect = AsyncMock(side_effect=mock_connect)
                        mock_ws_manager.disconnect = AsyncMock()

                        # Mock Celery task updates
                        with patch("packages.webui.tasks.redis.from_url", return_value=mock_redis):
                            # Create Celery task updater
                            celery_updater = CeleryTaskWithOperationUpdates(operation_id)
                            celery_updater._redis_client = mock_redis

                            # Start WebSocket connection in background
                            mock_websocket.receive_json = AsyncMock(side_effect=asyncio.CancelledError())
                            ws_task = asyncio.create_task(operation_websocket(mock_websocket, operation_id))

                            # Wait for connection
                            await asyncio.sleep(0.1)

                            # Simulate Celery task sending updates
                            await celery_updater.send_update("start", {"status": "started", "total_files": 100})

                            # Simulate progress update from WebSocket manager
                            if connected_ws:
                                await connected_ws.send_json(
                                    {"type": "progress", "data": {"progress": 25, "current_file": "doc1.pdf"}}
                                )

                            await asyncio.sleep(0.1)

                            # Cancel WebSocket task
                            ws_task.cancel()
                            try:
                                await ws_task
                            except asyncio.CancelledError:
                                pass

                            # Verify messages were received
                            assert len(received_messages) > 0
                            progress_messages = [msg for msg in received_messages if msg.get("type") == "progress"]
                            assert len(progress_messages) > 0
                            assert progress_messages[0]["data"]["progress"] == 25

                            # Verify Redis received the update
                            assert len(mock_redis.published_messages) > 0
                            assert any(
                                msg["channel"] == f"operation-progress:{operation_id}"
                                for msg in mock_redis.published_messages
                            )
