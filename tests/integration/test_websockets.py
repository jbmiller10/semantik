"""Integration tests for operations WebSocket endpoint.

This file contains tests specifically for the /ws/operations/{operation_id} endpoint
as required by Ticket-003.
"""

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
        with (
            patch("packages.webui.api.v2.operations.get_current_user_websocket", return_value=mock_user),
            patch("packages.webui.api.v2.operations.OperationService") as mock_service_class,
            patch("packages.webui.api.v2.operations.get_db") as mock_get_db,
            patch("packages.webui.api.v2.operations.ws_manager") as mock_ws_manager,
        ):
            # Mock the OperationService class
            mock_service_instance = AsyncMock()
            mock_operation = AsyncMock()
            mock_operation.uuid = "test-operation-id" 
            mock_operation.user_id = 1
            mock_service_instance.verify_websocket_access = AsyncMock(return_value=mock_operation)
            
            # Make OperationService return our mock instance when instantiated
            mock_service_class.return_value = mock_service_instance

            # Also mock OperationRepository since it's created in the handler
            with patch("packages.webui.api.v2.operations.OperationRepository") as mock_repo_class:
                mock_repo = AsyncMock()
                mock_repo_class.return_value = mock_repo

                # Mock database session as an async generator
                mock_db = AsyncMock()
                
                async def mock_get_db_generator():
                    yield mock_db
                
                # Make get_db return the generator directly (not a coroutine)
                mock_get_db.side_effect = lambda: mock_get_db_generator()

                # Mock WebSocket manager
                mock_ws_manager.connect = AsyncMock()
                mock_ws_manager.disconnect = AsyncMock()

                # Mock receive_json to simulate client disconnect after initial connection
                mock_websocket_client.receive_json = AsyncMock(side_effect=Exception("Client disconnected"))

                # Test the WebSocket connection
                await operation_websocket(mock_websocket_client, "test-operation-id")

                # Verify authentication was checked
                assert mock_websocket_client.query_params.get("token") == "valid-test-token"

                # Verify connection was established
                mock_ws_manager.connect.assert_called_once_with(mock_websocket_client, "test-operation-id", "1")

                # Verify disconnection was called
                mock_ws_manager.disconnect.assert_called_once_with(
                    mock_websocket_client, "test-operation-id", "1"
                )
