"""Example tests demonstrating WebSocket testing patterns."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from packages.webui.tasks import CeleryTaskWithOperationUpdates
from packages.webui.websocket_manager import RedisStreamWebSocketManager
from tests.webui.test_websocket_helpers import (
    WebSocketTestHarness,
    assert_message_order,
    count_message_types,
    simulate_job_updates,
)


class TestWebSocketExamples:
    """Example tests showing how to test WebSocket functionality."""

    @pytest.mark.asyncio()
    async def test_simple_websocket_flow(self, mock_redis_client):
        """Example: Test a simple WebSocket connection and message flow."""
        # Setup
        manager = RedisStreamWebSocketManager()
        manager.redis = mock_redis_client

        harness = WebSocketTestHarness(manager)

        # Mock operation repository
        with patch("shared.database.factory.create_operation_repository") as mock_create_repo:
            from unittest.mock import AsyncMock, MagicMock
            from enum import Enum

            # Create mock enums
            class MockStatus(Enum):
                PROCESSING = "processing"
            
            class MockType(Enum):
                INDEX = "index"

            mock_repo = AsyncMock()
            mock_operation = MagicMock()
            mock_operation.status = MockStatus.PROCESSING
            mock_operation.type = MockType.INDEX
            mock_operation.created_at = datetime.now(UTC)
            mock_operation.started_at = datetime.now(UTC)
            mock_operation.completed_at = None
            mock_operation.error_message = None
            mock_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
            mock_create_repo.return_value = mock_repo

            # Connect a client
            clients = await harness.connect_clients("operation123", num_clients=1)
            client = clients[0]

            # Client should receive initial state
            initial_messages = client.get_received_messages("current_state")
            assert len(initial_messages) == 1
            assert initial_messages[0]["data"]["status"] == "processing"
            assert initial_messages[0]["data"]["operation_type"] == "index"

            # Send an update
            await manager.send_update("operation123", "progress", {"progress": 50, "processed_files": 2})

            await asyncio.sleep(0.1)

            # Verify client received the update
            progress_messages = client.get_received_messages("progress")
            assert len(progress_messages) >= 1
            assert progress_messages[-1]["data"]["progress"] == 50

            # Cleanup
            await harness.cleanup()

    @pytest.mark.asyncio()
    async def test_multiple_clients_broadcast(self, mock_redis_client):
        """Example: Test broadcasting to multiple clients."""
        # Setup
        manager = RedisStreamWebSocketManager()
        manager.redis = mock_redis_client

        harness = WebSocketTestHarness(manager)

        with patch("shared.database.factory.create_operation_repository") as mock_create_repo:
            from unittest.mock import AsyncMock, MagicMock
            from enum import Enum

            # Create mock enums
            class MockStatus(Enum):
                PROCESSING = "processing"
            
            class MockType(Enum):
                INDEX = "index"

            mock_repo = AsyncMock()
            mock_operation = MagicMock()
            mock_operation.status = MockStatus.PROCESSING
            mock_operation.type = MockType.INDEX
            mock_operation.created_at = datetime.now(UTC)
            mock_operation.started_at = datetime.now(UTC)
            mock_operation.completed_at = None
            mock_operation.error_message = None
            mock_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
            mock_create_repo.return_value = mock_repo

            # Connect multiple clients
            await harness.connect_clients("operation456", num_clients=3)

            # Broadcast a message
            results = await harness.broadcast_and_verify("operation456", "announcement", {"message": "Processing started"})

            # Verify all clients received it
            for client_id, result in results.items():
                assert result["received"], f"Client {client_id} didn't receive message"
                assert result["messages"][0]["data"]["message"] == "Processing started"

            # Cleanup
            await harness.cleanup()

    @pytest.mark.asyncio()
    async def test_operation_lifecycle_updates(self, mock_redis_client):
        """Example: Test complete operation lifecycle with updates."""
        # Setup
        manager = RedisStreamWebSocketManager()
        manager.redis = mock_redis_client

        harness = WebSocketTestHarness(manager)

        with patch("shared.database.factory.create_operation_repository") as mock_create_repo:
            from unittest.mock import AsyncMock, MagicMock
            from enum import Enum

            # Create mock enums
            class MockStatus(Enum):
                PENDING = "pending"
            
            class MockType(Enum):
                INDEX = "index"

            mock_repo = AsyncMock()
            mock_operation = MagicMock()
            mock_operation.status = MockStatus.PENDING
            mock_operation.type = MockType.INDEX
            mock_operation.created_at = datetime.now(UTC)
            mock_operation.started_at = None
            mock_operation.completed_at = None
            mock_operation.error_message = None
            mock_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
            mock_create_repo.return_value = mock_repo

            # Connect client
            clients = await harness.connect_clients("operation789", num_clients=1)
            client = clients[0]

            # Clear initial state message
            client.clear_messages()

            # Simulate operation updates
            updater = CeleryTaskWithOperationUpdates("operation789")
            updater._redis_client = mock_redis_client

            await simulate_job_updates(updater, delays=[0.05] * 6)

            # Allow final propagation
            await asyncio.sleep(0.2)

            # Verify message order
            all_messages = client.received_messages
            assert_message_order(all_messages, ["start", "progress", "complete"])

            # Verify message counts
            type_counts = count_message_types(all_messages)
            assert type_counts.get("start", 0) >= 1
            assert type_counts.get("progress", 0) >= 4
            assert type_counts.get("complete", 0) >= 1

            # Verify final state
            complete_messages = client.get_received_messages("complete")
            assert complete_messages[-1]["data"]["status"] == "completed"
            assert complete_messages[-1]["data"]["processed_files"] == 10

            # Cleanup
            await harness.cleanup()

    @pytest.mark.asyncio()
    async def test_error_handling_example(self, mock_redis_client):
        """Example: Test error handling in WebSocket communication."""
        # Setup manager with a client that will fail
        manager = RedisStreamWebSocketManager()
        manager.redis = mock_redis_client

        harness = WebSocketTestHarness(manager)

        with patch("shared.database.factory.create_operation_repository") as mock_create_repo:
            from unittest.mock import AsyncMock, MagicMock
            from enum import Enum

            # Create mock enums
            class MockStatus(Enum):
                PROCESSING = "processing"
            
            class MockType(Enum):
                INDEX = "index"

            mock_repo = AsyncMock()
            mock_operation = MagicMock()
            mock_operation.status = MockStatus.PROCESSING
            mock_operation.type = MockType.INDEX
            mock_operation.created_at = datetime.now(UTC)
            mock_operation.started_at = datetime.now(UTC)
            mock_operation.completed_at = None
            mock_operation.error_message = None
            mock_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
            mock_create_repo.return_value = mock_repo

            # Connect clients
            good_clients = await harness.connect_clients("operation_error", num_clients=2)

            # Make one client fail on send
            bad_client = good_clients[0]
            bad_client.websocket.send_json.side_effect = Exception("Connection lost")

            # Send update - should handle the failure gracefully
            await manager.send_update("operation_error", "test", {"data": "test"})
            await asyncio.sleep(0.1)

            # Good client should still receive the message
            good_client = good_clients[1]
            messages = good_client.get_received_messages("test")
            assert len(messages) >= 1

            # Bad client should be removed from connections
            connection_found = False
            for websockets in manager.connections.values():
                if bad_client.websocket in websockets:
                    connection_found = True
                    break
            assert not connection_found, "Failed client wasn't removed"

            # Cleanup
            await harness.cleanup()

    @pytest.mark.asyncio()
    async def test_concurrent_operations_isolation(self, mock_redis_client):
        """Example: Test that different operations are isolated from each other."""
        # Setup
        manager = RedisStreamWebSocketManager()
        manager.redis = mock_redis_client

        harness = WebSocketTestHarness(manager)

        with patch("shared.database.factory.create_operation_repository") as mock_create_repo:
            from unittest.mock import AsyncMock, MagicMock
            from enum import Enum

            # Create mock enums
            class MockStatus(Enum):
                PROCESSING = "processing"
            
            class MockType(Enum):
                INDEX = "index"

            mock_repo = AsyncMock()
            mock_operation = MagicMock()
            mock_operation.status = MockStatus.PROCESSING
            mock_operation.type = MockType.INDEX
            mock_operation.created_at = datetime.now(UTC)
            mock_operation.started_at = datetime.now(UTC)
            mock_operation.completed_at = None
            mock_operation.error_message = None
            mock_repo.get_by_uuid = AsyncMock(return_value=mock_operation)
            mock_create_repo.return_value = mock_repo

            # Connect clients to different operations
            op1_clients = await harness.connect_clients("operation_A", num_clients=2)
            op2_clients = await harness.connect_clients("operation_B", num_clients=2)

            # Clear initial messages
            for client in op1_clients + op2_clients:
                client.clear_messages()

            # Send updates to different operations
            await manager.send_update("operation_A", "update_A", {"operation": "A"})
            await manager.send_update("operation_B", "update_B", {"operation": "B"})

            await asyncio.sleep(0.1)

            # Verify operation A clients only got operation A updates
            for client in op1_clients:
                assert len(client.get_received_messages("update_A")) >= 1
                assert len(client.get_received_messages("update_B")) == 0

            # Verify operation B clients only got operation B updates
            for client in op2_clients:
                assert len(client.get_received_messages("update_B")) >= 1
                assert len(client.get_received_messages("update_A")) == 0

            # Cleanup
            await harness.cleanup()
