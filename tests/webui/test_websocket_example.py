"""Example tests demonstrating WebSocket testing patterns."""

import asyncio
from datetime import UTC, datetime

import pytest

from packages.webui.websocket_manager import RedisStreamWebSocketManager
from tests.webui.test_websocket_helpers import (
    WebSocketTestHarness,
    assert_message_order,
    count_message_types,
)


class TestWebSocketExamples:
    """Example tests showing how to test WebSocket functionality."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Ensure clean state before and after each test."""
        # Setup - nothing needed before
        return
        # Teardown - ensure no lingering tasks
        # Note: We don't need async teardown here since the harness cleanup handles it

    @pytest.mark.asyncio()
    async def test_simple_websocket_flow(self, mock_redis_client):
        """Example: Test a simple WebSocket connection and message flow."""
        # Setup
        manager = RedisStreamWebSocketManager()
        # Use None to test direct broadcast mode first
        manager.redis = None
        # Mark as already attempted to prevent reconnection
        manager._startup_attempted = True

        harness = WebSocketTestHarness(manager)

        from enum import Enum
        from unittest.mock import MagicMock

        # Create mock enums
        class MockStatus(Enum):
            PROCESSING = "processing"

        class MockType(Enum):
            INDEX = "index"

        # Create mock operation
        mock_operation = MagicMock()
        mock_operation.status = MockStatus.PROCESSING
        mock_operation.type = MockType.INDEX
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = datetime.now(UTC)
        mock_operation.completed_at = None
        mock_operation.error_message = None

        # Set up the operation getter function
        async def mock_get_operation(operation_id):
            if operation_id == "operation123":
                return mock_operation
            return None

        manager.set_operation_getter(mock_get_operation)

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

        # Small delay for async operations
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
        # Use None to test direct broadcast mode
        manager.redis = None
        # Mark as already attempted to prevent reconnection
        manager._startup_attempted = True

        harness = WebSocketTestHarness(manager)

        from enum import Enum
        from unittest.mock import MagicMock

        # Create mock enums
        class MockStatus(Enum):
            PROCESSING = "processing"

        class MockType(Enum):
            INDEX = "index"

        # Create mock operation
        mock_operation = MagicMock()
        mock_operation.status = MockStatus.PROCESSING
        mock_operation.type = MockType.INDEX
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = datetime.now(UTC)
        mock_operation.completed_at = None
        mock_operation.error_message = None

        # Set up the operation getter function
        async def mock_get_operation(operation_id):
            if operation_id == "operation456":
                return mock_operation
            return None

        manager.set_operation_getter(mock_get_operation)

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
        # Use None to test direct broadcast mode
        manager.redis = None
        # Mark as already attempted to prevent reconnection
        manager._startup_attempted = True

        harness = WebSocketTestHarness(manager)

        from enum import Enum
        from unittest.mock import MagicMock

        # Create mock enums
        class MockStatus(Enum):
            PENDING = "pending"

        class MockType(Enum):
            INDEX = "index"

        # Create mock operation
        mock_operation = MagicMock()
        mock_operation.status = MockStatus.PENDING
        mock_operation.type = MockType.INDEX
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = None
        mock_operation.completed_at = None
        mock_operation.error_message = None

        # Set up the operation getter function
        async def mock_get_operation(operation_id):
            if operation_id == "operation789":
                return mock_operation
            return None

        manager.set_operation_getter(mock_get_operation)

        # Connect client
        clients = await harness.connect_clients("operation789", num_clients=1)
        client = clients[0]

        # Get initial state message and verify
        initial_state = client.get_received_messages("current_state")
        assert len(initial_state) == 1

        # Clear messages after verifying initial state
        client.clear_messages()

        # Simulate operation updates directly through manager
        await manager.send_update("operation789", "start", {"status": "started", "total_files": 10})
        await asyncio.sleep(0.05)
        await manager.send_update("operation789", "progress", {"progress": 20, "processed_files": 2})
        await asyncio.sleep(0.05)
        await manager.send_update("operation789", "progress", {"progress": 40, "processed_files": 4})
        await asyncio.sleep(0.05)
        await manager.send_update("operation789", "progress", {"progress": 60, "processed_files": 6})
        await asyncio.sleep(0.05)
        await manager.send_update("operation789", "progress", {"progress": 80, "processed_files": 8})
        await asyncio.sleep(0.05)
        await manager.send_update("operation789", "complete", {"status": "completed", "processed_files": 10})

        # Allow final propagation
        await asyncio.sleep(0.1)

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
        # Use None to test direct broadcast mode
        manager.redis = None
        # Mark as already attempted to prevent reconnection
        manager._startup_attempted = True

        harness = WebSocketTestHarness(manager)

        from enum import Enum
        from unittest.mock import MagicMock

        # Create mock enums
        class MockStatus(Enum):
            PROCESSING = "processing"

        class MockType(Enum):
            INDEX = "index"

        # Create mock operation
        mock_operation = MagicMock()
        mock_operation.status = MockStatus.PROCESSING
        mock_operation.type = MockType.INDEX
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = datetime.now(UTC)
        mock_operation.completed_at = None
        mock_operation.error_message = None

        # Set up the operation getter function
        async def mock_get_operation(operation_id):
            if operation_id == "operation_error":
                return mock_operation
            return None

        manager.set_operation_getter(mock_get_operation)

        # Connect clients
        good_clients = await harness.connect_clients("operation_error", num_clients=2)

        # Verify both clients received initial state
        for client in good_clients:
            initial_state = client.get_received_messages("current_state")
            assert len(initial_state) == 1
            client.clear_messages()

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
        # Use None to test direct broadcast mode
        manager.redis = None
        # Mark as already attempted to prevent reconnection
        manager._startup_attempted = True

        harness = WebSocketTestHarness(manager)

        from enum import Enum
        from unittest.mock import MagicMock

        # Create mock enums
        class MockStatus(Enum):
            PROCESSING = "processing"

        class MockType(Enum):
            INDEX = "index"

        # Create mock operation
        mock_operation = MagicMock()
        mock_operation.status = MockStatus.PROCESSING
        mock_operation.type = MockType.INDEX
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = datetime.now(UTC)
        mock_operation.completed_at = None
        mock_operation.error_message = None

        # Set up the operation getter function
        async def mock_get_operation(operation_id):
            if operation_id in ["operation_A", "operation_B"]:
                return mock_operation
            return None

        manager.set_operation_getter(mock_get_operation)

        # Connect clients to different operations
        op1_clients = await harness.connect_clients("operation_A", num_clients=2)
        op2_clients = await harness.connect_clients("operation_B", num_clients=2)

        # Clear initial messages
        for client in op1_clients + op2_clients:
            client.clear_messages()

        # Send updates to different operations
        await manager.send_update("operation_A", "update_A", {"operation": "A"})
        await manager.send_update("operation_B", "update_B", {"operation": "B"})

        # Give more time for message propagation
        await asyncio.sleep(0.3)

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
