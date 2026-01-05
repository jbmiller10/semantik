"""Example tests demonstrating WebSocket testing patterns."""

import asyncio
from datetime import UTC, datetime
from enum import Enum
from unittest.mock import MagicMock

import pytest

from tests.webui.test_websocket_helpers import (
    BASE_DELAY,
    WebSocketTestHarness,
    assert_message_order,
    count_message_types,
)
from webui.websocket.legacy_stream_manager import RedisStreamWebSocketManager, ws_manager


class TestWebSocketExamples:
    """Example tests showing how to test WebSocket functionality."""

    @pytest.fixture(autouse=True)
    def _setup_and_teardown(self) -> None:
        """Ensure clean state before and after each test."""
        # Setup - Reset any global state

        # Clear any existing connections and tasks
        ws_manager.connections.clear()
        ws_manager.consumer_tasks.clear()
        ws_manager._get_operation_func = None

        # Also reset the global instance if it exists
        if hasattr(ws_manager, "redis") and ws_manager.redis:
            # Close any existing Redis connection
            ws_manager.redis = None
            ws_manager._startup_attempted = False

        return

        # Teardown - ensure no lingering tasks
        # Note: The harness cleanup should handle most of this

    @pytest.mark.asyncio()
    async def test_operation_lifecycle_updates(self, mock_redis_client) -> None:
        """Example: Test complete operation lifecycle with updates."""
        # Setup - Create a fresh manager instance to avoid state pollution
        manager = RedisStreamWebSocketManager()
        # Use None to test direct broadcast mode
        manager.redis = None
        # Mark as already attempted to prevent reconnection
        manager._startup_attempted = True
        # Ensure clean connections
        manager.connections.clear()
        manager.consumer_tasks.clear()

        harness = WebSocketTestHarness(manager)

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
        async def mock_get_operation(operation_id: str) -> MagicMock | None:
            if operation_id == "operation789":
                return mock_operation
            return None

        manager.set_operation_getter(mock_get_operation)

        # Connect client
        clients = await harness.connect_clients("operation789", num_clients=1)
        client = clients[0]

        # Get initial state message and verify
        initial_state = await client.get_received_messages("current_state")
        assert len(initial_state) == 1

        # Don't clear messages - track all messages from start

        # Simulate operation updates directly through manager
        await manager.send_update("operation789", "start", {"status": "started", "total_files": 10})
        await asyncio.sleep(BASE_DELAY / 2)  # Shorter delay between updates
        await manager.send_update("operation789", "progress", {"progress": 20, "processed_files": 2})
        await asyncio.sleep(BASE_DELAY / 2)
        await manager.send_update("operation789", "progress", {"progress": 40, "processed_files": 4})
        await asyncio.sleep(BASE_DELAY / 2)
        await manager.send_update("operation789", "progress", {"progress": 60, "processed_files": 6})
        await asyncio.sleep(BASE_DELAY / 2)
        await manager.send_update("operation789", "progress", {"progress": 80, "processed_files": 8})
        await asyncio.sleep(BASE_DELAY / 2)
        await manager.send_update("operation789", "complete", {"status": "completed", "processed_files": 10})

        # Allow final propagation
        await asyncio.sleep(BASE_DELAY)

        # Verify message order - exclude initial state message
        all_messages = await client.get_received_messages()
        lifecycle_messages = [msg for msg in all_messages if msg.get("type") != "current_state"]
        assert_message_order(lifecycle_messages, ["start", "progress", "complete"])

        # Verify message counts
        type_counts = count_message_types(lifecycle_messages)
        assert type_counts.get("start", 0) >= 1
        assert type_counts.get("progress", 0) >= 4
        assert type_counts.get("complete", 0) >= 1

        # Verify final state
        complete_messages = await client.get_received_messages("complete")
        assert complete_messages[-1]["data"]["status"] == "completed"
        assert complete_messages[-1]["data"]["processed_files"] == 10

        # Cleanup
        await harness.cleanup()
