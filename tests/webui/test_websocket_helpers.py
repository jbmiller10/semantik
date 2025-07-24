"""Helper utilities for WebSocket testing."""

import asyncio
import contextlib
import os
from typing import Any
from unittest.mock import AsyncMock

from fastapi import WebSocket

# CI environments need more time for async operations
IS_CI = os.environ.get("CI", "false").lower() == "true"
BASE_DELAY = 0.5 if IS_CI else 0.1
LONG_DELAY = 1.0 if IS_CI else 0.3


class MockWebSocketClient:
    """Mock WebSocket client for testing."""

    def __init__(self, client_id: str) -> None:
        """Initialize mock WebSocket client."""
        self.client_id = client_id

        # Create a fresh WebSocket mock
        self.websocket = AsyncMock(spec=WebSocket)
        self.websocket.accept = AsyncMock()
        self.websocket.close = AsyncMock()
        self.websocket.receive_json = AsyncMock()

        # Track sent and received messages - instance specific
        self.sent_messages: list[dict[str, Any]] = []
        self.received_messages: list[dict[str, Any]] = []
        self._message_lock = asyncio.Lock()

        # Create a special mock for send_json that captures messages
        client_ref = self

        # Create an AsyncMock that captures messages
        async def send_json_impl(data):
            async with client_ref._message_lock:
                client_ref.received_messages.append(data.copy() if isinstance(data, dict) else data)

        # Replace send_json with an AsyncMock that has our implementation
        self.websocket.send_json = AsyncMock()
        self.websocket.send_json.side_effect = send_json_impl

    async def connect(self, manager, operation_id: str, user_id: str):
        """Connect to WebSocket manager."""
        await manager.connect(self.websocket, operation_id, user_id)

    async def disconnect(self, manager, operation_id: str, user_id: str):
        """Disconnect from WebSocket manager."""
        await manager.disconnect(self.websocket, operation_id, user_id)

    async def send_message(self, message: dict[str, Any]):
        """Send a message (simulate client sending)."""
        self.sent_messages.append(message)
        # In real implementation, this would trigger manager handlers

    def _ensure_messages_tracked(self):
        """Legacy method kept for compatibility."""
        # Messages are now captured directly in send_json side effect
        pass

    async def get_received_messages(self, message_type: str = None) -> list[dict[str, Any]]:
        """Get received messages, optionally filtered by type."""
        # Use lock to ensure thread safety
        async with self._message_lock:
            # Always ensure we have all messages from mock calls
            self._ensure_messages_tracked()

            # Debug: print all received messages if none match the filter
            if message_type:
                filtered = [msg for msg in self.received_messages if msg.get("type") == message_type]
                if not filtered and self.received_messages:
                    # This helps debug when messages are received but don't match the expected type
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.debug(
                        f"No messages of type '{message_type}' found. Received messages: {self.received_messages}"
                    )
                return filtered
            return self.received_messages.copy()

    async def clear_messages(self) -> None:
        """Clear all tracked messages."""
        async with self._message_lock:
            self.sent_messages.clear()
            self.received_messages.clear()


class WebSocketTestHarness:
    """Test harness for WebSocket integration testing."""

    def __init__(self, manager) -> None:
        """Initialize test harness."""
        self.manager = manager
        self.clients: dict[str, MockWebSocketClient] = {}

    async def create_client(self, client_id: str) -> MockWebSocketClient:
        """Create a new mock client."""
        client = MockWebSocketClient(client_id)
        self.clients[client_id] = client
        return client

    async def connect_clients(self, operation_id: str, num_clients: int = 1, user_prefix: str = "user"):
        """Connect multiple clients to an operation."""
        connected_clients = []
        for i in range(num_clients):
            client_id = f"client_{i}"
            user_id = f"{user_prefix}_{i}"

            client = await self.create_client(client_id)
            await client.connect(self.manager, operation_id, user_id)
            connected_clients.append(client)

        # Allow connections to stabilize
        await asyncio.sleep(BASE_DELAY)
        return connected_clients

    async def broadcast_and_verify(self, operation_id: str, message_type: str, data: dict[str, Any]):
        """Broadcast a message and verify all clients received it."""
        await self.manager.send_update(operation_id, message_type, data)

        # Allow message propagation
        await asyncio.sleep(BASE_DELAY)

        # Verify all clients for this operation received the message
        results = {}
        for client_id, client in self.clients.items():
            messages = await client.get_received_messages(message_type)
            results[client_id] = {"received": len(messages) > 0, "messages": messages}

        return results

    async def cleanup(self):
        """Clean up all connections and consumer tasks."""

        # First, cancel all consumer tasks to prevent event loop errors
        for _, task in list(self.manager.consumer_tasks.items()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.manager.consumer_tasks.clear()

        # Disconnect all clients
        for client in self.clients.values():
            # Find the connection info from manager
            for key, websockets in list(self.manager.connections.items()):
                if client.websocket in websockets:
                    parts = key.split(":")
                    if len(parts) == 3:  # user_id:operation:operation_id
                        user_id = parts[0]
                        operation_id = parts[2]
                        await client.disconnect(self.manager, operation_id, user_id)

        self.clients.clear()

        # Clear all remaining connections
        self.manager.connections.clear()


async def simulate_operation_updates(updater, delays: list[float] = None):
    """Simulate a sequence of operation updates with optional delays."""
    if delays is None:
        delays = [0.1] * 6  # Default delays between updates

    update_sequence = [
        ("start", {"status": "started", "total_files": 10}),
        ("progress", {"progress": 20, "processed_files": 2}),
        ("progress", {"progress": 40, "processed_files": 4}),
        ("progress", {"progress": 60, "processed_files": 6}),
        ("progress", {"progress": 80, "processed_files": 8}),
        ("complete", {"status": "completed", "processed_files": 10}),
    ]

    for i, (update_type, data) in enumerate(update_sequence):
        await updater.send_update(update_type, data)
        if i < len(delays):
            await asyncio.sleep(delays[i])


def assert_message_order(messages: list[dict[str, Any]], expected_types: list[str]) -> None:
    """Assert that messages appear in the expected order."""
    actual_types = [msg.get("type") for msg in messages]

    # Find expected types in order (allowing other messages in between)
    last_index = -1
    for expected_type in expected_types:
        try:
            index = actual_types.index(expected_type, last_index + 1)
            last_index = index
        except ValueError as err:
            raise AssertionError(
                f"Expected message type '{expected_type}' not found in correct order. Actual types: {actual_types}"
            ) from err


def count_message_types(messages: list[dict[str, Any]]) -> dict[str, int]:
    """Count occurrences of each message type."""
    counts = {}
    for msg in messages:
        msg_type = msg.get("type", "unknown")
        counts[msg_type] = counts.get(msg_type, 0) + 1
    return counts
