"""Helper utilities for WebSocket testing."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

from fastapi import WebSocket


class MockWebSocketClient:
    """Mock WebSocket client for testing."""

    def __init__(self, client_id: str):
        """Initialize mock WebSocket client."""
        self.client_id = client_id
        self.websocket = AsyncMock(spec=WebSocket)
        self.websocket.accept = AsyncMock()
        self.websocket.send_json = AsyncMock()
        self.websocket.close = AsyncMock()
        self.websocket.receive_json = AsyncMock()

        # Track sent and received messages
        self.sent_messages: list[dict[str, Any]] = []
        self.received_messages: list[dict[str, Any]] = []

        # Configure send_json to track messages
        async def track_send(data):
            self.received_messages.append(data)

        self.websocket.send_json.side_effect = track_send

    async def connect(self, manager, job_id: str, user_id: str):
        """Connect to WebSocket manager."""
        await manager.connect(self.websocket, job_id, user_id)

    async def disconnect(self, manager, job_id: str, user_id: str):
        """Disconnect from WebSocket manager."""
        await manager.disconnect(self.websocket, job_id, user_id)

    async def send_message(self, message: dict[str, Any]):
        """Send a message (simulate client sending)."""
        self.sent_messages.append(message)
        # In real implementation, this would trigger manager handlers

    def get_received_messages(self, message_type: str = None) -> list[dict[str, Any]]:
        """Get received messages, optionally filtered by type."""
        if message_type:
            return [msg for msg in self.received_messages if msg.get("type") == message_type]
        return self.received_messages

    def clear_messages(self):
        """Clear all tracked messages."""
        self.sent_messages.clear()
        self.received_messages.clear()


class WebSocketTestHarness:
    """Test harness for WebSocket integration testing."""

    def __init__(self, manager):
        """Initialize test harness."""
        self.manager = manager
        self.clients: dict[str, MockWebSocketClient] = {}

    async def create_client(self, client_id: str) -> MockWebSocketClient:
        """Create a new mock client."""
        client = MockWebSocketClient(client_id)
        self.clients[client_id] = client
        return client

    async def connect_clients(self, job_id: str, num_clients: int = 1, user_prefix: str = "user"):
        """Connect multiple clients to a job."""
        connected_clients = []
        for i in range(num_clients):
            client_id = f"client_{i}"
            user_id = f"{user_prefix}_{i}"

            client = await self.create_client(client_id)
            await client.connect(self.manager, job_id, user_id)
            connected_clients.append(client)

        # Allow connections to stabilize
        await asyncio.sleep(0.1)
        return connected_clients

    async def broadcast_and_verify(self, job_id: str, message_type: str, data: dict[str, Any]):
        """Broadcast a message and verify all clients received it."""
        await self.manager.send_update(job_id, message_type, data)

        # Allow message propagation
        await asyncio.sleep(0.1)

        # Verify all clients for this job received the message
        results = {}
        for client_id, client in self.clients.items():
            messages = client.get_received_messages(message_type)
            results[client_id] = {"received": len(messages) > 0, "messages": messages}

        return results

    async def cleanup(self):
        """Clean up all connections."""
        # Disconnect all clients
        for client in self.clients.values():
            # Find the connection info from manager
            for key, websockets in list(self.manager.connections.items()):
                if client.websocket in websockets:
                    user_id, job_id = key.split(":", 1)
                    await client.disconnect(self.manager, job_id, user_id)

        self.clients.clear()


async def simulate_job_updates(updater, delays: list[float] = None):
    """Simulate a sequence of job updates with optional delays."""
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


def assert_message_order(messages: list[dict[str, Any]], expected_types: list[str]):
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
