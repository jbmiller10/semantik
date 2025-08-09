"""Server-Sent Events (SSE) fallback for WebSocket-restricted environments."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SSEMessage(BaseModel):
    """Server-Sent Event message format."""

    id: str | None = None
    event: str | None = None
    data: Any
    retry: int | None = None


class SSEConnection:
    """Manages an SSE connection."""

    def __init__(self, request: Request, connection_id: str, user_id: str, channel: str) -> None:
        """
        Initialize SSE connection.

        Args:
            request: FastAPI request
            connection_id: Unique connection identifier
            user_id: User identifier
            channel: Channel to subscribe to
        """
        self.request = request
        self.connection_id = connection_id
        self.user_id = user_id
        self.channel = channel
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.connected_at = datetime.now(UTC)
        self.last_event_id: str | None = None
        self._closed = False

    async def send(self, message: SSEMessage) -> None:
        """
        Queue a message to send.

        Args:
            message: SSE message to send
        """
        if self._closed:
            return

        try:
            await self.queue.put(message)
        except asyncio.QueueFull:
            # Drop oldest message
            try:
                self.queue.get_nowait()
                await self.queue.put(message)
            except asyncio.QueueEmpty:
                pass

    async def close(self) -> None:
        """Close the SSE connection."""
        self._closed = True
        # Send close message
        await self.send(SSEMessage(event="close", data={"reason": "Connection closed"}))

    def is_closed(self) -> bool:
        """Check if connection is closed."""
        return self._closed or self.request.is_disconnected()


class SSEManager:
    """
    Manages Server-Sent Events as WebSocket fallback.

    Features:
    - Long-polling with automatic reconnection
    - Message queuing and replay
    - Compatible with WebSocket message format
    - Works behind restrictive proxies
    """

    def __init__(self) -> None:
        """Initialize SSE manager."""
        self.connections: dict[str, SSEConnection] = {}
        self.message_history: dict[str, list[SSEMessage]] = {}
        self.history_size = 100
        self.heartbeat_interval = 30

    def create_connection(self, request: Request, user_id: str, channel: str) -> SSEConnection:
        """
        Create a new SSE connection.

        Args:
            request: FastAPI request
            user_id: User identifier
            channel: Channel to subscribe to

        Returns:
            SSE connection object
        """
        connection_id = f"sse-{user_id}-{datetime.now(UTC).timestamp()}"
        connection = SSEConnection(request, connection_id, user_id, channel)

        # Store connection
        self.connections[connection_id] = connection

        # Check for Last-Event-ID header for reconnection
        last_event_id = request.headers.get("Last-Event-ID")
        if last_event_id:
            connection.last_event_id = last_event_id
            # Queue missed messages
            asyncio.create_task(self._replay_messages(connection))

        logger.info(f"SSE connection created: {connection_id}")
        return connection

    def remove_connection(self, connection_id: str) -> None:
        """
        Remove an SSE connection.

        Args:
            connection_id: Connection identifier
        """
        connection = self.connections.pop(connection_id, None)
        if connection:
            asyncio.create_task(connection.close())
            logger.info(f"SSE connection removed: {connection_id}")

    async def send_to_channel(self, channel: str, event: str, data: dict[str, Any]) -> int:
        """
        Send message to all connections on a channel.

        Args:
            channel: Target channel
            event: Event type
            data: Event data

        Returns:
            Number of messages sent
        """
        message = SSEMessage(id=str(datetime.now(UTC).timestamp()), event=event, data=data)

        # Store in history
        if channel not in self.message_history:
            self.message_history[channel] = []
        self.message_history[channel].append(message)

        # Trim history
        if len(self.message_history[channel]) > self.history_size:
            self.message_history[channel] = self.message_history[channel][-self.history_size :]

        # Send to connections
        count = 0
        for connection in list(self.connections.values()):
            if connection.channel == channel and not connection.is_closed():
                await connection.send(message)
                count += 1

        return count

    async def send_to_user(self, user_id: str, event: str, data: dict[str, Any]) -> int:
        """
        Send message to all connections for a user.

        Args:
            user_id: Target user
            event: Event type
            data: Event data

        Returns:
            Number of messages sent
        """
        message = SSEMessage(id=str(datetime.now(UTC).timestamp()), event=event, data=data)

        count = 0
        for connection in list(self.connections.values()):
            if connection.user_id == user_id and not connection.is_closed():
                await connection.send(message)
                count += 1

        return count

    async def stream_events(self, connection: SSEConnection) -> AsyncGenerator[str, None]:
        """
        Stream events for an SSE connection.

        Args:
            connection: SSE connection

        Yields:
            SSE formatted strings
        """
        # Send initial connection message
        yield self._format_sse(
            SSEMessage(
                event="connected",
                data={
                    "connection_id": connection.connection_id,
                    "channel": connection.channel,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
        )

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(connection))

        try:
            while not connection.is_closed():
                try:
                    # Wait for message with timeout for heartbeat
                    message = await asyncio.wait_for(connection.queue.get(), timeout=1.0)

                    # Format and yield message
                    yield self._format_sse(message)

                    # Update last event ID
                    if message.id:
                        connection.last_event_id = message.id

                except TimeoutError:
                    # Check if client disconnected
                    if connection.request.is_disconnected():
                        break
                    continue

        except Exception as e:
            logger.error(f"Error streaming events: {e}")
        finally:
            # Cleanup
            heartbeat_task.cancel()
            self.remove_connection(connection.connection_id)

    def _format_sse(self, message: SSEMessage) -> str:
        """
        Format message as SSE.

        Args:
            message: SSE message

        Returns:
            SSE formatted string
        """
        lines = []

        if message.id:
            lines.append(f"id: {message.id}")

        if message.event:
            lines.append(f"event: {message.event}")

        if message.retry:
            lines.append(f"retry: {message.retry}")

        # Format data
        data = json.dumps(message.data) if isinstance(message.data, dict) else str(message.data)

        # Split data by lines
        for line in data.split("\n"):
            lines.append(f"data: {line}")

        # End with double newline
        return "\n".join(lines) + "\n\n"

    async def _replay_messages(self, connection: SSEConnection) -> None:
        """
        Replay missed messages for a reconnecting client.

        Args:
            connection: SSE connection
        """
        if not connection.last_event_id:
            return

        # Find messages after last event ID
        history = self.message_history.get(connection.channel, [])

        found_last = False
        for message in history:
            if found_last:
                await connection.send(message)
            elif message.id == connection.last_event_id:
                found_last = True

    async def _heartbeat_loop(self, connection: SSEConnection) -> None:
        """
        Send periodic heartbeats to keep connection alive.

        Args:
            connection: SSE connection
        """
        while not connection.is_closed():
            try:
                await asyncio.sleep(self.heartbeat_interval)

                # Send heartbeat
                await connection.send(SSEMessage(event="heartbeat", data={"timestamp": datetime.now(UTC).isoformat()}))

            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")
                break


class SSEEndpoint:
    """FastAPI endpoint for SSE connections."""

    def __init__(self, manager: SSEManager) -> None:
        """
        Initialize SSE endpoint.

        Args:
            manager: SSE manager instance
        """
        self.manager = manager

    async def connect(self, request: Request, user_id: str, channel: str) -> StreamingResponse:
        """
        Handle SSE connection request.

        Args:
            request: FastAPI request
            user_id: User identifier
            channel: Channel to subscribe to

        Returns:
            Streaming response with SSE events
        """
        # Create connection
        connection = self.manager.create_connection(request, user_id, channel)

        # Return streaming response
        return StreamingResponse(
            self.manager.stream_events(connection),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable Nginx buffering
                "Access-Control-Allow-Origin": "*",  # Configure appropriately
            },
        )


# Global SSE manager instance
sse_manager = SSEManager()
sse_endpoint = SSEEndpoint(sse_manager)
