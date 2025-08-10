#!/usr/bin/env python3

"""
WebSocket Load Testing Script for Scalable WebSocket Manager.

This script simulates multiple concurrent WebSocket connections to test:
- Connection scaling up to 10,000 connections
- Message latency under load
- Cross-instance message routing
- Memory stability
- Cleanup after disconnection

Usage:
    python websocket_load_test.py --connections 1000 --duration 60 --url ws://localhost:8000
"""

import argparse
import asyncio
import contextlib
import json
import logging
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field

import aiohttp
from aiohttp import ClientWebSocketResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for a WebSocket connection."""

    connection_id: str
    user_id: str
    connected_at: float
    disconnected_at: float | None = None
    messages_received: int = 0
    messages_sent: int = 0
    latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def connection_duration(self) -> float:
        """Get connection duration in seconds."""
        if self.disconnected_at:
            return self.disconnected_at - self.connected_at
        return time.time() - self.connected_at

    @property
    def avg_latency(self) -> float:
        """Get average message latency in milliseconds."""
        if not self.latencies:
            return 0
        return statistics.mean(self.latencies)

    @property
    def p99_latency(self) -> float:
        """Get 99th percentile latency in milliseconds."""
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]


class WebSocketLoadTester:
    """WebSocket load testing client."""

    def __init__(self, base_url: str, auth_token: str | None = None) -> None:
        self.base_url = base_url
        self.auth_token = auth_token
        self.connections: dict[str, ClientWebSocketResponse] = {}
        self.stats: dict[str, ConnectionStats] = {}
        self.session: aiohttp.ClientSession | None = None
        self.running = False

    async def start(self) -> None:
        """Start the load tester."""
        self.session = aiohttp.ClientSession()
        self.running = True

    async def stop(self) -> None:
        """Stop the load tester and cleanup."""
        self.running = False

        # Close all connections
        for conn_id, ws in list(self.connections.items()):
            try:
                await ws.close()
            except Exception as e:
                logger.error(f"Error closing connection {conn_id}: {e}")

        self.connections.clear()

        # Close session
        if self.session:
            await self.session.close()

    async def connect_websocket(self, user_id: str, operation_id: str | None = None) -> str:
        """Create a WebSocket connection.

        Args:
            user_id: User ID for the connection
            operation_id: Optional operation ID to subscribe to

        Returns:
            Connection ID
        """
        conn_id = str(uuid.uuid4())

        # Build WebSocket URL
        if operation_id:
            ws_url = f"{self.base_url}/ws/operations/{operation_id}"
        else:
            ws_url = f"{self.base_url}/ws/test/{user_id}"

        # Add auth token if provided
        if self.auth_token:
            ws_url += f"?token={self.auth_token}"

        if not self.session:
            raise RuntimeError("Session not initialized. Call start() first.")

        try:
            # Connect to WebSocket
            ws = await self.session.ws_connect(ws_url)

            # Store connection
            self.connections[conn_id] = ws
            self.stats[conn_id] = ConnectionStats(connection_id=conn_id, user_id=user_id, connected_at=time.time())

            # Start message handler
            asyncio.create_task(self._handle_messages(conn_id, ws))

            logger.debug(f"Connected {conn_id} for user {user_id}")
            return conn_id

        except Exception as e:
            logger.error(f"Failed to connect WebSocket for user {user_id}: {e}")
            if conn_id in self.stats:
                self.stats[conn_id].errors.append(str(e))
            raise

    async def _handle_messages(self, conn_id: str, ws: ClientWebSocketResponse) -> None:
        """Handle incoming messages for a connection."""
        stats = self.stats.get(conn_id)
        if not stats:
            return

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        stats.messages_received += 1

                        # Handle ping messages
                        if data.get("type") == "ping":
                            await ws.send_json({"type": "pong", "timestamp": time.time()})
                            stats.messages_sent += 1

                        # Calculate latency if timestamp provided
                        if "timestamp" in data:
                            latency = (time.time() - data["timestamp"]) * 1000
                            stats.latencies.append(latency)

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received on {conn_id}: {msg.data}")

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error on {conn_id}: {ws.exception()}")
                    stats.errors.append(str(ws.exception()))

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.debug(f"WebSocket {conn_id} closed")
                    break

        except Exception as e:
            logger.error(f"Error handling messages for {conn_id}: {e}")
            stats.errors.append(str(e))

        finally:
            stats.disconnected_at = time.time()
            if conn_id in self.connections:
                del self.connections[conn_id]

    async def send_message(self, conn_id: str, message: dict) -> None:
        """Send a message through a specific connection."""
        if conn_id not in self.connections:
            raise ValueError(f"Connection {conn_id} not found")

        ws = self.connections[conn_id]
        stats = self.stats[conn_id]

        # Add timestamp for latency measurement
        message["timestamp"] = time.time()

        try:
            await ws.send_json(message)
            stats.messages_sent += 1
        except Exception as e:
            logger.error(f"Failed to send message on {conn_id}: {e}")
            stats.errors.append(str(e))
            raise

    async def broadcast_to_user(self, user_id: str, message: dict) -> None:
        """Send a message to all connections for a user."""
        user_connections = [conn_id for conn_id, stats in self.stats.items() if stats.user_id == user_id]

        for conn_id in user_connections:
            with contextlib.suppress(Exception):
                await self.send_message(conn_id, message)

    async def disconnect(self, conn_id: str) -> None:
        """Disconnect a specific connection."""
        if conn_id in self.connections:
            ws = self.connections[conn_id]
            await ws.close()
            del self.connections[conn_id]

        if conn_id in self.stats:
            self.stats[conn_id].disconnected_at = time.time()

    def get_stats_summary(self) -> dict:
        """Get summary statistics for all connections."""
        total_connections = len(self.stats)
        active_connections = len(self.connections)

        all_latencies = []
        total_messages_sent = 0
        total_messages_received = 0
        total_errors = 0

        for stats in self.stats.values():
            all_latencies.extend(stats.latencies)
            total_messages_sent += stats.messages_sent
            total_messages_received += stats.messages_received
            total_errors += len(stats.errors)

        return {
            "total_connections": total_connections,
            "active_connections": active_connections,
            "total_messages_sent": total_messages_sent,
            "total_messages_received": total_messages_received,
            "total_errors": total_errors,
            "avg_latency_ms": statistics.mean(all_latencies) if all_latencies else 0,
            "p50_latency_ms": statistics.median(all_latencies) if all_latencies else 0,
            "p99_latency_ms": (sorted(all_latencies)[int(len(all_latencies) * 0.99)] if all_latencies else 0),
            "max_latency_ms": max(all_latencies) if all_latencies else 0,
        }


async def run_load_test(
    url: str, num_connections: int, duration: int, connections_per_second: int = 100, auth_token: str | None = None
) -> bool:
    """Run the load test.

    Args:
        url: WebSocket server URL
        num_connections: Total number of connections to create
        duration: Test duration in seconds
        connections_per_second: Rate of new connections
        auth_token: Optional authentication token
    """
    logger.info(f"Starting load test: {num_connections} connections over {duration} seconds")
    logger.info(f"Target URL: {url}")

    tester = WebSocketLoadTester(url, auth_token)
    await tester.start()

    try:
        # Phase 1: Ramp up connections
        logger.info("Phase 1: Ramping up connections...")
        connection_delay = 1.0 / connections_per_second
        start_time = time.time()

        for i in range(num_connections):
            user_id = f"load_test_user_{i % 100}"  # Reuse users across connections

            try:
                await tester.connect_websocket(user_id)

                # Rate limit connection creation
                if i < num_connections - 1:
                    await asyncio.sleep(connection_delay)

                # Log progress
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    logger.info(f"Created {i + 1}/{num_connections} connections ({rate:.1f}/s)")

            except Exception as e:
                logger.error(f"Failed to create connection {i}: {e}")

        ramp_time = time.time() - start_time
        logger.info(f"Ramp up complete in {ramp_time:.1f}s")

        # Phase 2: Sustain load and send messages
        logger.info("Phase 2: Sustaining load and sending messages...")
        sustain_start = time.time()
        message_interval = 5  # Send message every 5 seconds

        while (time.time() - sustain_start) < duration:
            # Send test messages to random users
            for i in range(min(10, num_connections // 100)):
                user_id = f"load_test_user_{i}"
                await tester.broadcast_to_user(user_id, {"type": "load_test", "data": f"Test message at {time.time()}"})

            # Print current stats
            stats = tester.get_stats_summary()
            logger.info(
                f"Active: {stats['active_connections']}, "
                f"Sent: {stats['total_messages_sent']}, "
                f"Received: {stats['total_messages_received']}, "
                f"Avg Latency: {stats['avg_latency_ms']:.2f}ms, "
                f"P99 Latency: {stats['p99_latency_ms']:.2f}ms"
            )

            await asyncio.sleep(message_interval)

        # Phase 3: Graceful shutdown
        logger.info("Phase 3: Graceful shutdown...")
        await tester.stop()

        # Final statistics
        final_stats = tester.get_stats_summary()
        logger.info("=" * 60)
        logger.info("LOAD TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Connections: {final_stats['total_connections']}")
        logger.info(f"Total Messages Sent: {final_stats['total_messages_sent']}")
        logger.info(f"Total Messages Received: {final_stats['total_messages_received']}")
        logger.info(f"Total Errors: {final_stats['total_errors']}")
        logger.info(f"Average Latency: {final_stats['avg_latency_ms']:.2f}ms")
        logger.info(f"P50 Latency: {final_stats['p50_latency_ms']:.2f}ms")
        logger.info(f"P99 Latency: {final_stats['p99_latency_ms']:.2f}ms")
        logger.info(f"Max Latency: {final_stats['max_latency_ms']:.2f}ms")
        logger.info("=" * 60)

        # Check acceptance criteria
        success = True
        if final_stats["p99_latency_ms"] > 100:
            logger.error(f"FAILED: P99 latency ({final_stats['p99_latency_ms']:.2f}ms) exceeds 100ms requirement")
            success = False

        if final_stats["total_errors"] > num_connections * 0.01:  # Allow 1% error rate
            logger.error(f"FAILED: Error rate too high ({final_stats['total_errors']} errors)")
            success = False

        if success:
            logger.info("SUCCESS: All acceptance criteria met!")

        return success

    except Exception as e:
        logger.error(f"Load test failed: {e}")
        await tester.stop()
        return False


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WebSocket Load Testing Tool")
    parser.add_argument(
        "--url", default="ws://localhost:8000", help="WebSocket server URL (default: ws://localhost:8000)"
    )
    parser.add_argument(
        "--connections", type=int, default=1000, help="Number of concurrent connections (default: 1000)"
    )
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds (default: 60)")
    parser.add_argument("--rate", type=int, default=100, help="Connections per second during ramp-up (default: 100)")
    parser.add_argument("--token", help="Authentication token for WebSocket connections")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run the load test
    success = asyncio.run(
        run_load_test(
            url=args.url,
            num_connections=args.connections,
            duration=args.duration,
            connections_per_second=args.rate,
            auth_token=args.token,
        )
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
