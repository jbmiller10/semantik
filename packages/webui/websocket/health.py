"""Health monitoring for WebSocket connections."""

import asyncio
import contextlib
import logging
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ConnectionHealth(BaseModel):
    """Health status of a WebSocket connection."""

    connection_id: str
    latency_ms: float
    last_ping: datetime
    last_pong: datetime
    message_count: int
    error_count: int
    is_healthy: bool
    metadata: dict[str, Any] = {}


class HealthMetrics(BaseModel):
    """Aggregated health metrics."""

    total_connections: int
    healthy_connections: int
    unhealthy_connections: int
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    messages_per_second: float
    errors_per_second: float
    uptime_seconds: float


class WebSocketHealthMonitor:
    """
    Monitors health of WebSocket connections.

    Features:
    - Latency measurement
    - Connection health checks
    - Performance metrics
    - Automatic recovery
    - Alert generation
    """

    def __init__(self) -> None:
        """Initialize the health monitor."""
        self.connection_health: dict[str, ConnectionHealth] = {}
        self.latency_samples: defaultdict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.message_counts: defaultdict[str, int] = defaultdict(int)
        self.error_counts: defaultdict[str, int] = defaultdict(int)
        self.ping_times: dict[str, float] = {}

        # Metrics tracking
        self.start_time = time.time()
        self.total_messages = 0
        self.total_errors = 0
        self.metrics_window = deque(maxlen=60)  # Last 60 seconds

        # Health check settings
        self.ping_interval = 30  # seconds
        self.ping_timeout = 10  # seconds
        self.unhealthy_threshold = 3  # missed pings

        # Monitoring task
        self.monitor_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return

        self._running = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("WebSocket health monitor started")

    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False

        if self.monitor_task:
            self.monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitor_task

        logger.info("WebSocket health monitor stopped")

    def register_connection(self, connection_id: str) -> None:
        """
        Register a new connection for monitoring.

        Args:
            connection_id: Connection identifier
        """
        self.connection_health[connection_id] = ConnectionHealth(
            connection_id=connection_id,
            latency_ms=0,
            last_ping=datetime.now(UTC),
            last_pong=datetime.now(UTC),
            message_count=0,
            error_count=0,
            is_healthy=True,
        )
        logger.debug(f"Registered connection for monitoring: {connection_id}")

    def unregister_connection(self, connection_id: str) -> None:
        """
        Unregister a connection from monitoring.

        Args:
            connection_id: Connection identifier
        """
        self.connection_health.pop(connection_id, None)
        self.latency_samples.pop(connection_id, None)
        self.message_counts.pop(connection_id, None)
        self.error_counts.pop(connection_id, None)
        self.ping_times.pop(connection_id, None)
        logger.debug(f"Unregistered connection from monitoring: {connection_id}")

    def record_ping_sent(self, connection_id: str) -> None:
        """
        Record that a ping was sent.

        Args:
            connection_id: Connection identifier
        """
        self.ping_times[connection_id] = time.time()

        if connection_id in self.connection_health:
            self.connection_health[connection_id].last_ping = datetime.now(UTC)

    def record_pong_received(self, connection_id: str) -> float:
        """
        Record that a pong was received.

        Args:
            connection_id: Connection identifier

        Returns:
            Latency in milliseconds
        """
        if connection_id not in self.ping_times:
            return 0

        # Calculate latency
        latency_ms = (time.time() - self.ping_times[connection_id]) * 1000

        # Store latency sample
        self.latency_samples[connection_id].append(latency_ms)

        # Update health record
        if connection_id in self.connection_health:
            health = self.connection_health[connection_id]
            health.last_pong = datetime.now(UTC)
            health.latency_ms = latency_ms
            health.is_healthy = True

        # Clear ping time
        self.ping_times.pop(connection_id, None)

        return latency_ms

    def record_message(self, connection_id: str) -> None:
        """
        Record a message sent/received.

        Args:
            connection_id: Connection identifier
        """
        self.message_counts[connection_id] += 1
        self.total_messages += 1

        if connection_id in self.connection_health:
            self.connection_health[connection_id].message_count += 1

    def record_error(self, connection_id: str, error: str) -> None:
        """
        Record an error for a connection.

        Args:
            connection_id: Connection identifier
            error: Error description
        """
        self.error_counts[connection_id] += 1
        self.total_errors += 1

        if connection_id in self.connection_health:
            health = self.connection_health[connection_id]
            health.error_count += 1
            health.metadata.setdefault("recent_errors", []).append(
                {"timestamp": datetime.now(UTC).isoformat(), "error": error}
            )

            # Keep only last 10 errors
            health.metadata["recent_errors"] = health.metadata["recent_errors"][-10:]

    def get_connection_health(self, connection_id: str) -> ConnectionHealth | None:
        """
        Get health status for a connection.

        Args:
            connection_id: Connection identifier

        Returns:
            Connection health status
        """
        return self.connection_health.get(connection_id)

    def get_metrics(self) -> HealthMetrics:
        """
        Get aggregated health metrics.

        Returns:
            Health metrics
        """
        # Count healthy/unhealthy connections
        healthy = sum(1 for h in self.connection_health.values() if h.is_healthy)
        unhealthy = len(self.connection_health) - healthy

        # Calculate latency percentiles
        all_latencies = []
        for samples in self.latency_samples.values():
            all_latencies.extend(samples)

        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        p95_latency = self._percentile(all_latencies, 95) if all_latencies else 0
        p99_latency = self._percentile(all_latencies, 99) if all_latencies else 0

        # Calculate message rate
        uptime = time.time() - self.start_time
        messages_per_second = self.total_messages / uptime if uptime > 0 else 0
        errors_per_second = self.total_errors / uptime if uptime > 0 else 0

        return HealthMetrics(
            total_connections=len(self.connection_health),
            healthy_connections=healthy,
            unhealthy_connections=unhealthy,
            average_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            messages_per_second=messages_per_second,
            errors_per_second=errors_per_second,
            uptime_seconds=uptime,
        )

    def get_unhealthy_connections(self) -> list[ConnectionHealth]:
        """
        Get list of unhealthy connections.

        Returns:
            List of unhealthy connection health records
        """
        return [health for health in self.connection_health.values() if not health.is_healthy]

    def check_connection_health(self, connection_id: str) -> bool:
        """
        Check if a connection is healthy.

        Args:
            connection_id: Connection identifier

        Returns:
            True if healthy
        """
        health = self.connection_health.get(connection_id)
        if not health:
            return False

        # Check last pong time
        now = datetime.now(UTC)
        time_since_pong = (now - health.last_pong).total_seconds()

        if time_since_pong > self.ping_interval * self.unhealthy_threshold:
            health.is_healthy = False
            return False

        # Check error rate
        if health.error_count > 10 and health.error_count > health.message_count * 0.1:
            health.is_healthy = False
            return False

        health.is_healthy = True
        return True

    async def ping_connection(self, connection_id: str, send_func) -> bool:
        """
        Send ping to a connection.

        Args:
            connection_id: Connection identifier
            send_func: Async function to send ping message

        Returns:
            True if pong received within timeout
        """
        # Record ping sent
        self.record_ping_sent(connection_id)

        # Send ping
        try:
            await send_func({"type": "ping", "timestamp": datetime.now(UTC).isoformat()})
        except Exception as e:
            logger.error(f"Failed to send ping to {connection_id}: {e}")
            self.record_error(connection_id, f"Ping failed: {e}")
            return False

        # Wait for pong
        start_time = time.time()
        while time.time() - start_time < self.ping_timeout:
            if connection_id not in self.ping_times:
                # Pong received
                return True
            await asyncio.sleep(0.1)

        # Timeout
        self.record_error(connection_id, "Ping timeout")
        return False

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                # Collect metrics snapshot
                snapshot = {
                    "timestamp": time.time(),
                    "connections": len(self.connection_health),
                    "messages": self.total_messages,
                    "errors": self.total_errors,
                }
                self.metrics_window.append(snapshot)

                # Check connection health
                for connection_id in list(self.connection_health.keys()):
                    self.check_connection_health(connection_id)

                # Generate alerts for unhealthy connections
                unhealthy = self.get_unhealthy_connections()
                if unhealthy:
                    logger.warning(f"Found {len(unhealthy)} unhealthy connections")
                    for health in unhealthy[:5]:  # Log first 5
                        logger.warning(
                            f"Unhealthy connection: {health.connection_id} "
                            f"(latency={health.latency_ms:.1f}ms, errors={health.error_count})"
                        )

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(5)

    def _percentile(self, values: list[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def get_detailed_stats(self) -> dict[str, Any]:
        """
        Get detailed statistics.

        Returns:
            Dictionary of detailed statistics
        """
        metrics = self.get_metrics()

        # Connection details
        connection_details = []
        for health in self.connection_health.values():
            latencies = list(self.latency_samples.get(health.connection_id, []))
            connection_details.append(
                {
                    "connection_id": health.connection_id,
                    "is_healthy": health.is_healthy,
                    "latency_ms": health.latency_ms,
                    "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                    "message_count": health.message_count,
                    "error_count": health.error_count,
                    "uptime": (datetime.now(UTC) - health.last_ping).total_seconds(),
                }
            )

        return {
            "summary": metrics.model_dump(),
            "connections": connection_details,
            "rates": {
                "messages_per_second": metrics.messages_per_second,
                "errors_per_second": metrics.errors_per_second,
                "error_rate": (
                    metrics.errors_per_second / metrics.messages_per_second if metrics.messages_per_second > 0 else 0
                ),
            },
            "latency": {
                "average_ms": metrics.average_latency_ms,
                "p95_ms": metrics.p95_latency_ms,
                "p99_ms": metrics.p99_latency_ms,
            },
        }
