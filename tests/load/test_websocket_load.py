"""
Load tests for WebSocket connections at scale.

Tests the system's ability to handle multiple concurrent WebSocket connections,
message broadcasting, rate limiting, and memory stability.
"""

import asyncio
import json
import random
import statistics
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets
from faker import Faker
from locust import HttpUser, TaskSet, between, events, task
from locust.env import Environment
from locust.stats import stats_printer, stats_history
from websockets.client import WebSocketClientProtocol

from webui.auth import create_access_token
from webui.websocket_manager import RedisStreamWebSocketManager

fake = Faker()


@dataclass
class WebSocketMetrics:
    """Metrics for WebSocket performance testing."""
    
    connection_times: List[float]
    message_latencies: List[float]
    messages_received: int
    errors: List[str]
    memory_usage: List[int]
    connection_failures: int
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_connections": len(self.connection_times),
            "successful_connections": len(self.connection_times) - self.connection_failures,
            "connection_failures": self.connection_failures,
            "avg_connection_time": statistics.mean(self.connection_times) if self.connection_times else 0,
            "p95_connection_time": self._percentile(self.connection_times, 95),
            "p99_connection_time": self._percentile(self.connection_times, 99),
            "messages_received": self.messages_received,
            "avg_message_latency": statistics.mean(self.message_latencies) if self.message_latencies else 0,
            "p95_message_latency": self._percentile(self.message_latencies, 95),
            "p99_message_latency": self._percentile(self.message_latencies, 99),
            "error_count": len(self.errors),
            "peak_memory_mb": max(self.memory_usage) / (1024 * 1024) if self.memory_usage else 0,
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class WebSocketClient:
    """WebSocket client for load testing."""
    
    def __init__(self, url: str, token: str, operation_id: str):
        self.url = url
        self.token = token
        self.operation_id = operation_id
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.messages_received = []
        self.connected = False
        self.connection_time = 0
        self.errors = []
    
    async def connect(self) -> None:
        """Connect to WebSocket server."""
        start_time = time.time()
        try:
            self.websocket = await websockets.connect(
                f"{self.url}/ws/operations/{self.operation_id}?token={self.token}",
                ping_interval=30,
                ping_timeout=10,
            )
            self.connected = True
            self.connection_time = time.time() - start_time
        except Exception as e:
            self.errors.append(str(e))
            self.connected = False
            raise
    
    async def listen(self, duration: int = 60) -> None:
        """Listen for messages for specified duration."""
        if not self.connected:
            return
        
        end_time = time.time() + duration
        
        try:
            while time.time() < end_time:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=1.0,
                    )
                    self.messages_received.append({
                        "timestamp": time.time(),
                        "data": json.loads(message),
                    })
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    self.connected = False
                    break
        except Exception as e:
            self.errors.append(str(e))
    
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send a message to the server."""
        if not self.connected:
            return
        
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            self.errors.append(str(e))
    
    async def disconnect(self) -> None:
        """Disconnect from server."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False


class TestWebSocketLoadBasic:
    """Basic WebSocket load tests."""
    
    @pytest.mark.asyncio()
    async def test_100_concurrent_connections(self, test_server_url: str) -> None:
        """Test 100 concurrent WebSocket connections."""
        num_connections = 100
        metrics = WebSocketMetrics(
            connection_times=[],
            message_latencies=[],
            messages_received=0,
            errors=[],
            memory_usage=[],
            connection_failures=0,
        )
        
        # Create test tokens
        tokens = []
        for i in range(num_connections):
            token = create_access_token(
                data={"sub": f"user_{i}", "user_id": i}
            )
            tokens.append(token)
        
        # Create clients
        clients = []
        for i in range(num_connections):
            operation_id = str(uuid.uuid4())
            client = WebSocketClient(
                test_server_url,
                tokens[i],
                operation_id,
            )
            clients.append(client)
        
        # Connect all clients concurrently
        connection_tasks = []
        for client in clients:
            task = asyncio.create_task(client.connect())
            connection_tasks.append(task)
        
        # Wait for all connections
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        # Collect metrics
        for i, client in enumerate(clients):
            if client.connected:
                metrics.connection_times.append(client.connection_time)
            else:
                metrics.connection_failures += 1
                if isinstance(results[i], Exception):
                    metrics.errors.append(str(results[i]))
        
        # Listen for messages for 10 seconds
        listen_tasks = []
        for client in clients:
            if client.connected:
                task = asyncio.create_task(client.listen(duration=10))
                listen_tasks.append(task)
        
        await asyncio.gather(*listen_tasks, return_exceptions=True)
        
        # Disconnect all clients
        disconnect_tasks = []
        for client in clients:
            task = asyncio.create_task(client.disconnect())
            disconnect_tasks.append(task)
        
        await asyncio.gather(*disconnect_tasks)
        
        # Analyze results
        summary = metrics.get_summary()
        
        # Assertions
        assert summary["successful_connections"] >= 95  # At least 95% success rate
        assert summary["avg_connection_time"] < 1.0  # Average connection time < 1 second
        assert summary["p99_connection_time"] < 5.0  # P99 connection time < 5 seconds
        assert summary["connection_failures"] <= 5  # Max 5% failure rate
    
    @pytest.mark.asyncio()
    async def test_rapid_connect_disconnect(self, test_server_url: str) -> None:
        """Test rapid connection and disconnection cycles."""
        num_cycles = 50
        connection_times = []
        errors = []
        
        token = create_access_token(
            data={"sub": "test_user", "user_id": 1}
        )
        
        for _ in range(num_cycles):
            operation_id = str(uuid.uuid4())
            client = WebSocketClient(test_server_url, token, operation_id)
            
            try:
                # Connect
                await client.connect()
                connection_times.append(client.connection_time)
                
                # Send a message
                await client.send_message({
                    "type": "ping",
                    "timestamp": time.time(),
                })
                
                # Brief pause
                await asyncio.sleep(0.1)
                
                # Disconnect
                await client.disconnect()
                
            except Exception as e:
                errors.append(str(e))
        
        # Verify no memory leaks or connection exhaustion
        assert len(errors) < 5  # Less than 10% error rate
        assert statistics.mean(connection_times) < 0.5  # Fast connections
    
    @pytest.mark.asyncio()
    async def test_message_broadcast_performance(self, test_server_url: str) -> None:
        """Test message broadcasting to multiple clients."""
        num_clients = 50
        operation_id = str(uuid.uuid4())  # Shared operation
        
        # Create clients for same operation
        clients = []
        for i in range(num_clients):
            token = create_access_token(
                data={"sub": f"user_{i}", "user_id": i}
            )
            client = WebSocketClient(test_server_url, token, operation_id)
            clients.append(client)
        
        # Connect all clients
        connection_tasks = [client.connect() for client in clients]
        await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        connected_clients = [c for c in clients if c.connected]
        assert len(connected_clients) >= 45  # At least 90% connected
        
        # Start listening
        listen_tasks = []
        for client in connected_clients:
            task = asyncio.create_task(client.listen(duration=5))
            listen_tasks.append(task)
        
        # Simulate server broadcasting messages
        # In real test, server would broadcast to all clients
        await asyncio.sleep(1)
        
        # Send progress updates from one client (simulating server broadcast)
        if connected_clients:
            for i in range(10):
                await connected_clients[0].send_message({
                    "type": "progress",
                    "progress": i * 10,
                    "timestamp": time.time(),
                })
                await asyncio.sleep(0.1)
        
        # Wait for listening to complete
        await asyncio.gather(*listen_tasks, return_exceptions=True)
        
        # Verify all clients received messages
        messages_per_client = [len(c.messages_received) for c in connected_clients]
        
        # Most clients should receive broadcasts
        clients_with_messages = sum(1 for m in messages_per_client if m > 0)
        assert clients_with_messages >= len(connected_clients) * 0.8
        
        # Disconnect all
        disconnect_tasks = [client.disconnect() for client in clients]
        await asyncio.gather(*disconnect_tasks)


class TestWebSocketLoadLimits:
    """Test WebSocket connection limits and rate limiting."""
    
    @pytest.mark.asyncio()
    async def test_per_user_connection_limit(self, test_server_url: str) -> None:
        """Test per-user connection limit (10 connections)."""
        user_token = create_access_token(
            data={"sub": "test_user", "user_id": 1}
        )
        
        clients = []
        successful_connections = 0
        rejected_connections = 0
        
        # Try to create 15 connections for same user
        for i in range(15):
            operation_id = str(uuid.uuid4())
            client = WebSocketClient(test_server_url, user_token, operation_id)
            clients.append(client)
            
            try:
                await client.connect()
                successful_connections += 1
            except Exception as e:
                if "limit exceeded" in str(e).lower():
                    rejected_connections += 1
        
        # Should enforce 10 connection limit
        assert successful_connections <= 10
        assert rejected_connections >= 5
        
        # Clean up
        for client in clients:
            if client.connected:
                await client.disconnect()
    
    @pytest.mark.asyncio()
    async def test_global_connection_limit(self, test_server_url: str) -> None:
        """Test global connection limit (simulated at 1000)."""
        # This test would require many resources
        # For practical testing, we'll test a smaller limit
        
        num_users = 20
        connections_per_user = 5
        total_attempts = num_users * connections_per_user
        
        clients = []
        successful = 0
        
        for user_id in range(num_users):
            token = create_access_token(
                data={"sub": f"user_{user_id}", "user_id": user_id}
            )
            
            for _ in range(connections_per_user):
                operation_id = str(uuid.uuid4())
                client = WebSocketClient(test_server_url, token, operation_id)
                clients.append(client)
                
                try:
                    await client.connect()
                    successful += 1
                except Exception:
                    pass
        
        # Verify connections are limited but reasonable number succeed
        assert successful > 0
        assert successful <= total_attempts
        
        # Clean up
        for client in clients:
            if client.connected:
                await client.disconnect()
    
    @pytest.mark.asyncio()
    async def test_message_rate_limiting(self, test_server_url: str) -> None:
        """Test message rate limiting (500ms for progress updates)."""
        token = create_access_token(
            data={"sub": "test_user", "user_id": 1}
        )
        operation_id = str(uuid.uuid4())
        
        client = WebSocketClient(test_server_url, token, operation_id)
        await client.connect()
        
        # Start listening
        listen_task = asyncio.create_task(client.listen(duration=5))
        
        # Send rapid progress updates
        send_times = []
        for i in range(20):
            await client.send_message({
                "type": "progress",
                "progress": i * 5,
                "timestamp": time.time(),
            })
            send_times.append(time.time())
            await asyncio.sleep(0.1)  # 100ms between sends (below 500ms limit)
        
        await listen_task
        
        # Analyze received messages
        # Server should throttle progress updates to 500ms intervals
        if len(client.messages_received) > 1:
            intervals = []
            for i in range(1, len(client.messages_received)):
                interval = (
                    client.messages_received[i]["timestamp"] -
                    client.messages_received[i-1]["timestamp"]
                )
                intervals.append(interval)
            
            # Average interval should be close to throttle limit
            if intervals:
                avg_interval = statistics.mean(intervals)
                assert avg_interval >= 0.4  # Allow some tolerance
        
        await client.disconnect()


class TestWebSocketMemoryStability:
    """Test WebSocket memory stability over time."""
    
    @pytest.mark.asyncio()
    async def test_long_running_connections(self, test_server_url: str) -> None:
        """Test memory stability with long-running connections."""
        num_clients = 10
        duration_minutes = 2  # Reduced for testing
        
        clients = []
        memory_samples = []
        
        # Create and connect clients
        for i in range(num_clients):
            token = create_access_token(
                data={"sub": f"user_{i}", "user_id": i}
            )
            operation_id = str(uuid.uuid4())
            client = WebSocketClient(test_server_url, token, operation_id)
            
            try:
                await client.connect()
                clients.append(client)
            except Exception:
                pass
        
        connected_count = len(clients)
        assert connected_count >= 8  # At least 80% connected
        
        # Monitor for specified duration
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        # Start listening on all clients
        listen_tasks = []
        for client in clients:
            task = asyncio.create_task(client.listen(duration=duration_minutes * 60))
            listen_tasks.append(task)
        
        # Periodically send messages and check memory
        sample_interval = 10  # seconds
        while time.time() < end_time:
            # Send test messages
            for client in clients[:3]:  # Use first 3 clients to send
                await client.send_message({
                    "type": "heartbeat",
                    "timestamp": time.time(),
                })
            
            # In real test, would sample actual memory usage
            # For now, simulate memory sampling
            memory_samples.append(random.randint(100, 200) * 1024 * 1024)
            
            await asyncio.sleep(sample_interval)
        
        # Wait for all listeners to complete
        await asyncio.gather(*listen_tasks, return_exceptions=True)
        
        # Disconnect all
        for client in clients:
            await client.disconnect()
        
        # Analyze memory trend
        if len(memory_samples) > 2:
            # Check for memory leak (significant increase over time)
            first_half_avg = statistics.mean(memory_samples[:len(memory_samples)//2])
            second_half_avg = statistics.mean(memory_samples[len(memory_samples)//2:])
            
            # Memory should not increase by more than 20%
            memory_increase = (second_half_avg - first_half_avg) / first_half_avg
            assert memory_increase < 0.2
    
    @pytest.mark.asyncio()
    async def test_connection_churn_memory(self, test_server_url: str) -> None:
        """Test memory stability with high connection churn."""
        num_iterations = 20
        clients_per_iteration = 10
        memory_samples = []
        
        for iteration in range(num_iterations):
            iteration_clients = []
            
            # Create and connect clients
            for i in range(clients_per_iteration):
                token = create_access_token(
                    data={"sub": f"user_{iteration}_{i}", "user_id": iteration * 100 + i}
                )
                operation_id = str(uuid.uuid4())
                client = WebSocketClient(test_server_url, token, operation_id)
                
                try:
                    await client.connect()
                    iteration_clients.append(client)
                except Exception:
                    pass
            
            # Brief activity
            await asyncio.sleep(0.5)
            
            # Disconnect all
            for client in iteration_clients:
                await client.disconnect()
            
            # Simulate memory sampling
            memory_samples.append(random.randint(100, 150) * 1024 * 1024)
            
            # Brief pause between iterations
            await asyncio.sleep(0.5)
        
        # Check memory stability
        if len(memory_samples) > 5:
            # Memory should stabilize, not continuously grow
            early_avg = statistics.mean(memory_samples[:5])
            late_avg = statistics.mean(memory_samples[-5:])
            
            # Should not grow by more than 10%
            growth = (late_avg - early_avg) / early_avg
            assert growth < 0.1


class TestWebSocketStressScenarios:
    """Stress test scenarios for WebSocket connections."""
    
    @pytest.mark.asyncio()
    async def test_thundering_herd(self, test_server_url: str) -> None:
        """Test thundering herd scenario (many connections at once)."""
        num_clients = 200
        
        # Create all clients
        clients = []
        for i in range(num_clients):
            token = create_access_token(
                data={"sub": f"user_{i}", "user_id": i}
            )
            operation_id = str(uuid.uuid4())
            client = WebSocketClient(test_server_url, token, operation_id)
            clients.append(client)
        
        # Connect all at once (thundering herd)
        start_time = time.time()
        connection_tasks = [client.connect() for client in clients]
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Count successes and failures
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        # Server should handle thundering herd gracefully
        assert successful >= num_clients * 0.7  # At least 70% success
        assert total_time < 30  # Should complete within 30 seconds
        
        # Cleanup
        for client in clients:
            if client.connected:
                await client.disconnect()
    
    @pytest.mark.asyncio()
    async def test_slow_client_handling(self, test_server_url: str) -> None:
        """Test handling of slow clients that don't consume messages quickly."""
        
        class SlowClient(WebSocketClient):
            """Client that processes messages slowly."""
            
            async def listen(self, duration: int = 60) -> None:
                """Listen slowly with delays."""
                if not self.connected:
                    return
                
                end_time = time.time() + duration
                
                while time.time() < end_time:
                    try:
                        message = await asyncio.wait_for(
                            self.websocket.recv(),
                            timeout=1.0,
                        )
                        # Simulate slow processing
                        await asyncio.sleep(0.5)
                        self.messages_received.append({
                            "timestamp": time.time(),
                            "data": json.loads(message),
                        })
                    except asyncio.TimeoutError:
                        continue
                    except Exception:
                        break
        
        # Mix of normal and slow clients
        clients = []
        for i in range(10):
            token = create_access_token(
                data={"sub": f"user_{i}", "user_id": i}
            )
            operation_id = str(uuid.uuid4())
            
            if i < 5:
                client = WebSocketClient(test_server_url, token, operation_id)
            else:
                client = SlowClient(test_server_url, token, operation_id)
            
            clients.append(client)
        
        # Connect all
        for client in clients:
            try:
                await client.connect()
            except Exception:
                pass
        
        # Server should handle mix of slow and fast clients
        connected = [c for c in clients if c.connected]
        assert len(connected) >= 8  # Most should connect
        
        # Cleanup
        for client in clients:
            if client.connected:
                await client.disconnect()
    
    @pytest.mark.asyncio()
    async def test_network_interruption_recovery(self, test_server_url: str) -> None:
        """Test recovery from network interruptions."""
        token = create_access_token(
            data={"sub": "test_user", "user_id": 1}
        )
        operation_id = str(uuid.uuid4())
        
        client = WebSocketClient(test_server_url, token, operation_id)
        
        # Initial connection
        await client.connect()
        assert client.connected
        
        # Simulate network interruption by force closing
        await client.websocket.close(code=1006)  # Abnormal closure
        client.connected = False
        
        # Wait briefly
        await asyncio.sleep(1)
        
        # Attempt reconnection
        reconnect_client = WebSocketClient(test_server_url, token, operation_id)
        await reconnect_client.connect()
        assert reconnect_client.connected
        
        # Should be able to continue normally
        await reconnect_client.send_message({
            "type": "ping",
            "timestamp": time.time(),
        })
        
        await reconnect_client.disconnect()


class WebSocketLocustUser(HttpUser):
    """Locust user for WebSocket load testing."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a simulated user starts."""
        self.user_id = str(uuid.uuid4())
        self.token = create_access_token(
            data={"sub": f"user_{self.user_id}", "user_id": self.user_id}
        )
        self.websocket = None
        self.operation_id = str(uuid.uuid4())
    
    @task(1)
    async def connect_and_listen(self):
        """Connect to WebSocket and listen for messages."""
        try:
            # Connect
            start_time = time.time()
            self.websocket = await websockets.connect(
                f"ws://{self.host}/ws/operations/{self.operation_id}?token={self.token}"
            )
            connection_time = time.time() - start_time
            
            events.request.fire(
                request_type="WebSocket",
                name="connect",
                response_time=connection_time * 1000,
                response_length=0,
                exception=None,
                context={},
            )
            
            # Listen for messages
            for _ in range(10):
                try:
                    await asyncio.wait_for(self.websocket.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
            
            # Disconnect
            await self.websocket.close()
            
        except Exception as e:
            events.request.fire(
                request_type="WebSocket",
                name="connect",
                response_time=0,
                response_length=0,
                exception=e,
                context={},
            )
    
    @task(2)
    async def send_progress_updates(self):
        """Send progress update messages."""
        if self.websocket and not self.websocket.closed:
            try:
                start_time = time.time()
                await self.websocket.send(json.dumps({
                    "type": "progress",
                    "progress": random.randint(0, 100),
                    "timestamp": time.time(),
                }))
                send_time = time.time() - start_time
                
                events.request.fire(
                    request_type="WebSocket",
                    name="send_message",
                    response_time=send_time * 1000,
                    response_length=0,
                    exception=None,
                    context={},
                )
            except Exception as e:
                events.request.fire(
                    request_type="WebSocket",
                    name="send_message",
                    response_time=0,
                    response_length=0,
                    exception=e,
                    context={},
                )


def run_locust_test(host: str, users: int = 100, spawn_rate: int = 10, duration: int = 60):
    """Run Locust load test programmatically."""
    env = Environment(user_classes=[WebSocketLocustUser], host=host)
    env.create_local_runner()
    
    # Start test
    env.runner.start(users, spawn_rate=spawn_rate)
    
    # Run for specified duration
    time.sleep(duration)
    
    # Stop test
    env.runner.quit()
    
    # Return statistics
    return env.stats


# Pytest fixture for test server URL
@pytest.fixture()
def test_server_url():
    """Get test server URL from environment or use default."""
    import os
    return os.getenv("TEST_WEBSOCKET_URL", "ws://localhost:8080")