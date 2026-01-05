#!/usr/bin/env python3
"""
Stress test to validate WebSocket race condition fixes.

This script performs aggressive concurrent operations to detect race conditions.
"""

import asyncio
import logging
import random
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from webui.websocket.legacy_stream_manager import RedisStreamWebSocketManager
from webui.websocket.scalable_manager import ScalableWebSocketManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StressTestResults:
    """Track results of stress testing."""

    def __init__(self):
        self.errors: list[str] = []
        self.race_conditions: list[str] = []
        self.performance_metrics: dict[str, float] = {}
        self.operations_count = 0
        self.start_time = time.time()

    def add_error(self, error: str):
        self.errors.append(error)

    def add_race_condition(self, description: str):
        self.race_conditions.append(description)

    def add_metric(self, name: str, value: float):
        self.performance_metrics[name] = value

    def get_duration(self) -> float:
        return time.time() - self.start_time

    def print_summary(self):
        duration = self.get_duration()
        print("\n" + "=" * 60)
        print("STRESS TEST RESULTS")
        print("=" * 60)
        print(f"Duration: {duration:.2f} seconds")
        print(f"Total operations: {self.operations_count}")
        print(f"Operations/second: {self.operations_count / duration:.2f}")
        print(f"Errors found: {len(self.errors)}")
        print(f"Race conditions detected: {len(self.race_conditions)}")

        if self.errors:
            print("\nErrors:")
            for i, error in enumerate(self.errors[:5], 1):
                print(f"  {i}. {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")

        if self.race_conditions:
            print("\nRace Conditions:")
            for i, race in enumerate(self.race_conditions[:5], 1):
                print(f"  {i}. {race}")
            if len(self.race_conditions) > 5:
                print(f"  ... and {len(self.race_conditions) - 5} more")

        print("\nPerformance Metrics:")
        for metric, value in self.performance_metrics.items():
            print(f"  {metric}: {value:.3f}")

        print("=" * 60)

        return len(self.errors) == 0 and len(self.race_conditions) == 0


def create_mock_websocket(conn_id: str):
    """Create a mock WebSocket for testing."""
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    ws.ping = AsyncMock()
    ws._id = conn_id
    return ws


async def stress_test_redis_stream_manager():
    """Stress test the RedisStreamWebSocketManager."""
    results = StressTestResults()
    manager = RedisStreamWebSocketManager()

    print("\nTesting RedisStreamWebSocketManager (WITH locking)")
    print("-" * 50)

    with patch("webui.websocket.legacy_stream_manager.redis") as mock_redis:
        mock_redis_client = AsyncMock()
        mock_redis.from_url = AsyncMock(return_value=mock_redis_client)
        mock_redis_client.ping = AsyncMock()

        await manager.startup()

        # Track state for validation
        active_connections: set[str] = set()
        connection_lock = asyncio.Lock()

        async def aggressive_connect(worker_id: int, iterations: int):
            """Aggressively add connections."""
            for i in range(iterations):
                user_id = f"user_{worker_id}_{i % 10}"
                operation_id = f"op_{worker_id}_{i}"
                conn_id = f"conn_{worker_id}_{i}"

                ws = create_mock_websocket(conn_id)

                try:
                    await manager.connect(ws, operation_id, user_id)
                    async with connection_lock:
                        active_connections.add(conn_id)
                    results.operations_count += 1

                    # Random delay to vary timing
                    await asyncio.sleep(random.uniform(0.0001, 0.001))

                except Exception as e:
                    results.add_error(f"Connect error: {e}")

        async def aggressive_disconnect(_worker_id: int, iterations: int):
            """Aggressively remove connections."""
            await asyncio.sleep(0.01)  # Let some connections establish

            for _ in range(iterations):
                # Try to disconnect random connections
                async with connection_lock:
                    if active_connections:
                        conn_id = random.choice(list(active_connections))
                        active_connections.discard(conn_id)

                # Find and disconnect a random connection
                if manager.connections:
                    try:
                        key = random.choice(list(manager.connections.keys()))
                        if key in manager.connections and manager.connections[key]:
                            ws = next(iter(manager.connections[key]))
                            user_id = key.split(":")[0]
                            operation_id = key.split(":")[2] if len(key.split(":")) > 2 else "unknown"

                            await manager.disconnect(ws, operation_id, user_id)
                            results.operations_count += 1
                    except (KeyError, IndexError) as e:
                        results.add_race_condition(f"Disconnect race: {e}")
                    except Exception as e:
                        results.add_error(f"Disconnect error: {e}")

                await asyncio.sleep(random.uniform(0.0001, 0.001))

        async def aggressive_broadcast(worker_id: int, iterations: int):
            """Aggressively send broadcasts."""
            await asyncio.sleep(0.005)  # Let some connections establish

            for i in range(iterations):
                if manager.connections:
                    try:
                        # Pick a random operation to broadcast to
                        keys = [k for k in manager.connections if "operation:" in k]
                        if keys:
                            key = random.choice(keys)
                            operation_id = key.split(":")[2] if len(key.split(":")) > 2 else "unknown"

                            message = {"worker": worker_id, "msg": i, "timestamp": time.time()}
                            await manager._broadcast(operation_id, message)
                            results.operations_count += 1
                    except RuntimeError as e:
                        if "dictionary changed size during iteration" in str(e):
                            results.add_race_condition(f"Broadcast iteration race: {e}")
                        else:
                            results.add_error(f"Broadcast error: {e}")
                    except Exception as e:
                        results.add_error(f"Broadcast error: {e}")

                await asyncio.sleep(random.uniform(0.0001, 0.001))

        async def aggressive_cleanup(iterations: int):
            """Aggressively run cleanup operations."""
            await asyncio.sleep(0.01)  # Let some connections establish

            for _ in range(iterations):
                try:
                    await manager.cleanup_stale_connections()
                    results.operations_count += 1
                except RuntimeError as e:
                    if "dictionary changed size during iteration" in str(e):
                        results.add_race_condition(f"Cleanup iteration race: {e}")
                    else:
                        results.add_error(f"Cleanup error: {e}")
                except Exception as e:
                    results.add_error(f"Cleanup error: {e}")

                await asyncio.sleep(random.uniform(0.01, 0.02))

        # Launch many concurrent workers
        num_workers = 20
        iterations_per_worker = 50

        print(f"Launching {num_workers} workers, {iterations_per_worker} iterations each...")

        tasks = []

        # Mix of different operations running concurrently
        for i in range(num_workers // 4):
            tasks.append(aggressive_connect(i, iterations_per_worker))
        for i in range(num_workers // 4):
            tasks.append(aggressive_disconnect(i, iterations_per_worker))
        for i in range(num_workers // 4):
            tasks.append(aggressive_broadcast(i, iterations_per_worker))
        for i in range(num_workers // 4):
            tasks.append(aggressive_connect(i + 100, iterations_per_worker))  # More connects

        # Add cleanup task
        tasks.append(aggressive_cleanup(20))

        # Run all tasks concurrently
        start_time = time.time()
        await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        results.add_metric("total_time", elapsed)
        results.add_metric("ops_per_second", results.operations_count / elapsed)

        # Validate final state
        for key, websockets in manager.connections.items():
            if not isinstance(websockets, set):
                results.add_race_condition(f"Connection value corrupted: {key} -> {type(websockets)}")

        # Check for orphaned consumer tasks
        for operation_id, task in manager.consumer_tasks.items():
            if not any(operation_id in k for k in manager.connections) and not task.done():
                results.add_race_condition(f"Orphaned consumer task: {operation_id}")

    return results


async def stress_test_scalable_manager():
    """Stress test the ScalableWebSocketManager (demonstrating issues)."""
    results = StressTestResults()
    manager = ScalableWebSocketManager()

    print("\nTesting ScalableWebSocketManager (WITHOUT proper locking)")
    print("-" * 50)

    with patch("webui.websocket.scalable_manager.redis") as mock_redis:
        mock_redis_client = AsyncMock()
        mock_redis.from_url = AsyncMock(return_value=mock_redis_client)
        mock_redis_client.ping = AsyncMock()
        mock_redis_client.pubsub = MagicMock()
        mock_redis_client.pubsub.return_value.subscribe = AsyncMock()

        await manager.startup()

        async def aggressive_operations(worker_id: int, iterations: int):
            """Perform mixed operations aggressively."""
            for i in range(iterations):
                op = random.choice(["add", "remove", "iterate"])

                try:
                    if op == "add":
                        conn_id = f"conn_{worker_id}_{i}"
                        ws = create_mock_websocket(conn_id)
                        manager.local_connections[conn_id] = ws
                        manager.connection_metadata[conn_id] = {"user_id": f"user_{worker_id}"}
                        results.operations_count += 1

                    elif op == "remove":
                        if manager.local_connections:
                            conn_id = random.choice(list(manager.local_connections.keys()))
                            del manager.local_connections[conn_id]
                            if conn_id in manager.connection_metadata:
                                del manager.connection_metadata[conn_id]
                            results.operations_count += 1

                    elif op == "iterate":
                        # This can cause "dictionary changed size during iteration" errors
                        for _, (_, metadata) in enumerate(manager.connection_metadata.items()):
                            _ = metadata.get("user_id")
                        results.operations_count += 1

                except RuntimeError as e:
                    if "dictionary changed size during iteration" in str(e):
                        results.add_race_condition(f"Iteration race detected: {e}")
                    else:
                        results.add_error(str(e))
                except KeyError as e:
                    results.add_race_condition(f"Key error race: {e}")
                except Exception as e:
                    results.add_error(str(e))

                # No delay to maximize race condition probability
                if i % 10 == 0:
                    await asyncio.sleep(0)  # Yield to other tasks

        # Launch many concurrent workers
        num_workers = 20
        iterations_per_worker = 100

        print(f"Launching {num_workers} workers, {iterations_per_worker} iterations each...")

        tasks = [aggressive_operations(i, iterations_per_worker) for i in range(num_workers)]

        start_time = time.time()
        await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        results.add_metric("total_time", elapsed)
        results.add_metric("ops_per_second", results.operations_count / elapsed)

        # Check for inconsistencies
        if len(manager.local_connections) != len(manager.connection_metadata):
            results.add_race_condition(
                f"State inconsistency: {len(manager.local_connections)} connections vs "
                f"{len(manager.connection_metadata)} metadata entries"
            )

    return results


async def main():
    """Run all stress tests."""
    print("\n" + "=" * 60)
    print("WEBSOCKET RACE CONDITION STRESS TEST")
    print("=" * 60)

    # Test RedisStreamWebSocketManager (with locks)
    redis_results = await stress_test_redis_stream_manager()
    redis_passed = redis_results.print_summary()

    # Test ScalableWebSocketManager (without locks)
    scalable_results = await stress_test_scalable_manager()
    scalable_passed = scalable_results.print_summary()

    # Final verdict
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    if redis_passed:
        print("✅ RedisStreamWebSocketManager: PASSED (No race conditions detected)")
    else:
        print("❌ RedisStreamWebSocketManager: FAILED (Race conditions found)")

    if not scalable_passed:  # Expected to fail
        print("⚠️  ScalableWebSocketManager: FAILED (Expected - lacks proper locking)")
    else:
        print("🤔 ScalableWebSocketManager: PASSED (Unexpected - should have race conditions)")

    print("\nCONCLUSION:")
    if redis_passed:
        print("The RedisStreamWebSocketManager's locking implementation successfully")
        print("prevents race conditions under heavy concurrent load.")
    else:
        print("The RedisStreamWebSocketManager still has race condition issues.")

    if not scalable_passed:
        print("\nThe ScalableWebSocketManager needs similar locking mechanisms")
        print("to handle concurrent access safely.")

    print("=" * 60)

    return redis_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
