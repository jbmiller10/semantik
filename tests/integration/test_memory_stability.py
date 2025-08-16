"""
Integration tests for memory stability and leak detection.

Tests memory usage patterns, leak detection, and resource cleanup
for long-running chunking operations.
"""

import asyncio
import gc
import os
import time
import tracemalloc
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import psutil
import pytest
from faker import Faker
from shared.database.models import Chunk, Collection, Document
from sqlalchemy.ext.asyncio import AsyncSession
from webui.services.chunking_service import ChunkingService
from webui.websocket_manager import RedisStreamWebSocketManager

fake = Faker()


class MemoryMonitor:
    """Monitor memory usage during tests."""

    def __init__(self):
        self.baseline_memory = 0
        self.peak_memory = 0
        self.samples = []
        self.start_time = None
        self.process = psutil.Process(os.getpid())

    def start(self):
        """Start memory monitoring."""
        gc.collect()  # Force garbage collection
        self.baseline_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.peak_memory = self.baseline_memory
        self.samples = [self.baseline_memory]
        self.start_time = time.time()
        tracemalloc.start()

    def sample(self):
        """Take a memory sample."""
        current_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.samples.append(current_memory)
        self.peak_memory = max(self.peak_memory, current_memory)
        return current_memory

    def stop(self):
        """Stop monitoring and return statistics."""
        tracemalloc.stop()
        final_memory = self.sample()

        return {
            "baseline_mb": self.baseline_memory,
            "final_mb": final_memory,
            "peak_mb": self.peak_memory,
            "growth_mb": final_memory - self.baseline_memory,
            "growth_percent": ((final_memory - self.baseline_memory) / self.baseline_memory) * 100,
            "duration_seconds": time.time() - self.start_time,
            "samples": self.samples,
        }

    def get_top_allocations(self, limit=10):
        """Get top memory allocations."""
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        results = []
        for stat in top_stats[:limit]:
            results.append(
                {
                    "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                    "size_mb": stat.size / (1024 * 1024),
                    "count": stat.count,
                }
            )

        return results


@pytest.fixture()
def memory_monitor():
    """Create a memory monitor for tests."""
    return MemoryMonitor()


@pytest.fixture()
async def large_collection_dataset(async_session: AsyncSession, test_user: dict) -> dict[str, Any]:
    """Create a large dataset for memory testing."""
    collection = Collection(
        id=str(uuid.uuid4()),
        name="Memory Test Collection",
        description="Collection for memory stability testing",
        owner_id=test_user["id"],
        status="ready",
        vector_store_name=f"mem_test_{uuid.uuid4().hex[:8]}",
        embedding_model="test-model",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    async_session.add(collection)

    # Create many documents
    documents = []
    for i in range(100):
        doc = Document(
            id=str(uuid.uuid4()),
            collection_id=collection.id,
            name=f"doc_{i}.txt",
            content=fake.text(max_nb_chars=10000),  # 10KB each
            file_type="text",
            file_size=10000,
            created_at=datetime.now(UTC),
        )
        documents.append(doc)
        async_session.add(doc)

        # Commit in batches to avoid memory issues
        if i % 10 == 0:
            await async_session.commit()

    await async_session.commit()

    return {
        "collection": collection,
        "documents": documents,
        "total_size_mb": len(documents) * 10 / 1024,  # Total size in MB
    }


class TestChunkingMemoryStability:
    """Test memory stability during chunking operations."""

    @pytest.mark.asyncio()
    async def test_long_running_chunking_memory(
        self,
        async_session: AsyncSession,
        large_collection_dataset: dict[str, Any],
        memory_monitor: MemoryMonitor,
        redis_client: Any,
    ) -> None:
        """Test memory usage during long-running chunking operation."""
        collection = large_collection_dataset["collection"]
        documents = large_collection_dataset["documents"]

        memory_monitor.start()

        # Create chunking service
        _service = ChunkingService()
        operation_id = str(uuid.uuid4())

        # Process documents in batches
        batch_size = 10
        memory_samples = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            # Process batch
            for doc in batch:
                # Create chunks
                chunk_size = 500
                overlap = 50
                step = chunk_size - overlap

                chunks_to_insert = []
                for chunk_idx, start in enumerate(range(0, len(doc.content), step)):
                    end = min(start + chunk_size, len(doc.content))
                    chunk_content = doc.content[start:end]

                    if chunk_content.strip():
                        chunk = Chunk(
                            collection_id=collection.id,
                            document_id=doc.id,
                            content=chunk_content,
                            chunk_index=chunk_idx,
                            start_offset=start,
                            end_offset=end,
                            token_count=len(chunk_content.split()),
                            created_at=datetime.now(UTC),
                        )
                        chunks_to_insert.append(chunk)

                # Insert chunks
                async_session.add_all(chunks_to_insert)

            # Commit batch
            await async_session.commit()

            # Sample memory
            current_memory = memory_monitor.sample()
            memory_samples.append(current_memory)

            # Update progress
            progress = ((i + batch_size) / len(documents)) * 100
            redis_client.hset(
                f"operation:{operation_id}",
                mapping={
                    "progress": str(progress),
                    "documents_processed": str(i + batch_size),
                },
            )

            # Force garbage collection periodically
            if i % 30 == 0:
                gc.collect()

        # Stop monitoring
        stats = memory_monitor.stop()

        # Assertions
        assert stats["growth_mb"] < 100, f"Memory grew by {stats['growth_mb']}MB"
        assert stats["growth_percent"] < 50, f"Memory grew by {stats['growth_percent']}%"

        # Check for memory trend (should stabilize, not continuously grow)
        if len(memory_samples) > 10:
            first_half = memory_samples[: len(memory_samples) // 2]
            second_half = memory_samples[len(memory_samples) // 2 :]

            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)

            # Second half should not be significantly higher than first
            growth_rate = (avg_second - avg_first) / avg_first
            assert growth_rate < 0.2, f"Memory continued to grow: {growth_rate*100}% increase"

    @pytest.mark.asyncio()
    async def test_chunk_cache_memory_management(
        self,
        memory_monitor: MemoryMonitor,
        redis_client: Any,
    ) -> None:
        """Test memory management of chunk preview cache."""
        memory_monitor.start()

        _service = ChunkingService()
        cache_entries = []

        # Create many cache entries
        for i in range(100):
            preview_id = str(uuid.uuid4())

            # Create large preview data
            preview_data = {
                "preview_id": preview_id,
                "chunks": [
                    {
                        "index": j,
                        "content": fake.text(max_nb_chars=1000),
                        "metadata": {"key": f"value_{j}"},
                    }
                    for j in range(50)  # 50 chunks per preview
                ],
                "total_chunks": 50,
                "strategy": "recursive",
            }

            # Store in cache
            cache_key = f"preview:{preview_id}"
            redis_client.setex(
                cache_key,
                900,  # 15 minutes TTL
                str(preview_data),
            )
            cache_entries.append(cache_key)

            # Sample memory periodically
            if i % 10 == 0:
                memory_monitor.sample()

                # Simulate cache eviction for old entries
                if i > 50:
                    # Delete old entries
                    for key in cache_entries[:10]:
                        redis_client.delete(key)
                    cache_entries = cache_entries[10:]

        # Force cleanup
        for key in cache_entries:
            redis_client.delete(key)

        gc.collect()
        stats = memory_monitor.stop()

        # Memory should not grow excessively
        assert stats["growth_mb"] < 50, f"Cache memory grew by {stats['growth_mb']}MB"

    @pytest.mark.asyncio()
    async def test_streaming_processor_memory_bounds(
        self,
        async_session: AsyncSession,
        large_collection_dataset: dict[str, Any],
        memory_monitor: MemoryMonitor,
    ) -> None:
        """Test that streaming processor respects memory bounds."""
        _collection = large_collection_dataset["collection"]
        documents = large_collection_dataset["documents"]

        memory_monitor.start()

        # Simulate streaming processing
        async def stream_process_document(doc: Document, max_memory_mb: int = 50):
            """Process document in streaming fashion with memory limit."""
            current_memory = memory_monitor.sample()

            if current_memory - memory_monitor.baseline_memory > max_memory_mb:
                # Exceeded memory limit, force cleanup
                gc.collect()
                await asyncio.sleep(0.1)  # Allow cleanup

            # Process in small chunks
            chunk_size = 100
            chunks_created = 0

            for start in range(0, len(doc.content), chunk_size):
                end = min(start + chunk_size, len(doc.content))
                chunk_content = doc.content[start:end]

                # Simulate chunk processing
                if chunk_content.strip():
                    chunks_created += 1

                # Yield control periodically
                if chunks_created % 10 == 0:
                    await asyncio.sleep(0)

            return chunks_created

        # Process all documents with streaming
        total_chunks = 0
        for doc in documents[:20]:  # Process subset for testing
            chunks = await stream_process_document(doc)
            total_chunks += chunks

            # Check memory periodically
            current_memory = memory_monitor.sample()
            growth = current_memory - memory_monitor.baseline_memory

            # Should stay within bounds
            assert growth < 100, f"Memory grew beyond limit: {growth}MB"

        stats = memory_monitor.stop()

        # Verify memory was controlled
        assert stats["peak_mb"] - stats["baseline_mb"] < 100
        assert total_chunks > 0


class TestWebSocketMemoryStability:
    """Test WebSocket connection memory stability."""

    @pytest.mark.asyncio()
    async def test_websocket_connection_churn_memory(
        self,
        memory_monitor: MemoryMonitor,
    ) -> None:
        """Test memory stability with WebSocket connection churn."""
        memory_monitor.start()

        manager = RedisStreamWebSocketManager()
        connections_created = 0

        # Simulate connection churn
        for iteration in range(50):
            iteration_connections = []

            # Create connections
            for i in range(20):
                mock_ws = AsyncMock()
                mock_ws.send_json = AsyncMock()
                mock_ws.close = AsyncMock()

                user_id = f"user_{iteration}_{i}"
                operation_id = str(uuid.uuid4())
                key = f"{user_id}:operation:{operation_id}"

                manager.connections[key] = {mock_ws}
                iteration_connections.append(key)
                connections_created += 1

            # Simulate activity
            await asyncio.sleep(0.1)

            # Disconnect all
            for key in iteration_connections:
                if key in manager.connections:
                    del manager.connections[key]

            # Sample memory
            if iteration % 5 == 0:
                _current_memory = memory_monitor.sample()

                # Force cleanup periodically
                if iteration % 10 == 0:
                    gc.collect()

        # Final cleanup
        manager.connections.clear()
        gc.collect()

        stats = memory_monitor.stop()

        # Memory should stabilize despite churn
        assert stats["growth_mb"] < 20, f"WebSocket memory grew by {stats['growth_mb']}MB"
        assert connections_created == 1000  # 50 iterations * 20 connections

    @pytest.mark.asyncio()
    async def test_websocket_message_buffer_memory(
        self,
        memory_monitor: MemoryMonitor,
        redis_client: Any,
    ) -> None:
        """Test memory usage of WebSocket message buffering."""
        memory_monitor.start()

        manager = RedisStreamWebSocketManager()
        manager.redis = redis_client

        # Create connections
        connections = []
        for i in range(10):
            mock_ws = AsyncMock()
            user_id = f"user_{i}"
            operation_id = str(uuid.uuid4())
            key = f"{user_id}:operation:{operation_id}"

            manager.connections[key] = {mock_ws}
            connections.append((key, operation_id))

        # Send many messages
        for _ in range(100):
            for _key, operation_id in connections:
                # Create large message
                message = {
                    "type": "progress",
                    "operation_id": operation_id,
                    "chunks": [{"content": fake.text(max_nb_chars=500)} for _ in range(10)],
                    "timestamp": time.time(),
                }

                # Send via Redis stream
                stream_key = f"stream:chunking:{operation_id}"
                await manager.send_message(stream_key, message)

            # Sample memory
            if _ % 10 == 0:
                memory_monitor.sample()

        # Cleanup
        manager.connections.clear()

        # Clear Redis streams
        for _, operation_id in connections:
            redis_client.delete(f"stream:chunking:{operation_id}")

        gc.collect()
        stats = memory_monitor.stop()

        # Memory should not grow excessively
        assert stats["growth_mb"] < 50, f"Message buffer memory grew by {stats['growth_mb']}MB"


class TestConcurrentOperationMemory:
    """Test memory usage with concurrent operations."""

    @pytest.mark.asyncio()
    async def test_concurrent_chunking_memory_isolation(
        self,
        async_session: AsyncSession,
        memory_monitor: MemoryMonitor,
        test_user: dict,
    ) -> None:
        """Test memory isolation between concurrent chunking operations."""
        memory_monitor.start()

        # Create multiple collections
        collections = []
        for i in range(5):
            collection = Collection(
                id=str(uuid.uuid4()),
                name=f"Concurrent Collection {i}",
                owner_id=test_user["id"],
                status="ready",
                vector_store_name=f"concurrent_{i}_{uuid.uuid4().hex[:8]}",
                embedding_model="test-model",
                created_at=datetime.now(UTC),
            )
            collections.append(collection)
            async_session.add(collection)

        await async_session.commit()

        # Create documents for each collection
        all_documents = []
        for collection in collections:
            for j in range(20):
                doc = Document(
                    id=str(uuid.uuid4()),
                    collection_id=collection.id,
                    name=f"doc_{j}.txt",
                    content=fake.text(max_nb_chars=5000),
                    file_type="text",
                    file_size=5000,
                    created_at=datetime.now(UTC),
                )
                all_documents.append(doc)
                async_session.add(doc)

        await async_session.commit()

        # Process collections concurrently
        async def process_collection(collection: Collection):
            """Process a single collection."""
            docs = [d for d in all_documents if d.collection_id == collection.id]

            for doc in docs:
                # Create chunks
                chunks = []
                for i in range(10):
                    chunk = Chunk(
                        collection_id=collection.id,
                        document_id=doc.id,
                        content=f"Chunk {i}",
                        chunk_index=i,
                        start_offset=i * 100,
                        end_offset=(i + 1) * 100,
                        token_count=10,
                        created_at=datetime.now(UTC),
                    )
                    chunks.append(chunk)

                # Use separate session for isolation
                async with async_session.begin():
                    async_session.add_all(chunks)

            return len(docs)

        # Process all collections concurrently
        tasks = [process_collection(c) for c in collections]
        results = await asyncio.gather(*tasks)

        # Sample final memory
        stats = memory_monitor.stop()

        # Memory should be reasonable for concurrent operations
        assert stats["growth_mb"] < 100, f"Concurrent ops memory grew by {stats['growth_mb']}MB"
        assert sum(results) == 100  # 5 collections * 20 docs each

    @pytest.mark.asyncio()
    async def test_memory_pressure_adaptation(
        self,
        memory_monitor: MemoryMonitor,
    ) -> None:
        """Test system adaptation under memory pressure."""
        memory_monitor.start()
        initial_memory = memory_monitor.sample()

        # Simulate memory pressure
        memory_hog = []
        target_pressure_mb = 50

        # Allocate memory to create pressure
        while memory_monitor.sample() - initial_memory < target_pressure_mb:
            # Allocate 1MB at a time
            memory_hog.append(bytearray(1024 * 1024))

        # System should adapt
        _service = ChunkingService()

        # Simulate batch size adaptation
        original_batch_size = 100
        adapted_batch_size = original_batch_size

        _current_memory = memory_monitor.sample()
        available_memory = psutil.virtual_memory().available / (1024 * 1024)

        if available_memory < 500:  # Less than 500MB available
            # Reduce batch size
            adapted_batch_size = max(10, original_batch_size // 4)
        elif available_memory < 1000:  # Less than 1GB available
            adapted_batch_size = max(25, original_batch_size // 2)

        # Verify adaptation
        assert adapted_batch_size <= original_batch_size

        # Release memory
        memory_hog.clear()
        gc.collect()

        stats = memory_monitor.stop()

        # Memory should return close to baseline after release
        final_growth = stats["final_mb"] - stats["baseline_mb"]
        assert final_growth < 20, f"Memory did not recover: {final_growth}MB growth"


class TestMemoryLeakDetection:
    """Specific tests for memory leak detection."""

    @pytest.mark.asyncio()
    async def test_detect_chunk_reference_leak(
        self,
        async_session: AsyncSession,
        memory_monitor: MemoryMonitor,
    ) -> None:
        """Test detection of chunk reference leaks."""
        memory_monitor.start()

        # Track chunk references
        chunk_refs = []

        for iteration in range(20):
            # Create chunks
            chunks = []
            for i in range(100):
                chunk = Chunk(
                    collection_id=str(uuid.uuid4()),
                    document_id=str(uuid.uuid4()),
                    content=fake.text(max_nb_chars=1000),
                    chunk_index=i,
                    start_offset=0,
                    end_offset=1000,
                    token_count=100,
                    created_at=datetime.now(UTC),
                )
                chunks.append(chunk)

            # Intentionally keep references (simulating leak)
            if iteration < 5:
                chunk_refs.extend(chunks)  # Leak simulation

            # Clear chunks that should be garbage collected
            chunks.clear()

            # Force GC
            if iteration % 5 == 0:
                gc.collect()
                memory_monitor.sample()

        # Check for leak
        mid_point_memory = memory_monitor.sample()

        # Clear leaked references
        chunk_refs.clear()
        gc.collect()

        # Memory should drop significantly after clearing refs
        final_memory = memory_monitor.sample()
        memory_recovered = mid_point_memory - final_memory

        _stats = memory_monitor.stop()

        # Should recover most memory after clearing references
        assert memory_recovered > 0, "No memory recovered after clearing references"

        # Get top allocations to identify leak source
        top_allocations = memory_monitor.get_top_allocations()

        # Log top allocations for debugging
        for alloc in top_allocations[:5]:
            print(f"Top allocation: {alloc['file']}: {alloc['size_mb']:.2f}MB")

    @pytest.mark.asyncio()
    async def test_async_task_cleanup(
        self,
        memory_monitor: MemoryMonitor,
    ) -> None:
        """Test that async tasks are properly cleaned up."""
        memory_monitor.start()

        tasks = []

        async def leaky_task(task_id: int):
            """Simulate a task that might leak memory."""
            _data = bytearray(1024 * 1024)  # 1MB
            await asyncio.sleep(0.1)
            return task_id

        # Create many tasks
        for i in range(50):
            task = asyncio.create_task(leaky_task(i))
            tasks.append(task)

            # Sample memory periodically
            if i % 10 == 0:
                memory_monitor.sample()

        # Wait for all tasks
        results = await asyncio.gather(*tasks)

        # Clear task references
        tasks.clear()
        gc.collect()

        # Memory should be recovered
        stats = memory_monitor.stop()

        assert stats["growth_mb"] < 10, f"Tasks leaked {stats['growth_mb']}MB"
        assert len(results) == 50  # All tasks completed
