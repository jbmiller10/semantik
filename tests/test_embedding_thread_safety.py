#!/usr/bin/env python3
"""
Thread safety tests for embedding service singleton
"""
import asyncio
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from unittest.mock import MagicMock

# Mock metrics before importing
sys.modules["shared.metrics.prometheus"] = MagicMock()


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of the embedding service."""

    def test_singleton_thread_safety(self) -> None:
        """Test that singleton is thread-safe."""
        from shared.embedding import get_embedding_service_sync

        services: set[int] = set()
        lock = threading.Lock()

        def get_service_id() -> int:
            service = get_embedding_service_sync()
            service_id = id(service)
            with lock:
                services.add(service_id)
            return service_id

        # Get service from multiple threads
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(get_service_id) for _ in range(100)]
            for future in as_completed(futures):
                future.result()

        # Should only have one unique service instance
        assert len(services) == 1, "Multiple service instances created - not thread safe!"

    def test_concurrent_model_loading(self) -> None:
        """Test concurrent model loading doesn't cause issues."""
        from shared.embedding import EmbeddingService

        results = []
        errors = []
        lock = threading.Lock()

        def load_model(thread_id: int) -> None:
            try:
                service = EmbeddingService(mock_mode=True)
                success = service.load_model("test-model")
                with lock:
                    results.append((thread_id, success))
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))

        # Attempt concurrent model loading
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(load_model, i) for i in range(10)]
            for future in as_completed(futures):
                future.result()

        # All should succeed
        assert len(errors) == 0, f"Errors during concurrent loading: {errors}"
        assert len(results) == 10
        assert all(success for _, success in results)

    def test_concurrent_embedding_generation(self) -> None:
        """Test concurrent embedding generation is safe."""
        from shared.embedding import EmbeddingService

        service = EmbeddingService(mock_mode=True)
        service.load_model("test-model")

        results = []
        errors = []
        lock = threading.Lock()

        def generate_embeddings(thread_id: int) -> None:
            try:
                texts = [f"Thread {thread_id} text {i}" for i in range(100)]
                embeddings = service.generate_embeddings(texts, "test-model")
                with lock:
                    results.append((thread_id, embeddings.shape if embeddings is not None else None))
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))

        # Generate embeddings concurrently
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(generate_embeddings, i) for i in range(50)]
            for future in as_completed(futures):
                future.result()

        # All should succeed
        assert len(errors) == 0, f"Errors during concurrent generation: {errors}"
        assert len(results) == 50

        # All should have correct shape
        for thread_id, shape in results:
            assert shape == (100, 384), f"Thread {thread_id} got wrong shape: {shape}"

    def test_async_singleton_thread_safety(self) -> None:
        """Test async singleton is thread-safe across event loops."""
        from shared.embedding import get_embedding_service

        services: set[int] = set()
        lock = threading.Lock()

        def get_service_in_new_loop() -> int:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                service = loop.run_until_complete(get_embedding_service())
                service_id = id(service)
                with lock:
                    services.add(service_id)
                return service_id
            finally:
                loop.close()

        # Get service from multiple threads with different event loops
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_service_in_new_loop) for _ in range(20)]
            for future in as_completed(futures):
                future.result()

        # Should only have one unique service instance
        assert len(services) == 1, "Multiple async service instances created!"

    def test_race_condition_in_initialization(self) -> None:
        """Test no race conditions during service initialization."""
        from shared.embedding import cleanup, get_embedding_service

        async def test_concurrent_init() -> None:
            # Clean up any existing service
            await cleanup()

            # Try to initialize concurrently
            tasks = []
            for _ in range(10):
                tasks.append(get_embedding_service())

            services = await asyncio.gather(*tasks)

            # All should be the same instance
            first_id = id(services[0])
            for i, service in enumerate(services):
                assert id(service) == first_id, f"Service {i} has different id"

        asyncio.run(test_concurrent_init())

    def test_cleanup_during_active_requests(self) -> None:
        """Test cleanup behavior with active requests."""
        from shared.embedding import EmbeddingService

        service = EmbeddingService(mock_mode=True)
        service.load_model("test-model")

        results = []
        errors = []
        lock = threading.Lock()
        cleanup_called = threading.Event()

        def long_running_request() -> None:
            try:
                texts = ["text"] * 1000
                # Wait for cleanup to be called
                cleanup_called.wait()
                embeddings = service.generate_embeddings(texts, "test-model")
                with lock:
                    results.append(embeddings is not None)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        # Start long-running requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(long_running_request) for _ in range(5)]

            # Give threads time to start
            time.sleep(0.1)

            # Call cleanup while requests are active
            cleanup_called.set()
            service.unload_model()

            # Wait for all to complete
            for future in as_completed(futures):
                future.result()

        # Requests might fail or succeed depending on timing, but shouldn't crash
        total_completed = len(results) + len(errors)
        assert total_completed == 5, "Not all requests completed"


class TestAsyncConcurrency(unittest.TestCase):
    """Test async concurrency patterns."""

    def test_async_concurrent_embeddings(self) -> None:
        """Test concurrent async embedding generation."""

        async def run_test() -> None:
            from shared.embedding import get_embedding_service, initialize_embedding_service

            # Initialize service
            await initialize_embedding_service("test-model", mock_mode=True)
            service = await get_embedding_service()

            # Create concurrent tasks
            tasks = []
            for i in range(50):
                texts = [f"Task {i} text {j}" for j in range(100)]
                tasks.append(service.embed_texts(texts))

            # Run all concurrently
            start = time.time()
            results = await asyncio.gather(*tasks)
            duration = time.time() - start

            # Verify results
            assert len(results) == 50
            for i, embeddings in enumerate(results):
                assert embeddings.shape == (100, 384), f"Task {i} got wrong shape"

            # Should complete reasonably fast (concurrent execution)
            assert duration < 2.0, f"Concurrent execution too slow: {duration}s"

        asyncio.run(run_test())

    def test_async_exception_isolation(self) -> None:
        """Test that exceptions in one request don't affect others."""

        async def run_test() -> None:
            from shared.embedding import get_embedding_service, initialize_embedding_service

            # Create a custom service that sometimes fails
            class FailingService:
                def __init__(self, service: Any) -> None:
                    self.service = service
                    self.call_count = 0

                async def embed_texts(self, texts: list[str], **kwargs: Any) -> Any:
                    self.call_count += 1
                    # Fail every 3rd call
                    if self.call_count % 3 == 0:
                        raise RuntimeError("Simulated failure")
                    return await self.service.embed_texts(texts, **kwargs)

            await initialize_embedding_service("test-model", mock_mode=True)
            real_service = await get_embedding_service()

            # Wrap with failing service
            failing_service = FailingService(real_service)

            # Create tasks, some will fail
            tasks = []
            for i in range(10):
                texts = [f"Task {i} text"]
                tasks.append(failing_service.embed_texts(texts))

            # Gather with return_exceptions
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check results
            successes = sum(1 for r in results if not isinstance(r, Exception))
            failures = sum(1 for r in results if isinstance(r, Exception))

            # Should have some of each
            assert successes > 0, "No successful requests"
            assert failures > 0, "No failed requests"
            assert successes + failures == 10, "Wrong total count"

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
