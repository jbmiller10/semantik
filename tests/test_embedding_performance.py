#!/usr/bin/env python3
"""
Performance benchmarks for embedding service
"""
import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

# Mock metrics before importing
sys.modules["shared.metrics.prometheus"] = MagicMock()


def benchmark_sync_embeddings() -> dict[str, Any]:
    """Benchmark synchronous embedding generation."""
    from shared.embedding import EmbeddingService

    service = EmbeddingService(mock_mode=True)

    # Test different batch sizes
    results = {}
    text_counts = [1, 10, 100, 1000]

    for count in text_counts:
        texts = [f"This is test text number {i} for benchmarking" for i in range(count)]

        # Warm up
        service.generate_embeddings(texts[:1], "test-model", batch_size=32)

        # Benchmark
        start = time.time()
        embeddings = service.generate_embeddings(texts, "test-model", batch_size=32)
        duration = time.time() - start

        results[f"{count}_texts"] = {
            "duration": duration,
            "texts_per_second": count / duration,
            "embeddings_shape": embeddings.shape if embeddings is not None else None,
        }

    return results


async def benchmark_async_embeddings() -> dict[str, Any]:
    """Benchmark asynchronous embedding generation."""
    from shared.embedding import initialize_embedding_service

    # Initialize with mock model
    service = await initialize_embedding_service("test-model", mock_mode=True)

    results = {}
    text_counts = [1, 10, 100, 1000]

    for count in text_counts:
        texts = [f"This is test text number {i} for benchmarking" for i in range(count)]

        # Warm up
        await service.embed_texts(texts[:1], batch_size=32)

        # Benchmark
        start = time.time()
        embeddings = await service.embed_texts(texts, batch_size=32)
        duration = time.time() - start

        results[f"{count}_texts"] = {
            "duration": duration,
            "texts_per_second": count / duration,
            "embeddings_shape": embeddings.shape,
        }

    return results


def benchmark_concurrent_requests() -> dict[str, Any]:
    """Benchmark concurrent request handling."""
    from shared.embedding import EmbeddingService

    service = EmbeddingService(mock_mode=True)

    def make_request(request_id: int) -> float:
        texts = [f"Request {request_id} text {i}" for i in range(10)]
        start = time.time()
        service.generate_embeddings(texts, "test-model")
        return time.time() - start

    results = {}
    worker_counts = [1, 5, 10, 20]

    for workers in worker_counts:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            start = time.time()
            futures = [executor.submit(make_request, i) for i in range(workers * 10)]
            durations = [f.result() for f in futures]
            total_duration = time.time() - start

            results[f"{workers}_workers"] = {
                "total_duration": total_duration,
                "average_request_duration": sum(durations) / len(durations),
                "requests_per_second": len(futures) / total_duration,
                "total_requests": len(futures),
            }

    return results


def benchmark_memory_usage() -> dict[str, Any]:
    """Benchmark memory usage patterns."""
    import gc
    import os

    import psutil
    from shared.embedding import EmbeddingService

    process = psutil.Process(os.getpid())
    service = EmbeddingService(mock_mode=True)

    results = {}

    # Baseline memory
    gc.collect()
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Test increasing batch sizes
    batch_sizes = [10, 100, 1000]

    for size in batch_sizes:
        texts = [f"Text {i}" for i in range(size)]

        gc.collect()
        before_memory = process.memory_info().rss / 1024 / 1024

        embeddings = service.generate_embeddings(texts, "test-model", batch_size=32)

        after_memory = process.memory_info().rss / 1024 / 1024

        # Clean up
        del embeddings
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024

        results[f"batch_{size}"] = {
            "memory_before_mb": before_memory - baseline_memory,
            "memory_during_mb": after_memory - baseline_memory,
            "memory_after_mb": final_memory - baseline_memory,
            "peak_increase_mb": after_memory - before_memory,
        }

    return results


async def benchmark_service_lifecycle() -> dict[str, float]:
    """Benchmark service initialization and cleanup."""
    from shared.embedding import cleanup, get_embedding_service, initialize_embedding_service

    results = {}

    # Benchmark initialization
    start = time.time()
    await initialize_embedding_service("test-model", mock_mode=True)
    results["initialization_time"] = time.time() - start

    # Benchmark singleton retrieval
    start = time.time()
    for _ in range(100):
        _ = await get_embedding_service()
    results["singleton_retrieval_time_per_100"] = time.time() - start

    # Benchmark cleanup
    start = time.time()
    await cleanup()
    results["cleanup_time"] = time.time() - start

    return results


def run_all_benchmarks() -> None:
    """Run all benchmarks and print results."""
    print("=== Embedding Service Performance Benchmarks ===\n")

    # Synchronous benchmarks
    print("1. Synchronous Embedding Generation:")
    sync_results = benchmark_sync_embeddings()
    for key, value in sync_results.items():
        print(f"   {key}: {value}")
    print()

    # Asynchronous benchmarks
    print("2. Asynchronous Embedding Generation:")
    async_results = asyncio.run(benchmark_async_embeddings())
    for key, value in async_results.items():
        print(f"   {key}: {value}")
    print()

    # Concurrent request handling
    print("3. Concurrent Request Handling:")
    concurrent_results = benchmark_concurrent_requests()
    for key, value in concurrent_results.items():
        print(f"   {key}: {value}")
    print()

    # Memory usage
    print("4. Memory Usage Patterns:")
    memory_results = benchmark_memory_usage()
    for key, value in memory_results.items():
        print(f"   {key}: {value}")
    print()

    # Service lifecycle
    print("5. Service Lifecycle:")
    lifecycle_results = asyncio.run(benchmark_service_lifecycle())
    for key, value in lifecycle_results.items():
        print(f"   {key}: {value:.4f}s")
    print()

    # Performance summary
    print("=== Performance Summary ===")
    print(f"- Sync embedding throughput (1000 texts): {sync_results['1000_texts']['texts_per_second']:.2f} texts/sec")
    print(f"- Async embedding throughput (1000 texts): {async_results['1000_texts']['texts_per_second']:.2f} texts/sec")
    print(f"- Concurrent handling (20 workers): {concurrent_results['20_workers']['requests_per_second']:.2f} req/sec")
    print(f"- Memory overhead (1000 texts): {memory_results['batch_1000']['peak_increase_mb']:.2f} MB")


if __name__ == "__main__":
    run_all_benchmarks()
