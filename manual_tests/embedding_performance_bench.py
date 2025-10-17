#!/usr/bin/env python3
"""
Manual performance benchmarks for the embedding service.

Relocated from pytest on 2025-10-16. Run this script to profile batch speeds,
concurrency, and memory usage with mock embeddings.
"""

from __future__ import annotations

import asyncio
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock

import psutil

from packages.shared.embedding import EmbeddingService, cleanup, get_embedding_service, initialize_embedding_service

# Mock metrics before importing embedding internals
sys.modules["packages.shared.metrics.prometheus"] = MagicMock()


def benchmark_sync_embeddings() -> dict[str, Any]:
    """Benchmark synchronous embedding generation."""
    service = EmbeddingService(mock_mode=True)
    results: dict[str, Any] = {}

    for count in (1, 10, 100, 1000):
        texts = [f"This is test text number {i} for benchmarking" for i in range(count)]
        service.generate_embeddings(texts[:1], "test-model", batch_size=32)
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
    service = await initialize_embedding_service("test-model", mock_mode=True)
    results: dict[str, Any] = {}

    for count in (1, 10, 100, 1000):
        texts = [f"This is test text number {i} for benchmarking" for i in range(count)]
        await service.embed_texts(texts[:1], batch_size=32)
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
    service = EmbeddingService(mock_mode=True)
    results: dict[str, Any] = {}

    def make_request(request_id: int) -> float:
        texts = [f"Request {request_id} text {i}" for i in range(10)]
        start = time.time()
        service.generate_embeddings(texts, "test-model")
        return time.time() - start

    for workers in (1, 5, 10, 20):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            start = time.time()
            durations = [future.result() for future in (executor.submit(make_request, i) for i in range(workers * 10))]
            total_duration = time.time() - start
            results[f"{workers}_workers"] = {
                "total_duration": total_duration,
                "average_request_duration": sum(durations) / len(durations),
                "requests_per_second": len(durations) / total_duration,
                "total_requests": len(durations),
            }

    return results


def benchmark_memory_usage() -> dict[str, Any]:
    """Benchmark memory usage patterns."""
    process = psutil.Process(os.getpid())
    service = EmbeddingService(mock_mode=True)
    results: dict[str, Any] = {}

    gc.collect()
    baseline_memory = process.memory_info().rss / 1024 / 1024

    for size in (10, 100, 1000):
        texts = [f"Text {i}" for i in range(size)]
        gc.collect()
        before_memory = process.memory_info().rss / 1024 / 1024
        embeddings = service.generate_embeddings(texts, "test-model", batch_size=32)
        after_memory = process.memory_info().rss / 1024 / 1024
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
    results: dict[str, float] = {}
    start = time.time()
    await initialize_embedding_service("test-model", mock_mode=True)
    results["initialization_time"] = time.time() - start

    start = time.time()
    for _ in range(100):
        await get_embedding_service()
    results["singleton_retrieval_time_per_100"] = time.time() - start

    start = time.time()
    await cleanup()
    results["cleanup_time"] = time.time() - start

    return results


def run_all_benchmarks() -> None:
    """Run all benchmarks and print formatted results."""
    print("=== Embedding Service Performance Benchmarks ===\n")

    print("1. Synchronous Embedding Generation:")
    sync_results = benchmark_sync_embeddings()
    for key, value in sync_results.items():
        print(f"   {key}: {value}")
    print()

    print("2. Asynchronous Embedding Generation:")
    async_results = asyncio.run(benchmark_async_embeddings())
    for key, value in async_results.items():
        print(f"   {key}: {value}")
    print()

    print("3. Concurrent Request Handling:")
    concurrent_results = benchmark_concurrent_requests()
    for key, value in concurrent_results.items():
        print(f"   {key}: {value}")
    print()

    print("4. Memory Usage Patterns:")
    memory_results = benchmark_memory_usage()
    for key, value in memory_results.items():
        print(f"   {key}: {value}")
    print()

    print("5. Service Lifecycle:")
    lifecycle_results = asyncio.run(benchmark_service_lifecycle())
    for key, value in lifecycle_results.items():
        print(f"   {key}: {value:.4f}s")
    print()

    print("=== Performance Summary ===")
    print(f"- Sync embedding throughput (1000 texts): {sync_results['1000_texts']['texts_per_second']:.2f} texts/sec")
    print(f"- Async embedding throughput (1000 texts): {async_results['1000_texts']['texts_per_second']:.2f} texts/sec")
    print(f"- Concurrent handling (20 workers): {concurrent_results['20_workers']['requests_per_second']:.2f} req/sec")
    print(f"- Memory overhead (1000 texts): {memory_results['batch_1000']['peak_increase_mb']:.2f} MB")


if __name__ == "__main__":
    run_all_benchmarks()
