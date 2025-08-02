#!/usr/bin/env python3
"""
Performance benchmarks for chunking strategies.

This module provides comprehensive performance testing for all chunking
strategies, measuring speed, memory usage, and scalability.
"""

import asyncio
import json
import logging
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Any

import psutil
import pytest

from packages.shared.text_processing.chunking_factory import ChunkingFactory

logger = logging.getLogger(__name__)

# Constants
KB = 1024
MB = 1024 * KB


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""

    strategy: str
    document_size: str
    text_length: int
    chunks_created: int
    duration_seconds: float
    chunks_per_second: float
    memory_used_mb: float
    avg_chunk_size: float


class ChunkingBenchmarks:
    """Performance benchmarks for chunking strategies."""

    HARDWARE_BASELINE = {
        "cpu": "4 cores",
        "memory": "8GB",
        "description": "Standard container specs",
    }

    PERFORMANCE_TARGETS = {
        "character": {  # TokenTextSplitter
            "single_thread": 1000,  # chunks/sec
            "parallel_4": 3500,  # chunks/sec with 4 workers
            "memory_per_mb": 50,  # MB memory per MB document
        },
        "recursive": {  # SentenceSplitter
            "single_thread": 800,
            "parallel_4": 3000,
            "memory_per_mb": 60,
        },
        "markdown": {  # MarkdownNodeParser
            "single_thread": 600,
            "parallel_4": 2200,
            "memory_per_mb": 80,
        },
        "semantic": {  # SemanticSplitterNodeParser
            "single_thread": 150,  # Lower due to embeddings
            "parallel_4": 400,  # Limited by embedding model
            "memory_per_mb": 200,
        },
        "hierarchical": {  # HierarchicalNodeParser
            "single_thread": 400,  # Multiple passes
            "parallel_4": 1500,
            "memory_per_mb": 150,
        },
    }

    DOCUMENT_PROFILES = [
        {"name": "small", "size_bytes": 1 * KB, "size": "1KB", "chunks_expected": 2},
        {"name": "medium", "size_bytes": 100 * KB, "size": "100KB", "chunks_expected": 100},
        {"name": "large", "size_bytes": 10 * MB, "size": "10MB", "chunks_expected": 10000},
        {"name": "xlarge", "size_bytes": 100 * MB, "size": "100MB", "chunks_expected": 100000},
    ]

    @staticmethod
    def generate_test_document(size_bytes: int, doc_type: str = "text") -> str:
        """Generate test document of specified size.

        Args:
            size_bytes: Target size in bytes
            doc_type: Type of document (text, markdown, code)

        Returns:
            Generated document text
        """
        if doc_type == "text":
            # Generate sentences
            sentences = []
            words = [
                "the",
                "quick",
                "brown",
                "fox",
                "jumps",
                "over",
                "lazy",
                "dog",
                "and",
                "then",
                "runs",
                "away",
                "quickly",
                "through",
                "forest",
            ]

            current_size = 0
            while current_size < size_bytes:
                sentence = " ".join(random.choices(words, k=random.randint(5, 15)))
                sentence = sentence.capitalize() + ". "
                sentences.append(sentence)
                current_size += len(sentence)

            return "".join(sentences)[:size_bytes]

        elif doc_type == "markdown":
            # Generate markdown document
            sections = []
            current_size = 0
            section_num = 1

            while current_size < size_bytes:
                section = f"\n# Section {section_num}\n\n"
                section += "This is a paragraph in the section. " * random.randint(5, 10)
                section += "\n\n## Subsection {}.1\n\n".format(section_num)
                section += "More content here. " * random.randint(3, 8)
                section += "\n\n"

                sections.append(section)
                current_size += len(section)
                section_num += 1

            return "".join(sections)[:size_bytes]

        elif doc_type == "code":
            # Generate Python-like code
            code_lines = []
            current_size = 0

            code_lines.append("#!/usr/bin/env python3\n")
            code_lines.append('"""Generated test code."""\n\n')

            func_num = 1
            while current_size < size_bytes:
                func = f"def function_{func_num}(param1, param2):\n"
                func += '    """Function docstring."""\n'
                func += "    result = param1 + param2\n"
                func += "    for i in range(10):\n"
                func += "        result += i\n"
                func += "    return result\n\n"

                code_lines.append(func)
                current_size += len(func)
                func_num += 1

            return "".join(code_lines)[:size_bytes]

        else:
            # Default to text
            return ChunkingBenchmarks.generate_test_document(size_bytes, "text")


class PerformanceMonitor:
    """Monitor resource usage during tests."""

    def __init__(self) -> None:
        """Initialize the monitor."""
        self.process = psutil.Process()
        self.start_memory = 0
        self.peak_memory = 0

    def start(self) -> None:
        """Start monitoring."""
        self.start_memory = self.process.memory_info().rss / MB
        self.peak_memory = self.start_memory

    def update(self) -> None:
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss / MB
        self.peak_memory = max(self.peak_memory, current_memory)

    def stop(self) -> dict[str, float]:
        """Stop monitoring and return metrics."""
        final_memory = self.process.memory_info().rss / MB
        return {
            "start_memory_mb": self.start_memory,
            "peak_memory_mb": self.peak_memory,
            "final_memory_mb": final_memory,
            "memory_used_mb": self.peak_memory - self.start_memory,
        }


class TestChunkingPerformance:
    """Performance tests for chunking strategies."""

    @pytest.fixture
    def performance_monitor(self) -> PerformanceMonitor:
        """Provide performance monitor."""
        return PerformanceMonitor()

    @pytest.mark.parametrize(
        "strategy,document_size,expected_rate",
        [
            ("character", "1MB", 1000),
            ("recursive", "1MB", 800),
            ("markdown", "1MB", 600),
        ],
    )
    async def test_single_thread_performance(
        self,
        strategy: str,
        document_size: str,
        expected_rate: int,
        performance_monitor: PerformanceMonitor,
    ) -> None:
        """Test single-threaded chunking performance.

        Args:
            strategy: Chunking strategy to test
            document_size: Size of test document
            expected_rate: Expected chunks per second
            performance_monitor: Performance monitor fixture
        """
        # Generate test document
        size_bytes = 1 * MB  # 1MB for this test
        doc_type = "markdown" if strategy == "markdown" else "text"
        document = ChunkingBenchmarks.generate_test_document(size_bytes, doc_type)

        # Create chunker
        config = self.get_test_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)

        # Start monitoring
        performance_monitor.start()

        # Warm up
        await chunker.chunk_text_async(document[:1000], "warmup")

        # Measure chunking
        start_time = time.time()
        chunks = await chunker.chunk_text_async(document, "test_doc")
        duration = time.time() - start_time

        # Update monitoring
        performance_monitor.update()

        # Stop monitoring
        metrics = performance_monitor.stop()

        # Calculate rate
        chunks_per_second = len(chunks) / duration if duration > 0 else 0

        # Log results
        result = BenchmarkResult(
            strategy=strategy,
            document_size=document_size,
            text_length=len(document),
            chunks_created=len(chunks),
            duration_seconds=duration,
            chunks_per_second=chunks_per_second,
            memory_used_mb=metrics["memory_used_mb"],
            avg_chunk_size=sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0,
        )

        logger.info(f"Benchmark result: {result}")

        # Assertions
        assert (
            chunks_per_second >= expected_rate * 0.9
        ), f"{strategy} performance {chunks_per_second:.1f} below target {expected_rate}"

        assert metrics["memory_used_mb"] < 100, f"Memory usage {metrics['memory_used_mb']}MB exceeds limit"

    @pytest.mark.parametrize("num_workers", [2, 4, 8])
    async def test_parallel_performance(
        self,
        num_workers: int,
    ) -> None:
        """Test parallel chunking scalability.

        Args:
            num_workers: Number of parallel workers
        """
        # Generate test documents
        documents = [ChunkingBenchmarks.generate_test_document(100 * KB, "text") for _ in range(100)]

        # Test with recursive strategy
        config = {"strategy": "recursive", "params": {"chunk_size": 600}}
        chunker = ChunkingFactory.create_chunker(config)

        # Single worker baseline
        single_start = time.time()
        for doc in documents:
            await chunker.chunk_text_async(doc, "test")
        single_duration = time.time() - single_start

        # Parallel processing
        parallel_start = time.time()
        tasks = []

        # Process in batches to simulate parallel workers
        batch_size = len(documents) // num_workers
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            for doc in batch:
                task = chunker.chunk_text_async(doc, f"test_{i}")
                tasks.append(task)

        await asyncio.gather(*tasks)
        parallel_duration = time.time() - parallel_start

        # Calculate speedup
        speedup = single_duration / parallel_duration
        efficiency = speedup / num_workers

        logger.info(
            f"Parallel test with {num_workers} workers: " f"speedup={speedup:.2f}x, efficiency={efficiency:.2%}"
        )

        # Should achieve at least 70% efficiency
        assert efficiency >= 0.7, f"Parallel efficiency {efficiency:.2f} below threshold"

    @pytest.mark.parametrize("strategy", ["character", "recursive", "markdown"])
    async def test_memory_scaling(
        self,
        strategy: str,
        performance_monitor: PerformanceMonitor,
    ) -> None:
        """Test memory usage scaling with document size.

        Args:
            strategy: Chunking strategy to test
            performance_monitor: Performance monitor fixture
        """
        config = self.get_test_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)

        memory_results = []

        for profile in ChunkingBenchmarks.DOCUMENT_PROFILES[:3]:  # Skip xlarge for memory tests
            # Generate document
            doc_type = "markdown" if strategy == "markdown" else "text"
            document = ChunkingBenchmarks.generate_test_document(
                profile["size_bytes"],
                doc_type,
            )

            # Monitor memory
            performance_monitor.start()

            # Process document
            chunks = await chunker.chunk_text_async(document, f"test_{profile['name']}")

            performance_monitor.update()
            metrics = performance_monitor.stop()

            # Record results
            memory_per_mb = metrics["memory_used_mb"] / (profile["size_bytes"] / MB)
            memory_results.append(
                {
                    "size": profile["size"],
                    "memory_used_mb": metrics["memory_used_mb"],
                    "memory_per_mb": memory_per_mb,
                    "chunks": len(chunks),
                }
            )

            # Check against targets
            target = ChunkingBenchmarks.PERFORMANCE_TARGETS[strategy]["memory_per_mb"]
            assert memory_per_mb <= target * 1.2, f"Memory usage {memory_per_mb:.1f}MB/MB exceeds target {target}MB/MB"

        logger.info(f"Memory scaling for {strategy}: {json.dumps(memory_results, indent=2)}")

    def get_test_config(self, strategy: str) -> dict[str, Any]:
        """Get test configuration for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            Configuration dictionary
        """
        configs = {
            "character": {
                "strategy": "character",
                "params": {"chunk_size": 1000, "chunk_overlap": 200},
            },
            "recursive": {
                "strategy": "recursive",
                "params": {"chunk_size": 600, "chunk_overlap": 100},
            },
            "markdown": {
                "strategy": "markdown",
                "params": {},
            },
        }

        config = configs.get(strategy, configs["recursive"])

        # Use mock embeddings for testing
        os.environ["TESTING"] = "true"

        return config


# Standalone benchmark runner
async def run_benchmarks() -> None:
    """Run all benchmarks and generate report."""
    benchmarks = ChunkingBenchmarks()
    results = []

    strategies = ["character", "recursive", "markdown"]

    for strategy in strategies:
        config = {"strategy": strategy, "params": {}}
        if strategy == "character":
            config["params"] = {"chunk_size": 1000, "chunk_overlap": 200}
        elif strategy == "recursive":
            config["params"] = {"chunk_size": 600, "chunk_overlap": 100}

        chunker = ChunkingFactory.create_chunker(config)

        for profile in ChunkingBenchmarks.DOCUMENT_PROFILES[:3]:
            # Generate document
            doc_type = "markdown" if strategy == "markdown" else "text"
            document = benchmarks.generate_test_document(profile["size_bytes"], doc_type)

            # Benchmark
            monitor = PerformanceMonitor()
            monitor.start()

            start_time = time.time()
            chunks = await chunker.chunk_text_async(document, f"bench_{profile['name']}")
            duration = time.time() - start_time

            monitor.update()
            metrics = monitor.stop()

            # Record result
            result = BenchmarkResult(
                strategy=strategy,
                document_size=profile["size"],
                text_length=len(document),
                chunks_created=len(chunks),
                duration_seconds=duration,
                chunks_per_second=len(chunks) / duration if duration > 0 else 0,
                memory_used_mb=metrics["memory_used_mb"],
                avg_chunk_size=sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0,
            )

            results.append(result)
            print(
                f"{strategy} - {profile['size']}: {result.chunks_per_second:.1f} chunks/sec, "
                f"{result.memory_used_mb:.1f}MB memory"
            )

    # Generate summary report
    print("\n=== Chunking Performance Benchmark Report ===")
    print(f"Hardware: {ChunkingBenchmarks.HARDWARE_BASELINE}")
    print("\nResults by Strategy:")

    for strategy in strategies:
        strategy_results = [r for r in results if r.strategy == strategy]
        avg_rate = sum(r.chunks_per_second for r in strategy_results) / len(strategy_results)
        avg_memory = sum(r.memory_used_mb for r in strategy_results) / len(strategy_results)

        print(f"\n{strategy.capitalize()}:")
        print(f"  Average rate: {avg_rate:.1f} chunks/sec")
        print(f"  Average memory: {avg_memory:.1f}MB")
        print(f"  Target rate: {ChunkingBenchmarks.PERFORMANCE_TARGETS[strategy]['single_thread']} chunks/sec")


if __name__ == "__main__":
    # Run benchmarks when executed directly
    asyncio.run(run_benchmarks())
