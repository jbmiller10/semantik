#!/usr/bin/env python3
"""
Generate performance baseline for chunking strategies.

This script runs the performance tests and saves baseline results
for future regression detection.
"""

import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set testing environment
os.environ["TESTING"] = "true"

# Import after setting up path
from packages.shared.text_processing.chunking_factory import ChunkingFactory  # noqa: E402
from tests.performance.chunking_benchmarks import ChunkingBenchmarks, PerformanceMonitor  # noqa: E402
from tests.performance.performance_utils import (  # noqa: E402
    PerformanceBaseline,
    PerformanceBaselineManager,
    get_git_commit_hash,
    get_hardware_info,
)


async def generate_baseline() -> None:
    """Generate performance baseline for all strategies."""
    strategies = ["character", "recursive", "markdown"]
    baselines = []

    # Get system info
    git_commit = get_git_commit_hash()
    hardware_info = get_hardware_info()

    print("Generating performance baselines...")
    print(f"Git commit: {git_commit}")
    print(f"Hardware: {hardware_info}")
    print()

    for strategy in strategies:
        print(f"Testing {strategy} strategy...")

        # Generate 1MB test document
        doc_type = "markdown" if strategy == "markdown" else "text"
        document = ChunkingBenchmarks.generate_test_document(1024 * 1024, doc_type)

        # Create chunker
        config = {
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
        }[strategy]

        chunker = ChunkingFactory.create_chunker(config)

        # Warmup
        await chunker.chunk_text_async(document[:1000], "warmup")

        # Performance monitor
        monitor = PerformanceMonitor(sample_interval=0.05)

        # Run multiple iterations
        iterations = 5
        chunk_counts = []
        durations = []

        for i in range(iterations):
            monitor.start()

            start_time = asyncio.get_event_loop().time()
            chunks = await chunker.chunk_text_async(document, f"test_{i}")
            duration = asyncio.get_event_loop().time() - start_time

            metrics = monitor.stop()

            chunk_counts.append(len(chunks))
            durations.append(duration)

        # Calculate averages
        avg_chunks = sum(chunk_counts) / len(chunk_counts)
        avg_duration = sum(durations) / len(durations)
        chunks_per_second = avg_chunks / avg_duration

        # Memory metrics (from last run)
        memory_per_mb = metrics["memory_used_mb"] / 1.0  # 1MB document
        cpu_avg = metrics["cpu_stats"]["avg"]

        # Create baseline
        baseline = PerformanceBaseline(
            strategy=strategy,
            chunks_per_second=chunks_per_second,
            memory_per_mb=memory_per_mb,
            cpu_avg_percent=cpu_avg,
            test_date=datetime.now(UTC).isoformat(),
            git_commit=git_commit,
            hardware_info=hardware_info,
        )

        baselines.append(baseline)

        print(f"  Chunks per second: {chunks_per_second:.1f}")
        print(f"  Memory per MB: {memory_per_mb:.1f} MB")
        print(f"  Average CPU: {cpu_avg:.1f}%")
        print()

    # Save baselines
    manager = PerformanceBaselineManager()
    manager.save_baseline(baselines)

    print(f"Baselines saved to: {manager.baseline_file}")


if __name__ == "__main__":
    asyncio.run(generate_baseline())
