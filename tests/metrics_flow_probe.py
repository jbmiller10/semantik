#!/usr/bin/env python3
"""
Manual harness for exercising Prometheus metric helpers end-to-end.

Run this script to simulate pipeline operations and inspect the resulting
metric counters. It intentionally mirrors the legacy pytest harness that used
to live under tests/test_metrics_flow.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import prometheus_client  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("WEBUI_METRICS_PORT", "9092")

from packages.shared.metrics.prometheus import (  # noqa: E402
    record_chunks_created,
    record_embeddings_generated,
    record_file_failed,
    record_file_processed,
    record_operation_completed,
    record_operation_failed,
    record_operation_started,
    registry,
)


def print_current_metrics() -> None:
    """Print a filtered snapshot of current Prometheus metrics."""
    metrics_data = prometheus_client.generate_latest(registry).decode("utf-8")
    print("Current Metrics:")
    print("-" * 60)

    interesting_prefixes = (
        "embedding_operations_created_total",
        "embedding_operations_completed_total",
        "embedding_operations_failed_total",
        "embedding_chunks_created_total",
        "embedding_vectors_generated_total",
        "embedding_files_processed_total",
        "embedding_files_failed_total",
    )

    for line in metrics_data.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith(interesting_prefixes):
            print(line)

    print("-" * 60)


def simulate_operation_processing() -> None:
    """Simulate operation processing and print metric deltas."""
    print("Initial metrics state:")
    print_current_metrics()

    print("\nSimulating operation creation...")
    record_operation_started()

    print("Simulating file processing...")
    record_file_processed("extraction")
    record_file_processed("chunking")

    print("Simulating chunk creation (100 chunks)...")
    record_chunks_created(100)

    print("Simulating embedding generation (100 embeddings)...")
    record_embeddings_generated(100)

    print("Simulating operation completion (duration: 60 seconds)...")
    record_operation_completed(60.0)

    print("\nFinal metrics state:")
    print_current_metrics()

    print("\nSimulating a failed operation...")
    record_operation_started()
    record_file_failed("extraction", "io_error")
    record_operation_failed()

    print("\nMetrics after failure:")
    print_current_metrics()


if __name__ == "__main__":
    print("Testing Semantik Metrics Collection")
    print("=" * 60)
    simulate_operation_processing()
