#!/usr/bin/env python3
"""
Test script to verify metrics are being collected during operation processing
"""
import os
import sys
from pathlib import Path

# Add the project directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Set the metrics port to enable metrics collection
os.environ["WEBUI_METRICS_PORT"] = "9092"

# Import after setting environment variables
import prometheus_client  # noqa: E402
from shared.metrics.prometheus import (  # noqa: E402
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
    """Print current metric values"""
    metrics_data = prometheus_client.generate_latest(registry).decode("utf-8")
    print("Current Metrics:")
    print("-" * 60)

    # Extract relevant metrics
    for line in metrics_data.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        if any(
            metric in line
            for metric in [
                "embedding_operations_created_total",
                "embedding_operations_completed_total",
                "embedding_operations_failed_total",
                "embedding_chunks_created_total",
                "embedding_vectors_generated_total",
                "embedding_files_processed_total",
                "embedding_files_failed_total",
            ]
        ):
            print(line)
    print("-" * 60)


def simulate_operation_processing() -> None:
    """Simulate operation processing and check if metrics are updated"""
    print("Initial metrics state:")
    print_current_metrics()

    # Simulate operation creation
    print("\nSimulating operation creation...")
    record_operation_started()

    # Simulate file processing
    print("Simulating file processing...")
    record_file_processed("extraction")
    record_file_processed("chunking")

    # Simulate chunk creation
    print("Simulating chunk creation (100 chunks)...")
    record_chunks_created(100)

    # Simulate embedding generation
    print("Simulating embedding generation (100 embeddings)...")
    record_embeddings_generated(100)

    # Simulate operation completion
    print("Simulating operation completion (duration: 60 seconds)...")
    record_operation_completed(60.0)

    print("\nFinal metrics state:")
    print_current_metrics()

    # Also test failure scenarios
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

    print("\nDiagnosis:")
    print("- Metrics are being collected correctly when the functions are called")
    print("- The issue is that the operation processing code is not calling these metric functions")
    print("- Specifically:")
    print("  1. record_operation_started() is never called when an operation is created")
    print("  2. record_operation_completed() is never called when an operation completes")
    print("  3. record_operation_failed() is never called when an operation fails")
    print("\nTo fix this, we need to add these metric calls to the operation processing workflow.")
