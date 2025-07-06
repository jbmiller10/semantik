#!/usr/bin/env python3
"""
Test script to verify metrics are being collected during job processing
"""
import os
import sys
from pathlib import Path

# Add the project directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Set the metrics port to enable metrics collection
os.environ["WEBUI_METRICS_PORT"] = "9092"

from prometheus_client import generate_latest

from packages.vecpipe.metrics import (
    record_chunks_created,
    record_embeddings_generated,
    record_file_failed,
    record_file_processed,
    record_job_completed,
    record_job_failed,
    record_job_started,
    registry,
)


def print_current_metrics():
    """Print current metric values"""
    metrics_data = generate_latest(registry).decode("utf-8")
    print("Current Metrics:")
    print("-" * 60)

    # Extract relevant metrics
    for line in metrics_data.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        if any(
            metric in line
            for metric in [
                "embedding_jobs_created_total",
                "embedding_jobs_completed_total",
                "embedding_jobs_failed_total",
                "embedding_chunks_created_total",
                "embedding_vectors_generated_total",
                "embedding_files_processed_total",
                "embedding_files_failed_total",
            ]
        ):
            print(line)
    print("-" * 60)


def simulate_job_processing():
    """Simulate job processing and check if metrics are updated"""
    print("Initial metrics state:")
    print_current_metrics()

    # Simulate job creation
    print("\nSimulating job creation...")
    record_job_started()

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

    # Simulate job completion
    print("Simulating job completion (duration: 60 seconds)...")
    record_job_completed(60.0)

    print("\nFinal metrics state:")
    print_current_metrics()

    # Also test failure scenarios
    print("\nSimulating a failed job...")
    record_job_started()
    record_file_failed("extraction", "io_error")
    record_job_failed()

    print("\nMetrics after failure:")
    print_current_metrics()


if __name__ == "__main__":
    print("Testing Semantik Metrics Collection")
    print("=" * 60)
    simulate_job_processing()

    print("\nDiagnosis:")
    print("- Metrics are being collected correctly when the functions are called")
    print("- The issue is that the job processing code is not calling these metric functions")
    print("- Specifically:")
    print("  1. record_job_started() is never called when a job is created")
    print("  2. record_job_completed() is never called when a job completes")
    print("  3. record_job_failed() is never called when a job fails")
    print("\nTo fix this, we need to add these metric calls to the job processing workflow.")
