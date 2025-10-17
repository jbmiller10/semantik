#!/usr/bin/env python3
"""
Prometheus metrics for document embedding pipeline
Provides observability into system performance
"""

import logging
import os
import threading
import time
from typing import Any

import GPUtil
import psutil
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info, start_http_server

logger = logging.getLogger(__name__)

# Re-export prometheus_client types for convenience
__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsCollector",
    "TimingContext",
    "registry",
    "system_info",
    "operations_created",
    "operations_completed",
    "operations_failed",
    "operation_duration",
    "files_processed",
    "files_failed",
    "chunks_created",
    "embeddings_generated",
    "queue_length",
    "processing_lag",
    "extraction_duration",
    "chunking_duration",
    "embedding_batch_duration",
    "ingestion_duration",
    "gpu_memory_used",
    "gpu_memory_total",
    "gpu_utilization",
    "cpu_utilization",
    "memory_utilization",
    "memory_used",
    "memory_total",
    "qdrant_points",
    "qdrant_upload_errors",
    "embedding_oom_errors_total",
    "embedding_batch_size_reductions_total",
    "embedding_current_batch_size",
    "gpu_memory_usage_bytes",
    "metrics_collector",
    "record_operation_started",
    "record_operation_completed",
    "record_operation_failed",
    "record_file_processed",
    "record_file_failed",
    "record_chunks_created",
    "record_embeddings_generated",
    "update_queue_length",
    "update_processing_lag",
    "record_oom_error",
    "record_batch_size_reduction",
    "update_current_batch_size",
    "update_gpu_memory_usage",
    "start_metrics_server",
]

# Create custom registry
registry = CollectorRegistry()

_METRICS_SERVER_STARTED = False

# System Info
system_info = Info("embedding_system", "Document embedding system information", registry=registry)
system_info.info({"version": "2.0", "pipeline": "tiktoken_parallel"})

# Operation Metrics
operations_created = Counter(
    "embedding_operations_created_total", "Total number of operations created", registry=registry
)
operations_completed = Counter(
    "embedding_operations_completed_total", "Total number of operations completed", registry=registry
)
operations_failed = Counter("embedding_operations_failed_total", "Total number of operations failed", registry=registry)
operation_duration = Histogram(
    "embedding_operation_duration_seconds",
    "Operation processing duration",
    buckets=(60, 300, 600, 1800, 3600, 7200),
    registry=registry,
)

# Pipeline Stage Metrics
files_processed = Counter("embedding_files_processed_total", "Total files processed", ["stage"], registry=registry)
files_failed = Counter("embedding_files_failed_total", "Total files failed", ["stage", "error_type"], registry=registry)
chunks_created = Counter("embedding_chunks_created_total", "Total chunks created", registry=registry)
embeddings_generated = Counter("embedding_vectors_generated_total", "Total embeddings generated", registry=registry)

# Queue Metrics
queue_length = Gauge("embedding_queue_length", "Current queue length", ["stage"], registry=registry)
processing_lag = Gauge("embedding_processing_lag_seconds", "Processing lag in seconds", ["stage"], registry=registry)

# Performance Metrics
extraction_duration = Histogram(
    "embedding_extraction_duration_seconds",
    "Text extraction duration",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30),
    registry=registry,
)
chunking_duration = Histogram(
    "embedding_chunking_duration_seconds",
    "Text chunking duration",
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2),
    registry=registry,
)
embedding_batch_duration = Histogram(
    "embedding_generation_duration_seconds",
    "Embedding generation duration",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30),
    registry=registry,
)
ingestion_duration = Histogram(
    "embedding_ingestion_duration_seconds",
    "Qdrant ingestion duration",
    buckets=(0.1, 0.5, 1, 2, 5, 10),
    registry=registry,
)

# Resource Metrics
gpu_memory_used = Gauge("embedding_gpu_memory_used_bytes", "GPU memory used", ["gpu_index"], registry=registry)
gpu_memory_total = Gauge("embedding_gpu_memory_total_bytes", "GPU memory total", ["gpu_index"], registry=registry)
gpu_utilization = Gauge(
    "embedding_gpu_utilization_percent", "GPU utilization percentage", ["gpu_index"], registry=registry
)
cpu_utilization = Gauge("embedding_cpu_utilization_percent", "CPU utilization percentage", registry=registry)
memory_utilization = Gauge("embedding_memory_utilization_percent", "Memory utilization percentage", registry=registry)
memory_used = Gauge("embedding_memory_used_bytes", "System memory used", registry=registry)
memory_total = Gauge("embedding_memory_total_bytes", "System memory total", registry=registry)

# Qdrant Metrics
qdrant_points = Gauge("embedding_qdrant_points_total", "Total points in Qdrant", ["collection"], registry=registry)
qdrant_upload_errors = Counter("embedding_qdrant_upload_errors_total", "Qdrant upload errors", registry=registry)

# Adaptive Batch Sizing Metrics
embedding_oom_errors_total = Counter(
    "embedding_oom_errors_total",
    "Total out of memory errors during embedding generation",
    ["model", "quantization"],
    registry=registry,
)
embedding_batch_size_reductions_total = Counter(
    "embedding_batch_size_reductions_total",
    "Total batch size reductions due to OOM errors",
    ["model", "quantization"],
    registry=registry,
)
embedding_current_batch_size = Gauge(
    "embedding_current_batch_size",
    "Current batch size being used for embeddings",
    ["model", "quantization"],
    registry=registry,
)
gpu_memory_usage_bytes = Gauge(
    "gpu_memory_usage_bytes",
    "Current GPU memory usage in bytes",
    registry=registry,
)


class MetricsCollector:
    """Collector for system and GPU metrics"""

    def __init__(self) -> None:
        self.last_update: float = 0.0
        self.update_interval = 1  # seconds - reduced for more responsive metrics
        self.lock = threading.Lock()
        self.first_cpu_call = True

    def update_resource_metrics(self, force: bool = False) -> None:
        """Update resource utilization metrics"""
        current_time = time.time()

        # Check if update is needed (without lock for performance)
        if not force and current_time - self.last_update < self.update_interval:
            return

        # Acquire lock for thread safety
        with self.lock:
            # Double-check after acquiring lock
            if not force and current_time - self.last_update < self.update_interval:
                return

            try:
                # CPU and Memory
                # Use interval=None for non-blocking, but handle first call
                if self.first_cpu_call:
                    # First call needs to establish baseline
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    self.first_cpu_call = False
                else:
                    # Subsequent calls use accumulated data (non-blocking)
                    cpu_percent = psutil.cpu_percent(interval=None)

                mem_info = psutil.virtual_memory()

                # Debug logging
                import logging

                logger = logging.getLogger(__name__)
                logger.debug(f"Updating metrics: CPU={cpu_percent}%, Memory={mem_info.percent}%")

                cpu_utilization.set(cpu_percent)
                memory_utilization.set(mem_info.percent)
                memory_used.set(mem_info.used)
                memory_total.set(mem_info.total)

                # GPU metrics (if available)
                try:
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        gpu_memory_used.labels(gpu_index=str(i)).set(
                            gpu.memoryUsed * 1024 * 1024 * 1024
                        )  # Convert to bytes
                        gpu_memory_total.labels(gpu_index=str(i)).set(
                            gpu.memoryTotal * 1024 * 1024 * 1024
                        )  # Convert to bytes
                        gpu_utilization.labels(gpu_index=str(i)).set(gpu.load * 100)
                        logger.debug(f"GPU {i}: Load={gpu.load * 100}%, Memory Used={gpu.memoryUsed}GB")
                except Exception as gpu_err:
                    logger.debug(f"No GPU available or error reading GPU: {gpu_err}")

                self.last_update = current_time
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f"Error updating resource metrics: {e}")


# Global metrics collector
metrics_collector = MetricsCollector()


# Context managers for timing
class TimingContext:
    """Context manager for timing operations"""

    def __init__(self, histogram: Histogram) -> None:
        self.histogram = histogram
        self.start_time: float | None = None

    def __enter__(self) -> "TimingContext":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time:
            duration = time.time() - self.start_time
            self.histogram.observe(duration)


# Helper functions
def record_operation_started() -> None:
    """Record that an operation has started"""
    operations_created.inc()


def record_operation_completed(duration_seconds: float) -> None:
    """Record that an operation has completed"""
    operations_completed.inc()
    operation_duration.observe(duration_seconds)


def record_operation_failed() -> None:
    """Record that an operation has failed"""
    operations_failed.inc()


def record_file_processed(stage: str) -> None:
    """Record successful file processing"""
    files_processed.labels(stage=stage).inc()


def record_file_failed(stage: str, error_type: str) -> None:
    """Record failed file processing"""
    files_failed.labels(stage=stage, error_type=error_type).inc()


def record_chunks_created(count: int) -> None:
    """Record number of chunks created"""
    chunks_created.inc(count)


def record_embeddings_generated(count: int) -> None:
    """Record number of embeddings generated"""
    embeddings_generated.inc(count)


def update_queue_length(stage: str, length: int) -> None:
    """Update queue length for a stage"""
    queue_length.labels(stage=stage).set(length)


def update_processing_lag(stage: str, lag_seconds: float) -> None:
    """Update processing lag for a stage"""
    processing_lag.labels(stage=stage).set(lag_seconds)


# Removed unused functions: update_qdrant_points and record_qdrant_error
# These were defined but never called in the codebase


def record_oom_error(model: str, quantization: str) -> None:
    """Record an out of memory error during embedding generation"""
    embedding_oom_errors_total.labels(model=model, quantization=quantization).inc()


def record_batch_size_reduction(model: str, quantization: str) -> None:
    """Record a batch size reduction due to OOM error"""
    embedding_batch_size_reductions_total.labels(model=model, quantization=quantization).inc()


def update_current_batch_size(model: str, quantization: str, size: int) -> None:
    """Update the current batch size being used for embeddings"""
    embedding_current_batch_size.labels(model=model, quantization=quantization).set(size)


def update_gpu_memory_usage(bytes_used: int) -> None:
    """Update the current GPU memory usage in bytes"""
    gpu_memory_usage_bytes.set(bytes_used)


def _should_skip_metrics_server() -> bool:
    """Determine whether metrics server startup should be skipped."""

    if os.getenv("PROMETHEUS_DISABLE_SERVER", "").lower() in {"1", "true", "yes"}:
        return True

    if os.getenv("TESTING", "").lower() in {"1", "true", "yes"}:
        return True

    return False


def start_metrics_server(port: int = 9090) -> None:
    """Start Prometheus metrics server unless disabled or already running."""

    global _METRICS_SERVER_STARTED

    if _METRICS_SERVER_STARTED:
        return

    if _should_skip_metrics_server():
        logger.info("Prometheus metrics server startup skipped for testing environment")
        _METRICS_SERVER_STARTED = True
        return

    try:
        start_http_server(port, registry=registry)
        logger.info("Metrics server started on port %s", port)
        _METRICS_SERVER_STARTED = True
    except OSError as exc:
        logger.warning("Failed to start metrics server on port %s: %s", port, exc)


if __name__ == "__main__":
    # Example usage
    start_metrics_server(9090)

    # Simulate some metrics
    import random

    while True:
        # Update resource metrics
        metrics_collector.update_resource_metrics()

        # Simulate operation metrics
        if random.random() < 0.1:
            record_operation_started()

            if random.random() < 0.9:
                record_operation_completed(random.uniform(300, 3600))
            else:
                record_operation_failed()

        # Simulate file processing
        if random.random() < 0.3:
            stage = random.choice(["extraction", "chunking", "embedding", "ingestion"])
            if random.random() < 0.95:
                record_file_processed(stage)
            else:
                record_file_failed(stage, random.choice(["io_error", "parse_error", "oom"]))

        # Simulate queue metrics
        for stage in ["extraction", "embedding", "ingestion"]:
            update_queue_length(stage, random.randint(0, 100))
            update_processing_lag(stage, random.uniform(0, 60))

        time.sleep(5)
