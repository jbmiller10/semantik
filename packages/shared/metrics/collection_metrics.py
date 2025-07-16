#!/usr/bin/env python3
"""
Prometheus metrics for collection operations.
Provides comprehensive monitoring for collection management and operations.
"""

from prometheus_client import Counter, Gauge, Histogram

# Import the existing registry to ensure all metrics are registered in one place
from .prometheus import registry

# Collection operation metrics
collection_operations_total = Counter(
    "semantik_collection_operations_total",
    "Total collection operations",
    ["operation_type", "status"],
    registry=registry,
)

collection_operation_duration = Histogram(
    "semantik_collection_operation_duration_seconds",
    "Collection operation duration",
    ["operation_type"],
    buckets=(1, 5, 10, 30, 60, 300, 600, 1800, 3600, 7200),
    registry=registry,
)

# Collection status metrics
collections_total = Gauge(
    "semantik_collections_total",
    "Total number of collections",
    ["status"],
    registry=registry,
)

collection_documents_total = Gauge(
    "semantik_collection_documents_total",
    "Total documents per collection",
    ["collection_id"],
    registry=registry,
)

collection_vectors_total = Gauge(
    "semantik_collection_vectors_total",
    "Total vectors per collection",
    ["collection_id"],
    registry=registry,
)

collection_size_bytes = Gauge(
    "semantik_collection_size_bytes",
    "Total size of collection in bytes",
    ["collection_id"],
    registry=registry,
)

# Reindex-specific metrics
reindex_checkpoints = Counter(
    "semantik_reindex_checkpoints_total",
    "Reindex checkpoints reached",
    ["collection_id", "checkpoint"],
    registry=registry,
)

reindex_validation_duration = Histogram(
    "semantik_reindex_validation_duration_seconds",
    "Time spent validating reindex results",
    buckets=(0.5, 1, 2, 5, 10, 30, 60),
    registry=registry,
)

reindex_switch_duration = Histogram(
    "semantik_reindex_switch_duration_seconds",
    "Time to perform atomic collection switch",
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5),
    registry=registry,
)

# Resource usage metrics
collection_memory_usage_bytes = Gauge(
    "semantik_collection_memory_usage_bytes",
    "Memory usage during collection operations",
    ["operation_type"],
    registry=registry,
)

collection_cpu_seconds_total = Counter(
    "semantik_collection_cpu_seconds_total",
    "Total CPU time used by collection operations",
    ["operation_type"],
    registry=registry,
)

# Error and retry metrics
collection_operation_errors_total = Counter(
    "semantik_collection_operation_errors_total",
    "Total collection operation errors",
    ["operation_type", "error_type"],
    registry=registry,
)

collection_operation_retries_total = Counter(
    "semantik_collection_operation_retries_total",
    "Total collection operation retries",
    ["operation_type"],
    registry=registry,
)

# Search performance metrics
search_latency = Histogram(
    "semantik_search_latency_seconds",
    "Search request latency",
    ["collection_count"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
    registry=registry,
)

search_results_total = Histogram(
    "semantik_search_results_total",
    "Number of search results returned",
    buckets=(0, 1, 5, 10, 25, 50, 100, 250, 500, 1000),
    registry=registry,
)

# Document processing metrics
documents_processed_total = Counter(
    "semantik_documents_processed_total",
    "Total documents processed",
    ["operation_type", "status"],
    registry=registry,
)

document_processing_duration = Histogram(
    "semantik_document_processing_duration_seconds",
    "Document processing duration",
    ["operation_type"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60),
    registry=registry,
)

# Queue metrics for operations
operation_queue_size = Gauge(
    "semantik_operation_queue_size",
    "Number of operations in queue",
    ["operation_type"],
    registry=registry,
)

operation_queue_wait_time = Histogram(
    "semantik_operation_queue_wait_time_seconds",
    "Time operations spend in queue",
    ["operation_type"],
    buckets=(0, 1, 5, 10, 30, 60, 300, 600, 1800),
    registry=registry,
)

# Qdrant-specific metrics
qdrant_collection_operations_total = Counter(
    "semantik_qdrant_collection_operations_total",
    "Total Qdrant collection operations",
    ["operation", "status"],
    registry=registry,
)

qdrant_points_upserted_total = Counter(
    "semantik_qdrant_points_upserted_total",
    "Total points upserted to Qdrant",
    ["collection_name"],
    registry=registry,
)

qdrant_operation_duration = Histogram(
    "semantik_qdrant_operation_duration_seconds",
    "Qdrant operation duration",
    ["operation"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10),
    registry=registry,
)


# Helper functions for metric recording
def record_operation_started(operation_type: str) -> None:
    """Record that a collection operation has started."""
    collection_operations_total.labels(operation_type=operation_type, status="started").inc()


def record_operation_completed(operation_type: str, duration_seconds: float) -> None:
    """Record that a collection operation has completed successfully."""
    collection_operations_total.labels(operation_type=operation_type, status="completed").inc()
    collection_operation_duration.labels(operation_type=operation_type).observe(duration_seconds)


def record_operation_failed(operation_type: str, error_type: str) -> None:
    """Record that a collection operation has failed."""
    collection_operations_total.labels(operation_type=operation_type, status="failed").inc()
    collection_operation_errors_total.labels(operation_type=operation_type, error_type=error_type).inc()


def record_operation_retry(operation_type: str) -> None:
    """Record that a collection operation is being retried."""
    collection_operation_retries_total.labels(operation_type=operation_type).inc()


def update_collection_stats(collection_id: str, documents: int, vectors: int, size_bytes: int) -> None:
    """Update collection statistics."""
    collection_documents_total.labels(collection_id=collection_id).set(documents)
    collection_vectors_total.labels(collection_id=collection_id).set(vectors)
    collection_size_bytes.labels(collection_id=collection_id).set(size_bytes)


def record_reindex_checkpoint(collection_id: str, checkpoint: str) -> None:
    """Record reaching a reindex checkpoint."""
    reindex_checkpoints.labels(collection_id=collection_id, checkpoint=checkpoint).inc()


def record_document_processed(operation_type: str, status: str) -> None:
    """Record document processing completion."""
    documents_processed_total.labels(operation_type=operation_type, status=status).inc()


def update_operation_queue_size(operation_type: str, size: int) -> None:
    """Update the operation queue size."""
    operation_queue_size.labels(operation_type=operation_type).set(size)


def record_qdrant_operation(operation: str, status: str, duration: float | None = None) -> None:
    """Record a Qdrant operation."""
    qdrant_collection_operations_total.labels(operation=operation, status=status).inc()
    if duration is not None:
        qdrant_operation_duration.labels(operation=operation).observe(duration)


def record_qdrant_points_upserted(collection_name: str, count: int) -> None:
    """Record points upserted to Qdrant."""
    qdrant_points_upserted_total.labels(collection_name=collection_name).inc(count)


# Context managers for timing operations
class OperationTimer:
    """Context manager for timing collection operations."""

    def __init__(self, operation_type: str):
        self.operation_type = operation_type
        self.start_time: float | None = None

    def __enter__(self) -> "OperationTimer":
        import time

        self.start_time = time.time()
        record_operation_started(self.operation_type)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        import time

        if self.start_time:
            duration = time.time() - self.start_time
            if exc_type is None:
                record_operation_completed(self.operation_type, duration)
            else:
                error_type = exc_type.__name__ if exc_type else "Unknown"
                record_operation_failed(self.operation_type, error_type)


class QdrantOperationTimer:
    """Context manager for timing Qdrant operations."""

    def __init__(self, operation: str):
        self.operation = operation
        self.start_time: float | None = None

    def __enter__(self) -> "QdrantOperationTimer":
        import time

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        import time

        if self.start_time:
            duration = time.time() - self.start_time
            status = "success" if exc_type is None else "failed"
            record_qdrant_operation(self.operation, status, duration)