#!/usr/bin/env python3
"""
Prometheus metrics for chunking error tracking.

This module provides comprehensive metrics for monitoring chunking errors,
recovery operations, and resource usage patterns.
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

from packages.shared.metrics.prometheus import registry as default_registry

_metrics_registry: CollectorRegistry = default_registry

chunking_errors_total: Counter
chunking_error_recovery_attempts: Counter
chunking_error_recovery_duration: Histogram
chunking_operations_active: Gauge
chunking_operations_failed: Gauge
chunking_operations_queued: Gauge
chunking_memory_usage_bytes: Histogram
chunking_cpu_usage_seconds: Histogram
chunking_document_processing_duration: Histogram
chunking_chunks_created_total: Counter
chunking_chunk_size_bytes: Histogram
chunking_retry_count: Counter
chunking_max_retries_exceeded: Counter
chunking_dead_letter_queue_size: Gauge
chunking_circuit_breaker_state: Gauge
chunking_circuit_breaker_trips: Counter
chunking_partial_failures_total: Counter
chunking_partial_failure_document_ratio: Histogram
chunking_cleanup_operations_total: Counter
chunking_cleanup_duration_seconds: Histogram
chunking_error_handler_info: Info


def init_metrics(registry_override: CollectorRegistry | None = None) -> None:
    """Initialize all chunking error metrics, optionally using an isolated registry."""
    global _metrics_registry
    global chunking_errors_total
    global chunking_error_recovery_attempts
    global chunking_error_recovery_duration
    global chunking_operations_active
    global chunking_operations_failed
    global chunking_operations_queued
    global chunking_memory_usage_bytes
    global chunking_cpu_usage_seconds
    global chunking_document_processing_duration
    global chunking_chunks_created_total
    global chunking_chunk_size_bytes
    global chunking_retry_count
    global chunking_max_retries_exceeded
    global chunking_dead_letter_queue_size
    global chunking_circuit_breaker_state
    global chunking_circuit_breaker_trips
    global chunking_partial_failures_total
    global chunking_partial_failure_document_ratio
    global chunking_cleanup_operations_total
    global chunking_cleanup_duration_seconds
    global chunking_error_handler_info

    _metrics_registry = registry_override or default_registry

    # Error tracking metrics
    chunking_errors_total = Counter(
        "chunking_errors_total",
        "Total number of chunking errors by type and strategy",
        ["error_type", "strategy", "recoverable"],
        registry=_metrics_registry,
    )

    chunking_error_recovery_attempts = Counter(
        "chunking_error_recovery_attempts_total",
        "Total number of error recovery attempts",
        ["error_type", "recovery_strategy", "success"],
        registry=_metrics_registry,
    )

    chunking_error_recovery_duration = Histogram(
        "chunking_error_recovery_duration_seconds",
        "Time taken to recover from errors",
        ["error_type", "recovery_strategy"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
        registry=_metrics_registry,
    )

    # Operation status metrics
    chunking_operations_active = Gauge(
        "chunking_operations_active",
        "Number of currently active chunking operations",
        ["strategy", "operation_type"],
        registry=_metrics_registry,
    )

    chunking_operations_failed = Gauge(
        "chunking_operations_failed",
        "Current number of failed operations pending retry",
        ["strategy", "error_type"],
        registry=_metrics_registry,
    )

    chunking_operations_queued = Gauge(
        "chunking_operations_queued",
        "Number of operations queued due to resource limits",
        ["resource_type"],
        registry=_metrics_registry,
    )

    # Resource usage metrics
    chunking_memory_usage_bytes = Histogram(
        "chunking_memory_usage_bytes",
        "Memory usage per chunking operation",
        ["strategy", "status"],
        buckets=[
            1_000_000,  # 1MB
            10_000_000,  # 10MB
            50_000_000,  # 50MB
            100_000_000,  # 100MB
            250_000_000,  # 250MB
            500_000_000,  # 500MB
            1_000_000_000,  # 1GB
        ],
        registry=_metrics_registry,
    )

    chunking_cpu_usage_seconds = Histogram(
        "chunking_cpu_usage_seconds",
        "CPU time used per chunking operation",
        ["strategy", "status"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
        registry=_metrics_registry,
    )

    chunking_document_processing_duration = Histogram(
        "chunking_document_processing_duration_seconds",
        "Time taken to process individual documents",
        ["strategy", "document_type", "status"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60],
        registry=_metrics_registry,
    )

    # Chunk metrics
    chunking_chunks_created_total = Counter(
        "chunking_chunks_created_total",
        "Total number of chunks created",
        ["strategy", "document_type"],
        registry=_metrics_registry,
    )

    chunking_chunk_size_bytes = Histogram(
        "chunking_chunk_size_bytes",
        "Size distribution of created chunks",
        ["strategy"],
        buckets=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000],
        registry=_metrics_registry,
    )

    # Retry and failure metrics
    chunking_retry_count = Counter(
        "chunking_retry_count_total",
        "Total number of retry attempts",
        ["operation_type", "retry_reason"],
        registry=_metrics_registry,
    )

    chunking_max_retries_exceeded = Counter(
        "chunking_max_retries_exceeded_total",
        "Number of operations that exceeded max retries",
        ["strategy", "final_error_type"],
        registry=_metrics_registry,
    )

    chunking_dead_letter_queue_size = Gauge(
        "chunking_dead_letter_queue_size",
        "Number of messages in the dead letter queue",
        ["queue_name"],
        registry=_metrics_registry,
    )

    # Circuit breaker metrics
    chunking_circuit_breaker_state = Gauge(
        "chunking_circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=open, 2=half-open)",
        ["service"],
        registry=_metrics_registry,
    )

    chunking_circuit_breaker_trips = Counter(
        "chunking_circuit_breaker_trips_total",
        "Number of times circuit breaker has tripped",
        ["service", "reason"],
        registry=_metrics_registry,
    )

    # Partial failure metrics
    chunking_partial_failures_total = Counter(
        "chunking_partial_failures_total",
        "Total number of partial failures",
        ["strategy"],
        registry=_metrics_registry,
    )

    chunking_partial_failure_document_ratio = Histogram(
        "chunking_partial_failure_document_ratio",
        "Ratio of failed documents in partial failures",
        ["strategy"],
        buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        registry=_metrics_registry,
    )

    # Cleanup metrics
    chunking_cleanup_operations_total = Counter(
        "chunking_cleanup_operations_total",
        "Total number of cleanup operations",
        ["cleanup_strategy", "success"],
        registry=_metrics_registry,
    )

    chunking_cleanup_duration_seconds = Histogram(
        "chunking_cleanup_duration_seconds",
        "Time taken for cleanup operations",
        ["cleanup_strategy"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
        registry=_metrics_registry,
    )

    # System health metrics
    chunking_error_handler_info = Info(
        "chunking_error_handler",
        "Information about the error handler configuration",
        registry=_metrics_registry,
    )


def get_metrics_registry() -> CollectorRegistry:
    """Expose the active registry to facilitate testing."""
    return _metrics_registry


init_metrics()


# Helper functions for metric updates


def record_chunking_error(
    error_type: str,
    strategy: str,
    recoverable: bool = True,
) -> None:
    """Record a chunking error occurrence.

    Args:
        error_type: Type of error (e.g., 'memory', 'timeout')
        strategy: Chunking strategy in use
        recoverable: Whether the error is recoverable
    """
    chunking_errors_total.labels(
        error_type=error_type,
        strategy=strategy,
        recoverable=str(recoverable),
    ).inc()


def record_recovery_attempt(
    error_type: str,
    recovery_strategy: str,
    success: bool,
    duration: float,
) -> None:
    """Record an error recovery attempt.

    Args:
        error_type: Type of error being recovered
        recovery_strategy: Strategy used for recovery
        success: Whether recovery was successful
        duration: Time taken for recovery
    """
    chunking_error_recovery_attempts.labels(
        error_type=error_type,
        recovery_strategy=recovery_strategy,
        success=str(success),
    ).inc()

    if success:
        chunking_error_recovery_duration.labels(
            error_type=error_type,
            recovery_strategy=recovery_strategy,
        ).observe(duration)


def record_resource_usage(
    strategy: str,
    status: str,
    memory_bytes: float,
    cpu_seconds: float,
) -> None:
    """Record resource usage for a chunking operation.

    Args:
        strategy: Chunking strategy used
        status: Operation status (e.g., 'success', 'failure')
        memory_bytes: Memory usage in bytes
        cpu_seconds: CPU time used
    """
    chunking_memory_usage_bytes.labels(strategy=strategy, status=status).observe(memory_bytes)
    chunking_cpu_usage_seconds.labels(strategy=strategy, status=status).observe(cpu_seconds)


def record_partial_failure(
    strategy: str,
    document_ratio: float,
    chunk_count: int,
) -> None:
    """Record a partial failure event.

    Args:
        strategy: Chunking strategy in use
        document_ratio: Ratio of failed documents
        chunk_count: Number of chunks created before failure
    """
    chunking_partial_failures_total.labels(strategy=strategy).inc()
    chunking_partial_failure_document_ratio.labels(strategy=strategy).observe(document_ratio)
    chunking_chunks_created_total.labels(strategy=strategy, document_type="partial_failure").inc(chunk_count)


def update_operation_status(
    strategy: str,
    operation_type: str,
    status: str,
    error_type: str | None = None,
) -> None:
    """Update gauges tracking operation status.

    Args:
        strategy: Chunking strategy in use
        operation_type: Type of operation (e.g., 'append', 'reindex')
        status: Current status ('active', 'failed', 'queued')
        error_type: Optional error classification
    """
    if status == "active":
        chunking_operations_active.labels(strategy=strategy, operation_type=operation_type).inc()
    elif status == "inactive":
        chunking_operations_active.labels(strategy=strategy, operation_type=operation_type).dec()
    elif status == "failed" and error_type:
        chunking_operations_failed.labels(strategy=strategy, error_type=error_type).inc()
    elif status == "queued":
        chunking_operations_queued.labels(resource_type=operation_type).inc()
    elif status == "dequeued":
        chunking_operations_queued.labels(resource_type=operation_type).dec()


def update_circuit_breaker_state(service: str, state: str) -> None:
    """Update circuit breaker state gauge.

    Args:
        service: Service name
        state: Circuit breaker state ('closed', 'open', 'half_open')
    """
    state_mapping = {"closed": 0, "open": 1, "half_open": 2}
    chunking_circuit_breaker_state.labels(service=service).set(state_mapping.get(state, 0))


def chunking_dead_letter_enqueue(queue_name: str) -> None:
    """Increment dead letter queue message count for a queue."""
    chunking_dead_letter_queue_size.labels(queue_name=queue_name).inc()


def chunking_dead_letter_dequeue(queue_name: str) -> None:
    """Decrement dead letter queue message count for a queue."""
    chunking_dead_letter_queue_size.labels(queue_name=queue_name).dec()


def chunking_cleanup_operation_executed(cleanup_strategy: str, success: bool, duration: float) -> None:
    """Record cleanup operation metrics."""
    chunking_cleanup_operations_total.labels(cleanup_strategy=cleanup_strategy, success=str(success)).inc()
    chunking_cleanup_duration_seconds.labels(cleanup_strategy=cleanup_strategy).observe(duration)


def chunking_retry_recorded(operation_type: str, retry_reason: str, exceeded: bool = False) -> None:
    """Record a retry attempt and optional max retry exceedance."""
    chunking_retry_count.labels(operation_type=operation_type, retry_reason=retry_reason).inc()
    if exceeded:
        chunking_max_retries_exceeded.labels(strategy=operation_type, final_error_type=retry_reason).inc()


def chunking_circuit_breaker_trip(service: str, reason: str) -> None:
    """Record a circuit breaker trip."""
    chunking_circuit_breaker_trips.labels(service=service, reason=reason).inc()


def chunking_error_handler_configured(**details: str) -> None:
    """Update info metric describing error handler configuration."""
    chunking_error_handler_info.info(details)
