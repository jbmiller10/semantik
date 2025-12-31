#!/usr/bin/env python3
"""
Prometheus metrics for chunking error tracking.

This module provides comprehensive metrics for monitoring chunking errors,
recovery operations, and resource usage patterns.
"""

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

from shared.metrics.prometheus import registry

MetricT = TypeVar("MetricT")


def _get_or_create_metric(
    metric_cls: type[MetricT],
    name: str,
    documentation: str,
    *,
    registry: CollectorRegistry,
    labelnames: list[str] | tuple[str, ...] | None = None,
    **kwargs: Any,
) -> MetricT:
    """Return existing metric from registry or create a new one."""
    existing: MetricT | None = None
    if hasattr(registry, "_names_to_collectors"):
        names_to_collectors = getattr(registry, "_names_to_collectors", None)
        if isinstance(names_to_collectors, Mapping):
            existing = cast(MetricT | None, names_to_collectors.get(name))

    if existing is not None:
        return existing

    if metric_cls is Info:
        return metric_cls(name, documentation, registry=registry, **kwargs)

    labels = tuple(labelnames or ())
    return metric_cls(name, documentation, labelnames=labels, registry=registry, **kwargs)


# Error tracking metrics
chunking_error_events_total = _get_or_create_metric(
    Counter,
    "chunking_error_events_total",
    "Total number of chunking errors by type and strategy",
    registry=registry,
    labelnames=["error_type", "strategy", "recoverable"],
)

chunking_error_recovery_attempts = _get_or_create_metric(
    Counter,
    "chunking_error_recovery_attempts_total",
    "Total number of error recovery attempts",
    registry=registry,
    labelnames=["error_type", "recovery_strategy", "success"],
)

chunking_error_recovery_duration = _get_or_create_metric(
    Histogram,
    "chunking_error_recovery_duration_seconds",
    "Time taken to recover from errors",
    registry=registry,
    labelnames=["error_type", "recovery_strategy"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
)

# Operation status metrics
chunking_operations_active = _get_or_create_metric(
    Gauge,
    "chunking_operations_active",
    "Number of currently active chunking operations",
    registry=registry,
    labelnames=["strategy", "operation_type"],
)

chunking_operations_failed = _get_or_create_metric(
    Gauge,
    "chunking_operations_failed",
    "Current number of failed operations pending retry",
    registry=registry,
    labelnames=["strategy", "error_type"],
)

chunking_operations_queued = _get_or_create_metric(
    Gauge,
    "chunking_operations_queued",
    "Number of operations queued due to resource limits",
    registry=registry,
    labelnames=["resource_type"],
)

# Resource usage metrics
chunking_memory_usage_bytes = _get_or_create_metric(
    Histogram,
    "chunking_memory_usage_bytes",
    "Memory usage per chunking operation",
    registry=registry,
    labelnames=["strategy", "status"],
    buckets=[
        1_000_000,  # 1MB
        10_000_000,  # 10MB
        50_000_000,  # 50MB
        100_000_000,  # 100MB
        250_000_000,  # 250MB
        500_000_000,  # 500MB
        1_000_000_000,  # 1GB
    ],
)

chunking_cpu_usage_seconds = _get_or_create_metric(
    Histogram,
    "chunking_cpu_usage_seconds",
    "CPU time used per chunking operation",
    registry=registry,
    labelnames=["strategy", "status"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
)

chunking_document_processing_duration = _get_or_create_metric(
    Histogram,
    "chunking_document_processing_duration_seconds",
    "Time taken to process individual documents",
    registry=registry,
    labelnames=["strategy", "document_type", "status"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

# Chunk metrics
chunking_chunks_created_total = _get_or_create_metric(
    Counter,
    "chunking_chunks_created_total",
    "Total number of chunks created",
    registry=registry,
    labelnames=["strategy", "document_type"],
)

chunking_chunk_size_bytes = _get_or_create_metric(
    Histogram,
    "chunking_chunk_size_bytes",
    "Size distribution of created chunks",
    registry=registry,
    labelnames=["strategy"],
    buckets=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000],
)

# Retry and failure metrics
chunking_retry_count = _get_or_create_metric(
    Counter,
    "chunking_retry_count_total",
    "Total number of retry attempts",
    registry=registry,
    labelnames=["operation_type", "retry_reason"],
)

chunking_max_retries_exceeded = _get_or_create_metric(
    Counter,
    "chunking_max_retries_exceeded_total",
    "Number of operations that exceeded max retries",
    registry=registry,
    labelnames=["strategy", "final_error_type"],
)

chunking_dead_letter_queue_size = _get_or_create_metric(
    Gauge,
    "chunking_dead_letter_queue_size",
    "Number of messages in the dead letter queue",
    registry=registry,
    labelnames=["queue_name"],
)

# Circuit breaker metrics
chunking_circuit_breaker_state = _get_or_create_metric(
    Gauge,
    "chunking_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    registry=registry,
    labelnames=["service"],
)

chunking_circuit_breaker_trips = _get_or_create_metric(
    Counter,
    "chunking_circuit_breaker_trips_total",
    "Number of times circuit breaker has tripped",
    registry=registry,
    labelnames=["service", "reason"],
)

# Partial failure metrics
chunking_partial_failures_total = _get_or_create_metric(
    Counter,
    "chunking_partial_failures_total",
    "Total number of partial failures",
    registry=registry,
    labelnames=["strategy"],
)

chunking_partial_failure_document_ratio = _get_or_create_metric(
    Histogram,
    "chunking_partial_failure_document_ratio",
    "Ratio of failed documents in partial failures",
    registry=registry,
    labelnames=["strategy"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
)

# Cleanup metrics
chunking_cleanup_operations_total = _get_or_create_metric(
    Counter,
    "chunking_cleanup_operations_total",
    "Total number of cleanup operations",
    registry=registry,
    labelnames=["cleanup_strategy", "success"],
)

chunking_cleanup_duration_seconds = _get_or_create_metric(
    Histogram,
    "chunking_cleanup_duration_seconds",
    "Time taken for cleanup operations",
    registry=registry,
    labelnames=["cleanup_strategy"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

# System health metrics
chunking_error_handler_info = _get_or_create_metric(
    Info,
    "chunking_error_handler",
    "Information about the error handler configuration",
    registry=registry,
)

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
    chunking_error_events_total.labels(
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


def update_operation_status(
    strategy: str,
    operation_type: str,
    active_delta: int = 0,
    failed_delta: int = 0,
    error_type: str | None = None,
) -> None:
    """Update operation status gauges.

    Args:
        strategy: Chunking strategy
        operation_type: Type of operation
        active_delta: Change in active operations (+1 or -1)
        failed_delta: Change in failed operations
        error_type: Type of error for failed operations
    """
    if active_delta:
        chunking_operations_active.labels(
            strategy=strategy,
            operation_type=operation_type,
        ).inc(active_delta)

    if failed_delta and error_type:
        chunking_operations_failed.labels(
            strategy=strategy,
            error_type=error_type,
        ).inc(failed_delta)


def record_resource_usage(
    strategy: str,
    status: str,
    memory_bytes: int,
    cpu_seconds: float,
) -> None:
    """Record resource usage for an operation.

    Args:
        strategy: Chunking strategy used
        status: Operation status (success/failure)
        memory_bytes: Memory used in bytes
        cpu_seconds: CPU time in seconds
    """
    chunking_memory_usage_bytes.labels(
        strategy=strategy,
        status=status,
    ).observe(memory_bytes)

    chunking_cpu_usage_seconds.labels(
        strategy=strategy,
        status=status,
    ).observe(cpu_seconds)


def update_circuit_breaker_state(
    service: str,
    state: int,
    trip_reason: str | None = None,
) -> None:
    """Update circuit breaker metrics.

    Args:
        service: Service name
        state: Circuit breaker state (0=closed, 1=open, 2=half-open)
        trip_reason: Reason for tripping (if applicable)
    """
    chunking_circuit_breaker_state.labels(service=service).set(state)

    if state == 1 and trip_reason:  # Circuit opened
        chunking_circuit_breaker_trips.labels(
            service=service,
            reason=trip_reason,
        ).inc()


def record_partial_failure(
    strategy: str,
    total_documents: int,
    failed_documents: int,
) -> None:
    """Record a partial failure event.

    Args:
        strategy: Chunking strategy
        total_documents: Total documents processed
        failed_documents: Number of failed documents
    """
    chunking_partial_failures_total.labels(strategy=strategy).inc()

    if total_documents > 0:
        failure_ratio = failed_documents / total_documents
        chunking_partial_failure_document_ratio.labels(strategy=strategy).observe(failure_ratio)


# Initialize error handler info
chunking_error_handler_info.info(
    {
        "version": "1.0.0",
        "max_retries": "3",
        "retry_backoff": "exponential",
        "circuit_breaker_enabled": "true",
        "dead_letter_queue_enabled": "true",
    }
)
