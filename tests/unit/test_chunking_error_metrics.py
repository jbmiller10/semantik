#!/usr/bin/env python3
"""Focused unit tests for chunking error metrics helpers."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from packages.webui.services import chunking_error_metrics as metrics


@pytest.fixture()
def isolated_registry() -> CollectorRegistry:
    """Provide a fresh Prometheus registry for each test."""
    registry = CollectorRegistry()
    metrics.init_metrics(registry)
    yield registry
    metrics.init_metrics()  # Restore default registry after test execution


def test_record_chunking_error_increments_counter(isolated_registry: CollectorRegistry) -> None:
    """record_chunking_error should increment the counter with normalized labels."""
    metrics.record_chunking_error(error_type="memory", strategy="semantic", recoverable=False)

    value = isolated_registry.get_sample_value(
        "chunking_errors_total",
        {"error_type": "memory", "strategy": "semantic", "recoverable": "False"},
    )
    assert value == 1.0


def test_record_recovery_attempt_tracks_attempts_and_duration(isolated_registry: CollectorRegistry) -> None:
    """record_recovery_attempt should increment attempts counter and histogram when successful."""
    metrics.record_recovery_attempt(
        error_type="timeout",
        recovery_strategy="retry",
        success=True,
        duration=2.5,
    )

    attempts_count = isolated_registry.get_sample_value(
        "chunking_error_recovery_attempts_total",
        {"error_type": "timeout", "recovery_strategy": "retry", "success": "True"},
    )
    assert attempts_count == 1.0

    duration_count = isolated_registry.get_sample_value(
        "chunking_error_recovery_duration_seconds_count",
        {"error_type": "timeout", "recovery_strategy": "retry"},
    )
    duration_sum = isolated_registry.get_sample_value(
        "chunking_error_recovery_duration_seconds_sum",
        {"error_type": "timeout", "recovery_strategy": "retry"},
    )
    assert duration_count == 1.0
    assert duration_sum == pytest.approx(2.5)


def test_record_resource_usage_observes_histograms(isolated_registry: CollectorRegistry) -> None:
    """record_resource_usage should observe memory and CPU histograms."""
    metrics.record_resource_usage(strategy="hierarchical", status="success", memory_bytes=12_000_000, cpu_seconds=5.5)

    memory_count = isolated_registry.get_sample_value(
        "chunking_memory_usage_bytes_count",
        {"strategy": "hierarchical", "status": "success"},
    )
    cpu_sum = isolated_registry.get_sample_value(
        "chunking_cpu_usage_seconds_sum",
        {"strategy": "hierarchical", "status": "success"},
    )

    assert memory_count == 1.0
    assert cpu_sum == pytest.approx(5.5)


def test_record_partial_failure_updates_counters(isolated_registry: CollectorRegistry) -> None:
    """record_partial_failure should record counts and ratios for partial failures."""
    metrics.record_partial_failure(strategy="hybrid", document_ratio=0.25, chunk_count=4)

    failure_total = isolated_registry.get_sample_value(
        "chunking_partial_failures_total",
        {"strategy": "hybrid"},
    )
    ratio_sum = isolated_registry.get_sample_value(
        "chunking_partial_failure_document_ratio_sum",
        {"strategy": "hybrid"},
    )
    chunk_total = isolated_registry.get_sample_value(
        "chunking_chunks_created_total",
        {"strategy": "hybrid", "document_type": "partial_failure"},
    )

    assert failure_total == 1.0
    assert ratio_sum == pytest.approx(0.25)
    assert chunk_total == 4.0


def test_update_operation_status_manages_gauges(isolated_registry: CollectorRegistry) -> None:
    """update_operation_status should increment and decrement appropriate gauges."""
    metrics.update_operation_status(strategy="semantic", operation_type="append", status="active")
    metrics.update_operation_status(strategy="semantic", operation_type="append", status="inactive")
    metrics.update_operation_status(strategy="semantic", operation_type="append", status="failed", error_type="timeout")
    metrics.update_operation_status(strategy="semantic", operation_type="append", status="queued")
    metrics.update_operation_status(strategy="semantic", operation_type="append", status="dequeued")

    failed_value = isolated_registry.get_sample_value(
        "chunking_operations_failed",
        {"strategy": "semantic", "error_type": "timeout"},
    )
    queued_value = isolated_registry.get_sample_value(
        "chunking_operations_queued",
        {"resource_type": "append"},
    )

    assert failed_value == 1.0
    # Gauge should return to 0 after enqueue/dequeue pair
    assert queued_value == 0.0


def test_update_circuit_breaker_state_sets_numeric_value(isolated_registry: CollectorRegistry) -> None:
    """Circuit breaker gauge should reflect textual states."""
    metrics.update_circuit_breaker_state(service="qdrant", state="open")
    metrics.update_circuit_breaker_state(service="qdrant", state="half_open")

    half_open_value = isolated_registry.get_sample_value(
        "chunking_circuit_breaker_state",
        {"service": "qdrant"},
    )
    assert half_open_value == 2.0  # half_open -> 2 according to state mapping


def test_chunking_cleanup_operation_records_duration(isolated_registry: CollectorRegistry) -> None:
    """Cleanup metrics should track both count and duration."""
    metrics.chunking_cleanup_operation_executed(cleanup_strategy="redis", success=True, duration=3.0)

    cleanup_total = isolated_registry.get_sample_value(
        "chunking_cleanup_operations_total",
        {"cleanup_strategy": "redis", "success": "True"},
    )
    cleanup_sum = isolated_registry.get_sample_value(
        "chunking_cleanup_duration_seconds_sum",
        {"cleanup_strategy": "redis"},
    )

    assert cleanup_total == 1.0
    assert cleanup_sum == pytest.approx(3.0)


def test_chunking_retry_recorded_tracks_exceeded(isolated_registry: CollectorRegistry) -> None:
    """chunking_retry_recorded should increment retry counts and optionally max retries exceeded."""
    metrics.chunking_retry_recorded(operation_type="append", retry_reason="oom", exceeded=True)

    retry_total = isolated_registry.get_sample_value(
        "chunking_retry_count_total",
        {"operation_type": "append", "retry_reason": "oom"},
    )
    exceeded_total = isolated_registry.get_sample_value(
        "chunking_max_retries_exceeded_total",
        {"strategy": "append", "final_error_type": "oom"},
    )

    assert retry_total == 1.0
    assert exceeded_total == 1.0


def test_circuit_breaker_trip_increments_counter(isolated_registry: CollectorRegistry) -> None:
    """chunking_circuit_breaker_trip should increment the trips counter."""
    metrics.chunking_circuit_breaker_trip(service="chunking", reason="redis_down")

    trips_total = isolated_registry.get_sample_value(
        "chunking_circuit_breaker_trips_total",
        {"service": "chunking", "reason": "redis_down"},
    )
    assert trips_total == 1.0


def test_error_handler_configured_exposes_info_metric(isolated_registry: CollectorRegistry) -> None:
    """chunking_error_handler_configured should update info metric labels."""
    metrics.chunking_error_handler_configured(mode="async", retries="2")

    info_value = isolated_registry.get_sample_value(
        "chunking_error_handler_info",
        {"mode": "async", "retries": "2"},
    )
    assert info_value == 1.0
