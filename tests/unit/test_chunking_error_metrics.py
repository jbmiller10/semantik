#!/usr/bin/env python3

"""
Unit tests for chunking error metrics.

This module tests metric recording functions, verifies metric labels
and values, and tests all metric types (Counter, Gauge, Histogram).
"""

from unittest.mock import MagicMock, patch

from prometheus_client import Counter, Gauge, Histogram, Info

from packages.webui.services.chunking_error_metrics import (
    chunking_chunk_size_bytes,
    chunking_chunks_created_total,
    chunking_circuit_breaker_state,
    chunking_circuit_breaker_trips,
    chunking_cleanup_duration_seconds,
    chunking_cleanup_operations_total,
    chunking_cpu_usage_seconds,
    chunking_dead_letter_queue_size,
    chunking_document_processing_duration,
    chunking_error_handler_info,
    chunking_error_recovery_attempts,
    chunking_error_recovery_duration,
    chunking_errors_total,
    chunking_max_retries_exceeded,
    chunking_memory_usage_bytes,
    chunking_operations_active,
    chunking_operations_failed,
    chunking_operations_queued,
    chunking_partial_failure_document_ratio,
    chunking_partial_failures_total,
    chunking_retry_count,
    record_chunking_error,
    record_partial_failure,
    record_recovery_attempt,
    record_resource_usage,
    update_circuit_breaker_state,
    update_operation_status,
)


class TestChunkingErrorMetrics:
    """Test suite for chunking error metrics."""

    def test_metric_initialization(self) -> None:
        """Test that all metrics are properly initialized."""
        # Test Counters
        assert isinstance(chunking_errors_total, Counter)
        assert chunking_errors_total._name == "chunking_errors"
        assert set(chunking_errors_total._labelnames) == {"error_type", "strategy", "recoverable"}

        assert isinstance(chunking_error_recovery_attempts, Counter)
        assert set(chunking_error_recovery_attempts._labelnames) == {"error_type", "recovery_strategy", "success"}

        assert isinstance(chunking_chunks_created_total, Counter)
        assert set(chunking_chunks_created_total._labelnames) == {"strategy", "document_type"}

        assert isinstance(chunking_retry_count, Counter)
        assert set(chunking_retry_count._labelnames) == {"operation_type", "retry_reason"}

        assert isinstance(chunking_max_retries_exceeded, Counter)
        assert set(chunking_max_retries_exceeded._labelnames) == {"strategy", "final_error_type"}

        assert isinstance(chunking_circuit_breaker_trips, Counter)
        assert set(chunking_circuit_breaker_trips._labelnames) == {"service", "reason"}

        assert isinstance(chunking_partial_failures_total, Counter)
        assert set(chunking_partial_failures_total._labelnames) == {"strategy"}

        assert isinstance(chunking_cleanup_operations_total, Counter)
        assert set(chunking_cleanup_operations_total._labelnames) == {"cleanup_strategy", "success"}

        # Test Gauges
        assert isinstance(chunking_operations_active, Gauge)
        assert set(chunking_operations_active._labelnames) == {"strategy", "operation_type"}

        assert isinstance(chunking_operations_failed, Gauge)
        assert set(chunking_operations_failed._labelnames) == {"strategy", "error_type"}

        assert isinstance(chunking_operations_queued, Gauge)
        assert set(chunking_operations_queued._labelnames) == {"resource_type"}

        assert isinstance(chunking_dead_letter_queue_size, Gauge)
        assert set(chunking_dead_letter_queue_size._labelnames) == {"queue_name"}

        assert isinstance(chunking_circuit_breaker_state, Gauge)
        assert set(chunking_circuit_breaker_state._labelnames) == {"service"}

        # Test Histograms
        assert isinstance(chunking_error_recovery_duration, Histogram)
        assert set(chunking_error_recovery_duration._labelnames) == {"error_type", "recovery_strategy"}

        assert isinstance(chunking_memory_usage_bytes, Histogram)
        assert set(chunking_memory_usage_bytes._labelnames) == {"strategy", "status"}

        assert isinstance(chunking_cpu_usage_seconds, Histogram)
        assert set(chunking_cpu_usage_seconds._labelnames) == {"strategy", "status"}

        assert isinstance(chunking_document_processing_duration, Histogram)
        assert set(chunking_document_processing_duration._labelnames) == {"strategy", "document_type", "status"}

        assert isinstance(chunking_chunk_size_bytes, Histogram)
        assert set(chunking_chunk_size_bytes._labelnames) == {"strategy"}

        assert isinstance(chunking_partial_failure_document_ratio, Histogram)
        assert set(chunking_partial_failure_document_ratio._labelnames) == {"strategy"}

        assert isinstance(chunking_cleanup_duration_seconds, Histogram)
        assert set(chunking_cleanup_duration_seconds._labelnames) == {"cleanup_strategy"}

        # Test Info
        assert isinstance(chunking_error_handler_info, Info)

    def test_record_chunking_error(self) -> None:
        """Test recording chunking errors."""
        with patch.object(chunking_errors_total, "labels") as mock_labels:
            mock_inc = MagicMock()
            mock_labels.return_value.inc = mock_inc

            record_chunking_error(
                error_type="memory",
                strategy="semantic",
                recoverable=True,
            )

            mock_labels.assert_called_once_with(
                error_type="memory",
                strategy="semantic",
                recoverable="True",
            )
            mock_inc.assert_called_once()

    def test_record_chunking_error_not_recoverable(self) -> None:
        """Test recording non-recoverable chunking errors."""
        with patch.object(chunking_errors_total, "labels") as mock_labels:
            mock_inc = MagicMock()
            mock_labels.return_value.inc = mock_inc

            record_chunking_error(
                error_type="fatal",
                strategy="recursive",
                recoverable=False,
            )

            mock_labels.assert_called_once_with(
                error_type="fatal",
                strategy="recursive",
                recoverable="False",
            )
            mock_inc.assert_called_once()

    def test_record_recovery_attempt_success(self) -> None:
        """Test recording successful recovery attempt."""
        with (
            patch.object(chunking_error_recovery_attempts, "labels") as mock_attempts_labels,
            patch.object(chunking_error_recovery_duration, "labels") as mock_duration_labels,
        ):
            mock_attempts_inc = MagicMock()
            mock_attempts_labels.return_value.inc = mock_attempts_inc

            mock_duration_observe = MagicMock()
            mock_duration_labels.return_value.observe = mock_duration_observe

            record_recovery_attempt(
                error_type="timeout",
                recovery_strategy="retry",
                success=True,
                duration=2.5,
            )

            # Verify attempt counter
            mock_attempts_labels.assert_called_once_with(
                error_type="timeout",
                recovery_strategy="retry",
                success="True",
            )
            mock_attempts_inc.assert_called_once()

            # Verify duration histogram (only for success)
            mock_duration_labels.assert_called_once_with(
                error_type="timeout",
                recovery_strategy="retry",
            )
            mock_duration_observe.assert_called_once_with(2.5)

    def test_record_recovery_attempt_failure(self) -> None:
        """Test recording failed recovery attempt."""
        with (
            patch.object(chunking_error_recovery_attempts, "labels") as mock_attempts_labels,
            patch.object(chunking_error_recovery_duration, "labels") as mock_duration_labels,
        ):
            mock_attempts_inc = MagicMock()
            mock_attempts_labels.return_value.inc = mock_attempts_inc

            record_recovery_attempt(
                error_type="memory",
                recovery_strategy="fallback",
                success=False,
                duration=1.0,
            )

            # Verify attempt counter
            mock_attempts_labels.assert_called_once_with(
                error_type="memory",
                recovery_strategy="fallback",
                success="False",
            )
            mock_attempts_inc.assert_called_once()

            # Duration should not be recorded for failures
            mock_duration_labels.assert_not_called()

    def test_update_operation_status_active(self) -> None:
        """Test updating active operation status."""
        with patch.object(chunking_operations_active, "labels") as mock_labels:
            mock_inc = MagicMock()
            mock_labels.return_value.inc = mock_inc

            update_operation_status(
                strategy="semantic",
                operation_type="preview",
                active_delta=1,
            )

            mock_labels.assert_called_once_with(
                strategy="semantic",
                operation_type="preview",
            )
            mock_inc.assert_called_once_with(1)

    def test_update_operation_status_failed(self) -> None:
        """Test updating failed operation status."""
        with patch.object(chunking_operations_failed, "labels") as mock_labels:
            mock_inc = MagicMock()
            mock_labels.return_value.inc = mock_inc

            update_operation_status(
                strategy="recursive",
                operation_type="process",
                failed_delta=1,
                error_type="timeout",
            )

            mock_labels.assert_called_once_with(
                strategy="recursive",
                error_type="timeout",
            )
            mock_inc.assert_called_once_with(1)

    def test_update_operation_status_both(self) -> None:
        """Test updating both active and failed status."""
        with (
            patch.object(chunking_operations_active, "labels") as mock_active_labels,
            patch.object(chunking_operations_failed, "labels") as mock_failed_labels,
        ):
            mock_active_inc = MagicMock()
            mock_active_labels.return_value.inc = mock_active_inc

            mock_failed_inc = MagicMock()
            mock_failed_labels.return_value.inc = mock_failed_inc

            update_operation_status(
                strategy="code",
                operation_type="batch",
                active_delta=-1,  # Decrement active
                failed_delta=1,  # Increment failed
                error_type="memory",
            )

            # Verify active decremented
            mock_active_labels.assert_called_once_with(
                strategy="code",
                operation_type="batch",
            )
            mock_active_inc.assert_called_once_with(-1)

            # Verify failed incremented
            mock_failed_labels.assert_called_once_with(
                strategy="code",
                error_type="memory",
            )
            mock_failed_inc.assert_called_once_with(1)

    def test_record_resource_usage(self) -> None:
        """Test recording resource usage metrics."""
        with (
            patch.object(chunking_memory_usage_bytes, "labels") as mock_memory_labels,
            patch.object(chunking_cpu_usage_seconds, "labels") as mock_cpu_labels,
        ):
            mock_memory_observe = MagicMock()
            mock_memory_labels.return_value.observe = mock_memory_observe

            mock_cpu_observe = MagicMock()
            mock_cpu_labels.return_value.observe = mock_cpu_observe

            record_resource_usage(
                strategy="semantic",
                status="success",
                memory_bytes=256 * 1024 * 1024,  # 256MB
                cpu_seconds=15.5,
            )

            # Verify memory usage
            mock_memory_labels.assert_called_once_with(
                strategy="semantic",
                status="success",
            )
            mock_memory_observe.assert_called_once_with(256 * 1024 * 1024)

            # Verify CPU usage
            mock_cpu_labels.assert_called_once_with(
                strategy="semantic",
                status="success",
            )
            mock_cpu_observe.assert_called_once_with(15.5)

    def test_update_circuit_breaker_state_closed(self) -> None:
        """Test updating circuit breaker to closed state."""
        with patch.object(chunking_circuit_breaker_state, "labels") as mock_labels:
            mock_set = MagicMock()
            mock_labels.return_value.set = mock_set

            update_circuit_breaker_state(
                service="embedding",
                state=0,  # Closed
            )

            mock_labels.assert_called_once_with(service="embedding")
            mock_set.assert_called_once_with(0)

    def test_update_circuit_breaker_state_open_with_reason(self) -> None:
        """Test updating circuit breaker to open state with trip reason."""
        with (
            patch.object(chunking_circuit_breaker_state, "labels") as mock_state_labels,
            patch.object(chunking_circuit_breaker_trips, "labels") as mock_trips_labels,
        ):
            mock_state_set = MagicMock()
            mock_state_labels.return_value.set = mock_state_set

            mock_trips_inc = MagicMock()
            mock_trips_labels.return_value.inc = mock_trips_inc

            update_circuit_breaker_state(
                service="qdrant",
                state=1,  # Open
                trip_reason="consecutive_failures",
            )

            # Verify state updated
            mock_state_labels.assert_called_once_with(service="qdrant")
            mock_state_set.assert_called_once_with(1)

            # Verify trip counter incremented
            mock_trips_labels.assert_called_once_with(
                service="qdrant",
                reason="consecutive_failures",
            )
            mock_trips_inc.assert_called_once()

    def test_update_circuit_breaker_state_half_open(self) -> None:
        """Test updating circuit breaker to half-open state."""
        with patch.object(chunking_circuit_breaker_state, "labels") as mock_labels:
            mock_set = MagicMock()
            mock_labels.return_value.set = mock_set

            update_circuit_breaker_state(
                service="redis",
                state=2,  # Half-open
            )

            mock_labels.assert_called_once_with(service="redis")
            mock_set.assert_called_once_with(2)

            # No trip counter update for half-open state

    def test_record_partial_failure(self) -> None:
        """Test recording partial failure metrics."""
        with (
            patch.object(chunking_partial_failures_total, "labels") as mock_total_labels,
            patch.object(chunking_partial_failure_document_ratio, "labels") as mock_ratio_labels,
        ):
            mock_total_inc = MagicMock()
            mock_total_labels.return_value.inc = mock_total_inc

            mock_ratio_observe = MagicMock()
            mock_ratio_labels.return_value.observe = mock_ratio_observe

            record_partial_failure(
                strategy="recursive",
                total_documents=100,
                failed_documents=25,
            )

            # Verify total counter
            mock_total_labels.assert_called_once_with(strategy="recursive")
            mock_total_inc.assert_called_once()

            # Verify ratio histogram
            mock_ratio_labels.assert_called_once_with(strategy="recursive")
            mock_ratio_observe.assert_called_once_with(0.25)  # 25/100

    def test_record_partial_failure_zero_documents(self) -> None:
        """Test recording partial failure with zero documents."""
        with (
            patch.object(chunking_partial_failures_total, "labels") as mock_total_labels,
            patch.object(chunking_partial_failure_document_ratio, "labels") as mock_ratio_labels,
        ):
            mock_total_inc = MagicMock()
            mock_total_labels.return_value.inc = mock_total_inc

            record_partial_failure(
                strategy="semantic",
                total_documents=0,
                failed_documents=0,
            )

            # Total counter should still be incremented
            mock_total_labels.assert_called_once_with(strategy="semantic")
            mock_total_inc.assert_called_once()

            # Ratio should not be recorded when total is 0
            mock_ratio_labels.assert_not_called()

    def test_histogram_buckets(self) -> None:
        """Test that histograms have appropriate bucket configurations."""
        # Test memory usage buckets (bytes)
        memory_buckets = chunking_memory_usage_bytes._upper_bounds
        assert 1_000_000 in memory_buckets  # 1MB
        assert 100_000_000 in memory_buckets  # 100MB
        assert 1_000_000_000 in memory_buckets  # 1GB

        # Test CPU usage buckets (seconds)
        cpu_buckets = chunking_cpu_usage_seconds._upper_bounds
        assert 0.1 in cpu_buckets
        assert 60 in cpu_buckets
        assert 300 in cpu_buckets

        # Test chunk size buckets (bytes)
        chunk_buckets = chunking_chunk_size_bytes._upper_bounds
        assert 100 in chunk_buckets
        assert 1000 in chunk_buckets
        assert 50000 in chunk_buckets

        # Test recovery duration buckets
        recovery_buckets = chunking_error_recovery_duration._upper_bounds
        assert 0.1 in recovery_buckets
        assert 60 in recovery_buckets
        assert 300 in recovery_buckets

    def test_info_metric_initialization(self) -> None:
        """Test that info metric is properly initialized with default values."""
        # Info metrics store their data differently
        # We need to check the metric's value
        info_data = chunking_error_handler_info._value

        assert info_data.get("version") == "1.0.0"
        assert info_data.get("max_retries") == "3"
        assert info_data.get("retry_backoff") == "exponential"
        assert info_data.get("circuit_breaker_enabled") == "true"
        assert info_data.get("dead_letter_queue_enabled") == "true"

    def test_metric_label_consistency(self) -> None:
        """Test that related metrics use consistent label names."""
        # Strategy label should be consistent across metrics
        strategy_metrics = [
            chunking_errors_total,
            chunking_operations_active,
            chunking_operations_failed,
            chunking_memory_usage_bytes,
            chunking_cpu_usage_seconds,
            chunking_document_processing_duration,
            chunking_chunk_size_bytes,
            chunking_chunks_created_total,
            chunking_partial_failures_total,
            chunking_partial_failure_document_ratio,
            chunking_max_retries_exceeded,
        ]

        for metric in strategy_metrics:
            assert "strategy" in metric._labelnames, f"{metric._name} missing 'strategy' label"

        # Error type label consistency
        error_type_metrics = [
            chunking_errors_total,
            chunking_error_recovery_attempts,
            chunking_error_recovery_duration,
            chunking_operations_failed,
        ]

        for metric in error_type_metrics:
            assert "error_type" in metric._labelnames, f"{metric._name} missing 'error_type' label"

    def test_gauge_metrics_can_decrease(self) -> None:
        """Test that gauge metrics support both increment and decrement."""
        gauges = [
            chunking_operations_active,
            chunking_operations_failed,
            chunking_operations_queued,
            chunking_dead_letter_queue_size,
            chunking_circuit_breaker_state,
        ]

        for gauge in gauges:
            # All gauges should support inc/dec operations
            assert hasattr(gauge, "inc")
            assert hasattr(gauge, "dec")
            assert hasattr(gauge, "set")

    def test_counter_metrics_only_increase(self) -> None:
        """Test that counter metrics only support increment."""
        counters = [
            chunking_errors_total,
            chunking_error_recovery_attempts,
            chunking_chunks_created_total,
            chunking_retry_count,
            chunking_max_retries_exceeded,
            chunking_circuit_breaker_trips,
            chunking_partial_failures_total,
            chunking_cleanup_operations_total,
        ]

        for counter in counters:
            # Counters should only have inc, not dec
            assert hasattr(counter, "inc")
            assert not hasattr(counter, "dec")
