#!/usr/bin/env python3
"""
Error handling framework for chunking operations.

This module provides comprehensive error handling, retry strategies,
and recovery mechanisms for chunking failures.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import psutil

from shared.database.models import CollectionStatus
from webui.api.chunking_exceptions import ResourceType
from webui.middleware.correlation import get_correlation_id

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from shared.text_processing.base_chunker import ChunkResult
    from webui.utils.error_classifier import ErrorClassificationResult, ErrorClassifier


class ChunkingErrorType(Enum):
    """Types of errors that can occur during chunking."""

    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    INVALID_ENCODING = "invalid_encoding"
    STRATEGY_ERROR = "strategy_error"
    PARTIAL_FAILURE = "partial_failure"
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    PERMISSION_ERROR = "permission_error"
    DEPENDENCY_ERROR = "dependency_error"
    RESOURCE_LIMIT_ERROR = "resource_limit_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from errors."""

    action: str  # retry, skip, fail, partial_save
    max_retries: int
    backoff_type: str  # linear, exponential
    recommendations: list[str]
    fallback_strategy: str | None = None


@dataclass
class ChunkingOperationResult:
    """Result of a chunking operation with error handling."""

    status: str  # success, partial_success, failed
    processed_count: int
    failed_count: int
    recovery_operation_id: str | None = None
    recommendations: list[str] | None = None
    error_details: dict[str, Any] | None = None


@dataclass
class StreamRecoveryAction:
    """Action to take for streaming failure recovery."""

    action: str  # retry_from_checkpoint, retry_with_extended_timeout, mark_failed
    checkpoint: int | None = None
    new_batch_size: int | None = None
    new_timeout: int | None = None
    error_details: str | None = None


@dataclass
class ErrorHandlingResult:
    """Result of error handling with correlation tracking."""

    handled: bool
    recovery_action: str
    correlation_id: str
    operation_id: str
    error_type: ChunkingErrorType
    retry_after: int | None = None
    fallback_strategy: str | None = None
    state_checkpoint: dict[str, Any] | None = None
    recommendations: list[str] = field(default_factory=list)


@dataclass
class ResourceRecoveryAction:
    """Action to take for resource exhaustion recovery."""

    action: str  # queue, reduce_batch, wait_and_retry, fail
    queue_position: int | None = None
    new_batch_size: int | None = None
    wait_time: int | None = None
    alternative_strategy: str | None = None
    resource_availability: dict[str, float] | None = None


@dataclass
class CleanupResult:
    """Result of cleanup operation after failure."""

    cleaned: bool
    partial_results_saved: bool
    resources_freed: dict[str, Any]
    rollback_performed: bool
    cleanup_errors: list[str] = field(default_factory=list)


@dataclass
class ErrorReport:
    """Comprehensive error report for an operation."""

    operation_id: str
    correlation_id: str
    total_errors: int
    error_timeline: list[dict[str, Any]]
    error_breakdown: dict[str, int]
    resource_usage: dict[str, Any]
    recovery_attempts: list[dict[str, Any]]
    recommendations: list[str]
    created_at: datetime


class ChunkingErrorHandler:
    """Handle errors during chunking operations with production-ready features.

    This handler provides:
    - Correlation ID tracking for distributed tracing
    - State management with Redis for resumable operations
    - Advanced recovery strategies with adaptive batch sizing
    - Resource tracking and queuing for overload protection
    - Comprehensive error reporting and analytics
    """

    # Maximum concurrent operations per resource type
    RESOURCE_LIMITS = {
        ResourceType.MEMORY: {"max_concurrent": 10, "memory_limit_gb": 8},
        ResourceType.CPU: {"max_concurrent": 20, "cpu_limit_percent": 80},
        ResourceType.CONNECTIONS: {"max_concurrent": 100},
    }

    # State TTL in Redis (24 hours)
    STATE_TTL = 86400

    # Retry strategies for different error types
    RETRY_STRATEGIES = {
        ChunkingErrorType.MEMORY_ERROR: RecoveryStrategy(
            action="retry",
            max_retries=2,
            backoff_type="exponential",
            recommendations=[
                "Reduce batch size for processing",
                "Consider using streaming for large documents",
                "Increase memory allocation if possible",
            ],
            fallback_strategy="character",  # Simpler strategy uses less memory
        ),
        ChunkingErrorType.TIMEOUT_ERROR: RecoveryStrategy(
            action="retry",
            max_retries=3,
            backoff_type="linear",
            recommendations=[
                "Process documents in smaller batches",
                "Consider using a simpler chunking strategy",
                "Check network connectivity for remote resources",
            ],
        ),
        ChunkingErrorType.INVALID_ENCODING: RecoveryStrategy(
            action="retry",
            max_retries=1,
            backoff_type="linear",
            recommendations=[
                "Attempt with different encoding (UTF-8 with errors='ignore')",
                "Clean the text data before processing",
                "Consider extracting text with unstructured library",
            ],
        ),
        ChunkingErrorType.STRATEGY_ERROR: RecoveryStrategy(
            action="retry",
            max_retries=1,
            backoff_type="linear",
            recommendations=[
                "Fall back to recursive chunking strategy",
                "Check strategy configuration parameters",
                "Validate input text format",
            ],
            fallback_strategy="recursive",
        ),
        ChunkingErrorType.PARTIAL_FAILURE: RecoveryStrategy(
            action="partial_save",
            max_retries=0,
            backoff_type="none",
            recommendations=[
                "Save successfully processed chunks",
                "Mark failed documents for manual review",
                "Continue with remaining documents",
            ],
        ),
        ChunkingErrorType.DEPENDENCY_ERROR: RecoveryStrategy(
            action="retry",
            max_retries=2,
            backoff_type="exponential",
            recommendations=[
                "Verify third-party services are reachable",
                "Check API credentials or tokens",
                "Consult dependency health dashboards",
            ],
        ),
        ChunkingErrorType.RESOURCE_LIMIT_ERROR: RecoveryStrategy(
            action="retry",
            max_retries=2,
            backoff_type="linear",
            recommendations=[
                "Scale worker resources or reduce concurrency",
                "Lower batch sizes for chunking operations",
                "Review resource quota alarms and thresholds",
            ],
            fallback_strategy="character",
        ),
    }

    def __init__(
        self,
        redis_client: Redis | None = None,
        *,
        error_classifier: ErrorClassifier | None = None,
    ) -> None:
        """Initialize the error handler.

        Args:
            redis_client: Optional Redis client for state management.
                         If not provided, state management features will be disabled.
            error_classifier: Optional shared classifier instance. When not
                provided the default chunking classifier will be used.
        """
        from webui.utils.error_classifier import get_default_chunking_error_classifier

        self.retry_counts: dict[str, int] = {}
        self.redis_client = redis_client
        self._resource_locks: dict[str, asyncio.Lock] = {}
        self._error_history: dict[str, list[dict[str, Any]]] = {}
        self._error_classifier = error_classifier or get_default_chunking_error_classifier()

    async def handle_partial_failure(
        self,
        operation_id: str,
        processed_chunks: list[ChunkResult],
        failed_documents: list[str],
        errors: list[Exception],
    ) -> ChunkingOperationResult:
        """Handle partial chunking failures gracefully.

        Args:
            operation_id: Operation identifier
            processed_chunks: Successfully processed chunks
            failed_documents: List of failed document IDs
            errors: List of exceptions encountered

        Returns:
            ChunkingOperationResult with recovery information
        """
        # Get correlation ID for tracking
        correlation_id = get_correlation_id() or "unknown"

        logger.info(
            f"Handling partial failure for operation {operation_id}",
            extra={
                "correlation_id": correlation_id,
                "operation_id": operation_id,
                "processed_count": len(processed_chunks),
                "failed_count": len(failed_documents),
            },
        )

        # Save successful chunks (this would be implemented with actual DB operations)
        await self.save_partial_results(operation_id, processed_chunks)

        # Analyze failure patterns
        failure_analysis = self.analyze_failures(errors)

        # Create recovery strategy
        recovery_strategy = self.create_recovery_strategy(
            failure_analysis,
            failed_documents,
        )

        # Track errors for reporting
        for error in errors:
            error_type = self.classify_error(error)
            await self._track_error(operation_id, correlation_id, error, error_type)

        # Update collection status (this would use actual repository)
        await self.update_collection_status(
            operation_id,
            CollectionStatus.DEGRADED,
            f"Partial failure: {len(failed_documents)} documents failed",
        )

        # Create recovery operation
        recovery_op = await self.create_recovery_operation(
            operation_id,
            recovery_strategy,
        )

        # Save state for potential retry if Redis available
        if self.redis_client and recovery_op:
            await self._save_operation_state(
                operation_id,
                correlation_id,
                {
                    "failed_documents": failed_documents,
                    "recovery_strategy": recovery_strategy.action,
                    "processed_chunks": len(processed_chunks),
                },
                ChunkingErrorType.PARTIAL_FAILURE,
            )

        return ChunkingOperationResult(
            status="partial_success",
            processed_count=len(processed_chunks),
            failed_count=len(failed_documents),
            recovery_operation_id=recovery_op.get("id") if recovery_op else None,
            recommendations=recovery_strategy.recommendations,
            error_details={
                "correlation_id": correlation_id,
                "failure_analysis": failure_analysis,
                "failed_documents": failed_documents[:10],  # First 10 for brevity
            },
        )

    async def handle_streaming_failure(
        self,
        document_id: str,
        bytes_processed: int,
        error: Exception,
        operation_id: str | None = None,
    ) -> StreamRecoveryAction:
        """Handle failures during streaming processing.

        Args:
            document_id: Document being processed
            bytes_processed: Bytes processed before failure
            error: Exception that occurred
            operation_id: Optional operation identifier for tracking

        Returns:
            StreamRecoveryAction with recovery instructions
        """
        correlation_id = get_correlation_id() or "unknown"

        logger.warning(
            f"Streaming failure for document {document_id}",
            extra={
                "correlation_id": correlation_id,
                "document_id": document_id,
                "bytes_processed": bytes_processed,
                "error_type": type(error).__name__,
                "operation_id": operation_id,
            },
        )

        error_type = self.classify_error(error)

        # Track error if operation ID provided
        if operation_id:
            await self._track_error(operation_id, correlation_id, error, error_type)

        if error_type == ChunkingErrorType.MEMORY_ERROR:
            # Check current memory before deciding on batch size
            if self.redis_client:
                resource_availability = await self._check_resource_availability(ResourceType.MEMORY)
                memory_percent = resource_availability.get("percent_used", 50)
                # Adaptive batch size based on current memory
                new_batch_size = self._calculate_adaptive_batch_size(
                    memory_percent,
                    100.0,
                )
            else:
                new_batch_size = self.calculate_reduced_batch_size(error)

            return StreamRecoveryAction(
                action="retry_from_checkpoint",
                checkpoint=bytes_processed,
                new_batch_size=new_batch_size,
            )

        if error_type == ChunkingErrorType.TIMEOUT_ERROR:
            # Progressive timeout increase based on retry count
            retry_key = f"{operation_id}:{error_type.value}" if operation_id else f"{document_id}:{error_type.value}"
            retry_count = self.retry_counts.get(retry_key, 0)

            # Exponential timeout increase: 450s, 675s, 1012s
            base_timeout = self.calculate_extended_timeout()
            new_timeout = int(base_timeout * (1.5**retry_count))

            return StreamRecoveryAction(
                action="retry_with_extended_timeout",
                checkpoint=bytes_processed,
                new_timeout=new_timeout,
            )

        # Unrecoverable - mark document as failed
        return StreamRecoveryAction(
            action="mark_failed",
            error_details=str(error),
        )

    def classify_error(self, error: Exception) -> ChunkingErrorType:
        """Classify an error into a specific type.

        Args:
            error: Exception to classify

        Returns:
            ChunkingErrorType classification
        """
        return cast(ChunkingErrorType, self._error_classifier.as_enum(error))

    def classify_error_detailed(self, error: Exception) -> ErrorClassificationResult:
        """Return the detailed classification result for an error."""

        return self._error_classifier.classify(error)

    def get_retry_strategy(self, error_type: ChunkingErrorType) -> RecoveryStrategy:
        """Get retry strategy for an error type.

        Args:
            error_type: Type of error

        Returns:
            RecoveryStrategy for the error type
        """
        return self.RETRY_STRATEGIES.get(
            error_type,
            RecoveryStrategy(
                action="fail",
                max_retries=0,
                backoff_type="none",
                recommendations=["Manual intervention required"],
            ),
        )

    def should_retry(self, operation_id: str, error_type: ChunkingErrorType) -> bool:
        """Determine if operation should be retried.

        Args:
            operation_id: Operation identifier
            error_type: Type of error

        Returns:
            True if should retry, False otherwise
        """
        retry_key = f"{operation_id}:{error_type.value}"
        current_retries = self.retry_counts.get(retry_key, 0)

        strategy = self.get_retry_strategy(error_type)

        if current_retries >= strategy.max_retries:
            return False

        # Increment retry count
        self.retry_counts[retry_key] = current_retries + 1

        return strategy.action == "retry"

    def calculate_retry_delay(
        self,
        operation_id: str,
        error_type: ChunkingErrorType,
    ) -> int:
        """Calculate delay before retry in seconds.

        Args:
            operation_id: Operation identifier
            error_type: Type of error

        Returns:
            Delay in seconds
        """
        retry_key = f"{operation_id}:{error_type.value}"
        current_retries = self.retry_counts.get(retry_key, 1)

        strategy = self.get_retry_strategy(error_type)

        if strategy.backoff_type == "exponential":
            # Exponential backoff: 10s, 20s, 40s, etc.
            delay = int(min(300, 10 * (2 ** (current_retries - 1))))
        elif strategy.backoff_type == "linear":
            # Linear backoff: 10s, 20s, 30s, etc.
            delay = int(min(300, 10 * current_retries))
        else:
            delay = 0

        logger.info(f"Retry {current_retries}/{strategy.max_retries} for {error_type.value}, delay: {delay}s")

        return delay

    def calculate_reduced_batch_size(self, error: Exception) -> int:  # noqa: ARG002
        """Calculate reduced batch size after memory error.

        Args:
            error: Memory error encountered

        Returns:
            New batch size
        """
        # Simple heuristic: reduce by 50%
        # In production, this could be smarter based on actual memory usage
        return 16  # Reduced from default 32

    def calculate_extended_timeout(self) -> int:
        """Calculate extended timeout after timeout error.

        Returns:
            New timeout in seconds
        """
        # Extend by 50%
        return 450  # Extended from default 300

    def analyze_failures(self, errors: list[Exception]) -> dict[str, Any]:
        """Analyze failure patterns.

        Args:
            errors: List of exceptions

        Returns:
            Analysis results
        """
        error_types: dict[str, int] = {}
        for error in errors:
            error_type = self.classify_error(error)
            error_types[error_type.value] = error_types.get(error_type.value, 0) + 1

        return {
            "total_errors": len(errors),
            "error_breakdown": error_types,
            "most_common": max(error_types, key=lambda k: error_types[k]) if error_types else None,
        }

    def create_recovery_strategy(
        self,
        failure_analysis: dict[str, Any],
        failed_documents: list[str],
    ) -> RecoveryStrategy:
        """Create recovery strategy based on failure analysis.

        Args:
            failure_analysis: Analysis of failures
            failed_documents: List of failed documents

        Returns:
            RecoveryStrategy for recovery
        """
        most_common_error = failure_analysis.get("most_common")

        if most_common_error:
            error_type = ChunkingErrorType(most_common_error)
            return self.get_retry_strategy(error_type)

        # Default strategy
        return RecoveryStrategy(
            action="partial_save",
            max_retries=0,
            backoff_type="none",
            recommendations=[
                f"Review {len(failed_documents)} failed documents",
                "Consider using a different chunking strategy",
                "Check document formats and encoding",
            ],
        )

    async def save_partial_results(
        self,
        operation_id: str,
        chunks: list[ChunkResult],
    ) -> None:
        """Save partial results to database.

        Args:
            operation_id: Operation identifier
            chunks: Chunks to save
        """
        # This would be implemented with actual database operations
        logger.info(f"Saved {len(chunks)} chunks for operation {operation_id}")

    async def update_collection_status(
        self,
        operation_id: str,  # noqa: ARG002
        status: CollectionStatus,
        message: str,
    ) -> None:
        """Update collection status.

        Args:
            operation_id: Operation identifier
            status: New status
            message: Status message
        """
        # This would be implemented with actual database operations
        logger.info(f"Updated collection status to {status.value}: {message}")

    async def create_recovery_operation(
        self,
        operation_id: str,
        strategy: RecoveryStrategy,
    ) -> dict[str, Any] | None:
        """Create recovery operation.

        Args:
            operation_id: Original operation ID
            strategy: Recovery strategy

        Returns:
            Recovery operation details or None
        """
        if strategy.action in ["retry", "partial_save"]:
            # This would create an actual operation in the database
            return {
                "id": f"recovery_{operation_id}",
                "original_operation_id": operation_id,
                "strategy": strategy.action,
                "max_retries": strategy.max_retries,
            }

        return None

    async def handle_with_correlation(
        self,
        operation_id: str,
        correlation_id: str,
        error: Exception,
        context: dict[str, Any],
    ) -> ErrorHandlingResult:
        """Handle error with correlation ID tracking for distributed tracing.

        Args:
            operation_id: Operation identifier
            correlation_id: Correlation ID for tracing
            error: Exception that occurred
            context: Additional context about the operation

        Returns:
            ErrorHandlingResult with recovery instructions
        """
        # Log error with correlation ID
        logger.error(
            f"Chunking error for operation {operation_id}",
            extra={
                "correlation_id": correlation_id,
                "operation_id": operation_id,
                "error_type": type(error).__name__,
                "context": context,
            },
            exc_info=error,
        )

        # Classify the error
        error_type = self.classify_error(error)

        # Track error in history
        await self._track_error(operation_id, correlation_id, error, error_type)

        # Get recovery strategy
        strategy = self.get_retry_strategy(error_type)

        # Check if we should retry
        should_retry = self.should_retry(operation_id, error_type)

        # Save operation state if Redis is available
        state_checkpoint = None
        if self.redis_client and should_retry:
            state_checkpoint = await self._save_operation_state(
                operation_id,
                correlation_id,
                context,
                error_type,
            )

        # Calculate retry delay
        retry_after = None
        if should_retry:
            retry_after = self.calculate_retry_delay(operation_id, error_type)

        return ErrorHandlingResult(
            handled=True,
            recovery_action=strategy.action if should_retry else "fail",
            correlation_id=correlation_id,
            operation_id=operation_id,
            error_type=error_type,
            retry_after=retry_after,
            fallback_strategy=strategy.fallback_strategy,
            state_checkpoint=state_checkpoint,
            recommendations=strategy.recommendations,
        )

    async def handle_resource_exhaustion(
        self,
        operation_id: str,
        resource_type: ResourceType,
        current_usage: float,
        limit: float,
    ) -> ResourceRecoveryAction:
        """Handle resource exhaustion with intelligent recovery strategies.

        Args:
            operation_id: Operation identifier
            resource_type: Type of resource exhausted
            current_usage: Current resource usage
            limit: Resource limit

        Returns:
            ResourceRecoveryAction with recovery instructions
        """
        logger.warning(
            f"Resource exhaustion for {resource_type.value}",
            extra={
                "operation_id": operation_id,
                "resource_type": resource_type.value,
                "current_usage": current_usage,
                "limit": limit,
                "usage_percent": (current_usage / limit * 100) if limit > 0 else 100,
            },
        )

        # Check current system resources
        resource_availability = await self._check_resource_availability(resource_type)

        # Check if we can queue the operation
        if self.redis_client:
            queue_position = await self._queue_operation(operation_id, resource_type)
            if queue_position is not None:
                return ResourceRecoveryAction(
                    action="queue",
                    queue_position=queue_position,
                    wait_time=queue_position * 30,  # Estimate 30s per queued operation
                    resource_availability=resource_availability,
                )

        # Determine recovery action based on resource type
        if resource_type == ResourceType.MEMORY:
            # Adaptive batch size reduction based on memory pressure
            new_batch_size = self._calculate_adaptive_batch_size(current_usage, limit)
            return ResourceRecoveryAction(
                action="reduce_batch",
                new_batch_size=new_batch_size,
                alternative_strategy="streaming",
                resource_availability=resource_availability,
            )

        if resource_type == ResourceType.CPU:
            # For CPU, suggest waiting or using simpler strategy
            return ResourceRecoveryAction(
                action="wait_and_retry",
                wait_time=60,  # Wait 60 seconds
                alternative_strategy="character",  # Simpler strategy
                resource_availability=resource_availability,
            )

        # For other resources, wait and retry
        return ResourceRecoveryAction(
            action="wait_and_retry",
            wait_time=30,
            resource_availability=resource_availability,
        )

    async def cleanup_failed_operation(
        self,
        operation_id: str,
        partial_results: list[ChunkResult] | None,
        cleanup_strategy: str = "save_partial",
    ) -> CleanupResult:
        """Clean up after a failed operation with configurable strategies.

        Args:
            operation_id: Operation identifier
            partial_results: Any partial results to handle
            cleanup_strategy: Strategy for cleanup (save_partial, rollback, discard)

        Returns:
            CleanupResult with cleanup details
        """
        logger.info(f"Cleaning up failed operation {operation_id} with strategy: {cleanup_strategy}")

        cleanup_errors = []
        resources_freed = {}
        partial_saved = False
        rollback_performed = False

        try:
            # Save partial results if requested
            if cleanup_strategy == "save_partial" and partial_results:
                try:
                    await self.save_partial_results(operation_id, partial_results)
                    partial_saved = True
                except Exception as e:
                    cleanup_errors.append(f"Failed to save partial results: {str(e)}")

            # Clear operation state from Redis
            if self.redis_client:
                try:
                    state_key = f"chunking:state:{operation_id}"
                    checkpoint_key = f"chunking:checkpoint:{operation_id}"
                    queue_key = "chunking:queue:*"

                    # Remove state and checkpoint
                    await self.redis_client.delete(state_key, checkpoint_key)

                    # Remove from any queues
                    async for key in self.redis_client.scan_iter(queue_key):
                        await self.redis_client.lrem(key, 0, operation_id)

                    resources_freed["redis_keys"] = [state_key, checkpoint_key]
                except Exception as e:
                    cleanup_errors.append(f"Failed to clear Redis state: {str(e)}")

            # Clear from retry counts
            keys_to_remove = [k for k in self.retry_counts if k.startswith(f"{operation_id}:")]
            for key in keys_to_remove:
                del self.retry_counts[key]
            resources_freed["retry_entries"] = len(keys_to_remove)  # type: ignore[assignment]

            # Clear from error history
            if operation_id in self._error_history:
                del self._error_history[operation_id]
                resources_freed["error_history"] = True  # type: ignore[assignment]

            # Perform rollback if requested
            if cleanup_strategy == "rollback":
                # This would involve database transactions in a real implementation
                rollback_performed = True
                logger.info(f"Performed rollback for operation {operation_id}")

        except Exception as e:
            cleanup_errors.append(f"Unexpected error during cleanup: {str(e)}")
            logger.error(f"Cleanup failed for operation {operation_id}", exc_info=e)

        return CleanupResult(
            cleaned=len(cleanup_errors) == 0,
            partial_results_saved=partial_saved,
            resources_freed=resources_freed,
            rollback_performed=rollback_performed,
            cleanup_errors=cleanup_errors,
        )

    def create_error_report(
        self,
        operation_id: str,
        errors: list[Exception] | None = None,
    ) -> ErrorReport:
        """Create a comprehensive error report for an operation.

        Args:
            operation_id: Operation identifier
            errors: Optional list of errors (uses history if not provided)

        Returns:
            ErrorReport with detailed error analysis
        """
        correlation_id = get_correlation_id() or "unknown"

        # Get error history for the operation
        error_history = self._error_history.get(operation_id, [])

        # If errors provided, use those; otherwise use history
        if errors:
            for error in errors:
                error_type = self.classify_error(error)
                error_history.append(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "error_type": error_type.value,
                        "error_message": str(error),
                        "error_class": type(error).__name__,
                    }
                )

        # Analyze error patterns
        error_breakdown: dict[str, int] = {}
        for entry in error_history:
            error_type = entry.get("error_type", "unknown")
            error_breakdown[error_type] = error_breakdown.get(error_type, 0) + 1

        # Get resource usage
        resource_usage = self._get_current_resource_usage()

        # Get recovery attempts
        recovery_attempts = []
        for key, count in self.retry_counts.items():
            if key.startswith(f"{operation_id}:"):
                error_type_str = key.split(":", 1)[1]
                try:
                    error_type_enum = ChunkingErrorType(error_type_str)
                    max_retries = self.RETRY_STRATEGIES.get(
                        error_type_enum,
                        RecoveryStrategy("fail", 0, "none", []),
                    ).max_retries
                except ValueError:
                    # Invalid error type, use default
                    max_retries = 0
                recovery_attempts.append(
                    {
                        "error_type": error_type_str,
                        "retry_count": count,
                        "max_retries": max_retries,
                    }
                )

        # Generate recommendations
        recommendations = self._generate_recommendations(error_breakdown, resource_usage)

        return ErrorReport(
            operation_id=operation_id,
            correlation_id=correlation_id,
            total_errors=len(error_history),
            error_timeline=error_history,
            error_breakdown=error_breakdown,
            resource_usage=resource_usage,
            recovery_attempts=recovery_attempts,
            recommendations=recommendations,
            created_at=datetime.now(UTC),
        )

    # Helper methods for new functionality

    async def _track_error(
        self,
        operation_id: str,
        correlation_id: str,
        error: Exception,
        error_type: ChunkingErrorType,
    ) -> None:
        """Track error in history for analysis."""
        if operation_id not in self._error_history:
            self._error_history[operation_id] = []

        self._error_history[operation_id].append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "correlation_id": correlation_id,
                "error_type": error_type.value,
                "error_message": str(error),
                "error_class": type(error).__name__,
                "stack_trace": None,  # Could add traceback if needed
            }
        )

        # Limit history size to prevent memory issues
        if len(self._error_history[operation_id]) > 100:
            self._error_history[operation_id] = self._error_history[operation_id][-100:]

    async def _save_operation_state(
        self,
        operation_id: str,
        correlation_id: str,
        context: dict[str, Any],
        error_type: ChunkingErrorType,
    ) -> dict[str, Any] | None:
        """Save operation state to Redis for resumability."""
        if not self.redis_client:
            return None

        try:
            state = {
                "operation_id": operation_id,
                "correlation_id": correlation_id,
                "error_type": error_type.value,
                "context": context,
                "timestamp": datetime.now(UTC).isoformat(),
                "retry_count": self.retry_counts.get(f"{operation_id}:{error_type.value}", 0),
            }

            # Create operation fingerprint for idempotency
            fingerprint = self._create_operation_fingerprint(operation_id, context)
            state["fingerprint"] = fingerprint

            # Save to Redis with TTL
            state_key = f"chunking:state:{operation_id}"
            await self.redis_client.setex(
                state_key,
                self.STATE_TTL,
                json.dumps(state),
            )

            # Save checkpoint if available
            if "checkpoint" in context:
                checkpoint_key = f"chunking:checkpoint:{operation_id}"
                await self.redis_client.setex(
                    checkpoint_key,
                    self.STATE_TTL,
                    json.dumps(context["checkpoint"]),
                )

            return state

        except Exception as e:
            logger.error(f"Failed to save operation state: {str(e)}", exc_info=e)
            return None

    async def _check_resource_availability(
        self,
        resource_type: ResourceType,
    ) -> dict[str, float]:
        """Check current resource availability."""
        availability = {}

        try:
            if resource_type == ResourceType.MEMORY:
                memory = psutil.virtual_memory()
                availability = {
                    "available_gb": memory.available / (1024**3),
                    "percent_used": memory.percent,
                    "total_gb": memory.total / (1024**3),
                }
            elif resource_type == ResourceType.CPU:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                availability = {
                    "percent_used": cpu_percent,
                    "percent_available": 100 - cpu_percent,
                    "cpu_count": psutil.cpu_count(),
                }
            elif resource_type == ResourceType.CONNECTIONS:
                # This would check actual connection pools in production
                availability = {
                    "estimated_available": 50,  # Placeholder
                }
        except Exception as e:
            logger.error(f"Failed to check resource availability: {str(e)}")

        return availability

    async def _queue_operation(
        self,
        operation_id: str,
        resource_type: ResourceType,
    ) -> int | None:
        """Queue operation for later execution when resources are available."""
        if not self.redis_client:
            return None

        try:
            queue_key = f"chunking:queue:{resource_type.value}"

            # Check if already queued
            position = await self.redis_client.lpos(queue_key, operation_id)
            if position is not None:
                return int(position)

            # Add to queue
            await self.redis_client.rpush(queue_key, operation_id)

            # Get position
            position = await self.redis_client.lpos(queue_key, operation_id)
            return int(position) if position is not None else None

        except Exception as e:
            logger.error(f"Failed to queue operation: {str(e)}")
            return None

    def _calculate_adaptive_batch_size(
        self,
        current_usage: float,
        limit: float,
    ) -> int:
        """Calculate adaptive batch size based on resource pressure."""
        usage_ratio = current_usage / limit if limit > 0 else 1.0

        if usage_ratio > 0.9:
            # Very high usage: minimize batch size
            return 4
        if usage_ratio > 0.8:
            # High usage: small batches
            return 8
        if usage_ratio > 0.7:
            # Moderate usage: medium batches
            return 16
        # Low usage: normal batches
        return 32

    def _create_operation_fingerprint(
        self,
        operation_id: str,
        context: dict[str, Any],
    ) -> str:
        """Create fingerprint for operation idempotency."""
        # Extract key fields for fingerprint
        fingerprint_data = {
            "operation_id": operation_id,
            "collection_id": context.get("collection_id"),
            "document_ids": sorted(context.get("document_ids", [])),
            "strategy": context.get("strategy"),
            "params": context.get("params"),
        }

        # Create hash
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()

    def _get_current_resource_usage(self) -> dict[str, Any]:
        """Get current system resource usage."""
        try:
            memory = psutil.virtual_memory()
            return {
                "memory": {
                    "used_gb": (memory.total - memory.available) / (1024**3),
                    "percent": memory.percent,
                },
                "cpu": {
                    "percent": psutil.cpu_percent(interval=0.1),
                },
                "operations": {
                    "active": len(self.retry_counts),
                    "queued": 0,  # Would check Redis queues in production
                },
            }
        except Exception as e:
            logger.error(f"Failed to get resource usage: {str(e)}")
            return {}

    def _generate_recommendations(
        self,
        error_breakdown: dict[str, int],
        resource_usage: dict[str, Any],
    ) -> list[str]:
        """Generate recommendations based on error patterns and resources."""
        recommendations = []

        # Analyze error patterns
        if error_breakdown.get("memory_error", 0) > 0:
            recommendations.append("Consider using streaming mode for large documents")
            recommendations.append("Reduce batch size for processing")
            if resource_usage.get("memory", {}).get("percent", 0) > 80:
                recommendations.append("System memory is high - consider scaling resources")

        if error_breakdown.get("timeout_error", 0) > 0:
            recommendations.append("Use simpler chunking strategies for faster processing")
            recommendations.append("Process documents in smaller batches")

        if error_breakdown.get("strategy_error", 0) > 0:
            recommendations.append("Review strategy configuration parameters")
            recommendations.append("Consider using recursive strategy as fallback")

        # Resource-based recommendations
        if resource_usage.get("cpu", {}).get("percent", 0) > 80:
            recommendations.append("High CPU usage detected - consider queuing operations")

        if len(recommendations) == 0:
            recommendations.append("Review logs for detailed error information")

        return recommendations

    async def acquire_resource_lock(
        self,
        resource_type: ResourceType,
        operation_id: str,
        timeout: float = 30.0,
    ) -> bool:
        """Acquire a lock for a specific resource type to prevent overload.

        Args:
            resource_type: Type of resource to lock
            operation_id: Operation requesting the lock
            timeout: Maximum time to wait for lock

        Returns:
            True if lock acquired, False otherwise
        """
        lock_key = f"resource_lock:{resource_type.value}"

        # Create lock if doesn't exist
        if lock_key not in self._resource_locks:
            self._resource_locks[lock_key] = asyncio.Lock()

        lock = self._resource_locks[lock_key]

        try:
            # Try to acquire with timeout
            await asyncio.wait_for(lock.acquire(), timeout=timeout)

            logger.info(
                f"Acquired resource lock for {resource_type.value}",
                extra={
                    "operation_id": operation_id,
                    "resource_type": resource_type.value,
                    "correlation_id": get_correlation_id(),
                },
            )
            return True

        except TimeoutError:
            logger.warning(
                f"Failed to acquire resource lock for {resource_type.value} within {timeout}s",
                extra={
                    "operation_id": operation_id,
                    "resource_type": resource_type.value,
                    "correlation_id": get_correlation_id(),
                },
            )
            return False

    def release_resource_lock(self, resource_type: ResourceType) -> None:
        """Release a resource lock.

        Args:
            resource_type: Type of resource to unlock
        """
        lock_key = f"resource_lock:{resource_type.value}"

        if lock_key in self._resource_locks:
            lock = self._resource_locks[lock_key]
            if lock.locked():
                lock.release()
                logger.info(
                    f"Released resource lock for {resource_type.value}",
                    extra={
                        "resource_type": resource_type.value,
                        "correlation_id": get_correlation_id(),
                    },
                )

    async def get_operation_state(
        self,
        operation_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve saved operation state from Redis.

        Args:
            operation_id: Operation identifier

        Returns:
            Saved state dictionary or None if not found
        """
        if not self.redis_client:
            return None

        try:
            state_key = f"chunking:state:{operation_id}"
            state_data = await self.redis_client.get(state_key)

            if state_data:
                state_dict: dict[str, Any] = json.loads(state_data)
                return state_dict

        except Exception as e:
            logger.error(f"Failed to retrieve operation state: {str(e)}", exc_info=e)

        return None

    async def resume_operation(
        self,
        operation_id: str,
        context_updates: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Resume a failed operation from saved state.

        Args:
            operation_id: Operation to resume
            context_updates: Optional updates to merge with saved context

        Returns:
            Combined state and context for resuming, or None if not found
        """
        state = await self.get_operation_state(operation_id)

        if not state:
            logger.warning(f"No saved state found for operation {operation_id}")
            return None

        # Check if checkpoint exists
        checkpoint = None
        if self.redis_client:
            checkpoint_key = f"chunking:checkpoint:{operation_id}"
            checkpoint_data = await self.redis_client.get(checkpoint_key)
            if checkpoint_data:
                checkpoint = json.loads(checkpoint_data)

        # Merge context updates
        if context_updates:
            state["context"].update(context_updates)

        # Add checkpoint if found
        if checkpoint:
            state["checkpoint"] = checkpoint

        logger.info(
            f"Resuming operation {operation_id}",
            extra={
                "operation_id": operation_id,
                "correlation_id": state.get("correlation_id"),
                "has_checkpoint": checkpoint is not None,
            },
        )

        return state
