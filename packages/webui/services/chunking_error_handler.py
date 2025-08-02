#!/usr/bin/env python3
"""
Error handling framework for chunking operations.

This module provides comprehensive error handling, retry strategies,
and recovery mechanisms for chunking failures.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from packages.shared.database.models import CollectionStatus
from packages.shared.text_processing.base_chunker import ChunkResult

logger = logging.getLogger(__name__)


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


class ChunkingErrorHandler:
    """Handle errors during chunking operations."""

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
    }

    def __init__(self) -> None:
        """Initialize the error handler."""
        self.retry_counts: dict[str, int] = {}

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
        # Save successful chunks (this would be implemented with actual DB operations)
        await self.save_partial_results(operation_id, processed_chunks)

        # Analyze failure patterns
        failure_analysis = self.analyze_failures(errors)

        # Create recovery strategy
        recovery_strategy = self.create_recovery_strategy(
            failure_analysis,
            failed_documents,
        )

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

        return ChunkingOperationResult(
            status="partial_success",
            processed_count=len(processed_chunks),
            failed_count=len(failed_documents),
            recovery_operation_id=recovery_op.get("id") if recovery_op else None,
            recommendations=recovery_strategy.recommendations,
            error_details={
                "failure_analysis": failure_analysis,
                "failed_documents": failed_documents[:10],  # First 10 for brevity
            },
        )

    async def handle_streaming_failure(
        self,
        document_id: str,  # noqa: ARG002
        bytes_processed: int,
        error: Exception,
    ) -> StreamRecoveryAction:
        """Handle failures during streaming processing.

        Args:
            document_id: Document being processed
            bytes_processed: Bytes processed before failure
            error: Exception that occurred

        Returns:
            StreamRecoveryAction with recovery instructions
        """
        error_type = self.classify_error(error)

        if error_type == ChunkingErrorType.MEMORY_ERROR:
            # Reduce batch size and retry from checkpoint
            new_batch_size = self.calculate_reduced_batch_size(error)
            return StreamRecoveryAction(
                action="retry_from_checkpoint",
                checkpoint=bytes_processed,
                new_batch_size=new_batch_size,
            )

        if error_type == ChunkingErrorType.TIMEOUT_ERROR:
            # Extend timeout and retry
            new_timeout = self.calculate_extended_timeout()
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
        error_str = str(error).lower()

        if isinstance(error, MemoryError) or "memory" in error_str:
            return ChunkingErrorType.MEMORY_ERROR

        if isinstance(error, UnicodeError | UnicodeDecodeError) or "encoding" in error_str:
            return ChunkingErrorType.INVALID_ENCODING

        if "permission" in error_str or "access denied" in error_str:
            return ChunkingErrorType.PERMISSION_ERROR

        if "connection" in error_str or "network" in error_str:
            return ChunkingErrorType.NETWORK_ERROR

        if isinstance(error, TimeoutError) or "timeout" in error_str:
            return ChunkingErrorType.TIMEOUT_ERROR

        if "validation" in error_str:
            return ChunkingErrorType.VALIDATION_ERROR

        if "strategy" in error_str or "chunker" in error_str:
            return ChunkingErrorType.STRATEGY_ERROR

        return ChunkingErrorType.UNKNOWN_ERROR

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

        logger.info(f"Retry {current_retries}/{strategy.max_retries} for {error_type.value}, " f"delay: {delay}s")

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
