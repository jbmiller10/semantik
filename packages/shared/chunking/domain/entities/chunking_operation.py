#!/usr/bin/env python3
"""
ChunkingOperation entity representing the core domain aggregate.

This module defines the main chunking operation entity that orchestrates
the chunking process and maintains operation-level invariants.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from shared.chunking.domain.entities.chunk import Chunk
from shared.chunking.domain.entities.chunk_collection import ChunkCollection
from shared.chunking.domain.exceptions import DocumentTooLargeError, InvalidStateError
from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.domain.value_objects.operation_status import OperationStatus

if TYPE_CHECKING:
    from shared.chunking.domain.services.chunking_strategies.base import ChunkingStrategy


class ChunkingOperation:
    """
    Core domain entity representing a chunking operation.

    This is the aggregate root that orchestrates the chunking process,
    manages state transitions, and enforces business rules.
    """

    # Business rule constants
    MAX_DOCUMENT_SIZE = 10_000_000  # 10MB character limit
    MAX_CHUNKS_PER_OPERATION = 10_000  # Maximum chunks allowed
    MAX_OPERATION_DURATION_SECONDS = 300  # 5 minute timeout

    def __init__(
        self,
        operation_id: str,
        document_id: str,
        document_content: str,
        config: ChunkConfig,
    ) -> None:
        """
        Initialize a chunking operation.

        Args:
            operation_id: Unique identifier for the operation
            document_id: ID of the document being chunked
            document_content: The text content to chunk
            config: Configuration for chunking

        Raises:
            DocumentTooLargeError: If document exceeds size limits
        """
        self._validate_document_size(document_content)

        self._id = operation_id
        self._document_id = document_id
        self._document_content = document_content
        self._config = config

        # Operation state
        self._status = OperationStatus.PENDING
        self._chunk_collection = ChunkCollection(document_id, document_content)

        # Timing information
        self._created_at = datetime.now(tz=UTC)
        self._started_at: datetime | None = None
        self._completed_at: datetime | None = None

        # Progress tracking
        self._progress_percentage: float = 0.0
        self._estimated_total_chunks: int = self._estimate_chunks()

        # Error information
        self._error_message: str | None = None
        self._error_details: dict[str, Any] | None = None

        # Performance metrics
        self._metrics: dict[str, Any] = {}

    @property
    def id(self) -> str:
        """Get the operation ID."""
        return self._id

    @property
    def document_id(self) -> str:
        """Get the document ID."""
        return self._document_id

    @property
    def status(self) -> OperationStatus:
        """Get the current operation status."""
        return self._status

    @property
    def config(self) -> ChunkConfig:
        """Get the chunking configuration."""
        return self._config

    @property
    def chunk_collection(self) -> ChunkCollection:
        """Get the chunk collection."""
        return self._chunk_collection

    @property
    def progress_percentage(self) -> float:
        """Get the progress percentage (0-100)."""
        return self._progress_percentage

    @property
    def error_message(self) -> str | None:
        """Get the error message if operation failed."""
        return self._error_message

    @property
    def metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        return self._metrics.copy()

    def start(self) -> None:
        """
        Start the chunking operation.

        Raises:
            InvalidStateError: If operation cannot be started from current state
        """
        if not self._status.can_transition_to(OperationStatus.PROCESSING):
            raise InvalidStateError(f"Cannot start operation in {self._status.value} state")

        self._status = OperationStatus.PROCESSING
        self._started_at = datetime.now(tz=UTC)
        self._progress_percentage = 0.0

    def execute(self, strategy: "ChunkingStrategy") -> None:
        """
        Execute the chunking using the provided strategy.

        Args:
            strategy: The chunking strategy to use

        Raises:
            InvalidStateError: If operation is not in PROCESSING state
        """
        if self._status != OperationStatus.PROCESSING:
            raise InvalidStateError(f"Cannot execute operation in {self._status.value} state")

        start_time = datetime.now(tz=UTC)

        try:
            # Perform the chunking
            chunks = strategy.chunk(
                self._document_content,
                self._config,
                progress_callback=self._update_progress,
            )

            # Validate chunk count
            if len(chunks) > self.MAX_CHUNKS_PER_OPERATION:
                raise InvalidStateError(
                    f"Operation produced {len(chunks)} chunks, exceeding limit of {self.MAX_CHUNKS_PER_OPERATION}"
                )

            # Add chunks to collection
            for chunk in chunks:
                self._chunk_collection.add_chunk(chunk)

            # Update metrics
            end_time = datetime.now(tz=UTC)
            self._update_metrics(start_time, end_time, len(chunks))

            # Mark as completed
            self._complete()

        except Exception as e:
            self._fail(str(e), {"exception_type": type(e).__name__})
            raise

    def add_chunk(self, chunk: Chunk) -> None:
        """
        Add a chunk to the operation.

        Args:
            chunk: The chunk to add

        Raises:
            InvalidStateError: If operation is not in PROCESSING state
        """
        if self._status != OperationStatus.PROCESSING:
            raise InvalidStateError(f"Cannot add chunks to operation in {self._status.value} state")

        self._chunk_collection.add_chunk(chunk)

        # Update progress based on chunks added
        if self._estimated_total_chunks > 0:
            self._progress_percentage = min(
                100.0, (self._chunk_collection.chunk_count / self._estimated_total_chunks) * 100
            )

    def cancel(self, reason: str | None = None) -> None:
        """
        Cancel the operation.

        Args:
            reason: Optional cancellation reason

        Raises:
            InvalidStateError: If operation cannot be cancelled from current state
        """
        if not self._status.can_transition_to(OperationStatus.CANCELLED):
            raise InvalidStateError(f"Cannot cancel operation in {self._status.value} state")

        self._status = OperationStatus.CANCELLED
        self._completed_at = datetime.now(tz=UTC)
        self._error_message = reason or "Operation cancelled by user"

    def validate_results(self) -> tuple[bool, list[str]]:
        """
        Validate the chunking results.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check if we have chunks
        if self._chunk_collection.chunk_count == 0:
            issues.append("No chunks were produced")

        # Validate collection completeness
        collection_valid, collection_issues = self._chunk_collection.validate_completeness()
        if not collection_valid:
            issues.extend(collection_issues)

        # Check coverage
        coverage = self._chunk_collection.calculate_coverage()
        if coverage < 0.9:  # Require at least 90% coverage
            issues.append(f"Insufficient coverage: {coverage:.1%}")

        # Check for timeout
        if self._started_at:
            duration = (datetime.now(tz=UTC) - self._started_at).total_seconds()
            if duration > self.MAX_OPERATION_DURATION_SECONDS:
                issues.append(f"Operation exceeded timeout: {duration:.1f}s")

        return len(issues) == 0, issues

    def get_statistics(self) -> dict[str, Any]:
        """
        Get comprehensive statistics about the operation.

        Returns:
            Dictionary with operation statistics
        """
        stats: dict[str, Any] = {
            "operation_id": self._id,
            "document_id": self._document_id,
            "status": self._status.value,
            "config": {
                "strategy": self._config.strategy_name,
                "min_tokens": self._config.min_tokens,
                "max_tokens": self._config.max_tokens,
                "overlap_tokens": self._config.overlap_tokens,
            },
            "chunks": {
                "total": self._chunk_collection.chunk_count,
                "estimated": self._estimated_total_chunks,
            },
            "coverage": self._chunk_collection.calculate_coverage(),
            "progress": self._progress_percentage,
        }

        # Add timing information
        if self._started_at:
            timing_info: dict[str, Any] = {
                "started_at": self._started_at.isoformat(),
                "duration_seconds": self._calculate_duration(),
            }
            if self._completed_at:
                timing_info["completed_at"] = self._completed_at.isoformat()
            stats["timing"] = timing_info

        # Add chunk statistics
        stats["chunk_stats"] = self._chunk_collection.calculate_size_statistics()
        stats["overlap_stats"] = self._chunk_collection.calculate_overlap_statistics()

        # Add metrics
        if self._metrics:
            stats["metrics"] = self._metrics

        # Add error information if failed
        if self._status == OperationStatus.FAILED:
            stats["error"] = {
                "message": self._error_message,
                "details": self._error_details,
            }

        return stats

    def _complete(self) -> None:
        """Mark the operation as completed."""
        if not self._status.can_transition_to(OperationStatus.COMPLETED):
            raise InvalidStateError(f"Cannot complete operation in {self._status.value} state")

        self._status = OperationStatus.COMPLETED
        self._completed_at = datetime.now(tz=UTC)
        self._progress_percentage = 100.0

    def _fail(self, error_message: str, error_details: dict[str, Any] | None = None) -> None:
        """
        Mark the operation as failed.

        Args:
            error_message: The error message
            error_details: Optional error details
        """
        if not self._status.can_transition_to(OperationStatus.FAILED):
            raise InvalidStateError(f"Cannot fail operation in {self._status.value} state")

        self._status = OperationStatus.FAILED
        self._completed_at = datetime.now(tz=UTC)
        self._error_message = error_message
        self._error_details = error_details or {}

    def _update_progress(self, percentage: float) -> None:
        """
        Update operation progress.

        Args:
            percentage: Progress percentage (0-100)
        """
        self._progress_percentage = max(0.0, min(100.0, percentage))

    def _update_metrics(self, start_time: datetime, end_time: datetime, chunk_count: int) -> None:
        """
        Update performance metrics.

        Args:
            start_time: When chunking started
            end_time: When chunking ended
            chunk_count: Number of chunks produced
        """
        duration = (end_time - start_time).total_seconds()

        self._metrics = {
            "duration_seconds": duration,
            "chunks_per_second": chunk_count / duration if duration > 0 else 0,
            "characters_per_second": len(self._document_content) / duration if duration > 0 else 0,
            "average_chunk_size": len(self._document_content) / chunk_count if chunk_count > 0 else 0,
        }

    def _estimate_chunks(self) -> int:
        """
        Estimate the number of chunks that will be produced.

        Returns:
            Estimated chunk count
        """
        # Estimate tokens in document (rough approximation)
        estimated_tokens = len(self._document_content) // 4  # ~4 chars per token

        result: int = self._config.estimate_chunks(estimated_tokens)
        return result

    def _calculate_duration(self) -> float:
        """
        Calculate operation duration in seconds.

        Returns:
            Duration in seconds, or 0 if not started
        """
        if not self._started_at:
            return 0.0

        end_time = self._completed_at or datetime.now(tz=UTC)
        return (end_time - self._started_at).total_seconds()

    def _validate_document_size(self, content: str) -> None:
        """
        Validate document size meets constraints.

        Args:
            content: Document content to validate

        Raises:
            DocumentTooLargeError: If document exceeds size limits
        """
        if len(content) > self.MAX_DOCUMENT_SIZE:
            raise DocumentTooLargeError(len(content), self.MAX_DOCUMENT_SIZE)

    def __repr__(self) -> str:
        """String representation of the operation."""
        return (
            f"ChunkingOperation(id={self._id}, "
            f"status={self._status.value}, "
            f"chunks={self._chunk_collection.chunk_count}, "
            f"progress={self._progress_percentage:.1f}%)"
        )
