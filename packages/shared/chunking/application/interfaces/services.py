"""
Service interfaces for chunking application layer.

These interfaces define contracts for infrastructure services that the
application layer depends on.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Self

# Import repository interfaces - forward reference to avoid circular imports
from shared.chunking.application.interfaces.repositories import (
    CheckpointRepository,
    ChunkingOperationRepository,
    ChunkRepository,
    DocumentRepository,
)


class DocumentFormat(str, Enum):
    """Supported document formats."""

    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    DOCX = "docx"
    RTF = "rtf"


class DocumentService(ABC):
    """
    Service interface for document operations.

    Handles document loading, parsing, and content extraction.
    """

    @abstractmethod
    async def load(self, file_path: str, max_size_bytes: int | None = None) -> Any:
        """
        Load a document from file system.

        Args:
            file_path: Path to the document
            max_size_bytes: Optional size limit for loading

        Returns:
            Document domain entity

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is too large or unsupported format
        """

    @abstractmethod
    async def load_partial(self, file_path: str, size_kb: int) -> Any:
        """
        Load a partial document (first N kilobytes).

        Args:
            file_path: Path to the document
            size_kb: Number of kilobytes to load

        Returns:
            Partial document domain entity
        """

    @abstractmethod
    async def extract_text(self, document: Any) -> str:
        """
        Extract plain text from a document.

        Args:
            document: Document entity

        Returns:
            Extracted text content
        """

    @abstractmethod
    async def detect_format(self, file_path: str) -> DocumentFormat:
        """
        Detect the format of a document.

        Args:
            file_path: Path to the document

        Returns:
            Detected document format
        """

    @abstractmethod
    async def get_metadata(self, file_path: str) -> dict[str, Any]:
        """
        Extract metadata from a document.

        Args:
            file_path: Path to the document

        Returns:
            Document metadata (size, creation date, etc.)
        """


class NotificationService(ABC):
    """
    Service interface for notifications and events.

    Handles event publishing and notification dispatch.
    """

    @abstractmethod
    async def notify_operation_started(self, operation_id: str, metadata: dict[str, Any]) -> None:
        """
        Notify that an operation has started.

        Args:
            operation_id: Operation identifier
            metadata: Additional operation metadata
        """

    @abstractmethod
    async def notify_operation_completed(self, operation_id: str, chunks_created: int) -> None:
        """
        Notify that an operation has completed.

        Args:
            operation_id: Operation identifier
            chunks_created: Number of chunks created
        """

    @abstractmethod
    async def notify_operation_failed(self, operation_id: str, error: Exception) -> None:
        """
        Notify that an operation has failed.

        Args:
            operation_id: Operation identifier
            error: The exception that caused failure
        """

    @abstractmethod
    async def notify_operation_cancelled(self, operation_id: str, reason: str | None) -> None:
        """
        Notify that an operation was cancelled.

        Args:
            operation_id: Operation identifier
            reason: Cancellation reason
        """

    @abstractmethod
    async def notify_progress(self, operation_id: str, progress_percentage: float) -> None:
        """
        Notify operation progress update.

        Args:
            operation_id: Operation identifier
            progress_percentage: Progress from 0 to 100
        """

    @abstractmethod
    async def notify_error(self, error: Exception, context: dict[str, Any] | None = None) -> None:
        """
        Notify about a general error.

        Args:
            error: The exception
            context: Additional error context
        """


class ChunkingStrategyFactory(ABC):
    """
    Factory interface for creating chunking strategies.

    Creates appropriate strategy instances based on configuration.
    """

    @abstractmethod
    def create_strategy(self, strategy_type: str, config: dict[str, Any]) -> Any:
        """
        Create a chunking strategy instance.

        Args:
            strategy_type: Type of strategy to create
            config: Strategy configuration

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy type is unknown
        """

    @abstractmethod
    def get_available_strategies(self) -> list[str]:
        """
        Get list of available strategy types.

        Returns:
            List of strategy type names
        """

    @abstractmethod
    def get_default_config(self, strategy_type: str) -> dict[str, Any]:
        """
        Get default configuration for a strategy.

        Args:
            strategy_type: Strategy type

        Returns:
            Default configuration dictionary
        """


class MetricsService(ABC):
    """
    Service interface for metrics collection.

    Collects and reports performance metrics.
    """

    @abstractmethod
    async def record_operation_duration(self, operation_id: str, duration_ms: float) -> None:
        """
        Record operation duration.

        Args:
            operation_id: Operation identifier
            duration_ms: Duration in milliseconds
        """

    @abstractmethod
    async def record_chunk_processing_time(self, operation_id: str, chunk_id: str, duration_ms: float) -> None:
        """
        Record individual chunk processing time.

        Args:
            operation_id: Operation identifier
            chunk_id: Chunk identifier
            duration_ms: Processing time in milliseconds
        """

    @abstractmethod
    async def record_memory_usage(self, operation_id: str, memory_mb: float) -> None:
        """
        Record memory usage for an operation.

        Args:
            operation_id: Operation identifier
            memory_mb: Memory usage in megabytes
        """

    @abstractmethod
    async def get_operation_metrics(self, operation_id: str) -> dict[str, Any]:
        """
        Get collected metrics for an operation.

        Args:
            operation_id: Operation identifier

        Returns:
            Dictionary of metrics
        """

    @abstractmethod
    async def record_strategy_performance(
        self, strategy_type: str, document_size: int, chunks_created: int, duration_ms: float
    ) -> None:
        """
        Record strategy performance metrics.

        Args:
            strategy_type: Type of chunking strategy
            document_size: Size of document in bytes
            chunks_created: Number of chunks created
            duration_ms: Processing duration
        """


class UnitOfWork(ABC):
    """
    Unit of Work pattern interface for transaction management.

    Ensures transactional consistency across repositories.
    """

    @abstractmethod
    async def __aenter__(self) -> Self:
        """Begin a unit of work (transaction)."""

    @abstractmethod
    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """End unit of work (commit or rollback)."""

    @abstractmethod
    async def commit(self) -> None:
        """Commit the current transaction."""

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback the current transaction."""

    @property
    @abstractmethod
    def chunks(self) -> ChunkRepository:
        """Get chunk repository within this unit of work."""

    @property
    @abstractmethod
    def operations(self) -> ChunkingOperationRepository:
        """Get operation repository within this unit of work."""

    @property
    @abstractmethod
    def checkpoints(self) -> CheckpointRepository:
        """Get checkpoint repository within this unit of work."""

    @property
    @abstractmethod
    def documents(self) -> DocumentRepository:
        """Get document repository within this unit of work."""
