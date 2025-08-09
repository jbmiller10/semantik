"""
Repository interfaces for chunking application layer.

These interfaces define contracts that infrastructure implementations must follow.
They provide abstraction from specific database technologies.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class ChunkRepository(ABC):
    """
    Repository interface for chunk persistence.

    This interface abstracts chunk storage operations, allowing
    infrastructure layer to implement using any storage mechanism.
    """

    @abstractmethod
    async def save(self, chunk: Any) -> str:
        """
        Save a single chunk to storage.

        Args:
            chunk: Domain chunk entity to save

        Returns:
            The ID of the saved chunk
        """

    @abstractmethod
    async def save_batch(self, chunks: list[Any]) -> list[str]:
        """
        Save multiple chunks in a batch operation.

        Args:
            chunks: List of domain chunk entities

        Returns:
            List of IDs for saved chunks
        """

    @abstractmethod
    async def find_by_id(self, chunk_id: str) -> Any | None:
        """
        Find a chunk by its ID.

        Args:
            chunk_id: Unique identifier of the chunk

        Returns:
            Chunk entity if found, None otherwise
        """

    @abstractmethod
    async def find_by_operation(self, operation_id: str) -> list[Any]:
        """
        Find all chunks associated with an operation.

        Args:
            operation_id: ID of the chunking operation

        Returns:
            List of chunks for the operation
        """

    @abstractmethod
    async def find_by_document(self, document_id: str) -> list[Any]:
        """
        Find all chunks for a document.

        Args:
            document_id: ID of the document

        Returns:
            List of chunks for the document
        """

    @abstractmethod
    async def delete_by_operation(self, operation_id: str) -> int:
        """
        Delete all chunks for an operation.

        Args:
            operation_id: ID of the operation

        Returns:
            Number of chunks deleted
        """

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: ID of the document

        Returns:
            Number of chunks deleted
        """

    @abstractmethod
    async def count_by_operation(self, operation_id: str) -> int:
        """
        Count chunks for an operation.

        Args:
            operation_id: ID of the operation

        Returns:
            Number of chunks
        """


class ChunkingOperationRepository(ABC):
    """
    Repository interface for chunking operation persistence.

    Manages the lifecycle and state of chunking operations.
    """

    @abstractmethod
    async def create(self, operation: Any) -> str:
        """
        Create a new chunking operation.

        Args:
            operation: Domain operation entity

        Returns:
            ID of created operation
        """

    @abstractmethod
    async def find_by_id(self, operation_id: str) -> Any | None:
        """
        Find an operation by ID.

        Args:
            operation_id: Unique identifier

        Returns:
            Operation entity if found
        """

    @abstractmethod
    async def update_status(self, operation_id: str, status: str,
                          error_message: str | None = None) -> None:
        """
        Update operation status.

        Args:
            operation_id: Operation to update
            status: New status value
            error_message: Optional error message if failed
        """

    @abstractmethod
    async def update_progress(self, operation_id: str, chunks_processed: int,
                            total_chunks: int | None = None) -> None:
        """
        Update operation progress.

        Args:
            operation_id: Operation to update
            chunks_processed: Number of chunks processed
            total_chunks: Total number of chunks (if known)
        """

    @abstractmethod
    async def find_active_operations(self) -> list[Any]:
        """
        Find all active (in-progress) operations.

        Returns:
            List of active operations
        """

    @abstractmethod
    async def find_by_document(self, document_id: str) -> list[Any]:
        """
        Find all operations for a document.

        Args:
            document_id: Document identifier

        Returns:
            List of operations
        """

    @abstractmethod
    async def mark_completed(self, operation_id: str,
                           completed_at: datetime) -> None:
        """
        Mark an operation as completed.

        Args:
            operation_id: Operation to complete
            completed_at: Completion timestamp
        """

    @abstractmethod
    async def mark_cancelled(self, operation_id: str, reason: str | None = None) -> None:
        """
        Mark an operation as cancelled.

        Args:
            operation_id: Operation to cancel
            reason: Optional cancellation reason
        """


class CheckpointRepository(ABC):
    """
    Repository interface for operation checkpoints.

    Manages checkpoints for resumable chunking operations.
    """

    @abstractmethod
    async def save_checkpoint(self, operation_id: str, position: int,
                             state: dict[str, Any]) -> None:
        """
        Save a checkpoint for an operation.

        Args:
            operation_id: Operation identifier
            position: Current processing position
            state: Serializable state data
        """

    @abstractmethod
    async def get_latest_checkpoint(self, operation_id: str) -> dict[str, Any] | None:
        """
        Get the latest checkpoint for an operation.

        Args:
            operation_id: Operation identifier

        Returns:
            Checkpoint data if exists
        """

    @abstractmethod
    async def delete_checkpoints(self, operation_id: str) -> int:
        """
        Delete all checkpoints for an operation.

        Args:
            operation_id: Operation identifier

        Returns:
            Number of checkpoints deleted
        """

    @abstractmethod
    async def count_checkpoints(self, operation_id: str) -> int:
        """
        Count checkpoints for an operation.

        Args:
            operation_id: Operation identifier

        Returns:
            Number of checkpoints
        """


class DocumentRepository(ABC):
    """
    Repository interface for document metadata.

    Manages document metadata and relationships.
    """

    @abstractmethod
    async def find_by_id(self, document_id: str) -> Any | None:
        """
        Find a document by ID.

        Args:
            document_id: Document identifier

        Returns:
            Document entity if found
        """

    @abstractmethod
    async def find_by_path(self, file_path: str) -> Any | None:
        """
        Find a document by file path.

        Args:
            file_path: Path to the document file

        Returns:
            Document entity if found
        """

    @abstractmethod
    async def create(self, document: Any) -> str:
        """
        Create a new document record.

        Args:
            document: Document entity

        Returns:
            ID of created document
        """

    @abstractmethod
    async def update_chunking_status(self, document_id: str,
                                    status: str) -> None:
        """
        Update document chunking status.

        Args:
            document_id: Document to update
            status: New chunking status
        """

    @abstractmethod
    async def get_or_create(self, file_path: str,
                          metadata: dict[str, Any]) -> Any:
        """
        Get existing document or create new one.

        Args:
            file_path: Path to document
            metadata: Document metadata

        Returns:
            Document entity
        """
