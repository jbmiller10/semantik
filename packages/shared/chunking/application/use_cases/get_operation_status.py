"""
Get Operation Status Use Case.

Queries the status and progress of chunking operations.
"""

from datetime import datetime
from typing import Any

from ..dto.requests import GetOperationStatusRequest
from ..dto.responses import ChunkDTO, GetOperationStatusResponse, OperationMetrics, OperationStatus
from ...domain.value_objects.operation_status import OperationStatus as DomainOperationStatus
from ..interfaces.repositories import ChunkingOperationRepository, ChunkRepository
from ..interfaces.services import MetricsService


class GetOperationStatusUseCase:
    """
    Use case for querying operation status.

    This use case:
    - Retrieves operation status and progress
    - Optionally includes chunk details
    - Provides performance metrics
    - Returns detailed error information for failed operations
    """

    def __init__(
        self,
        operation_repository: ChunkingOperationRepository,
        chunk_repository: ChunkRepository,
        metrics_service: MetricsService | None = None,
    ):
        """
        Initialize the use case with dependencies.

        Args:
            operation_repository: Repository for operation data
            chunk_repository: Repository for chunk data
            metrics_service: Optional service for metrics
        """
        self.operation_repository = operation_repository
        self.chunk_repository = chunk_repository
        self.metrics_service = metrics_service

    async def execute(self, request: GetOperationStatusRequest) -> GetOperationStatusResponse:
        """
        Execute the status query use case.

        This is a read-only operation that doesn't modify state.

        Args:
            request: Status request with operation ID

        Returns:
            GetOperationStatusResponse with current status

        Raises:
            ValueError: If operation not found
        """
        # 1. Validate request
        request.validate()

        # 2. Find operation
        operation = None
        
        # If document_id is provided, find operations by document_id
        if request.document_id:
            operations = await self.operation_repository.find_by_document_id(request.document_id)
            if operations:
                # Get the most recent operation
                operation = max(operations, key=lambda op: getattr(op, '_created_at', getattr(op, 'created_at', datetime.min)))
        elif request.operation_id:
            operation = await self.operation_repository.find_by_id(request.operation_id)
        
        if not operation:
            identifier = request.document_id if request.document_id else request.operation_id
            raise ValueError(f"Operation not found: {identifier}")

        # 3. Map operation status
        status = self._map_status(operation.status)

        # 4. Calculate progress percentage
        progress_percentage = 0.0
        # Handle potential mock or None values safely
        try:
            total_chunks = getattr(operation, 'total_chunks', None)
            chunks_processed = getattr(operation, 'chunks_processed', 0)
            
            if total_chunks and isinstance(total_chunks, (int, float)) and total_chunks > 0:
                if isinstance(chunks_processed, (int, float)):
                    progress_percentage = (chunks_processed / total_chunks) * 100
            elif chunks_processed and isinstance(chunks_processed, (int, float)) and chunks_processed > 0:
                # Estimate if total not known
                progress_percentage = min(99.0, chunks_processed)  # Cap at 99% if total unknown
        except (TypeError, AttributeError):
            # Handle mock objects or missing attributes gracefully
            progress_percentage = 0.0

        # 5. Load chunks if requested
        chunks = None
        if request.include_chunks:
            chunk_entities = await self.chunk_repository.find_by_operation(request.operation_id)
            chunks = self._map_chunks_to_dtos(chunk_entities)

        # 6. Get metrics if requested and available
        metrics = None
        if request.include_metrics and self.metrics_service:
            metrics_data = await self.metrics_service.get_operation_metrics(request.operation_id)
            if metrics_data:
                metrics = OperationMetrics(
                    chunks_per_second=metrics_data.get("chunks_per_second", 0),
                    avg_chunk_processing_time_ms=metrics_data.get("avg_processing_time_ms", 0),
                    memory_usage_mb=metrics_data.get("memory_usage_mb", 0),
                    checkpoint_recovery_count=metrics_data.get("checkpoint_recoveries", 0),
                    error_count=metrics_data.get("error_count", 0),
                    retry_count=metrics_data.get("retry_count", 0)
                )

        # 7. Build error details if operation failed
        error_details = None
        if status == OperationStatus.FAILED and operation.error_message:
            error_details = {
                "error_type": operation.error_type if hasattr(operation, 'error_type') else "Unknown",
                "error_message": operation.error_message,
                "failed_at": operation.failed_at.isoformat() if hasattr(operation, 'failed_at') else None,
                "last_checkpoint": operation.last_checkpoint if hasattr(operation, 'last_checkpoint') else None,
                "chunks_before_failure": operation.chunks_processed
            }

        # 8. Create and return response
        # Use the actual operation ID from the found operation
        operation_id = getattr(operation, 'id', getattr(operation, 'operation_id', request.operation_id))
        return GetOperationStatusResponse(
            operation_id=operation_id,
            status=status,
            progress_percentage=progress_percentage,
            chunks_processed=getattr(operation, 'chunks_processed', 0),
            total_chunks=getattr(operation, 'total_chunks', None),
            started_at=getattr(operation, 'created_at', getattr(operation, '_created_at', None)),
            updated_at=getattr(operation, 'updated_at', getattr(operation, '_updated_at', None)),
            completed_at=getattr(operation, 'completed_at', getattr(operation, '_completed_at', None)),
            error_message=getattr(operation, 'error_message', None),
            error_details=error_details,
            chunks=chunks,
            metrics=metrics
        )

    def _map_status(self, status_string: str | DomainOperationStatus | OperationStatus) -> OperationStatus:
        """
        Map string status or domain status to DTO enum.

        Args:
            status_string: Status as string, DomainOperationStatus, or OperationStatus enum

        Returns:
            OperationStatus enum value for DTO
        """
        # If already a DTO enum, return it
        if isinstance(status_string, OperationStatus):
            return status_string
        
        # Handle domain enum
        if isinstance(status_string, DomainOperationStatus):
            # Map domain status to DTO status
            domain_to_dto = {
                DomainOperationStatus.PENDING: OperationStatus.PENDING,
                DomainOperationStatus.PROCESSING: OperationStatus.IN_PROGRESS,  # Map PROCESSING to IN_PROGRESS
                DomainOperationStatus.COMPLETED: OperationStatus.COMPLETED,
                DomainOperationStatus.FAILED: OperationStatus.FAILED,
                DomainOperationStatus.CANCELLED: OperationStatus.CANCELLED,
            }
            return domain_to_dto.get(status_string, OperationStatus.PENDING)
        
        # Convert string to enum
        status_mapping = {
            "pending": OperationStatus.PENDING,
            "processing": OperationStatus.IN_PROGRESS,  # Map processing string to IN_PROGRESS
            "in_progress": OperationStatus.IN_PROGRESS,
            "completed": OperationStatus.COMPLETED,
            "failed": OperationStatus.FAILED,
            "cancelled": OperationStatus.CANCELLED,
            "partially_completed": OperationStatus.PARTIALLY_COMPLETED
        }
        
        # Handle string conversion
        if isinstance(status_string, str):
            return status_mapping.get(status_string.lower(), OperationStatus.PENDING)
        
        # Default for unknown types
        return OperationStatus.PENDING

    def _map_chunks_to_dtos(self, chunk_entities: list[Any]) -> list[ChunkDTO]:
        """
        Map chunk entities to DTOs.

        Args:
            chunk_entities: List of chunk entities from repository

        Returns:
            List of ChunkDTO objects
        """
        chunk_dtos = []
        for chunk in chunk_entities:
            # Truncate content for status response
            content = chunk.content
            if len(content) > 200:
                content = content[:200] + "..."

            chunk_dto = ChunkDTO(
                chunk_id=chunk.id,
                content=content,
                position=chunk.position,
                start_offset=chunk.start_offset,
                end_offset=chunk.end_offset,
                token_count=chunk.token_count,
                metadata={
                    "created_at": chunk.created_at.isoformat() if hasattr(chunk, 'created_at') else None,
                    "operation_id": chunk.operation_id,
                    "document_id": chunk.document_id if hasattr(chunk, 'document_id') else None
                }
            )
            chunk_dtos.append(chunk_dto)

        return chunk_dtos
