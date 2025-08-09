"""
Get Operation Status Use Case.

Queries the status and progress of chunking operations.
"""

from typing import Any

from ..dto.requests import GetOperationStatusRequest
from ..dto.responses import ChunkDTO, GetOperationStatusResponse, OperationMetrics, OperationStatus
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
        metrics_service: MetricsService | None = None
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
        operation = await self.operation_repository.find_by_id(request.operation_id)
        if not operation:
            raise ValueError(f"Operation not found: {request.operation_id}")

        # 3. Map operation status
        status = self._map_status(operation.status)

        # 4. Calculate progress percentage
        progress_percentage = 0.0
        if operation.total_chunks and operation.total_chunks > 0:
            progress_percentage = (operation.chunks_processed / operation.total_chunks) * 100
        elif operation.chunks_processed > 0:
            # Estimate if total not known
            progress_percentage = min(99.0, operation.chunks_processed)  # Cap at 99% if total unknown

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
        return GetOperationStatusResponse(
            operation_id=request.operation_id,
            status=status,
            progress_percentage=progress_percentage,
            chunks_processed=operation.chunks_processed,
            total_chunks=operation.total_chunks,
            started_at=operation.created_at,
            updated_at=operation.updated_at,
            completed_at=operation.completed_at if hasattr(operation, 'completed_at') else None,
            error_message=operation.error_message if hasattr(operation, 'error_message') else None,
            error_details=error_details,
            chunks=chunks,
            metrics=metrics
        )

    def _map_status(self, status_string: str) -> OperationStatus:
        """
        Map string status to enum.

        Args:
            status_string: Status as string

        Returns:
            OperationStatus enum value
        """
        status_mapping = {
            "pending": OperationStatus.PENDING,
            "in_progress": OperationStatus.IN_PROGRESS,
            "completed": OperationStatus.COMPLETED,
            "failed": OperationStatus.FAILED,
            "cancelled": OperationStatus.CANCELLED,
            "partially_completed": OperationStatus.PARTIALLY_COMPLETED
        }
        return status_mapping.get(status_string.lower(), OperationStatus.PENDING)

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
