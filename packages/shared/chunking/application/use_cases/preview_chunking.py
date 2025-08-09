"""
Preview Chunking Use Case.

Generates a preview of chunking results by processing a sample of the document.
This use case is designed for quick feedback without full processing overhead.
"""

import time
from uuid import uuid4

from ..dto.requests import PreviewRequest
from ..dto.responses import ChunkDTO, PreviewResponse
from ..interfaces.services import ChunkingStrategyFactory, DocumentService, MetricsService, NotificationService


class PreviewChunkingUseCase:
    """
    Use case for generating chunking preview.

    This use case:
    - Loads a sample of the document (first N KB)
    - Applies the selected chunking strategy
    - Returns a preview of the first chunks
    - Estimates total chunks for the full document
    """

    def __init__(
        self,
        document_service: DocumentService,
        strategy_factory: ChunkingStrategyFactory,
        notification_service: NotificationService,
        metrics_service: MetricsService | None = None
    ):
        """
        Initialize the use case with dependencies.

        Args:
            document_service: Service for document operations
            strategy_factory: Factory for creating chunking strategies
            notification_service: Service for notifications
            metrics_service: Optional service for metrics collection
        """
        self.document_service = document_service
        self.strategy_factory = strategy_factory
        self.notification_service = notification_service
        self.metrics_service = metrics_service

    async def execute(self, request: PreviewRequest) -> PreviewResponse:
        """
        Execute the preview chunking use case.

        This is the single public method that orchestrates the entire use case.

        Args:
            request: Preview request DTO with parameters

        Returns:
            PreviewResponse with sample chunks and estimates

        Raises:
            ValueError: If request validation fails
            FileNotFoundError: If document file doesn't exist
            Exception: For other infrastructure failures
        """
        start_time = time.time()
        operation_id = str(uuid4())

        try:
            # 1. Validate request parameters
            request.validate()

            # 2. Notify operation started
            await self.notification_service.notify_operation_started(
                operation_id=operation_id,
                metadata={
                    "type": "preview",
                    "file_path": request.file_path,
                    "strategy": request.strategy_type.value
                }
            )

            # 3. Load partial document (infrastructure)
            document = await self.document_service.load_partial(
                file_path=request.file_path,
                size_kb=request.preview_size_kb
            )

            # 4. Extract text content
            text_content = await self.document_service.extract_text(document)
            sample_size = len(text_content.encode('utf-8'))

            # 5. Get full document metadata for estimation
            full_metadata = await self.document_service.get_metadata(request.file_path)
            full_size = full_metadata.get("size_bytes", sample_size)

            # 6. Create and configure chunking strategy
            strategy_config = {
                "min_tokens": request.min_tokens,
                "max_tokens": request.max_tokens,
                "overlap": request.overlap
            }
            strategy = self.strategy_factory.create_strategy(
                strategy_type=request.strategy_type.value,
                config=strategy_config
            )

            # 7. Apply chunking strategy (domain logic)
            # Note: We assume the strategy has a chunk() method
            # The actual domain entity would be created in the domain layer
            chunks = strategy.chunk(text_content)

            # 8. Calculate estimates
            chunks_in_sample = len(chunks)
            size_ratio = full_size / sample_size if sample_size > 0 else 1
            total_chunks_estimate = int(chunks_in_sample * size_ratio)

            # 9. Take preview sample (first N chunks)
            preview_chunks = chunks[:request.max_preview_chunks]

            # 10. Map chunks to DTOs
            chunk_dtos = []
            for i, chunk in enumerate(preview_chunks):
                chunk_dto = ChunkDTO(
                    chunk_id=f"{operation_id}-preview-{i}",
                    content=chunk.content,
                    position=i,
                    start_offset=chunk.metadata.start_offset,
                    end_offset=chunk.metadata.end_offset,
                    token_count=chunk.metadata.token_count,
                    metadata={
                        "preview": True,
                        "strategy": request.strategy_type.value
                    }
                )
                chunk_dtos.append(chunk_dto)

            # 11. Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # 12. Record metrics if service available
            if self.metrics_service:
                await self.metrics_service.record_operation_duration(
                    operation_id=operation_id,
                    duration_ms=processing_time_ms
                )
                await self.metrics_service.record_strategy_performance(
                    strategy_type=request.strategy_type.value,
                    document_size=sample_size,
                    chunks_created=len(preview_chunks),
                    duration_ms=processing_time_ms
                )

            # 13. Notify completion
            await self.notification_service.notify_operation_completed(
                operation_id=operation_id,
                chunks_created=len(preview_chunks)
            )

            # 14. Return response DTO
            return PreviewResponse(
                operation_id=operation_id,
                chunks=chunk_dtos,
                total_chunks_estimate=total_chunks_estimate,
                strategy_used=request.strategy_type.value,
                document_sample_size=sample_size,
                processing_time_ms=processing_time_ms
            )

        except ValueError as e:
            # Domain validation errors
            await self.notification_service.notify_operation_failed(
                operation_id=operation_id,
                error=e
            )
            raise ValueError(f"Preview failed - validation error: {e}")

        except FileNotFoundError as e:
            # Document not found
            await self.notification_service.notify_operation_failed(
                operation_id=operation_id,
                error=e
            )
            raise FileNotFoundError(f"Document not found: {request.file_path}")

        except Exception as e:
            # Infrastructure or unexpected errors
            await self.notification_service.notify_error(
                error=e,
                context={
                    "operation_id": operation_id,
                    "use_case": "preview_chunking",
                    "file_path": request.file_path
                }
            )
            raise Exception(f"Preview operation failed: {e}") from e
