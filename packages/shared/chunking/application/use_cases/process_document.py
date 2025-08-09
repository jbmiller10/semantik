"""
Process Document Use Case.

Handles full document processing with chunking, persistence, and checkpointing.
This is the main use case for production document processing.
"""

import time
from datetime import datetime
from typing import Any
from uuid import uuid4

from ..dto.requests import ProcessDocumentRequest
from ..dto.responses import OperationStatus, ProcessDocumentResponse
from ..interfaces.services import (
    ChunkingStrategyFactory,
    DocumentService,
    MetricsService,
    NotificationService,
    UnitOfWork,
)


class ProcessDocumentUseCase:
    """
    Use case for full document processing.

    This use case:
    - Processes entire documents
    - Saves chunks to persistent storage
    - Manages checkpoints for resumability
    - Handles transactions properly
    - Emits progress events
    """

    def __init__(
        self,
        unit_of_work: UnitOfWork,
        document_service: DocumentService,
        strategy_factory: ChunkingStrategyFactory,
        notification_service: NotificationService,
        metrics_service: MetricsService | None = None
    ):
        """
        Initialize the use case with dependencies.

        Args:
            unit_of_work: UoW for transaction management
            document_service: Service for document operations
            strategy_factory: Factory for creating chunking strategies
            notification_service: Service for notifications
            metrics_service: Optional service for metrics
        """
        self.unit_of_work = unit_of_work
        self.document_service = document_service
        self.strategy_factory = strategy_factory
        self.notification_service = notification_service
        self.metrics_service = metrics_service

    async def execute(self, request: ProcessDocumentRequest) -> ProcessDocumentResponse:
        """
        Execute the document processing use case.

        Manages the entire document processing lifecycle with proper
        transaction boundaries and error handling.

        Args:
            request: Process request DTO with parameters

        Returns:
            ProcessDocumentResponse with operation results

        Raises:
            ValueError: If validation fails
            FileNotFoundError: If document doesn't exist
            Exception: For processing failures
        """
        operation_id = str(uuid4())
        processing_started_at = datetime.utcnow()
        chunks_saved = 0
        checkpoints_created = 0

        # Transaction boundary starts here
        async with self.unit_of_work:
            try:
                # 1. Validate request
                request.validate()

                # 2. Create or get document record
                document_entity = await self.unit_of_work.documents.get_or_create(
                    file_path=request.file_path,
                    metadata=request.metadata or {}
                )

                # 3. Check for existing operations on this document
                existing_ops = await self.unit_of_work.operations.find_by_document(
                    request.document_id
                )
                active_ops = [op for op in existing_ops if op.status == "in_progress"]
                if active_ops:
                    raise ValueError(f"Document already being processed: {request.document_id}")

                # 4. Create chunking operation record
                operation = self._create_operation_entity(
                    operation_id=operation_id,
                    document_id=request.document_id,
                    collection_id=request.collection_id,
                    strategy_type=request.strategy_type.value
                )
                await self.unit_of_work.operations.create(operation)

                # 5. Check for checkpoint (resuming previous operation)
                checkpoint = None
                if request.enable_checkpointing:
                    checkpoint = await self.unit_of_work.checkpoints.get_latest_checkpoint(
                        operation_id
                    )

                # 6. Notify operation started
                await self.notification_service.notify_operation_started(
                    operation_id=operation_id,
                    metadata={
                        "document_id": request.document_id,
                        "collection_id": request.collection_id,
                        "strategy": request.strategy_type.value,
                        "resuming": checkpoint is not None
                    }
                )

                # 7. Load full document
                document = await self.document_service.load(request.file_path)
                text_content = await self.document_service.extract_text(document)

                # 8. Create and configure chunking strategy
                strategy_config = {
                    "min_tokens": request.min_tokens,
                    "max_tokens": request.max_tokens,
                    "overlap": request.overlap
                }
                strategy = self.strategy_factory.create_strategy(
                    strategy_type=request.strategy_type.value,
                    config=strategy_config
                )

                # 9. Apply chunking strategy
                chunks = strategy.chunk(text_content)
                total_chunks = len(chunks)

                # 10. Update operation with total chunks
                await self.unit_of_work.operations.update_progress(
                    operation_id=operation_id,
                    chunks_processed=0,
                    total_chunks=total_chunks
                )

                # 11. Process and save chunks with checkpointing
                start_position = 0
                if checkpoint:
                    start_position = checkpoint.get("position", 0)
                    chunks_saved = checkpoint.get("chunks_saved", 0)

                batch_size = 10  # Process chunks in batches
                for i in range(start_position, total_chunks, batch_size):
                    batch_end = min(i + batch_size, total_chunks)
                    batch = chunks[i:batch_end]

                    # Save batch of chunks
                    chunk_entities = []
                    for j, chunk in enumerate(batch):
                        chunk_entity = self._create_chunk_entity(
                            chunk=chunk,
                            operation_id=operation_id,
                            document_id=request.document_id,
                            collection_id=request.collection_id,
                            position=i + j
                        )
                        chunk_entities.append(chunk_entity)

                    await self.unit_of_work.chunks.save_batch(chunk_entities)
                    chunks_saved += len(batch)

                    # Update progress
                    progress_percentage = (chunks_saved / total_chunks) * 100
                    await self.unit_of_work.operations.update_progress(
                        operation_id=operation_id,
                        chunks_processed=chunks_saved,
                        total_chunks=total_chunks
                    )

                    # Notify progress
                    await self.notification_service.notify_progress(
                        operation_id=operation_id,
                        progress_percentage=progress_percentage
                    )

                    # Create checkpoint if enabled
                    if request.enable_checkpointing and (i + batch_size) % request.checkpoint_interval == 0:
                        await self.unit_of_work.checkpoints.save_checkpoint(
                            operation_id=operation_id,
                            position=batch_end,
                            state={
                                "chunks_saved": chunks_saved,
                                "last_chunk_id": chunk_entities[-1].id if chunk_entities else None
                            }
                        )
                        checkpoints_created += 1

                    # Record metrics for batch
                    if self.metrics_service:
                        batch_time = time.time()
                        for chunk_entity in chunk_entities:
                            await self.metrics_service.record_chunk_processing_time(
                                operation_id=operation_id,
                                chunk_id=chunk_entity.id,
                                duration_ms=10  # Placeholder - would be actual time
                            )

                # 12. Mark operation as completed
                processing_completed_at = datetime.utcnow()
                await self.unit_of_work.operations.mark_completed(
                    operation_id=operation_id,
                    completed_at=processing_completed_at
                )

                # 13. Update document chunking status
                await self.unit_of_work.documents.update_chunking_status(
                    document_id=request.document_id,
                    status="completed"
                )

                # 14. Clean up checkpoints after successful completion
                if request.enable_checkpointing:
                    await self.unit_of_work.checkpoints.delete_checkpoints(operation_id)

                # Commit transaction
                await self.unit_of_work.commit()

                # 15. Send completion notification (after commit)
                await self.notification_service.notify_operation_completed(
                    operation_id=operation_id,
                    chunks_created=chunks_saved
                )

                # 16. Record final metrics
                if self.metrics_service:
                    total_time_ms = (processing_completed_at - processing_started_at).total_seconds() * 1000
                    await self.metrics_service.record_operation_duration(
                        operation_id=operation_id,
                        duration_ms=total_time_ms
                    )
                    await self.metrics_service.record_strategy_performance(
                        strategy_type=request.strategy_type.value,
                        document_size=len(text_content.encode('utf-8')),
                        chunks_created=chunks_saved,
                        duration_ms=total_time_ms
                    )

                # 17. Return response
                return ProcessDocumentResponse(
                    operation_id=operation_id,
                    document_id=request.document_id,
                    collection_id=request.collection_id,
                    status=OperationStatus.COMPLETED,
                    total_chunks=total_chunks,
                    chunks_processed=chunks_saved,
                    chunks_saved=chunks_saved,
                    processing_started_at=processing_started_at,
                    processing_completed_at=processing_completed_at,
                    error_message=None,
                    checkpoints_created=checkpoints_created
                )

            except Exception as e:
                # Rollback transaction on error
                await self.unit_of_work.rollback()

                # Try to update operation status in a new transaction
                # This needs to be done in a separate transaction since the current one was rolled back
                try:
                    async with self.unit_of_work:
                        await self.unit_of_work.operations.update_status(
                            operation_id=operation_id,
                            status="failed",
                            error_message=str(e)
                        )
                        await self.unit_of_work.commit()
                except Exception as update_error:
                    # If we can't update the operation status, log it but continue
                    # The operation record might not have been created yet if the error occurred early
                    await self.notification_service.notify_error(
                        error=update_error,
                        context={
                            "operation_id": operation_id,
                            "original_error": str(e),
                            "context": "Failed to update operation status after error"
                        }
                    )

                # Notify failure
                await self.notification_service.notify_operation_failed(
                    operation_id=operation_id,
                    error=e
                )

                # Log error
                await self.notification_service.notify_error(
                    error=e,
                    context={
                        "operation_id": operation_id,
                        "document_id": request.document_id,
                        "use_case": "process_document"
                    }
                )

                # Return failure response
                return ProcessDocumentResponse(
                    operation_id=operation_id,
                    document_id=request.document_id,
                    collection_id=request.collection_id,
                    status=OperationStatus.FAILED,
                    total_chunks=0,
                    chunks_processed=chunks_saved,
                    chunks_saved=chunks_saved,
                    processing_started_at=processing_started_at,
                    processing_completed_at=None,
                    error_message=str(e),
                    checkpoints_created=checkpoints_created
                )

    def _create_operation_entity(self, operation_id: str, document_id: str,
                                collection_id: str, strategy_type: str) -> Any:
        """
        Create a chunking operation entity.

        This would typically create a domain entity, but we're keeping it
        abstract here as the domain layer is being implemented separately.
        """
        # Placeholder - would create actual domain entity
        return {
            "id": operation_id,
            "document_id": document_id,
            "collection_id": collection_id,
            "strategy_type": strategy_type,
            "status": "pending",
            "created_at": datetime.utcnow()
        }

    def _create_chunk_entity(self, chunk: Any, operation_id: str,
                           document_id: str, collection_id: str,
                           position: int) -> Any:
        """
        Create a chunk entity for persistence.

        Maps from domain chunk to persistable entity.
        """
        # Placeholder - would create actual domain entity
        return {
            "id": str(uuid4()),
            "operation_id": operation_id,
            "document_id": document_id,
            "collection_id": collection_id,
            "content": chunk.content,
            "position": position,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "token_count": chunk.token_count,
            "created_at": datetime.utcnow()
        }
