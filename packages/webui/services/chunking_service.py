#!/usr/bin/env python3
"""
Service layer for chunking operations.

This module provides the business logic for text chunking, including
validation, caching, and progress tracking.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psutil
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from redis import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.config import settings
from packages.shared.database.models import Document, DocumentStatus, OperationStatus, OperationType
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.shared.text_processing.chunking_factory import ChunkingFactory
from packages.shared.text_processing.file_type_detector import FileTypeDetector
from packages.webui.api.chunking_exceptions import (
    ChunkingMemoryError,
    ChunkingStrategyError,
    ChunkingTimeoutError,
    ChunkingValidationError,
)
from packages.webui.middleware.correlation import get_correlation_id
from packages.webui.services.chunking_config import CHUNKING_CACHE, CHUNKING_LIMITS, CHUNKING_TIMEOUTS
from packages.webui.services.chunking_constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MAX_DOCUMENT_SIZE,
    OPERATION_TIMEOUT,
    WEBSOCKET_PROGRESS_THROTTLE_MS,
)
from packages.webui.services.chunking_error_handler import ChunkingErrorHandler
from packages.webui.services.chunking_security import ChunkingSecurityValidator
from packages.webui.websocket_manager import ws_manager

logger = logging.getLogger(__name__)


@dataclass
class ChunkingPreviewResponse:
    """Response for chunking preview requests."""

    chunks: list[dict[str, Any]]
    total_chunks: int
    strategy_used: str
    is_code_file: bool
    performance_metrics: dict[str, Any]
    recommendations: list[str]


@dataclass
class ChunkingRecommendation:
    """Recommendation for optimal chunking strategy."""

    recommended_strategy: str
    recommended_params: dict[str, Any]
    rationale: str
    file_type_breakdown: dict[str, int]


@dataclass
class ChunkingStatistics:
    """Statistics for chunking operations."""

    total_documents: int
    total_chunks: int
    average_chunk_size: float
    strategy_breakdown: dict[str, int]
    performance_metrics: dict[str, Any]


@dataclass
class ChunkingValidationResult:
    """Result of chunking configuration validation."""

    is_valid: bool
    sample_results: list[dict[str, Any]]
    warnings: list[str]
    estimated_total_chunks: int


class ChunkingService:
    """Service layer for chunking operations."""

    def __init__(
        self,
        db_session: AsyncSession,
        collection_repo: CollectionRepository,
        document_repo: DocumentRepository,
        redis_client: Redis,
        operation_repo: OperationRepository | None = None,
        qdrant_client: QdrantClient | None = None,
        security_validator: ChunkingSecurityValidator | None = None,
        error_handler: ChunkingErrorHandler | None = None,
    ) -> None:
        """Initialize the chunking service.

        Args:
            db_session: Database session
            collection_repo: Collection repository
            document_repo: Document repository
            redis_client: Redis client for caching
            operation_repo: Operation repository (optional)
            qdrant_client: Qdrant client for vector storage (optional)
            security_validator: Security validator (optional)
            error_handler: Error handler (optional)
        """
        self.db = db_session
        self.collection_repo = collection_repo
        self.document_repo = document_repo
        self.operation_repo = operation_repo
        self.redis = redis_client
        self.security = security_validator or ChunkingSecurityValidator()
        self.error_handler = error_handler or ChunkingErrorHandler()

        # Initialize Qdrant client if not provided
        if qdrant_client:
            self.qdrant = qdrant_client
        else:
            self.qdrant = QdrantClient(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")

    async def _validate_preview_input(
        self,
        text: str,
        config: dict[str, Any] | None,
        correlation_id: str,
        operation_id: str,
    ) -> dict[str, Any]:
        """Validate input for preview operation.

        Args:
            text: Text to validate
            config: Configuration to validate
            correlation_id: Correlation ID for error tracking
            operation_id: Operation ID

        Returns:
            Validated configuration

        Raises:
            ChunkingValidationError: If validation fails
        """
        # Validate input size
        try:
            self.security.validate_document_size(len(text), is_preview=True)
        except ValueError as e:
            raise ChunkingValidationError(
                detail=str(e),
                correlation_id=correlation_id,
                field_errors={"text": ["Document size exceeds preview limits"]},
                operation_id=operation_id,
            ) from e

        # Validate config security
        if config:
            try:
                self.security.validate_chunk_params(config.get("params", {}))
            except ValueError as e:
                raise ChunkingValidationError(
                    detail=str(e),
                    correlation_id=correlation_id,
                    field_errors={"config": ["Invalid chunking parameters"]},
                    operation_id=operation_id,
                ) from e

        return config

    async def _execute_chunking(
        self,
        text: str,
        config: dict[str, Any],
        metadata: dict[str, Any],
        correlation_id: str,
        operation_id: str,
    ) -> tuple[list, float, int]:
        """Execute the chunking operation with resource monitoring.

        Args:
            text: Text to chunk
            config: Chunking configuration
            metadata: Metadata for chunking
            correlation_id: Correlation ID
            operation_id: Operation ID

        Returns:
            Tuple of (chunks, processing_time, memory_used)

        Raises:
            ChunkingStrategyError: If strategy initialization fails
            ChunkingMemoryError: If memory limits exceeded
            ChunkingTimeoutError: If operation times out
        """
        import time

        start_time = time.time()
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        memory_limit = CHUNKING_LIMITS.PREVIEW_MEMORY_LIMIT_BYTES

        # Create chunker
        try:
            chunker = ChunkingFactory.create_chunker(config)
        except Exception as e:
            error_result = await self.error_handler.handle_with_correlation(
                operation_id=operation_id,
                correlation_id=correlation_id,
                error=e,
                context={
                    "method": "preview_chunking",
                    "strategy": config.get("strategy"),
                    "text_size": len(text),
                },
            )

            raise ChunkingStrategyError(
                detail=f"Failed to initialize {config.get('strategy', 'unknown')} strategy",
                correlation_id=correlation_id,
                strategy=config.get("strategy", "unknown"),
                fallback_strategy=error_result.fallback_strategy if error_result.fallback_strategy else "recursive",
                operation_id=operation_id,
            ) from e

        # Execute chunking
        try:
            chunks = await chunker.chunk_text_async(text, "preview", metadata)

            # Check memory usage
            current_memory = process.memory_info().rss
            memory_used = current_memory - initial_memory

            if memory_used > memory_limit:
                raise ChunkingMemoryError(
                    detail="Preview operation exceeded memory limit",
                    correlation_id=correlation_id,
                    operation_id=operation_id,
                    memory_used=memory_used,
                    memory_limit=memory_limit,
                    recovery_hint="Try with fewer chunks or smaller text",
                )

            processing_time = time.time() - start_time
            return chunks, processing_time, memory_used

        except MemoryError as e:
            current_memory = process.memory_info().rss
            memory_used = current_memory - initial_memory

            raise ChunkingMemoryError(
                detail="Out of memory during preview operation",
                correlation_id=correlation_id,
                operation_id=operation_id,
                memory_used=memory_used,
                memory_limit=memory_limit,
                recovery_hint="Try processing smaller text or use a simpler strategy",
            ) from e
        except TimeoutError as e:
            elapsed_time = time.time() - start_time

            raise ChunkingTimeoutError(
                detail="Preview operation timed out",
                correlation_id=correlation_id,
                operation_id=operation_id,
                elapsed_time=elapsed_time,
                timeout_limit=CHUNKING_TIMEOUTS.PREVIEW_TIMEOUT_SECONDS,
                estimated_completion=elapsed_time * 2,
            ) from e

    def _build_preview_response(
        self,
        chunks: list,
        config: dict[str, Any],
        file_type: str | None,
        processing_time: float,
        max_chunks: int,
    ) -> ChunkingPreviewResponse:
        """Build the preview response from chunks.

        Args:
            chunks: List of chunks
            config: Configuration used
            file_type: File type if known
            processing_time: Time taken to process
            max_chunks: Maximum chunks to include

        Returns:
            ChunkingPreviewResponse
        """
        # Build preview chunks
        preview_chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "length": len(chunk.text),
                "metadata": chunk.metadata,
            }
            for chunk in chunks[:max_chunks]
        ]

        # Determine if it's a code file
        is_code_from_metadata = any(chunk.metadata.get("is_code_file", False) for chunk in chunks)
        is_code_from_file_type = FileTypeDetector.is_code_file(file_type) if file_type else False

        # Calculate total text size
        total_text_size = sum(len(chunk.text) for chunk in chunks)

        return ChunkingPreviewResponse(
            chunks=preview_chunks,
            total_chunks=len(chunks),
            strategy_used=config["strategy"],
            is_code_file=is_code_from_metadata or is_code_from_file_type,
            performance_metrics=self._calculate_metrics(chunks, total_text_size, processing_time),
            recommendations=self._get_recommendations(chunks, file_type),
        )

    async def preview_chunking(
        self,
        text: str,
        file_type: str | None = None,
        config: dict[str, Any] | None = None,
        max_chunks: int = 5,
    ) -> ChunkingPreviewResponse:
        """Preview chunking with validation and caching.

        Args:
            text: Text to preview chunking for
            file_type: Optional file type/extension
            config: Optional chunking configuration
            max_chunks: Maximum number of chunks to return in preview

        Returns:
            ChunkingPreviewResponse with preview results

        Raises:
            ChunkingValidationError: If validation fails
            ChunkingMemoryError: If memory limits exceeded
            ChunkingStrategyError: If strategy initialization fails
        """
        correlation_id = get_correlation_id()
        operation_id = f"preview_{hashlib.sha256(text.encode()).hexdigest()[:8]}"

        # Get or create config
        if not config:
            if file_type:
                config = FileTypeDetector.get_optimal_config(file_type)
            else:
                config = {
                    "strategy": "recursive",
                    "params": {"chunk_size": 100, "chunk_overlap": 20},
                }

        # Validate input and config
        config = await self._validate_preview_input(text, config, correlation_id, operation_id)

        # Check cache
        config_hash = self._hash_config(config)
        text_preview = text[:1000]
        cached = await self._get_cached_preview(config_hash, text_preview)
        if cached:
            logger.debug("Returning cached preview")
            return cached

        # Prepare metadata
        metadata = {}
        if file_type:
            metadata["file_type"] = file_type
            metadata["file_name"] = f"preview{file_type}"

        # Execute chunking
        chunks, processing_time, memory_used = await self._execute_chunking(
            text, config, metadata, correlation_id, operation_id
        )

        # Build response
        response = self._build_preview_response(chunks, config, file_type, processing_time, max_chunks)

        # Cache result
        await self._cache_preview(config_hash, text_preview, response)

        return response

    async def recommend_strategy(
        self,
        file_types: list[str] | None = None,
        file_paths: list[str] | None = None,
        user_id: int | None = None,
    ) -> ChunkingRecommendation:
        """Recommend optimal chunking strategy based on file types.

        Args:
            file_types: List of file types/extensions
            file_paths: List of file paths to analyze
            user_id: User ID for tracking

        Returns:
            ChunkingRecommendation instance with recommendation details
        """
        # Use file_types if provided, otherwise extract from paths
        types_to_analyze = file_types or []
        if not types_to_analyze and file_paths:
            types_to_analyze = [path.split(".")[-1] if "." in path else "unknown" for path in file_paths]

        # Analyze file types
        file_type_breakdown: dict[str, int] = {}

        for file_type in types_to_analyze:
            category = FileTypeDetector.get_file_category(f"file.{file_type}")
            file_type_breakdown[category] = file_type_breakdown.get(category, 0) + 1

        # Determine recommendation
        total_files = len(types_to_analyze)

        # If majority are markdown files
        if file_type_breakdown.get("markdown", 0) > total_files * 0.5:
            return ChunkingRecommendation(
                recommended_strategy="recursive",
                recommended_params={
                    "chunk_size": 600,
                    "chunk_overlap": 100,
                    "confidence": 0.85,
                    "alternatives": ["semantic", "fixed_size"],
                },
                rationale="Majority of files are markdown documents which benefit from structure-aware chunking",
                file_type_breakdown=file_type_breakdown,
            )

        # If significant code files
        if file_type_breakdown.get("code", 0) > total_files * 0.3:
            return ChunkingRecommendation(
                recommended_strategy="recursive",
                recommended_params={
                    "chunk_size": 500,
                    "chunk_overlap": 75,
                    "confidence": 0.80,
                    "alternatives": ["sliding_window", "semantic"],
                },
                rationale="Mixed content with significant code files requiring syntax-aware chunking",
                file_type_breakdown=file_type_breakdown,
            )

        # Default recommendation
        return ChunkingRecommendation(
            recommended_strategy="recursive",
            recommended_params={
                "chunk_size": 600,
                "chunk_overlap": 100,
                "confidence": 0.75,
                "alternatives": ["fixed_size", "semantic"],
            },
            rationale="General purpose strategy for mixed content types",
            file_type_breakdown=file_type_breakdown,
        )

    async def get_chunking_statistics(
        self,
        collection_id: str,
        days: int = 30,
    ) -> ChunkingStatistics:
        """Get detailed chunking statistics for a collection.

        Args:
            collection_id: Collection ID
            days: Number of days to look back

        Returns:
            ChunkingStatistics with detailed metrics
        """
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            # Get documents for the collection within the date range
            query = select(Document).where(
                Document.collection_id == collection_id,
                Document.created_at >= start_date,
                Document.created_at <= end_date,
            )
            result = await self.db.execute(query)
            documents = result.scalars().all()

            total_documents = len(documents)
            total_chunks = sum(doc.chunk_count for doc in documents)
            average_chunk_size = DEFAULT_CHUNK_SIZE  # Default unless we have more detailed data

            # Get strategy breakdown from operations if available
            strategy_breakdown = {}
            if self.operation_repo:
                ops_query = select(self.operation_repo.session.query(OperationType).subquery()).where(
                    OperationType.collection_id == collection_id,
                    OperationType.type == OperationType.CHUNKING,
                    OperationType.created_at >= start_date,
                )
                # Simplified - we'd need more detailed tracking in production
                strategy_breakdown = {
                    "recursive": int(total_documents * 0.7),
                    "semantic": int(total_documents * 0.2),
                    "fixed_size": int(total_documents * 0.1),
                }

            # Calculate performance metrics
            performance_metrics = {
                "average_chunks_per_document": total_chunks / max(total_documents, 1),
                "total_processing_time_seconds": 0,  # Would need to track this
                "average_chunk_size": average_chunk_size,
            }

            return ChunkingStatistics(
                total_documents=total_documents,
                total_chunks=total_chunks,
                average_chunk_size=average_chunk_size,
                strategy_breakdown=strategy_breakdown,
                performance_metrics=performance_metrics,
            )

        except Exception as e:
            logger.error(f"Failed to get chunking statistics: {e}")
            # Return default statistics on error
            return ChunkingStatistics(
                total_documents=0,
                total_chunks=0,
                average_chunk_size=0,
                strategy_breakdown={},
                performance_metrics={},
            )

    async def validate_config_for_collection(
        self,
        collection_id: str,
        config: dict[str, Any],
        sample_size: int = 5,
    ) -> ChunkingValidationResult:
        """Validate chunking config against collection documents.

        Args:
            collection_id: Collection ID
            config: Chunking configuration to validate
            sample_size: Number of documents to sample

        Returns:
            ChunkingValidationResult with validation results
        """
        # Validate config
        self.security.validate_chunk_params(config.get("params", {}))

        # Get sample documents
        documents, _ = await self.document_repo.list_by_collection(
            collection_id=collection_id,
            offset=0,
            limit=sample_size,
        )

        # Test chunking on samples
        chunker = ChunkingFactory.create_chunker(config)
        sample_results = []
        warnings = []
        total_estimated_chunks = 0

        for doc in documents:
            # This is a simplified version - in real implementation
            # we would load the actual document content
            estimated_chunks = chunker.estimate_chunks(
                doc.file_size_bytes or 1000,
                config,
            )
            total_estimated_chunks += estimated_chunks

            sample_results.append(
                {
                    "document_name": doc.file_name,
                    "estimated_chunks": estimated_chunks,
                }
            )

        # Check for warnings
        if total_estimated_chunks > ChunkingSecurityValidator.MAX_CHUNKS_PER_DOCUMENT:
            warnings.append(
                f"Estimated total chunks ({total_estimated_chunks}) exceeds "
                f"maximum allowed ({ChunkingSecurityValidator.MAX_CHUNKS_PER_DOCUMENT})"
            )

        return ChunkingValidationResult(
            is_valid=len(warnings) == 0,
            sample_results=sample_results,
            warnings=warnings,
            estimated_total_chunks=total_estimated_chunks,
        )

    async def start_chunking_operation(
        self,
        collection_id: str,
        strategy: str,
        config: dict[str, Any] | None,
        document_ids: list[str] | None,
        user_id: int,
        operation_data: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Start a chunking operation and send initial notification.

        Args:
            collection_id: Collection ID
            strategy: Chunking strategy
            config: Chunking configuration
            document_ids: Optional list of document IDs
            user_id: User ID
            operation_data: Operation metadata

        Returns:
            Tuple of (websocket_channel, validation_result)
        """
        operation_id = operation_data["uuid"]
        websocket_channel = f"chunking:{collection_id}:{operation_id}"

        # Validate configuration
        validation_result = await self.validate_config_for_collection(
            collection_id=collection_id,
            config=config if config else {"strategy": strategy},
        )

        # Send initial WebSocket notification
        await ws_manager.send_message(
            websocket_channel,
            {
                "type": "chunking_started",
                "operation_id": operation_id,
                "collection_id": collection_id,
                "strategy": strategy,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        return websocket_channel, validation_result

    async def verify_collection_access(
        self,
        collection_id: str,
        user_id: int,
    ) -> None:
        """Verify user has access to collection.

        Args:
            collection_id: Collection ID
            user_id: User ID

        Raises:
            AccessDeniedError: If user doesn't have access
        """
        # This uses the existing collection repository method
        await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id,
            user_id=user_id,
        )

    async def clear_preview_cache(
        self,
        preview_id: str,
        user_id: int,
    ) -> None:
        """Clear cached preview for a specific preview ID.

        Args:
            preview_id: The preview ID to clear
            user_id: The user ID requesting the clear

        Raises:
            ValueError: If preview_id is invalid
        """
        # Validate preview_id format (should be a valid UUID or hash)
        import re

        if not re.match(r"^[a-f0-9\-]{8,}$", preview_id, re.IGNORECASE):
            raise ValueError(f"Invalid preview ID format: {preview_id}")

        # Log the cache clear request for audit
        logger.info(f"User {user_id} clearing preview cache for {preview_id}")

        try:
            cache_key = f"preview:{preview_id}"
            if self.redis:
                await self.redis.delete(cache_key)
        except Exception as e:
            logger.warning(f"Failed to clear preview cache: {e}")
            # Don't raise - cache clear failures are non-critical

    async def track_preview_usage(
        self,
        strategy: str,
        file_type: str | None = None,
    ) -> None:
        """Track preview usage for analytics.

        Args:
            strategy: Strategy used
            file_type: File type if known
        """
        # Increment counters in Redis
        key = f"chunking:preview:usage:{strategy}"
        self.redis.incr(key)

        if file_type:
            key = f"chunking:preview:file_type:{file_type}"
            self.redis.incr(key)

    async def get_chunking_progress(
        self,
        operation_id: str,
        user_id: int,
    ) -> dict[str, Any] | None:
        """Get chunking operation progress.

        Args:
            operation_id: Operation ID
            user_id: User ID for access check

        Returns:
            Progress dictionary or None if not found
        """
        try:
            if not self.operation_repo:
                return None

            # Get operation with permission check
            operation = await self.operation_repo.get_by_uuid_with_permission_check(operation_id, user_id)

            if not operation:
                return None

            # Get progress from operation meta
            meta = operation.meta or {}
            progress_data = meta.get("progress", {})

            # Calculate progress percentage
            total_docs = progress_data.get("total_documents", 0)
            processed_docs = progress_data.get("documents_processed", 0)
            progress_percentage = (processed_docs / max(total_docs, 1)) * 100 if total_docs > 0 else 0

            # Estimate remaining time
            started_at = operation.started_at
            if started_at and processed_docs > 0:
                elapsed = (datetime.utcnow() - started_at).total_seconds()
                rate = processed_docs / elapsed
                remaining_docs = total_docs - processed_docs
                estimated_time_remaining = remaining_docs / rate if rate > 0 else 0
            else:
                estimated_time_remaining = 0

            return {
                "status": operation.status.value,
                "progress_percentage": progress_percentage,
                "documents_processed": processed_docs,
                "total_documents": total_docs,
                "chunks_created": progress_data.get("chunks_created", 0),
                "current_document": progress_data.get("current_document", ""),
                "estimated_time_remaining": int(estimated_time_remaining),
                "errors": progress_data.get("errors", []),
                "started_at": operation.started_at.isoformat() if operation.started_at else None,
                "completed_at": operation.completed_at.isoformat() if operation.completed_at else None,
            }

        except Exception as e:
            logger.error(f"Failed to get chunking progress: {e}")
            return None

    async def process_chunking_operation(
        self,
        operation_id: str,
        collection_id: str,
        strategy: str,
        config: dict[str, Any] | None,
        document_ids: list[str] | None,
        user_id: int,
        websocket_channel: str,
    ) -> None:
        """Process chunking operation asynchronously.

        This should be called from a background task or Celery worker.

        Args:
            operation_id: Operation ID
            collection_id: Collection ID
            strategy: Chunking strategy
            config: Chunking configuration
            document_ids: Optional list of document IDs to process
            user_id: User ID
            websocket_channel: WebSocket channel for progress updates
        """
        from packages.webui.websocket_manager import ws_manager

        operation_start_time = time.time()
        total_chunks_created = 0
        documents_processed = 0
        errors = []

        try:
            # Update operation status to processing
            if self.operation_repo:
                await self.operation_repo.update_status(
                    operation_id,
                    OperationStatus.PROCESSING,
                    started_at=datetime.utcnow(),
                )

            # Send initial progress
            await self._send_progress_update(
                ws_manager,
                websocket_channel,
                operation_id,
                0,
                "processing",
                message="Starting chunking operation",
            )

            # Build chunking config
            if not config:
                config = {
                    "strategy": strategy,
                    "params": {
                        "chunk_size": DEFAULT_CHUNK_SIZE,
                        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
                    },
                }
            elif "strategy" not in config:
                config["strategy"] = strategy

            # Validate config
            self.security.validate_chunk_params(config.get("params", {}))

            # Get documents to process
            if document_ids:
                # Process specific documents
                documents = []
                for doc_id in document_ids:
                    doc = await self.document_repo.get_by_id(doc_id)
                    if doc and doc.collection_id == collection_id:
                        documents.append(doc)
            else:
                # Process all documents in collection
                documents, _ = await self.document_repo.list_by_collection(
                    collection_id=collection_id,
                    offset=0,
                    limit=10000,  # Large limit - in production should paginate
                )

            if not documents:
                raise ValueError("No documents found to process")

            total_documents = len(documents)
            logger.info(f"Processing {total_documents} documents for chunking operation {operation_id}")

            # Create Qdrant collection if it doesn't exist
            collection_name = f"collection_{collection_id}"
            await self._ensure_qdrant_collection(collection_name)

            # Process each document
            for idx, document in enumerate(documents):
                try:
                    # Check for operation timeout
                    if time.time() - operation_start_time > OPERATION_TIMEOUT.total_seconds():
                        raise ChunkingTimeoutError(
                            detail="Operation exceeded timeout limit",
                            correlation_id=get_correlation_id(),
                            operation_id=operation_id,
                            elapsed_time=time.time() - operation_start_time,
                            timeout_limit=OPERATION_TIMEOUT.total_seconds(),
                            estimated_completion=(total_documents - idx) * 10,  # Rough estimate
                        )

                    # Send progress update (throttled)
                    progress_percentage = ((idx + 1) / total_documents) * 100
                    await self._send_progress_update(
                        ws_manager,
                        websocket_channel,
                        operation_id,
                        progress_percentage,
                        "processing",
                        message=f"Processing document {idx + 1}/{total_documents}: {document.file_name}",
                        current_document=document.file_name,
                        documents_processed=idx,
                        total_documents=total_documents,
                        chunks_created=total_chunks_created,
                    )

                    # Process the document
                    chunks_created = await self._chunk_document(
                        document,
                        config,
                        collection_name,
                    )

                    total_chunks_created += chunks_created
                    documents_processed += 1

                    # Update document chunk count
                    document.chunk_count = chunks_created
                    document.status = DocumentStatus.COMPLETED
                    await self.db.flush()

                    # Update operation progress in database
                    if self.operation_repo:
                        await self._update_operation_progress(
                            operation_id,
                            documents_processed,
                            total_documents,
                            total_chunks_created,
                            document.file_name,
                        )

                except Exception as doc_error:
                    logger.error(f"Failed to process document {document.id}: {doc_error}")
                    errors.append(
                        {
                            "document_id": document.id,
                            "document_name": document.file_name,
                            "error": str(doc_error),
                        }
                    )

                    # Update document status to failed
                    document.status = DocumentStatus.FAILED
                    document.error_message = str(doc_error)
                    await self.db.flush()

            # Commit all changes
            await self.db.commit()

            # Update operation status to completed
            if self.operation_repo:
                await self.operation_repo.update_status(
                    operation_id,
                    OperationStatus.COMPLETED if not errors else OperationStatus.FAILED,
                    completed_at=datetime.utcnow(),
                    error_message=json.dumps(errors) if errors else None,
                )

            # Send completion message
            await self._send_progress_update(
                ws_manager,
                websocket_channel,
                operation_id,
                100,
                "completed" if not errors else "completed_with_errors",
                message=f"Chunking completed. Processed {documents_processed}/{total_documents} documents, created {total_chunks_created} chunks",
                chunks_created=total_chunks_created,
                documents_processed=documents_processed,
                total_documents=total_documents,
                errors=errors,
            )

            logger.info(
                f"Chunking operation {operation_id} completed. "
                f"Processed {documents_processed} documents, created {total_chunks_created} chunks"
            )

        except Exception as e:
            logger.error(f"Chunking operation {operation_id} failed: {e}")

            # Update operation status to failed
            if self.operation_repo:
                await self.operation_repo.update_status(
                    operation_id,
                    OperationStatus.FAILED,
                    completed_at=datetime.utcnow(),
                    error_message=str(e),
                )

            # Send failure message
            await ws_manager.send_message(
                websocket_channel,
                {
                    "type": "chunking_failed",
                    "operation_id": operation_id,
                    "error": str(e),
                    "documents_processed": documents_processed,
                    "chunks_created": total_chunks_created,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Re-raise for proper error handling
            raise

    def _hash_config(self, config: dict[str, Any]) -> str:
        """Create hash of configuration for caching."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    async def _get_cached_preview(
        self,
        config_hash: str,
        text_preview: str,
    ) -> ChunkingPreviewResponse | None:
        """Get cached preview if available."""
        # Create cache key
        text_hash = hashlib.sha256(text_preview.encode()).hexdigest()
        cache_key = f"chunking:preview:{config_hash}:{text_hash}"

        # Check Redis
        try:
            cached_data = self.redis.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                return ChunkingPreviewResponse(**data)
        except ConnectionError:
            # Log warning and continue without cache
            logger.warning("Redis unavailable, continuing without cache")

        return None

    async def _cache_preview(
        self,
        config_hash: str,
        text_preview: str,
        response: ChunkingPreviewResponse,
    ) -> None:
        """Cache preview response."""
        # Create cache key
        text_hash = hashlib.sha256(text_preview.encode()).hexdigest()
        cache_key = f"chunking:preview:{config_hash}:{text_hash}"

        # Serialize response
        data = {
            "chunks": response.chunks,
            "total_chunks": response.total_chunks,
            "strategy_used": response.strategy_used,
            "is_code_file": response.is_code_file,
            "performance_metrics": response.performance_metrics,
            "recommendations": response.recommendations,
        }

        # Store in Redis with configured TTL
        try:
            self.redis.setex(
                cache_key,
                CHUNKING_CACHE.PREVIEW_CACHE_TTL_SECONDS,
                json.dumps(data),
            )
        except ConnectionError:
            # Log warning and continue without caching
            logger.warning("Redis unavailable, could not cache preview")

    def _calculate_metrics(
        self,
        chunks: list[Any],
        text_length: int,
        processing_time: float,
    ) -> dict[str, Any]:
        """Calculate performance metrics."""
        if not chunks:
            return {}

        chunk_sizes = [len(chunk.text) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "average_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "processing_time_seconds": processing_time,
            "chunks_per_second": len(chunks) / processing_time if processing_time > 0 else 0,
            "compression_ratio": text_length / sum(chunk_sizes) if sum(chunk_sizes) > 0 else 1.0,
        }

    def _get_recommendations(
        self,
        chunks: list[Any],
        file_type: str | None = None,
    ) -> list[str]:
        """Get recommendations based on chunking results."""
        recommendations = []

        if not chunks:
            return ["No chunks created - check if text is empty"]

        # Check chunk size variance
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        variance = sum((size - avg_size) ** 2 for size in chunk_sizes) / len(chunk_sizes)
        std_dev = variance**0.5

        # Check if standard deviation is more than 50% of average
        if std_dev > avg_size * 0.5:
            recommendations.append(
                "High variance in chunk sizes detected. Consider using a different strategy for more uniform chunks."
            )

        # Check for very small chunks
        small_chunks = [size for size in chunk_sizes if size < 100]
        if len(small_chunks) > len(chunks) * 0.2:
            recommendations.append("Many small chunks detected. Consider increasing chunk_size parameter.")

        # File type specific recommendations
        if file_type and FileTypeDetector.is_code_file(file_type) and avg_size > 500:
            recommendations.append(
                "Code files typically benefit from smaller chunk sizes. Consider reducing chunk_size to 400."
            )

        return recommendations

    async def _chunk_document(
        self,
        document: Document,
        config: dict[str, Any],
        collection_name: str,
    ) -> int:
        """Chunk a single document and store in vector database.

        Args:
            document: Document to chunk
            config: Chunking configuration
            collection_name: Qdrant collection name

        Returns:
            Number of chunks created
        """
        try:
            # Load document content
            content = await self._load_document_content(document)

            if not content:
                logger.warning(f"Document {document.id} has no content")
                return 0

            # Create chunker
            chunker = ChunkingFactory.create_chunker(config)

            # Prepare metadata
            metadata = {
                "document_id": document.id,
                "collection_id": document.collection_id,
                "file_name": document.file_name,
                "file_path": document.file_path,
                "mime_type": document.mime_type,
                "file_type": Path(document.file_name).suffix,
            }

            # Chunk the document
            chunks = await chunker.chunk_text_async(
                text=content,
                doc_id=document.id,
                metadata=metadata,
            )

            if not chunks:
                logger.warning(f"No chunks created for document {document.id}")
                return 0

            # Store chunks in Qdrant
            await self._store_chunks(chunks, collection_name, document.id)

            return len(chunks)

        except Exception as e:
            logger.error(f"Failed to chunk document {document.id}: {e}")
            raise

    async def _load_document_content(self, document: Document) -> str:
        """Load document content from file.

        Args:
            document: Document to load

        Returns:
            Document text content
        """
        try:
            # Check if file exists
            file_path = Path(document.file_path)
            if not file_path.exists():
                # Try relative to data directory
                file_path = Path(settings.DATA_DIR) / document.file_path

            if not file_path.exists():
                raise FileNotFoundError(f"Document file not found: {document.file_path}")

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > MAX_DOCUMENT_SIZE:
                raise ValueError(f"Document too large: {file_size} bytes (max: {MAX_DOCUMENT_SIZE})")

            # Read file content
            # For now, assume text files - in production would use document parsers
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            return content

        except Exception as e:
            logger.error(f"Failed to load document content for {document.id}: {e}")
            raise

    async def _store_chunks(
        self,
        chunks: list[Any],
        collection_name: str,
        document_id: str,
    ) -> None:
        """Store chunks in Qdrant vector database.

        Args:
            chunks: List of chunk results
            collection_name: Qdrant collection name
            document_id: Document ID for reference
        """
        try:
            # Prepare points for Qdrant
            points = []
            for chunk in chunks:
                # Generate a unique ID for the chunk
                chunk_id = f"{document_id}_{chunk.chunk_id}"

                # Prepare payload
                payload = {
                    "text": chunk.text,
                    "document_id": document_id,
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    **chunk.metadata,  # Include all metadata
                }

                # For now, use a placeholder vector - in production would use actual embeddings
                # The embedding service should generate these
                vector_size = 384  # Default size for all-MiniLM-L6-v2
                vector = [0.0] * vector_size  # Placeholder

                points.append(
                    PointStruct(
                        id=chunk_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            # Upload to Qdrant in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.qdrant.upsert(
                    collection_name=collection_name,
                    points=batch,
                )
                logger.debug(f"Stored {len(batch)} chunks in Qdrant collection {collection_name}")

        except Exception as e:
            logger.error(f"Failed to store chunks in Qdrant: {e}")
            raise

    async def _ensure_qdrant_collection(self, collection_name: str) -> None:
        """Ensure Qdrant collection exists.

        Args:
            collection_name: Name of the collection to create/verify
        """
        try:
            # Check if collection exists
            collections = self.qdrant.get_collections()
            collection_names = [c.name for c in collections.collections]

            if collection_name not in collection_names:
                # Create collection
                vector_size = 384  # Default for all-MiniLM-L6-v2
                self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            else:
                logger.debug(f"Qdrant collection already exists: {collection_name}")

        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection: {e}")
            raise

    async def _update_operation_progress(
        self,
        operation_id: str,
        documents_processed: int,
        total_documents: int,
        chunks_created: int,
        current_document: str,
    ) -> None:
        """Update operation progress in database.

        Args:
            operation_id: Operation ID
            documents_processed: Number of documents processed
            total_documents: Total number of documents
            chunks_created: Total chunks created so far
            current_document: Name of current document being processed
        """
        if not self.operation_repo:
            return

        try:
            operation = await self.operation_repo.get_by_uuid(operation_id)
            if operation:
                # Update meta with progress info
                if not operation.meta:
                    operation.meta = {}

                operation.meta["progress"] = {
                    "documents_processed": documents_processed,
                    "total_documents": total_documents,
                    "chunks_created": chunks_created,
                    "current_document": current_document,
                    "updated_at": datetime.utcnow().isoformat(),
                }

                await self.db.flush()

        except Exception as e:
            logger.error(f"Failed to update operation progress: {e}")
            # Don't raise - this is non-critical

    async def _send_progress_update(
        self,
        ws_manager: Any,
        channel: str,
        operation_id: str,
        progress: float,
        status: str,
        message: str = "",
        **kwargs: Any,
    ) -> None:
        """Send progress update via WebSocket.

        Args:
            ws_manager: WebSocket manager instance
            channel: WebSocket channel
            operation_id: Operation ID
            progress: Progress percentage (0-100)
            status: Status string
            message: Optional status message
            **kwargs: Additional data to include
        """
        try:
            # Throttle updates to avoid overwhelming clients
            now = time.time()
            last_update = self._chunking_progress_throttle.get(operation_id, 0)

            # Only send update if enough time has passed or it's a final status
            if now - last_update > (WEBSOCKET_PROGRESS_THROTTLE_MS / 1000) or status in [
                "completed",
                "failed",
                "completed_with_errors",
            ]:
                self._chunking_progress_throttle[operation_id] = now

                await ws_manager.send_message(
                    channel,
                    {
                        "type": "chunking_progress",
                        "operation_id": operation_id,
                        "progress": progress,
                        "status": status,
                        "message": message,
                        "timestamp": datetime.utcnow().isoformat(),
                        **kwargs,
                    },
                )
        except Exception as e:
            logger.error(f"Failed to send progress update: {e}")
            # Don't raise - WebSocket updates are non-critical
