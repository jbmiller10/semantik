#!/usr/bin/env python3
"""
Service layer for chunking operations.

This module provides the business logic for text chunking, including
validation, caching, and progress tracking.
"""

import hashlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psutil
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
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
from packages.webui.services.chunking_error_handler import ChunkingErrorHandler
from packages.webui.services.chunking_security import ChunkingSecurityValidator

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
        security_validator: ChunkingSecurityValidator | None = None,
        error_handler: ChunkingErrorHandler | None = None,
    ) -> None:
        """Initialize the chunking service.

        Args:
            db_session: Database session
            collection_repo: Collection repository
            document_repo: Document repository
            redis_client: Redis client for caching
            security_validator: Security validator (optional)
            error_handler: Error handler (optional)
        """
        self.db = db_session
        self.collection_repo = collection_repo
        self.document_repo = document_repo
        self.redis = redis_client
        self.security = security_validator or ChunkingSecurityValidator()
        self.error_handler = error_handler or ChunkingErrorHandler()

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

        try:
            # Validate input size for preview
            self.security.validate_document_size(len(text), is_preview=True)
        except ValueError as e:
            raise ChunkingValidationError(
                detail=str(e),
                correlation_id=correlation_id,
                field_errors={"text": ["Document size exceeds preview limits"]},
                operation_id=operation_id,
            ) from e

        # Get or validate config
        if not config:
            if file_type:
                config = FileTypeDetector.get_optimal_config(file_type)
            else:
                # Default config with smaller chunk size for better splitting
                config = {
                    "strategy": "recursive",
                    "params": {"chunk_size": 100, "chunk_overlap": 20},
                }

        # Validate config security
        try:
            self.security.validate_chunk_params(config.get("params", {}))
        except ValueError as e:
            raise ChunkingValidationError(
                detail=str(e),
                correlation_id=correlation_id,
                field_errors={"config": ["Invalid chunking parameters"]},
                operation_id=operation_id,
            ) from e

        # Check cache
        config_hash = self._hash_config(config)
        text_preview = text[:1000]  # First 1000 chars for cache key
        cached = await self._get_cached_preview(config_hash, text_preview)
        if cached:
            logger.debug("Returning cached preview")
            return cached

        # Create chunker and process with error handling
        import time

        start_time = time.time()

        # Check memory before processing
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        memory_limit = CHUNKING_LIMITS.PREVIEW_MEMORY_LIMIT_BYTES

        # Prepare metadata for chunker
        metadata = {}
        if file_type:
            metadata["file_type"] = file_type
            metadata["file_name"] = f"preview{file_type}"

        try:
            chunker = ChunkingFactory.create_chunker(config)
        except Exception as e:
            # Handle strategy initialization errors
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

            # If we have a fallback strategy, raise a specific error
            if error_result.fallback_strategy:
                raise ChunkingStrategyError(
                    detail=f"Failed to initialize {config.get('strategy')} strategy",
                    correlation_id=correlation_id,
                    strategy=config.get("strategy", "unknown"),
                    fallback_strategy=error_result.fallback_strategy,
                    operation_id=operation_id,
                ) from e
            raise ChunkingStrategyError(
                detail="Failed to initialize chunking strategy",
                correlation_id=correlation_id,
                strategy=config.get("strategy", "unknown"),
                fallback_strategy="recursive",
                operation_id=operation_id,
            ) from e

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

        except MemoryError as e:
            # Handle out of memory errors
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
            # Handle timeout errors
            elapsed_time = time.time() - start_time

            raise ChunkingTimeoutError(
                detail="Preview operation timed out",
                correlation_id=correlation_id,
                operation_id=operation_id,
                elapsed_time=elapsed_time,
                timeout_limit=CHUNKING_TIMEOUTS.PREVIEW_TIMEOUT_SECONDS,
                estimated_completion=elapsed_time * 2,  # Rough estimate
            ) from e
        except Exception as e:
            # Handle any other errors
            error_result = await self.error_handler.handle_with_correlation(
                operation_id=operation_id,
                correlation_id=correlation_id,
                error=e,
                context={
                    "method": "preview_chunking",
                    "strategy": config.get("strategy"),
                    "text_size": len(text),
                    "chunks_processed": 0,
                },
            )
            raise  # Re-raise the original exception

        processing_time = time.time() - start_time

        # Build response
        preview_chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "length": len(chunk.text),
                "metadata": chunk.metadata,
            }
            for chunk in chunks[:max_chunks]
        ]

        # Check if any chunk indicates it's a code file from metadata
        is_code_from_metadata = any(chunk.metadata.get("is_code_file", False) for chunk in chunks)
        # Also check file type detection
        is_code_from_file_type = FileTypeDetector.is_code_file(file_type) if file_type else False

        response = ChunkingPreviewResponse(
            chunks=preview_chunks,
            total_chunks=len(chunks),
            strategy_used=config["strategy"],
            is_code_file=is_code_from_metadata or is_code_from_file_type,
            performance_metrics=self._calculate_metrics(chunks, len(text), processing_time),
            recommendations=self._get_recommendations(chunks, file_type),
        )

        # Cache result
        await self._cache_preview(config_hash, text_preview, response)

        return response

    async def recommend_strategy(
        self,
        file_types: list[str] | None = None,
        file_paths: list[str] | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        """Recommend optimal chunking strategy based on file types.

        Args:
            file_types: List of file types/extensions
            file_paths: List of file paths to analyze
            user_id: User ID for tracking

        Returns:
            Dictionary with recommendation details
        """
        # Use file_types if provided, otherwise extract from paths
        types_to_analyze = file_types or []
        if not types_to_analyze and file_paths:
            types_to_analyze = [path.split('.')[-1] if '.' in path else 'unknown' for path in file_paths]
        
        # Analyze file types
        file_type_breakdown: dict[str, int] = {}
        
        for file_type in types_to_analyze:
            category = FileTypeDetector.get_file_category(f"file.{file_type}")
            file_type_breakdown[category] = file_type_breakdown.get(category, 0) + 1
        
        # Determine recommendation
        total_files = len(types_to_analyze)
        
        # If majority are markdown files
        if file_type_breakdown.get("markdown", 0) > total_files * 0.5:
            return {
                "strategy": "recursive",  # Use recursive instead of markdown
                "confidence": 0.85,
                "reasoning": "Majority of files are markdown documents which benefit from structure-aware chunking",
                "alternatives": ["semantic", "fixed_size"],
                "chunk_size": 600,
                "chunk_overlap": 100,
            }
        
        # If significant code files
        if file_type_breakdown.get("code", 0) > total_files * 0.3:
            return {
                "strategy": "recursive",
                "confidence": 0.80,
                "reasoning": "Mixed content with significant code files requiring syntax-aware chunking",
                "alternatives": ["sliding_window", "semantic"],
                "chunk_size": 500,
                "chunk_overlap": 75,
            }
        
        # Default recommendation
        return {
            "strategy": "recursive",
            "confidence": 0.75,
            "reasoning": "General purpose strategy for mixed content types",
            "alternatives": ["fixed_size", "semantic"],
            "chunk_size": 600,
            "chunk_overlap": 100,
        }

    async def get_chunking_statistics(
        self,
        collection_id: str,  # noqa: ARG002
        days: int = 30,  # noqa: ARG002
    ) -> ChunkingStatistics:
        """Get detailed chunking statistics for a collection.

        Args:
            collection_id: Collection ID
            days: Number of days to look back

        Returns:
            ChunkingStatistics with detailed metrics
        """
        # This would query from the chunking_metrics table
        # For now, return mock data
        return ChunkingStatistics(
            total_documents=100,
            total_chunks=1000,
            average_chunk_size=600,
            strategy_breakdown={
                "recursive": 80,
                "markdown": 15,
                "character": 5,
            },
            performance_metrics={
                "average_chunks_per_second": 500,
                "peak_memory_usage_mb": 100,
            },
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
        if not re.match(r'^[a-f0-9\-]{8,}$', preview_id, re.IGNORECASE):
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
        # This would query the operation status from database
        # For now, return mock data for valid UUIDs
        import uuid
        try:
            uuid.UUID(operation_id)
            return {
                "status": "in_progress",
                "progress_percentage": 50.0,
                "documents_processed": 5,
                "total_documents": 10,
                "chunks_created": 250,
                "current_document": "document.pdf",
                "estimated_time_remaining": 30,
                "errors": [],
            }
        except ValueError:
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
        
        try:
            # Send initial progress
            await ws_manager.send_message(
                websocket_channel,
                {
                    "type": "chunking_progress",
                    "operation_id": operation_id,
                    "progress": 0,
                    "status": "in_progress",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
            
            # TODO: Implement actual chunking logic here
            # This would:
            # 1. Load documents from database
            # 2. Apply chunking strategy
            # 3. Store chunks in vector database
            # 4. Update operation status
            # 5. Send progress updates via WebSocket
            
            import asyncio
            await asyncio.sleep(1)  # Simulate processing
            
            # Send completion
            await ws_manager.send_message(
                websocket_channel,
                {
                    "type": "chunking_completed",
                    "operation_id": operation_id,
                    "progress": 100,
                    "status": "completed",
                    "chunks_created": 50,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
            
        except Exception as e:
            logger.error(f"Chunking operation failed: {e}")
            await ws_manager.send_message(
                websocket_channel,
                {
                    "type": "chunking_failed",
                    "operation_id": operation_id,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

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
