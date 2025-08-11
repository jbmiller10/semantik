"""
Service layer for chunking operations.

This service orchestrates all chunking-related operations including strategy
selection, preview generation, and actual document chunking.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.chunking.application.dto.requests import (
    ChunkingStrategy as ChunkingStrategyEnum,
)
from packages.shared.chunking.domain.exceptions import (
    ChunkingDomainError,
    InvalidConfigurationError,
    StrategyNotFoundError,
)
from packages.shared.chunking.domain.services.chunking_strategies import (
    STRATEGY_REGISTRY,
    get_strategy,
)
from packages.shared.chunking.infrastructure.exception_translator import (
    exception_translator,
)
from packages.shared.chunking.infrastructure.exceptions import (
    ApplicationException,
    ChunkingStrategyError,
    DocumentTooLargeError,
    DomainException,
    ResourceNotFoundException,
    ValidationException,
)
from packages.shared.chunking.infrastructure.exceptions import (
    PermissionDeniedException as InfraPermissionDeniedException,
)
from packages.shared.database.models import Operation
from packages.shared.database.repositories.collection_repository import (
    CollectionRepository,
)
from packages.shared.database.repositories.document_repository import DocumentRepository

# All exceptions now come from the new infrastructure layer
# Old chunking_exceptions module should be deleted as we're PRE-RELEASE
from .cache_manager import CacheManager, QueryMonitor
from .chunking_config_builder import ChunkingConfigBuilder
from .chunking_error_handler import ChunkingErrorHandler
from .chunking_strategies import ChunkingStrategyRegistry
from .chunking_strategy_factory import ChunkingStrategyFactory
from .chunking_validation import ChunkingInputValidator

logger = logging.getLogger(__name__)


class ChunkingStatistics:
    """Statistics for chunking operations."""

    def __init__(
        self,
        total_documents: int = 0,
        total_chunks: int = 0,
        average_chunk_size: float = 0,
        strategy_breakdown: dict[str, int] | None = None,
    ):
        """Initialize chunking statistics."""
        self.total_documents = total_documents
        self.total_chunks = total_chunks
        self.average_chunk_size = average_chunk_size
        self.strategy_breakdown = strategy_breakdown or {}


class SimpleChunkingStrategyFactory:
    """Simple implementation of ChunkingStrategyFactory interface."""

    def create_strategy(self, strategy_type: str, _config: dict[str, Any]) -> Any:
        """Create a chunking strategy instance."""
        return get_strategy(strategy_type)

    def get_available_strategies(self) -> list[str]:
        """Get list of available strategy types."""
        return list(STRATEGY_REGISTRY.keys())

    def get_default_config(self, strategy_type: str) -> dict[str, Any]:
        """Get default configuration for a strategy."""
        defaults: dict[str, dict[str, Any]] = {
            "character": {"chunk_size": 1000, "chunk_overlap": 200},
            "recursive": {"chunk_size": 1000, "chunk_overlap": 200},
            "markdown": {"chunk_size": 1000, "chunk_overlap": 200},
            "semantic": {"buffer_size": 1, "breakpoint_percentile_threshold": 95},
            "hierarchical": {"chunk_sizes": [2048, 512], "chunk_overlap": 50},
            "hybrid": {"primary_strategy": "recursive", "fallback_strategy": "character"},
        }
        return defaults.get(strategy_type, {})


class ChunkingService:
    """Service for managing chunking operations."""

    # Mapping from API strategy names to factory strategy names
    STRATEGY_MAPPING = {
        "fixed_size": "character",
        "sliding_window": "character",
        "semantic": "semantic",
        "recursive": "recursive",
        "document_structure": "markdown",
        "markdown": "markdown",
        "hierarchical": "hierarchical",
        "hybrid": "hybrid",
    }

    def __init__(
        self,
        db_session: AsyncSession,
        collection_repo: CollectionRepository,
        document_repo: DocumentRepository,
        redis_client: aioredis.Redis | None = None,
    ):
        """Initialize the chunking service.

        Args:
            db_session: Database session for operations
            collection_repo: Repository for collection operations
            document_repo: Repository for document operations
            redis_client: Optional Redis client for caching
        """
        self.db_session = db_session
        self.collection_repo = collection_repo
        self.document_repo = document_repo
        self.redis_client = redis_client
        self.error_handler = ChunkingErrorHandler()
        self.validator = ChunkingInputValidator()
        self.config_builder = ChunkingConfigBuilder()
        self.strategy_factory = ChunkingStrategyFactory()
        self.exception_translator = exception_translator

        # Initialize cache manager if Redis is available
        self.cache_manager = CacheManager(redis_client) if redis_client else None
        self.query_monitor = QueryMonitor()

    async def get_available_strategies(self) -> list[dict[str, Any]]:
        """Get list of available chunking strategies with details.

        This method provides the business logic for listing strategies,
        including their configurations and metadata.

        Returns:
            List of strategy information dictionaries
        """
        strategies = []

        for strategy in ChunkingStrategyEnum:
            strategy_def = ChunkingStrategyRegistry.get_strategy_definition(strategy)

            # Get default config from builder
            default_config = self.config_builder.get_default_config(strategy)

            # Get compatibility info from factory
            strategy_info = self.strategy_factory.get_strategy_info(strategy)

            strategies.append({
                "id": strategy.value,
                "name": strategy_def.get("name", strategy.value),
                "description": strategy_def.get("description", ""),
                "best_for": strategy_def.get("best_for", []),
                "pros": strategy_def.get("pros", []),
                "cons": strategy_def.get("cons", []),
                "default_config": default_config,
                "performance_characteristics": strategy_def.get("performance_characteristics", {}),
                "available": strategy_info.get("available", True),
            })

        return strategies

    async def apply_chunking(
        self,
        document_id: str,
        strategy: str | ChunkingStrategyEnum,
        config_overrides: dict[str, Any] | None = None,
        user_id: int | None = None,
    ) -> str:
        """Apply chunking to a document.

        This method contains the business logic for applying chunking
        to a document, including validation and operation creation.

        Args:
            document_id: Document to chunk
            strategy: Strategy to use
            config_overrides: Optional config overrides
            user_id: User initiating the operation

        Returns:
            Operation ID for tracking

        Raises:
            ValidationException: If validation fails
            InfraPermissionDeniedException: If user lacks access
            ResourceNotFoundException: If document doesn't exist
        """
        correlation_id = str(uuid.uuid4())

        # Validate user access
        if not user_id:
            raise InfraPermissionDeniedException(
                user_id="anonymous",
                resource="chunking_operation",
                action="create",
                correlation_id=correlation_id,
            )

        # Validate document access
        await self._validate_document_access(document_id, user_id)

        # Build and validate configuration
        config_result = self.config_builder.build_config(
            strategy=strategy,
            user_config=config_overrides,
        )

        if config_result.validation_errors:
            raise ValidationException(
                field="config",
                value=config_overrides,
                reason=f"Invalid configuration: {', '.join(config_result.validation_errors)}",
                correlation_id=correlation_id,
            )

        # Create operation
        operation_id = await self.start_chunking_operation(
            collection_id="",  # Would get from document
            strategy=str(config_result.strategy),
            config=config_result.config,
            user_id=user_id,
        )

        return operation_id.get("operation_id", str(uuid.uuid4()))

    async def recommend_strategy(
        self,
        content_size: int | None = None,
        file_types: list[str] | None = None,
        file_paths: list[str] | None = None,
        has_structure: bool | None = None,
    ) -> dict[str, Any]:
        """Recommend a chunking strategy based on content characteristics.

        Args:
            content_size: Size of content in bytes (optional)
            file_types: List of file types to analyze (optional)
            file_paths: List of file paths to analyze (optional)
            has_structure: Whether content has structure (markdown, etc.)

        Returns:
            Strategy recommendation with reasoning
        """

        # Extract file types from paths if provided
        if file_paths and not file_types:
            file_types = [Path(path).suffix for path in file_paths]

        # Analyze file type breakdown
        file_type_breakdown: dict[str, int] = {}
        if file_types:
            for ft in file_types:
                category = self._categorize_file_type(ft)
                file_type_breakdown[category] = file_type_breakdown.get(category, 0) + 1

        # Determine if content has structure
        if has_structure is None and file_types:
            has_structure = any(ft in [".md", ".markdown", ".mdx", ".rst"] for ft in file_types)

        # Determine strategy based on file type breakdown
        if file_type_breakdown.get("markdown", 0) > len(file_types) / 2 if file_types else False:
            recommended_strategy = ChunkingStrategyEnum.RECURSIVE
            reasoning = "Majority of files are markdown with structure to preserve"
            chunk_size = 600
        elif file_type_breakdown.get("code", 0) > 0:
            recommended_strategy = ChunkingStrategyEnum.RECURSIVE
            reasoning = "Code files detected, using optimized settings for code"
            chunk_size = 500
        elif has_structure:
            recommended_strategy = ChunkingStrategyEnum.RECURSIVE
            reasoning = "Content has markdown structure that should be preserved"
            chunk_size = 600
        elif content_size and content_size < 10000:  # Small documents
            recommended_strategy = ChunkingStrategyEnum.RECURSIVE
            reasoning = "Small document size works well with simple splitting"
            chunk_size = 600
        elif content_size and content_size > 1000000:  # Large documents
            recommended_strategy = ChunkingStrategyEnum.RECURSIVE
            reasoning = "Large documents benefit from intelligent sentence-aware splitting"
            chunk_size = 600
        else:
            recommended_strategy = ChunkingStrategyEnum.RECURSIVE
            reasoning = "Default recommendation for general mixed content"
            chunk_size = 600

        return {
            "strategy": recommended_strategy,
            "reasoning": reasoning,
            "params": {"chunk_size": chunk_size, "chunk_overlap": 100},
            "chunk_size": chunk_size,
            "chunk_overlap": 100,
            "confidence": 0.8 if file_type_breakdown else 0.6,
            "file_type_breakdown": file_type_breakdown,
            "alternatives": self._get_alternative_strategies(recommended_strategy),
        }

    async def preview_chunks(
        self,
        strategy: str | ChunkingStrategyEnum,
        content: str | None = None,
        document_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        user_id: int | None = None,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Preview chunking results with all business logic.

        This method contains all the business logic that was previously
        in the router, ensuring proper separation of concerns.

        Args:
            strategy: Chunking strategy to use
            content: Optional content to chunk
            document_id: Optional document ID to load content from
            config_overrides: Optional configuration overrides
            user_id: Optional user ID for access validation
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Preview results with chunks and metadata

        Raises:
            ApplicationException: Translated application-level exceptions
        """
        correlation_id = correlation_id or str(uuid.uuid4())

        try:
            # Input validation - must have either content or document_id
            if not content and not document_id:
                raise ValidationException(
                    field="input",
                    value=None,
                    reason="Either content or document_id must be provided",
                    correlation_id=correlation_id,
                )

            if content and document_id:
                raise ValidationException(
                    field="input",
                    value="both content and document_id",
                    reason="Cannot provide both content and document_id",
                    correlation_id=correlation_id,
                )

            # Document access validation if document_id provided
            if document_id:
                if not user_id:
                    raise InfraPermissionDeniedException(
                        user_id="anonymous",
                        resource=f"document:{document_id}",
                        action="read",
                        correlation_id=correlation_id,
                    )

                # Validate document access with exception handling
                try:
                    await self._validate_document_access(document_id, user_id)
                    content = await self._load_document_content(document_id)
                except ResourceNotFoundException:
                    # Already an infrastructure exception, just re-raise
                    raise
                except Exception as e:
                    if isinstance(e, ApplicationException):
                        raise
                    # Translate infrastructure exception
                    raise self.exception_translator.translate_infrastructure_to_application(
                        e,
                        {"resource_type": "Document", "resource_id": document_id},
                    )

            # Validate content size
            if content and len(content) > 10_000_000:  # 10MB limit
                raise DocumentTooLargeError(
                    size=len(content),
                    max_size=10_000_000,
                    correlation_id=correlation_id,
                )

            # Build and validate configuration
            config_result = self.config_builder.build_config(
                strategy=strategy,
                user_config=config_overrides,
            )

            if config_result.validation_errors:
                raise ValidationException(
                    field="config",
                    value=config_overrides,
                    reason=f"Invalid configuration: {', '.join(config_result.validation_errors)}",
                    correlation_id=correlation_id,
                )

            # Create strategy instance with domain exception handling
            try:
                chunking_strategy = self.strategy_factory.create_strategy(
                    strategy_name=config_result.strategy,
                    config=config_result.config,
                    correlation_id=correlation_id,
                )
            except StrategyNotFoundError as e:
                raise ChunkingStrategyError(
                    strategy=str(strategy),
                    reason=f"Strategy not found: {e.strategy_name}",
                    correlation_id=correlation_id,
                    cause=e,
                )
            except ChunkingDomainError as e:
                # Translate domain exception
                raise self.exception_translator.translate_domain_to_application(
                    e,
                    correlation_id,
                )
            except Exception as e:
                raise ChunkingStrategyError(
                    strategy=str(strategy),
                    reason=str(e),
                    correlation_id=correlation_id,
                    cause=e,
                )

            # Execute chunking with proper exception handling
            try:
                result = await self._execute_chunking(
                    strategy=chunking_strategy,
                    content=content or "",
                    config=config_result.config,
                    strategy_name=config_result.strategy,
                )
            except DomainException as e:
                # Translate domain exception
                raise self.exception_translator.translate_domain_to_application(
                    e,
                    correlation_id,
                )
            except TimeoutError as e:
                raise ChunkingStrategyError(
                    strategy=str(strategy),
                    reason="Processing timeout exceeded",
                    correlation_id=correlation_id,
                    cause=e,
                )
            except MemoryError as e:
                raise DocumentTooLargeError(
                    size=len(content) if content else 0,
                    max_size=10_000_000,
                    correlation_id=correlation_id,
                    cause=e,
                )
            except Exception as e:
                # Log unexpected error with full context
                logger.exception(
                    "Unexpected error during chunking",
                    extra={
                        "correlation_id": correlation_id,
                        "strategy": str(strategy),
                        "error_type": type(e).__name__,
                    },
                )
                raise ApplicationException(
                    message="Unexpected error during chunking",
                    code="CHUNKING_ERROR",
                    details={
                        "strategy": str(strategy),
                        "error": str(e),
                        "type": type(e).__name__,
                    },
                    correlation_id=correlation_id,
                    cause=e,
                )

            # Cache preview result
            preview_id = await self._cache_preview_result(result, str(strategy))
            result["preview_id"] = preview_id
            result["correlation_id"] = correlation_id

            return result

        except ApplicationException:
            # Already translated, just re-raise
            raise
        except DomainException:
            # Domain exceptions should be re-raised as-is
            # They'll be translated at the API layer
            raise
        except Exception as e:
            # Catch-all for unexpected errors
            logger.exception(
                "Unexpected error in preview_chunks",
                extra={"correlation_id": correlation_id},
            )
            raise ApplicationException(
                message="An unexpected error occurred",
                code="INTERNAL_ERROR",
                correlation_id=correlation_id,
                cause=e,
            )

    async def _validate_document_access(self, document_id: str, user_id: int) -> None:
        """Validate user has access to document."""
        # This would use actual repository methods
        # For now, simplified implementation
        document = await self.document_repo.get_by_id(document_id)
        if not document:
            raise ResourceNotFoundException(
                resource_type="Document",
                resource_id=document_id,
                correlation_id=str(uuid.uuid4()),
            )

        # Check collection permissions
        # This would check actual permissions in production
        # Simplified for now

    async def _load_document_content(self, document_id: str) -> str:
        """Load document content from storage."""
        document = await self.document_repo.get_by_id(document_id)
        if not document:
            raise ResourceNotFoundException(
                resource_type="Document",
                resource_id=document_id,
                correlation_id=str(uuid.uuid4()),
            )

        # Load from appropriate storage
        # Simplified - would handle different storage types
        return document.get("content", "")

    async def _execute_chunking(
        self,
        strategy: Any,
        content: str,
        config: dict[str, Any],
        strategy_name: Any,
    ) -> dict[str, Any]:
        """Execute chunking with resource limits."""
        import time

        # Apply resource limits
        max_content_size = 10_000_000  # 10MB
        if len(content) > max_content_size:
            raise DocumentTooLargeError(
                size=len(content),
                max_size=max_content_size,
                correlation_id=str(uuid.uuid4()),
            )

        # Execute with timeout
        start_time = time.time()

        try:
            # Import necessary components
            from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig

            # Build ChunkConfig
            chunk_size = config.get("chunk_size", 1000)
            chunk_overlap = config.get("chunk_overlap", 200)

            # Ensure overlap is not too large
            if chunk_overlap >= chunk_size:
                chunk_overlap = min(200, chunk_size // 4)

            # Build ChunkConfig with proper parameters
            min_tokens = min(100, chunk_size // 2)
            max_tokens = max(chunk_size, min_tokens + 1)
            overlap_tokens = min(chunk_overlap, min_tokens - 1)

            chunk_config = ChunkConfig(
                strategy_name=str(strategy_name),
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                overlap_tokens=max(0, overlap_tokens),
                preserve_structure=config.get("preserve_structure", True),
                semantic_threshold=config.get("semantic_threshold", 0.7),
                hierarchy_levels=config.get("hierarchy_levels", 3),
            )

            # Execute chunking with timeout
            chunks = await asyncio.wait_for(
                asyncio.to_thread(
                    strategy.chunk,
                    content,
                    chunk_config,
                ),
                timeout=30.0,
            )

            # Extract text content from chunk entities
            chunk_texts = [chunk.content for chunk in chunks]

        except TimeoutError:
            raise ChunkingStrategyError(
                strategy=str(strategy_name),
                reason="Chunking operation timed out",
                correlation_id=str(uuid.uuid4()),
            )
        except Exception as e:
            logger.warning(f"Strategy failed, falling back: {e}")
            # Fallback to simple chunking
            chunk_texts = self._simple_chunk_fallback(content, chunk_size, chunk_overlap)

        processing_time_ms = (time.time() - start_time) * 1000

        # Calculate statistics
        statistics = self._calculate_statistics(chunk_texts)

        # Format result
        return {
            "strategy": str(strategy_name),
            "config": config,
            "chunks": [
                {
                    "chunk_id": f"chunk_{idx:04d}",
                    "content": chunk,
                    "text": chunk,
                    "metadata": {"index": idx, "strategy": str(strategy_name)},
                    "index": idx,
                    "size": len(chunk),
                }
                for idx, chunk in enumerate(chunk_texts)
            ],
            "total_chunks": len(chunk_texts),
            "metrics": statistics,
            "statistics": statistics,
            "processing_time_ms": processing_time_ms,
            "cached": False,
            "expires_at": (datetime.now(UTC) + timedelta(minutes=15)).isoformat(),
        }

    def _simple_chunk_fallback(
        self, content: str, chunk_size: int, chunk_overlap: int
    ) -> list[str]:
        """Simple fallback chunking."""
        chunks = []
        if chunk_overlap >= chunk_size:
            chunk_overlap = min(chunk_overlap, chunk_size // 4)

        step_size = max(1, chunk_size - chunk_overlap)
        for i in range(0, len(content), step_size):
            chunk_content = content[i : i + chunk_size]
            if chunk_content.strip():
                chunks.append(chunk_content)

        return chunks

    def _calculate_statistics(self, chunks: list[str]) -> dict[str, Any]:
        """Calculate chunking statistics."""
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "size_variance": 0.0,
                "quality_score": 0.0,
            }

        sizes = [len(chunk) for chunk in chunks]
        avg_size = sum(sizes) / len(sizes)

        # Calculate variance
        variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes) if len(sizes) > 1 else 0

        # Simple quality score based on consistency
        quality_score = 1.0 - min(1.0, variance / (avg_size ** 2)) if avg_size > 0 else 0.0

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": avg_size,
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "size_variance": variance,
            "quality_score": quality_score,
        }

    async def _cache_preview_result(
        self, result: dict[str, Any], strategy: str
    ) -> str:
        """Cache preview result and return preview ID."""
        preview_id = str(uuid.uuid4())

        if self.redis_client:
            try:
                cache_key = f"preview:{preview_id}"
                await self.redis_client.setex(
                    cache_key,
                    1800,  # 30 minutes
                    json.dumps(result),
                )
            except Exception as e:
                logger.warning(f"Failed to cache preview: {e}")

        return preview_id

    async def preview_chunking(
        self,
        content: str,
        strategy: str | None = None,
        config: dict[str, Any] | None = None,
        file_type: str | None = None,
        max_chunks: int | None = None,
        cache_result: bool = True,
    ) -> dict[str, Any]:
        """Generate a preview of how content will be chunked.

        Args:
            content: Text content to chunk
            strategy: Chunking strategy to use (optional, defaults to recursive)
            config: Configuration for the strategy
            file_type: File type for determining chunking behavior
            max_chunks: Maximum number of chunks to return in preview
            cache_result: Whether to cache the result

        Returns:
            Preview response with chunks and metadata
        """
        from packages.webui.services.chunking_constants import MAX_PREVIEW_CONTENT_SIZE
        from packages.webui.services.chunking_security import ValidationError

        # Check size limit
        if len(content) > MAX_PREVIEW_CONTENT_SIZE:
            raise ValidationError("Document too large")

        # Validate chunk size if provided in config (do this before try block)
        if config and "params" in config and "chunk_size" in config["params"]:
            chunk_size = config["params"]["chunk_size"]
            if chunk_size <= 0 or chunk_size > 10000:
                raise ValidationError(f"Invalid chunk size: {chunk_size}. Must be between 1 and 10000.")

        try:
            # Check if cached result exists
            if cache_result and self.redis_client:
                # Get strategy string for cache key before conversion
                cache_strategy = strategy.value if strategy and hasattr(strategy, "value") else strategy or "recursive"
                cache_key = self._generate_cache_key(content, cache_strategy, config)
                cached = await self.redis_client.get(cache_key)
                if cached:
                    import json

                    result = json.loads(cached)
                    return dict(result)

            # Default strategy if not provided
            if not strategy:
                strategy = ChunkingStrategyEnum.RECURSIVE

            # Convert enum to string if necessary
            strategy_str = strategy.value if hasattr(strategy, "value") else str(strategy)

            # Map API strategy names to internal factory names
            internal_strategy = self.STRATEGY_MAPPING.get(strategy_str, strategy_str)

            # Use the actual strategy pattern for chunking
            strategy_instance = get_strategy(internal_strategy)

            import time

            start_time = time.time()

            # Create ChunkConfig from the provided config dictionary
            from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig

            # Use sensible defaults if no config provided
            chunk_size = config.get("chunk_size", 1000) if config else 1000
            chunk_overlap = config.get("chunk_overlap", 200) if config else 200

            # Ensure overlap is not too large
            if chunk_overlap >= chunk_size:
                chunk_overlap = min(200, chunk_size // 4)

            # Build ChunkConfig with proper parameters
            # Ensure min_tokens is less than max_tokens and overlap_tokens is valid
            min_tokens = min(100, chunk_size // 2)  # Set reasonable min
            max_tokens = max(chunk_size, min_tokens + 1)  # Ensure max > min
            overlap_tokens = min(chunk_overlap, min_tokens - 1)  # Ensure overlap < min

            chunk_config = ChunkConfig(
                strategy_name=strategy,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                overlap_tokens=max(0, overlap_tokens),  # Ensure non-negative
                preserve_structure=config.get("preserve_structure", True) if config else True,
                semantic_threshold=config.get("semantic_threshold", 0.7) if config else 0.7,
                hierarchy_levels=config.get("hierarchy_levels", 3) if config else 3,
            )

            # Use the strategy to generate chunks properly
            try:
                chunk_entities = strategy_instance.chunk(
                    content=content,
                    config=chunk_config,
                    progress_callback=None,  # Could add progress tracking if needed
                )

                # Extract text content from chunk entities
                chunks = [chunk.content for chunk in chunk_entities]

            except Exception as strategy_error:
                logger.warning(f"Strategy {strategy} failed, falling back to simple chunking: {strategy_error}")
                # Fallback to simple chunking only if strategy fails
                chunks = []
                if chunk_overlap >= chunk_size:
                    # If overlap is too large, reset to reasonable default
                    chunk_overlap = min(chunk_overlap, chunk_size // 4)

                step_size = max(1, chunk_size - chunk_overlap)  # Ensure positive step
                for i in range(0, len(content), step_size):
                    chunk_content = content[i : i + chunk_size]
                    if chunk_content.strip():  # Only add non-empty chunks
                        chunks.append(chunk_content)

            processing_time_ms = (time.time() - start_time) * 1000
            avg_chunk_size = sum(len(c) for c in chunks) / len(chunks) if chunks else 0

            # Determine if it's a code file
            is_code_file = False
            if file_type:
                code_extensions = [".py", ".js", ".ts", ".cpp", ".c", ".java", ".go", ".rs"]
                is_code_file = file_type in code_extensions

            # Limit chunks if max_chunks specified
            preview_chunks = chunks
            if max_chunks:
                preview_chunks = chunks[:max_chunks]

            # Get recommendations
            recommendations = self._get_recommendations(chunks, file_type)

            # Convert to response format
            result = {
                "preview_id": str(uuid.uuid4()),
                "strategy": strategy_str if hasattr(strategy, "value") else strategy,
                "strategy_used": strategy_str if hasattr(strategy, "value") else strategy,  # For compatibility
                "config": config or self._get_default_config(internal_strategy),
                "chunks": [
                    {
                        "chunk_id": f"chunk_{idx:04d}",
                        "text": chunk,
                        "content": chunk,
                        "metadata": {"index": idx, "strategy": strategy},
                        "index": idx,
                        "size": len(chunk),
                    }
                    for idx, chunk in enumerate(preview_chunks)
                ],
                "total_chunks": len(chunks),
                "avg_chunk_size": avg_chunk_size,
                "processing_time_ms": processing_time_ms,
                "is_code_file": is_code_file,
                "recommendations": recommendations,
                "performance_metrics": self._calculate_metrics(chunks, len(content), processing_time_ms / 1000),
                "expires_at": (datetime.now(UTC) + timedelta(minutes=30)).isoformat(),
            }

            # Cache if requested and Redis is available
            if cache_result and self.redis_client:
                cache_key = self._generate_cache_key(content, cache_strategy, config)
                await self._cache_preview(cache_key, result)

            return result

        except InvalidConfigurationError as e:
            logger.error(f"Invalid chunking configuration: {e}")
            # Return safe error message without exposing internal details
            return {
                "error": "Invalid chunking configuration provided",
                "strategy": strategy or "recursive",
                "chunks": [],
                "total_chunks": 0,
                "recommendations": ["Please check your chunking configuration parameters"],
            }
        except ValueError as e:
            logger.error(f"Invalid input value: {e}")
            # Return safe error message
            return {
                "error": "Invalid input parameters",
                "strategy": strategy or "recursive",
                "chunks": [],
                "total_chunks": 0,
                "recommendations": ["Please verify your input parameters"],
            }
        except ConnectionError as e:
            # Handle Redis connection errors specifically
            logger.error(f"Redis connection error during preview chunking: {e}")
            return {
                "error": str(e),  # Include the connection error message
                "strategy": strategy or "recursive",
                "chunks": [],
                "total_chunks": 0,
                "recommendations": ["Service temporarily unavailable. Please try again later"],
            }
        except Exception as e:
            # Log the actual error internally but don't expose details to client
            logger.error(f"Unexpected error during preview chunking: {type(e).__name__}: {e}")
            # Return generic safe error response
            return {
                "error": "An unexpected error occurred during chunking",
                "strategy": strategy or "recursive",
                "chunks": [],
                "total_chunks": 0,
                "recommendations": ["Please try again or contact support if the issue persists"],
            }

    async def compare_strategies(
        self,
        content: str,
        strategies: list[ChunkingStrategyEnum],
        configs: dict[str, dict[str, Any]] | None = None,
        max_chunks_per_strategy: int = 5,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        """Compare multiple chunking strategies with full business logic.

        This method contains all the comparison logic that was previously
        in the router, ensuring proper separation of concerns.

        Args:
            content: Content to chunk
            strategies: List of strategies to compare
            configs: Optional per-strategy configurations
            max_chunks_per_strategy: Maximum chunks to return per strategy
            user_id: Optional user ID for access validation

        Returns:
            Full comparison results with recommendations
        """
        from datetime import UTC, datetime

        correlation_id = str(uuid.uuid4())
        comparisons = []
        processing_start = datetime.now(UTC)

        # Process each strategy
        for strategy in strategies:
            try:
                # Get config for this strategy
                config = None
                if configs and strategy.value in configs:
                    config = configs[strategy.value]

                # Execute preview for this strategy
                result = await self.preview_chunking(
                    content=content,
                    strategy=strategy,
                    config=config,
                    max_chunks=max_chunks_per_strategy,
                )

                # Get strategy definition from registry
                strategy_def = ChunkingStrategyRegistry.get_strategy_definition(strategy)

                # Calculate quality metrics
                metrics = result.get("metrics", {})
                if not metrics:
                    # Calculate basic metrics if not provided
                    chunks = result.get("chunks", [])
                    if chunks:
                        sizes = [chunk.get("size", len(chunk.get("content", ""))) for chunk in chunks]
                        avg_size = sum(sizes) / len(sizes) if sizes else 0
                        variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes) if len(sizes) > 1 else 0
                        quality_score = 1.0 - min(1.0, variance / (avg_size ** 2)) if avg_size > 0 else 0.0
                    else:
                        avg_size = 0
                        variance = 0
                        quality_score = 0.0

                    metrics = {
                        "avg_chunk_size": avg_size,
                        "size_variance": variance,
                        "quality_score": quality_score,
                    }

                # Build comparison entry
                comparison_entry = {
                    "strategy": strategy,
                    "config": result.get("config", self._get_default_config(strategy.value)),
                    "sample_chunks": result.get("chunks", [])[:max_chunks_per_strategy],
                    "total_chunks": result.get("total_chunks", 0),
                    "avg_chunk_size": metrics.get("avg_chunk_size", 0),
                    "size_variance": metrics.get("size_variance", 0),
                    "quality_score": metrics.get("quality_score", 0),
                    "processing_time_ms": result.get("processing_time_ms", 0),
                    "pros": strategy_def.get("pros", []),
                    "cons": strategy_def.get("cons", []),
                }

                comparisons.append(comparison_entry)

            except Exception as e:
                logger.warning(f"Failed to process strategy {strategy}: {e}")
                # Add failed strategy with error info
                comparisons.append({
                    "strategy": strategy,
                    "config": self._get_default_config(strategy.value) if hasattr(strategy, "value") else {},
                    "sample_chunks": [],
                    "total_chunks": 0,
                    "avg_chunk_size": 0,
                    "size_variance": 0,
                    "quality_score": 0,
                    "processing_time_ms": 0,
                    "pros": [],
                    "cons": [],
                    "error": str(e),
                })

        # Generate recommendation based on comparison
        if comparisons:
            # Filter out failed comparisons
            valid_comparisons = [c for c in comparisons if "error" not in c]

            if valid_comparisons:
                # Find best strategy by quality score
                best_comparison = max(valid_comparisons, key=lambda x: x.get("quality_score", 0))
                best_strategy = best_comparison["strategy"]

                # Build recommendation
                recommendation = {
                    "recommended_strategy": best_strategy,
                    "confidence": best_comparison.get("quality_score", 0.5),
                    "reasoning": f"Based on quality score analysis, {best_strategy.value} provides the best chunking for this content",
                    "alternative_strategies": [
                        c["strategy"] for c in valid_comparisons
                        if c["strategy"] != best_strategy
                    ],
                    "suggested_config": best_comparison.get("config", {}),
                }
            else:
                # All strategies failed
                recommendation = {
                    "recommended_strategy": ChunkingStrategyEnum.RECURSIVE,
                    "confidence": 0.3,
                    "reasoning": "Unable to properly compare strategies, using default recommendation",
                    "alternative_strategies": [],
                    "suggested_config": self._get_default_config("recursive"),
                }
        else:
            # No comparisons at all
            recommendation = {
                "recommended_strategy": ChunkingStrategyEnum.RECURSIVE,
                "confidence": 0.3,
                "reasoning": "No strategies provided for comparison",
                "alternative_strategies": [],
                "suggested_config": self._get_default_config("recursive"),
            }

        processing_time = int((datetime.now(UTC) - processing_start).total_seconds() * 1000)

        return {
            "comparison_id": correlation_id,
            "comparisons": comparisons,
            "recommendation": recommendation,
            "processing_time_ms": processing_time,
        }

    async def start_chunking_operation(
        self,
        collection_id: str,
        strategy: str,
        config: dict[str, Any],
        user_id: int,
    ) -> dict[str, Any]:
        """Start a chunking operation for a collection.

        Args:
            collection_id: ID of the collection to chunk
            strategy: Chunking strategy to use
            config: Strategy configuration
            user_id: ID of the user initiating the operation

        Returns:
            Operation details
        """
        try:
            # Validate collection exists and user has access
            collection = await self.collection_repo.get_by_uuid(collection_id)
            if not collection:
                raise ValueError(f"Collection {collection_id} not found")

            # Create operation record
            operation_id = str(uuid.uuid4())
            operation = Operation(
                uuid=operation_id,
                collection_id=collection_id,
                type="chunking",
                status="pending",
                user_id=user_id,
                config={
                    "strategy": strategy,
                    "chunk_config": config,
                },
                created_at=datetime.now(UTC),
            )

            self.db_session.add(operation)
            await self.db_session.commit()

            # Invalidate cache for this collection
            if self.cache_manager:
                await self.cache_manager.invalidate_collection(collection_id)

            # Queue the operation for processing
            # This would typically trigger a Celery task
            logger.info(f"Queued chunking operation {operation_id} for collection {collection_id}")

            return {
                "operation_id": operation_id,
                "collection_id": collection_id,
                "status": "pending",
                "strategy": strategy,
                "config": config,
                "created_at": operation.created_at.isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to start chunking operation: {e}")
            await self.db_session.rollback()
            raise

    async def get_chunking_progress(self, operation_id: str) -> dict[str, Any]:
        """Get progress of a chunking operation.

        Args:
            operation_id: ID of the operation

        Returns:
            Progress information
        """
        try:
            # Get operation from database
            result = await self.db_session.execute(select(Operation).where(Operation.id == operation_id))
            operation = result.scalar_one_or_none()

            if not operation:
                raise ValueError(f"Operation {operation_id} not found")

            # Calculate progress based on operation status and metadata
            progress_pct = 0.0
            if operation.status == "completed":
                progress_pct = 100.0
            elif operation.status == "in_progress" and operation.meta:
                chunks_processed = operation.meta.get("chunks_processed", 0)
                total_chunks = operation.meta.get("total_chunks", 0)
                if total_chunks > 0:
                    progress_pct = (chunks_processed / total_chunks) * 100

            return {
                "operation_id": operation_id,
                "status": operation.status,
                "progress_percentage": progress_pct,
                "chunks_processed": operation.meta.get("chunks_processed", 0) if operation.meta else 0,
                "total_chunks": operation.meta.get("total_chunks", 0) if operation.meta else 0,
                "started_at": operation.started_at.isoformat() if operation.started_at else None,
                "error": operation.error_message,
            }

        except Exception as e:
            logger.error(f"Failed to get chunking progress: {e}")
            raise

    async def get_metrics_by_strategy(
        self,
        period_days: int = 30,
        user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get chunking metrics grouped by strategy.

        This method provides strategy-specific metrics for the specified period.

        Args:
            period_days: Number of days to look back
            user_id: Optional user ID for filtering

        Returns:
            List of strategy metrics dictionaries
        """
        metrics = []

        # Import the strategy enum here to avoid circular imports
        from packages.webui.api.v2.chunking_schemas import ChunkingStrategy

        for strategy in ChunkingStrategy:
            # Get strategy definition from registry
            strategy_def = ChunkingStrategyRegistry.get_strategy_definition(strategy)

            # In production, would fetch actual metrics from database
            # For now, returning placeholder data
            metrics.append({
                "strategy": strategy,
                "usage_count": 0,
                "avg_chunk_size": 512,
                "avg_processing_time": 1.5,
                "success_rate": 0.95,
                "avg_quality_score": 0.8,
                "best_for_types": strategy_def.get("best_for", []),
            })

        return metrics

    @QueryMonitor.monitor("get_chunking_statistics")
    async def get_chunking_statistics(self, collection_id: str) -> dict[str, Any]:
        """Get chunking statistics for a collection with optimized queries.

        Args:
            collection_id: ID of the collection

        Returns:
            Statistics about chunking with caching
        """
        from sqlalchemy import case

        # Try to get from cache first if available
        if self.cache_manager:
            cache_key = self.cache_manager._generate_cache_key(
                "statistics",
                {"collection_id": collection_id}
            )
            cached = await self.cache_manager.get(cache_key)
            if cached:
                return cached

        try:
            # Get collection
            collection = await self.collection_repo.get_by_uuid(collection_id)
            if not collection:
                raise ValueError(f"Collection {collection_id} not found")

            # Single aggregation query instead of loading all records
            stats_query = select(
                func.count(Operation.id).label('total_operations'),
                func.count(
                    case(
                        (Operation.status == 'completed', Operation.id),
                        else_=None
                    )
                ).label('completed_operations'),
                func.count(
                    case(
                        (Operation.status == 'failed', Operation.id),
                        else_=None
                    )
                ).label('failed_operations'),
                func.count(
                    case(
                        (Operation.status == 'processing', Operation.id),
                        else_=None
                    )
                ).label('processing_operations'),
                func.avg(
                    case(
                        (and_(Operation.status == 'completed', Operation.completed_at.isnot(None), Operation.started_at.isnot(None)),
                         func.extract('epoch', Operation.completed_at - Operation.started_at)),
                        else_=None
                    )
                ).label('avg_processing_time'),
                func.max(Operation.created_at).label('last_operation_at'),
                func.min(Operation.created_at).label('first_operation_at')
            ).where(
                and_(
                    Operation.collection_id == collection_id,
                    Operation.type == 'chunking'
                )
            )

            result = await self.db_session.execute(stats_query)
            stats = result.one()

            # Get latest strategy with a separate optimized query
            latest_strategy_query = select(
                Operation.config['strategy'].label('strategy')
            ).where(
                and_(
                    Operation.collection_id == collection_id,
                    Operation.type == 'chunking',
                    Operation.config.isnot(None)
                )
            ).order_by(
                Operation.created_at.desc()
            ).limit(1)

            strategy_result = await self.db_session.execute(latest_strategy_query)
            latest_strategy_row = strategy_result.one_or_none()
            latest_strategy = latest_strategy_row.strategy if latest_strategy_row else None

            result = {
                "collection_id": collection_id,
                "total_operations": stats.total_operations or 0,
                "completed_operations": stats.completed_operations or 0,
                "failed_operations": stats.failed_operations or 0,
                "in_progress_operations": stats.processing_operations or 0,
                "latest_strategy": latest_strategy,
                "last_operation_at": stats.last_operation_at.isoformat() if stats.last_operation_at else None,
                "avg_processing_time": float(stats.avg_processing_time or 0),
                "first_operation_at": stats.first_operation_at.isoformat() if stats.first_operation_at else None,
            }

            # Cache the result if cache manager is available
            if self.cache_manager:
                cache_key = self.cache_manager._generate_cache_key(
                    "statistics",
                    {"collection_id": collection_id}
                )
                await self.cache_manager.set(cache_key, result, ttl=60)  # Cache for 1 minute

            return result

        except Exception as e:
            logger.error(f"Failed to get chunking statistics: {e}")
            raise

    async def validate_config_for_collection(
        self, collection_id: str, strategy: str, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate chunking configuration for a collection.

        Args:
            collection_id: ID of the collection
            strategy: Chunking strategy
            config: Configuration to validate

        Returns:
            Validation results
        """
        try:
            # Get collection
            collection = await self.collection_repo.get_by_uuid(collection_id)
            if not collection:
                raise ValueError(f"Collection {collection_id} not found")

            # Validate configuration
            is_valid, errors = self.validator.validate_config(strategy, config)

            # Check collection-specific constraints
            if is_valid:
                # Add collection-specific validation if needed
                pass

            return {
                "valid": is_valid,
                "errors": errors,
                "warnings": [],
                "suggested_config": self._get_default_config(strategy) if not is_valid else config,
            }

        except Exception as e:
            logger.error(f"Failed to validate config: {e}")
            raise

    # Cache management methods

    async def get_cached_preview(
        self,
        preview_id: str,
        user_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Get cached preview by ID with proper user validation.

        This is a PUBLIC method that replaces direct cache key construction
        in the router layer. It handles all cache-related business logic.

        Args:
            preview_id: Preview ID to retrieve
            user_id: User ID for validation (optional)

        Returns:
            Preview data with all necessary fields or None if not found
        """
        if not self.redis_client:
            return None

        # Construct cache key with user ID if provided
        if user_id:
            cache_key = f"preview:{preview_id}:{user_id}"
        else:
            cache_key = f"preview:{preview_id}"

        try:
            data = await self.redis_client.get(cache_key)
            if data:
                result = json.loads(data)
                # Ensure all required fields are present
                from datetime import UTC, datetime, timedelta

                if "preview_id" not in result:
                    result["preview_id"] = preview_id

                if "expires_at" not in result:
                    result["expires_at"] = (datetime.now(UTC) + timedelta(minutes=15)).isoformat()

                if "cached" not in result:
                    result["cached"] = True

                return dict(result)
        except Exception as e:
            logger.warning(f"Failed to get cached preview: {e}")

        return None

    async def cache_preview_with_user(
        self,
        preview_id: str,
        preview_data: dict[str, Any],
        user_id: int | None = None,
        ttl: int = 1800,
    ) -> None:
        """Cache preview data with user context.

        Public method for caching preview data with proper user association.

        Args:
            preview_id: Preview ID
            preview_data: Data to cache
            user_id: User ID (optional)
            ttl: Time to live in seconds (default 30 minutes)
        """
        if not self.redis_client:
            return

        # Construct cache key with user ID if provided
        if user_id:
            cache_key = f"preview:{preview_id}:{user_id}"
        else:
            cache_key = f"preview:{preview_id}"

        try:
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(preview_data),
            )
        except Exception as e:
            logger.warning(f"Failed to cache preview: {e}")

    async def _cache_preview(self, cache_key: str, preview_data: dict[str, Any], ttl: int = 1800) -> None:
        """Cache preview data in Redis.

        PRIVATE method for internal use only.

        Args:
            cache_key: Cache key
            preview_data: Data to cache
            ttl: Time to live in seconds (default 30 minutes)
        """
        if not self.redis_client:
            return

        try:
            await self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(preview_data),
            )
        except Exception as e:
            logger.warning(f"Failed to cache preview: {e}")

    async def _get_cached_preview_by_key(self, cache_key: str) -> dict[str, Any] | None:
        """Get cached preview by key.

        PRIVATE method for internal use only.

        Args:
            cache_key: Cache key

        Returns:
            Cached preview data or None
        """
        if not self.redis_client:
            return None

        try:
            data = await self.redis_client.get(cache_key)
            if data:
                result = json.loads(data)
                return dict(result)
        except Exception as e:
            logger.warning(f"Failed to get cached preview: {e}")

        return None

    async def clear_preview_cache(self, pattern: str | None = None) -> int:
        """Clear preview cache.

        Args:
            pattern: Optional pattern to match keys

        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0

        try:
            if pattern:
                keys = await self.redis_client.keys(f"preview:{pattern}*")
            else:
                keys = await self.redis_client.keys("preview:*")

            if keys:
                return await self.redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Failed to clear preview cache: {e}")

        return 0

    async def track_preview_usage(
        self,
        user_id: int | None = None,
        preview_id: str | None = None,
        action: str | None = None,
        strategy: str | None = None,
        file_type: str | None = None,
    ) -> None:
        """Track usage of a preview.

        Args:
            user_id: User ID for tracking
            preview_id: Preview ID (optional)
            action: Action performed (viewed, applied, etc.)
            strategy: Strategy used
            file_type: File type being processed
        """
        if not self.redis_client:
            return

        try:
            # Track by user and strategy
            if user_id and strategy:
                await self.redis_client.incr(f"chunking:preview:user:{user_id}:{strategy}")

            # Track overall strategy usage
            if strategy:
                await self.redis_client.incr(f"chunking:preview:usage:{strategy}")

            # Track file type usage
            if file_type:
                await self.redis_client.incr(f"chunking:preview:file_type:{file_type}")

            # Track specific preview if ID provided
            if preview_id and action:
                key = f"preview_usage:{preview_id}"
                await self.redis_client.hincrby(key, action, 1)
                await self.redis_client.expire(key, 86400)  # 24 hours
        except Exception as e:
            logger.warning(f"Failed to track preview usage: {e}")

    async def verify_collection_access(
        self,
        collection_id: str,
        user_id: int,
    ) -> None:
        """Verify user has access to collection.

        Args:
            collection_id: Collection UUID
            user_id: User ID
        """
        await self.collection_repo.get_by_uuid_with_permission_check(
            collection_uuid=collection_id,
            user_id=user_id,
        )

    def _calculate_metrics(
        self,
        chunks: list[Any],
        text_length: int,
        processing_time: float,
    ) -> dict[str, Any]:
        """Calculate metrics for chunking results.

        Args:
            chunks: List of chunks
            text_length: Original text length
            processing_time: Processing time in seconds

        Returns:
            Metrics dictionary
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "average_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "chunks_per_second": 0,
                "compression_ratio": 0,
            }

        chunk_sizes = [len(c.text if hasattr(c, "text") else c) for c in chunks]
        total_chunk_chars = sum(chunk_sizes)

        return {
            "total_chunks": len(chunks),
            "average_chunk_size": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
            "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            "chunks_per_second": len(chunks) / processing_time if processing_time > 0 else 0,
            "compression_ratio": total_chunk_chars / text_length if text_length > 0 else 1,
        }

    def _get_recommendations(
        self,
        chunks: list[Any],
        file_type: str | None = None,
    ) -> list[str]:
        """Get recommendations based on chunking results.

        Args:
            chunks: List of chunks
            file_type: File type being processed

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if not chunks:
            return ["No chunks generated - check your content and settings"]

        # Calculate chunk size statistics
        chunk_sizes = [len(c.text if hasattr(c, "text") else c) for c in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        min_size = min(chunk_sizes) if chunk_sizes else 0
        max_size = max(chunk_sizes) if chunk_sizes else 0

        # Check for high variance
        if max_size > 0 and min_size > 0:
            variance_ratio = max_size / min_size
            if variance_ratio > 5:
                recommendations.append("High chunk size variance detected - consider adjusting parameters")

        # Check for many small chunks
        small_chunks = [s for s in chunk_sizes if s < 100]
        if len(small_chunks) > len(chunks) * 0.3:
            recommendations.append("Many small chunks detected - consider increasing chunk size")

        # File type specific recommendations
        if file_type == ".py" and avg_size > 1000:
            recommendations.append("Large chunks for Python code - consider smaller chunks for better granularity")

        # General recommendations
        if len(chunks) > 100:
            recommendations.append("Large number of chunks - may impact search performance")

        if not recommendations:
            recommendations.append("Chunking parameters appear well-balanced")

        return recommendations

    # Helper methods

    def _generate_cache_key(self, content: str, strategy: str, config: dict[str, Any] | None) -> str:
        """Generate a cache key for preview data.

        Args:
            content: Content being chunked
            strategy: Strategy being used
            config: Configuration

        Returns:
            Cache key string
        """
        # Create a hash of the content and parameters
        hasher = hashlib.sha256()
        hasher.update(content.encode())
        hasher.update(strategy.encode())
        hasher.update(json.dumps(config or {}, sort_keys=True).encode())
        return f"preview:{hasher.hexdigest()[:16]}"

    def _get_default_config(self, strategy: str) -> dict[str, Any]:
        """Get default configuration for a strategy.

        Args:
            strategy: Strategy name

        Returns:
            Default configuration
        """
        defaults: dict[str, dict[str, Any]] = {
            "character": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
            "recursive": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separators": ["\n\n", "\n", " ", ""],
            },
            "markdown": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "include_metadata": True,
            },
            "semantic": {
                "buffer_size": 1,
                "breakpoint_percentile_threshold": 95,
                "embedding_model": "text-embedding-ada-002",
            },
            "hierarchical": {
                "chunk_sizes": [2048, 512],
                "chunk_overlap": 50,
            },
            "hybrid": {
                "primary_strategy": "recursive",
                "fallback_strategy": "character",
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
        }
        return defaults.get(strategy, {})

    def _get_alternative_strategies(self, primary_strategy: str) -> list[dict[str, str]]:
        """Get alternative strategies to the primary one.

        Args:
            primary_strategy: The primary recommended strategy

        Returns:
            List of alternative strategies with reasons
        """
        alternatives = {
            "character": [
                {
                    "strategy": "recursive",
                    "reason": "Better sentence preservation for natural text",
                },
                {
                    "strategy": "markdown",
                    "reason": "If content has markdown structure",
                },
            ],
            "recursive": [
                {
                    "strategy": "semantic",
                    "reason": "For topic-based splitting using AI",
                },
                {
                    "strategy": "hierarchical",
                    "reason": "For creating parent-child chunk relationships",
                },
            ],
            "markdown": [
                {
                    "strategy": "recursive",
                    "reason": "For plain text without markdown",
                },
                {
                    "strategy": "hierarchical",
                    "reason": "To preserve document hierarchy",
                },
            ],
        }
        return alternatives.get(primary_strategy, [])

    def _categorize_file_type(self, file_type: str) -> str:
        """Categorize a file type."""
        # Get extension from path if needed
        if "/" in file_type or "\\" in file_type:
            file_type = Path(file_type).suffix

        # Ensure file_type starts with a dot
        if file_type and not file_type.startswith("."):
            file_type = "." + file_type

        code_extensions = [".py", ".js", ".ts", ".cpp", ".c", ".java", ".go", ".rs", ".sh"]
        markdown_extensions = [".md", ".markdown", ".mdx", ".rst"]

        if file_type in code_extensions:
            return "code"
        if file_type in markdown_extensions:
            return "markdown"
        return "text"

    def _get_strategy_pros(self, strategy: str) -> list[str]:
        """Get pros of a strategy.

        Args:
            strategy: Strategy name

        Returns:
            List of advantages
        """
        pros = {
            "character": [
                "Simple and fast",
                "Predictable chunk sizes",
                "Low computational overhead",
            ],
            "recursive": [
                "Preserves sentence boundaries",
                "Intelligent text splitting",
                "Good for general text",
            ],
            "markdown": [
                "Preserves document structure",
                "Respects markdown formatting",
                "Maintains header hierarchy",
            ],
            "semantic": [
                "Topic-aware splitting",
                "Uses AI for natural boundaries",
                "Best semantic coherence",
            ],
            "hierarchical": [
                "Creates parent-child relationships",
                "Good for hierarchical retrieval",
                "Preserves document structure",
            ],
            "hybrid": [
                "Adapts to content type",
                "Fallback mechanisms",
                "Flexible configuration",
            ],
        }
        return pros.get(strategy, [])

    def _get_strategy_cons(self, strategy: str) -> list[str]:
        """Get cons of a strategy.

        Args:
            strategy: Strategy name

        Returns:
            List of disadvantages
        """
        cons = {
            "character": [
                "May split mid-sentence",
                "No semantic awareness",
                "Can break words",
            ],
            "recursive": [
                "Slightly slower than character",
                "May produce variable chunk sizes",
            ],
            "markdown": [
                "Only suitable for markdown content",
                "Requires structured input",
            ],
            "semantic": [
                "Requires embedding API calls",
                "Higher computational cost",
                "Slower processing",
            ],
            "hierarchical": [
                "More complex to implement",
                "Higher storage requirements",
                "Complex retrieval logic",
            ],
            "hybrid": [
                "Configuration complexity",
                "Harder to predict behavior",
                "Debugging challenges",
            ],
        }
        return cons.get(strategy, [])

    def _map_strategy_to_factory_name(self, strategy: str) -> str:
        """Map a strategy name to its factory name.

        This method provides mapping between user-friendly strategy names
        and the internal factory names used by the chunking system.

        Args:
            strategy: User-provided strategy name

        Returns:
            Factory name for the strategy
        """
        # First check the primary STRATEGY_MAPPING
        if strategy in self.STRATEGY_MAPPING:
            return self.STRATEGY_MAPPING[strategy]

        # Map common variations to standard names
        strategy_mapping = {
            # Standard mappings
            "character": "character",
            "recursive": "recursive",
            "markdown": "markdown",
            "semantic": "semantic",
            "hierarchical": "hierarchical",
            "hybrid": "hybrid",
            # Alternative names and variations
            "char": "character",
            "simple": "character",
            "recursive_text": "recursive",
            "recursive_character": "recursive",
            "md": "markdown",
            "semantic_chunking": "semantic",
            "ai": "semantic",
            "hierarchy": "hierarchical",
            "multi_level": "hierarchical",
            "mixed": "hybrid",
            "adaptive": "hybrid",
            # Handle case variations
            "CHARACTER": "character",
            "RECURSIVE": "recursive",
            "MARKDOWN": "markdown",
            "SEMANTIC": "semantic",
            "HIERARCHICAL": "hierarchical",
            "HYBRID": "hybrid",
        }

        # Return mapped name or original if no mapping exists
        return strategy_mapping.get(strategy, strategy)

    # Additional methods for completeness

    async def create_operation(
        self, collection_id: str, operation_type: str, config: dict[str, Any], user_id: int
    ) -> dict[str, Any]:
        """Create a new operation.

        This is an alias for start_chunking_operation for compatibility.
        """
        if operation_type == "chunking":
            return await self.start_chunking_operation(
                collection_id=collection_id,
                strategy=config.get("strategy", "recursive"),
                config=config.get("chunk_config", {}),
                user_id=user_id,
            )
        raise ValueError(f"Unsupported operation type: {operation_type}")

    async def process_chunking_operation(self, operation_id: str) -> None:
        """Process a chunking operation.

        This would typically be called by a Celery task.

        Args:
            operation_id: ID of the operation to process
        """
        # This is a placeholder - actual implementation would be in a Celery task
        logger.info(f"Processing chunking operation {operation_id}")

    async def update_collection(self, collection_id: str, updates: dict[str, Any]) -> None:
        """Update collection after chunking.

        Args:
            collection_id: Collection ID
            updates: Updates to apply
        """
        # This would update collection metadata after chunking
        logger.info(f"Updating collection {collection_id} with {updates}")
