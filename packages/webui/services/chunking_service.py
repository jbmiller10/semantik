"""
Service layer for chunking operations.

This service orchestrates all chunking-related operations including strategy
selection, preview generation, and actual document chunking.
"""

import asyncio
import contextlib
import hashlib
import importlib
import json
import logging
import math
import re
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean
from typing import Any, cast

import redis.asyncio as aioredis
import shared.text_processing.chunking as token_chunking
from shared.config import settings as shared_settings
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.chunking.application.dto.requests import ChunkingStrategy as ChunkingStrategyEnum
from packages.shared.chunking.domain.exceptions import ChunkSizeViolationError
from packages.shared.chunking.domain.services.chunking_strategies import STRATEGY_REGISTRY, get_strategy
from packages.shared.chunking.infrastructure.exception_translator import exception_translator
from packages.shared.chunking.infrastructure.exceptions import (
    ChunkingStrategyError,
    DocumentTooLargeError,
    ResourceNotFoundError,
    ValidationError,
)
from packages.shared.chunking.infrastructure.exceptions import PermissionDeniedError as InfraPermissionDeniedError
from packages.shared.database.models import Operation
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.webui.services.chunking import (
    ChunkingCache,
    ChunkingConfigManager,
    ChunkingMetrics,
    ChunkingOrchestrator,
    ChunkingProcessor,
    ChunkingServiceAdapter,
    ChunkingValidator,
)
from packages.webui.services.chunking.strategy_registry import (
    get_api_to_internal_map,
    get_strategy_defaults,
    resolve_internal_strategy_name,
)
from packages.webui.services.dtos.api_models import ChunkingStrategy

# All exceptions now come from the new infrastructure layer
# Old chunking_exceptions module should be deleted as we're PRE-RELEASE
from .cache_manager import CacheManager, QueryMonitor
from .chunking_config_builder import ChunkingConfigBuilder
from .chunking_error_handler import ChunkingErrorHandler
from .chunking_strategies import ChunkingStrategyRegistry
from .chunking_strategy_factory import ChunkingStrategyFactory
from .chunking_validation import ChunkingInputValidator
from .dtos import (
    ServiceChunkingStats,
    ServiceChunkList,
    ServiceChunkPreview,
    ServiceChunkRecord,
    ServiceCompareResponse,
    ServiceDocumentAnalysis,
    ServiceGlobalMetrics,
    ServicePreviewResponse,
    ServiceQualityAnalysis,
    ServiceSavedConfiguration,
    ServiceStrategyComparison,
    ServiceStrategyInfo,
    ServiceStrategyMetrics,
    ServiceStrategyRecommendation,
)

logger = logging.getLogger(__name__)

# Constants for chunking configuration
DEFAULT_MIN_TOKEN_THRESHOLD = 100  # Minimum tokens to ensure meaningful chunks


def _read_json_file(path: Path) -> dict[str, Any]:
    """Safely read JSON content from a file."""
    try:
        if not path.exists():
            return {"configs": {}}
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to decode JSON from %s: %s", path, exc)
        return {"configs": {}}
    except OSError as exc:  # pragma: no cover - defensive
        logger.error("Unable to read %s: %s", path, exc)
        return {"configs": {}}


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    """Persist JSON payload to disk atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, default=_json_default, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _json_default(value: Any) -> Any:
    """JSON serializer that understands datetime objects."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


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
        return get_strategy_defaults(strategy_type, context="factory")


class ChunkingService:
    """Service for managing chunking operations."""

    # Mapping from API strategy names to factory strategy names
    STRATEGY_MAPPING = get_api_to_internal_map().copy()

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
        self._config_store_path = shared_settings.data_dir / "chunking_configs.json"

        # Orchestrator-based architecture for preview/compare flows
        self._orchestrator = ChunkingOrchestrator(
            processor=ChunkingProcessor(),
            cache=ChunkingCache(redis_client),
            metrics=ChunkingMetrics(),
            validator=ChunkingValidator(
                db_session=db_session,
                collection_repo=collection_repo,
                document_repo=document_repo,
            ),
            config_manager=ChunkingConfigManager(),
            db_session=db_session,
            collection_repo=collection_repo,
            document_repo=document_repo,
        )
        self._adapter = ChunkingServiceAdapter(
            orchestrator=self._orchestrator,
            db_session=db_session,
            collection_repo=collection_repo,
            document_repo=document_repo,
        )

    async def _load_config_store(self) -> dict[str, list[dict[str, Any]]]:
        """Load saved configurations grouped by user."""
        data = await asyncio.to_thread(_read_json_file, self._config_store_path)
        configs = data.get("configs", {}) if isinstance(data, dict) else {}
        # Ensure proper types
        sanitized: dict[str, list[dict[str, Any]]] = {}
        for user_id, entries in configs.items():
            if isinstance(user_id, str) and isinstance(entries, list):
                sanitized[user_id] = [entry for entry in entries if isinstance(entry, dict)]
        return sanitized

    async def _write_config_store(self, configs: dict[str, list[dict[str, Any]]]) -> None:
        """Persist configuration store to disk."""
        await asyncio.to_thread(_write_json_file, self._config_store_path, {"configs": configs})

    @staticmethod
    def _calculate_min_tokens(desired_chunk_size: int) -> int:
        """Derive a conservative min_tokens target that avoids over-constraining small chunk sizes."""

        normalized_size = max(10, desired_chunk_size)
        base_tokens = max(10, normalized_size // 4)
        return min(DEFAULT_MIN_TOKEN_THRESHOLD, base_tokens)

    @staticmethod
    def _strategy_to_string(strategy: str | ChunkingStrategyEnum) -> str:
        """Normalize strategy identifiers to string names."""

        if isinstance(strategy, ChunkingStrategyEnum):
            return strategy.value
        return str(strategy)

    @staticmethod
    def _record_metric(method_name: str, *args: Any, **kwargs: Any) -> None:
        """Dispatch metric recording to the dynamically reloaded chunking_metrics module."""

        metrics_module = importlib.import_module("packages.webui.services.chunking_metrics")
        record_func = getattr(metrics_module, method_name, None)
        if record_func:
            record_func(*args, **kwargs)

    def _deserialize_saved_config(self, payload: dict[str, Any]) -> ServiceSavedConfiguration:
        """Convert raw payload into ServiceSavedConfiguration."""
        created_at = self._parse_datetime(payload.get("created_at"))
        updated_at = self._parse_datetime(payload.get("updated_at"))
        tags = payload.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]

        return ServiceSavedConfiguration(
            id=str(payload.get("id")),
            name=str(payload.get("name", "Unnamed configuration")),
            description=payload.get("description"),
            strategy=payload.get("strategy", "fixed_size"),
            config=dict(payload.get("config", {})),
            created_by=int(payload.get("created_by", 0)),
            created_at=created_at,
            updated_at=updated_at,
            usage_count=int(payload.get("usage_count", 0)),
            is_default=bool(payload.get("is_default", False)),
            tags=[str(tag) for tag in tags],
        )

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        """Parse datetime from ISO string, defaulting to now if parsing fails."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass
        return datetime.now(UTC)

    async def get_available_strategies(self) -> list[dict[str, Any]]:
        """Get list of available chunking strategies with details.

        This method provides the business logic for listing strategies,
        including their configurations and metadata.

        Returns:
            List of strategy information dictionaries
        """
        strategies = []

        for strategy in ChunkingStrategyEnum:
            # Convert shared enum to webui enum for registry lookup
            from packages.webui.api.v2.chunking_schemas import ChunkingStrategy as WebUIChunkingStrategy

            webui_strategy = WebUIChunkingStrategy(strategy.value)
            strategy_def = ChunkingStrategyRegistry.get_strategy_definition(webui_strategy)

            # Get default config from builder
            default_config = self.config_builder.get_default_config(webui_strategy)

            # Get compatibility info from factory
            strategy_info = self.strategy_factory.get_strategy_info(strategy)

            strategies.append(
                {
                    "id": strategy.value,
                    "name": strategy_def.get("name", strategy.value),
                    "description": strategy_def.get("description", ""),
                    "best_for": strategy_def.get("best_for", []),
                    "pros": strategy_def.get("pros", []),
                    "cons": strategy_def.get("cons", []),
                    "default_config": default_config,
                    "performance_characteristics": strategy_def.get("performance_characteristics", {}),
                    "available": strategy_info.get("available", True),
                }
            )

        return strategies

    async def get_available_strategies_for_api(self) -> list[ServiceStrategyInfo]:
        """Get available strategies formatted for API response.

        This method handles all the transformation logic that was previously
        in the router, including strategy mapping and configuration building.

        Returns:
            List of ServiceStrategyInfo DTOs for API response
        """
        strategies_data = await self.get_available_strategies()
        strategies = []

        for strategy_data in strategies_data:
            try:
                # Use helper method to build DTO
                strategy_info = self._build_strategy_info(strategy_data)
                strategies.append(strategy_info)
            except Exception as e:
                # Log and skip invalid strategies
                logger.warning(f"Failed to build strategy info for {strategy_data.get('id')}: {e}")
                continue

        return strategies

    async def get_strategy_details(self, strategy_id: str) -> ServiceStrategyInfo | None:
        """Get detailed information about a specific strategy.

        This method handles strategy lookup with alias mapping and all
        transformation logic that was previously in the router.

        Args:
            strategy_id: Strategy identifier (supports aliases)

        Returns:
            ServiceStrategyInfo DTO or None if not found
        """
        # Get all strategies from service
        strategies_data = await self.get_available_strategies()

        # Fallback if service returns unexpected type
        if not isinstance(strategies_data, list) or not strategies_data:
            strategies_data = [
                {
                    "id": "character",
                    "name": "Fixed Size Chunking",
                    "description": "Simple fixed-size chunking with consistent chunk sizes",
                    "best_for": ["txt"],
                    "pros": ["Predictable"],
                    "cons": ["May split sentences"],
                    "default_config": {"strategy": "fixed_size", "chunk_size": 1000, "chunk_overlap": 200},
                    "performance_characteristics": {"speed": "fast"},
                }
            ]

        # Find the requested strategy
        strategy_data = None
        internal_id = resolve_internal_strategy_name(strategy_id) or strategy_id
        for s in strategies_data:
            if s["id"] == internal_id:
                strategy_data = s
                break

        if not strategy_data:
            return None

        try:
            # Use helper method to build DTO
            return self._build_strategy_info(strategy_data)
        except Exception as e:
            # Log and return None if building fails
            logger.warning(f"Failed to build strategy info for {strategy_id}: {e}")
            return None

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
            ValidationError: If validation fails
            InfraPermissionDeniedError: If user lacks access
            ResourceNotFoundError: If document doesn't exist
        """
        correlation_id = str(uuid.uuid4())

        # Validate user access
        if not user_id:
            raise InfraPermissionDeniedError(
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
            raise ValidationError(
                field="config",
                value=config_overrides,
                reason=f"Invalid configuration: {', '.join(config_result.validation_errors)}",
                correlation_id=correlation_id,
            )

        # Create operation
        operation_strategy = (
            config_result.strategy.value if hasattr(config_result.strategy, "value") else str(config_result.strategy)
        )
        operation_id = await self.start_chunking_operation(
            collection_id="",  # Would get from document
            strategy=operation_strategy,
            config=config_result.config,
            user_id=user_id,
        )

        return str(operation_id.get("operation_id", str(uuid.uuid4())))

    async def recommend_strategy(
        self,
        content_size: int | None = None,
        file_types: list[str] | None = None,
        file_paths: list[str] | None = None,
        has_structure: bool | None = None,
    ) -> ServiceStrategyRecommendation:
        """Recommend a chunking strategy based on content characteristics.

        Args:
            content_size: Size of content in bytes (optional)
            file_types: List of file types to analyze (optional)
            file_paths: List of file paths to analyze (optional)
            has_structure: Whether content has structure (markdown, etc.)

        Returns:
            ServiceStrategyRecommendation DTO with strategy and reasoning
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

        # Get alternative strategies
        alternatives = self._get_alternative_strategies(recommended_strategy)
        alternative_strategies = [alt.get("name", alt.get("strategy", "")) for alt in alternatives]

        # Build and return the DTO
        return ServiceStrategyRecommendation(
            strategy=(
                recommended_strategy.value if hasattr(recommended_strategy, "value") else str(recommended_strategy)
            ),
            confidence=0.8 if file_type_breakdown else 0.6,
            reasoning=reasoning,
            alternatives=alternative_strategies,
            chunk_size=chunk_size,
            chunk_overlap=100,
            preserve_sentences=True,
        )

    async def preview_chunks(
        self,
        strategy: str | ChunkingStrategyEnum,
        content: str | None = None,
        document_id: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        user_id: int | None = None,
        _correlation_id: str | None = None,
    ) -> dict[str, Any]:
        """Delegate preview computation to orchestrator-backed adapter."""

        strategy_name = self._strategy_to_string(strategy)
        return await self._adapter.preview_chunks(
            content=content,
            document_id=document_id,
            strategy=strategy_name,
            config=config_overrides,
            user_id=user_id,
        )

    async def _validate_document_access(self, document_id: str, _user_id: int) -> None:
        """Validate user has access to document."""
        # This would use actual repository methods
        # For now, simplified implementation
        document = await self.document_repo.get_by_id(document_id)
        if not document:
            raise ResourceNotFoundError(
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
            raise ResourceNotFoundError(
                resource_type="Document",
                resource_id=document_id,
                correlation_id=str(uuid.uuid4()),
            )

        # Load from appropriate storage
        # Simplified - would handle different storage types
        return str(document.get("content", ""))

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
            min_tokens = min(DEFAULT_MIN_TOKEN_THRESHOLD, chunk_size // 2)
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

        except TimeoutError as e:
            raise ChunkingStrategyError(
                strategy=str(strategy_name),
                reason="Chunking operation timed out",
                correlation_id=str(uuid.uuid4()),
            ) from e
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

    def _simple_chunk_fallback(self, content: str, chunk_size: int, chunk_overlap: int) -> list[str]:
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
        quality_score = 1.0 - min(1.0, variance / (avg_size**2)) if avg_size > 0 else 0.0

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": avg_size,
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "size_variance": variance,
            "quality_score": quality_score,
        }

    async def _cache_preview_result(self, result: dict[str, Any], _strategy: str) -> str:
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
        file_type: str | None = None,  # noqa: ARG002 - Reserved for future file-type-specific strategy selection
        max_chunks: int | None = None,
        cache_result: bool = True,
    ) -> ServicePreviewResponse:
        strategy_name = self._strategy_to_string(strategy or ChunkingStrategyEnum.RECURSIVE)
        return await self._orchestrator.preview_chunks(
            content=content,
            strategy=strategy_name,
            config=config,
            use_cache=cache_result,
            max_chunks=max_chunks,
        )

    async def validate_preview_content(self, content: str | None, document_id: str | None) -> None:
        """Validate preview request content.

        This method contains all the validation logic that was previously
        in the router, ensuring proper separation of concerns.

        Args:
            content: Optional content to validate
            document_id: Optional document ID

        Raises:
            ValidationError: If validation fails
            DocumentTooLargeError: If content exceeds size limit
        """
        from packages.shared.chunking.infrastructure.exceptions import DocumentTooLargeError, ValidationError

        # Must have either content or document_id
        if not content and not document_id:
            raise ValidationError(field="input", value=None, reason="document_id or content must be provided")

        if content is not None:
            # Check for null bytes
            if "\x00" in content:
                raise ValidationError(
                    field="content",
                    value=content[:100] + "..." if len(content) > 100 else content,  # Truncate for error message
                    reason="Content contains null bytes",
                )

            # Enforce ~10MB limit
            if len(content) > 10 * 1024 * 1024:
                raise DocumentTooLargeError(
                    size=len(content), max_size=10 * 1024 * 1024, correlation_id=str(uuid.uuid4())
                )

    async def compare_strategies_for_api(
        self,
        content: str,
        strategies: list[str],
        configs: dict[str, dict[str, Any]] | None = None,
        max_chunks_per_strategy: int = 5,
    ) -> ServiceCompareResponse:
        """Compare multiple chunking strategies formatted for API response.

        This method handles all the comparison and transformation logic
        that was previously in the router.

        Args:
            content: Content to chunk
            strategies: List of strategy names
            configs: Optional per-strategy configurations
            max_chunks_per_strategy: Maximum chunks to return per strategy

        Returns:
            ServiceCompareResponse DTO with comparison results
        """
        import time

        start_time = time.time()
        comparisons = []
        best_strategy = None
        best_quality = -1.0

        for strategy in strategies:
            # Get config for this strategy if provided
            config = None
            if configs and strategy in configs:
                strategy_config = configs.get(strategy)
                if strategy_config:
                    config = strategy_config.model_dump() if hasattr(strategy_config, "model_dump") else strategy_config

            # Execute preview - now returns ServicePreviewResponse
            preview = await self.preview_chunking(
                strategy=strategy,
                content=content or "",
                config=config,
                max_chunks=max_chunks_per_strategy,
            )

            # Get quality score from metrics
            quality = 0.8  # Default quality
            if preview.metrics and "quality_score" in preview.metrics:
                quality = float(preview.metrics["quality_score"])

            if quality > best_quality:
                best_quality = quality
                best_strategy = preview.strategy

            # Get pros and cons for this strategy
            pros = self._get_strategy_pros(preview.strategy)
            cons = self._get_strategy_cons(preview.strategy)

            # Calculate statistics from chunks
            avg_chunk_size = 0.0
            size_variance = 0.0
            if preview.chunks:
                sizes = [
                    c.char_count
                    for c in preview.chunks
                    if isinstance(c, ServiceChunkPreview) and c.char_count is not None
                ]
                if sizes:
                    avg_chunk_size = sum(sizes) / len(sizes)
                    if len(sizes) > 1:
                        mean = avg_chunk_size
                        size_variance = sum((s - mean) ** 2 for s in sizes) / len(sizes)

            # Build comparison DTO
            comparison = ServiceStrategyComparison(
                strategy=preview.strategy,
                config=preview.config,
                sample_chunks=preview.chunks[:max_chunks_per_strategy],  # Limit chunks
                total_chunks=preview.total_chunks,
                avg_chunk_size=avg_chunk_size,
                size_variance=size_variance,
                quality_score=quality,
                processing_time_ms=preview.processing_time_ms,
                pros=pros,
                cons=cons,
            )
            comparisons.append(comparison)

        # Build recommendation
        if best_strategy:
            alternatives = [s for s in strategies if s != best_strategy]
            recommendation = ServiceStrategyRecommendation(
                strategy=best_strategy,
                confidence=best_quality if best_quality >= 0 else 0.8,
                reasoning="Selected strategy with higher quality score",
                alternatives=alternatives,
                chunk_size=512,  # Default recommendation
                chunk_overlap=50,
                preserve_sentences=True,
            )
        else:
            recommendation = ServiceStrategyRecommendation(
                strategy=strategies[0] if strategies else "fixed_size",
                confidence=0.5,
                reasoning="Default recommendation",
                alternatives=[],
                chunk_size=512,
                chunk_overlap=50,
                preserve_sentences=True,
            )

        processing_time_ms = int((time.time() - start_time) * 1000)

        return ServiceCompareResponse(
            comparison_id=str(uuid.uuid4()),
            comparisons=cast(list[ServiceStrategyComparison | dict[str, Any]], comparisons),
            recommendation=recommendation,
            processing_time_ms=processing_time_ms,
        )

    async def compare_strategies(
        self,
        content: str,
        strategies: list[ChunkingStrategyEnum],
        configs: dict[str, dict[str, Any]] | None = None,
        max_chunks_per_strategy: int = 5,
        _user_id: int | None = None,
    ) -> dict[str, Any]:
        strategy_names = [self._strategy_to_string(strategy) for strategy in strategies]
        config_map = configs.copy() if configs else None

        return await self._adapter.compare_strategies(
            content=content,
            strategies=strategy_names,
            strategy_configs=config_map,
            max_chunks_per_strategy=max_chunks_per_strategy,
        )

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
        _period_days: int = 30,
        _user_id: int | None = None,
    ) -> list[ServiceStrategyMetrics]:
        """Get chunking metrics grouped by strategy.

        This method provides strategy-specific metrics for the specified period.
        If no metrics are available, returns default placeholder metrics.

        Args:
            period_days: Number of days to look back
            user_id: Optional user ID for filtering

        Returns:
            List of ServiceStrategyMetrics DTOs
        """
        # Try to get real metrics from database (placeholder for now)
        try:
            # In production, would fetch actual metrics from database
            # For now, check if we can get strategy definitions
            from packages.webui.api.v2.chunking_schemas import ChunkingStrategy

            metrics = []
            for strategy in ChunkingStrategy:
                # Get strategy definition from registry if available
                try:
                    strategy_def = ChunkingStrategyRegistry.get_strategy_definition(strategy)
                    best_for_types = strategy_def.get("best_for", []) if strategy_def else []
                except Exception:
                    best_for_types = []

                # Create metric DTO with placeholder data
                metrics.append(
                    ServiceStrategyMetrics(
                        strategy=strategy.value,
                        usage_count=0,
                        avg_chunk_size=512.0,
                        avg_processing_time=1.5,
                        success_rate=0.95,
                        avg_quality_score=0.8,
                        best_for_types=best_for_types,
                    )
                )

            return metrics

        except Exception as e:
            logger.warning(f"Failed to get metrics by strategy, using defaults: {e}")
            # Return default metrics using helper method
            return self._get_default_metrics()

    async def get_collection_chunks(
        self,
        collection_id: str,
        *,
        page: int = 1,
        page_size: int = 20,
        document_id: str | None = None,
    ) -> ServiceChunkList:
        """Return paginated chunk data for a collection."""
        from packages.shared.chunking.infrastructure.exceptions import ResourceNotFoundError
        from packages.shared.database.models import Chunk

        try:
            collection = await self.collection_repo.get_by_uuid(collection_id)
        except Exception as exc:  # pragma: no cover - defensive database failure handling
            logger.error("Failed to fetch collection %s: %s", collection_id, exc)
            raise

        if not collection:
            raise ResourceNotFoundError("Collection", str(collection_id))

        safe_page = max(page, 1)
        safe_page_size = max(1, min(page_size, 500))
        offset = (safe_page - 1) * safe_page_size

        filters = [Chunk.collection_id == collection.id]
        if document_id:
            filters.append(Chunk.document_id == document_id)

        total_result = await self.db_session.execute(select(func.count(Chunk.id)).where(*filters))
        total_chunks = int(total_result.scalar() or 0)

        if total_chunks == 0:
            return ServiceChunkList(chunks=[], total=0, page=safe_page, page_size=safe_page_size)

        chunk_query = select(Chunk).where(*filters).order_by(Chunk.chunk_index).offset(offset).limit(safe_page_size)
        chunk_rows = await self.db_session.execute(chunk_query)
        chunk_objects = chunk_rows.scalars().all()

        records: list[ServiceChunkRecord] = []
        for chunk in chunk_objects:
            metadata_raw = getattr(chunk, "meta", {}) or {}
            metadata = cast(dict[str, Any], metadata_raw) if isinstance(metadata_raw, dict) else {}

            records.append(
                ServiceChunkRecord(
                    id=cast(int, chunk.id),
                    collection_id=cast(str, chunk.collection_id),
                    document_id=cast(str | None, chunk.document_id),
                    chunk_index=cast(int, chunk.chunk_index),
                    content=cast(str, chunk.content) if chunk.content is not None else "",
                    token_count=cast(int | None, getattr(chunk, "token_count", None)),
                    metadata=metadata,
                    created_at=cast(datetime | None, getattr(chunk, "created_at", None)),
                    updated_at=cast(datetime | None, getattr(chunk, "updated_at", None)),
                )
            )

        return ServiceChunkList(
            chunks=records,
            total=total_chunks,
            page=safe_page,
            page_size=safe_page_size,
        )

    async def get_global_metrics(
        self,
        *,
        period_days: int = 30,
        user_id: int | None = None,  # noqa: ARG002 - reserved for future multi-tenant filtering
    ) -> ServiceGlobalMetrics:
        """Compute global chunking metrics for the requested period."""
        from packages.shared.database.models import Chunk, Collection, Document, Operation, OperationStatus

        period_end = datetime.now(UTC)
        safe_days = max(period_days, 1)
        period_start = period_end - timedelta(days=safe_days)

        # Count chunks created within the period
        chunk_count_result = await self.db_session.execute(
            select(func.count(Chunk.id)).where(Chunk.created_at >= period_start)
        )
        total_chunks_created = int(chunk_count_result.scalar() or 0)

        # Documents processed during the period
        documents_processed_result = await self.db_session.execute(
            select(func.count(Document.id)).where(
                Document.chunking_completed_at.isnot(None),
                Document.chunking_completed_at >= period_start,
            )
        )
        total_documents_processed = int(documents_processed_result.scalar() or 0)

        avg_chunks_per_document = total_chunks_created / total_documents_processed if total_documents_processed else 0.0

        # Operations executed during the period
        operation_rows = await self.db_session.execute(
            select(
                Operation.collection_id,
                Operation.status,
                Operation.started_at,
                Operation.completed_at,
            ).where(Operation.created_at >= period_start)
        )
        operations = operation_rows.all()

        total_operations = len(operations)
        completed_operations = sum(1 for _, status, _, _ in operations if status == OperationStatus.COMPLETED)
        success_rate = completed_operations / total_operations if total_operations else 1.0

        durations: list[float] = []
        for _, status, started_at, completed_at in operations:
            if status == OperationStatus.COMPLETED and started_at and completed_at:
                durations.append((completed_at - started_at).total_seconds())

        avg_processing_time = fmean(durations) if durations else 0.0
        processed_collection_ids = {
            collection_id for collection_id, status, _, _ in operations if status == OperationStatus.COMPLETED
        }
        total_collections_processed = len(processed_collection_ids)

        # Determine most used strategy from collections updated during the period
        strategy_rows = await self.db_session.execute(
            select(Collection.chunking_strategy).where(
                Collection.chunking_strategy.isnot(None), Collection.updated_at >= period_start
            )
        )
        strategies = [row[0] for row in strategy_rows if row[0]]
        if not strategies and processed_collection_ids:
            # Fallback to collections involved in operations if they lack recent updates
            fallback_rows = await self.db_session.execute(
                select(Collection.chunking_strategy).where(Collection.id.in_(processed_collection_ids))
            )
            strategies.extend(row[0] for row in fallback_rows if row[0])

        strategy_counter = Counter(str(strategy).lower() for strategy in strategies if strategy)
        most_used_strategy = strategy_counter.most_common(1)[0][0] if strategy_counter else "fixed_size"

        return ServiceGlobalMetrics(
            total_collections_processed=total_collections_processed,
            total_chunks_created=total_chunks_created,
            total_documents_processed=total_documents_processed,
            avg_chunks_per_document=avg_chunks_per_document,
            most_used_strategy=most_used_strategy,
            avg_processing_time=avg_processing_time,
            success_rate=min(max(success_rate, 0.0), 1.0),
            period_start=period_start,
            period_end=period_end,
        )

    async def get_collection_chunk_stats(self, collection_id: str) -> ServiceChunkingStats:
        """Get chunk-level statistics for a collection.

        This method returns chunk-level metrics suitable for ChunkingStats schema.

        Args:
            collection_id: ID of the collection

        Returns:
            ServiceChunkingStats DTO for the API response
        """
        try:
            # Get collection
            collection = await self.collection_repo.get_by_uuid(collection_id)
            if not collection:
                from packages.shared.chunking.infrastructure.exceptions import ResourceNotFoundError

                raise ResourceNotFoundError("Collection", str(collection_id))

            # Get chunk statistics from database
            from packages.shared.database.models import Chunk

            chunk_stats_query = select(
                func.count(Chunk.id).label("total_chunks"),
                func.avg(func.length(Chunk.content)).label("avg_chunk_size"),
                func.min(func.length(Chunk.content)).label("min_chunk_size"),
                func.max(func.length(Chunk.content)).label("max_chunk_size"),
                func.var_pop(func.length(Chunk.content)).label("size_variance"),
            ).where(Chunk.collection_id == collection.id)

            result = await self.db_session.execute(chunk_stats_query)
            stats = result.one()

            # Count documents
            from packages.shared.database.models import Document

            doc_count_query = select(func.count(Document.id)).where(Document.collection_id == collection.id)
            doc_result = await self.db_session.execute(doc_count_query)
            total_documents = doc_result.scalar() or 0

            # Get the latest operation to find processing time and strategy
            from packages.shared.database.models import Operation, OperationType

            latest_op_query = (
                select(Operation)
                .where(
                    and_(
                        Operation.collection_id == collection.id,
                        Operation.type == OperationType.INDEX,
                        Operation.status == "completed",
                    )
                )
                .order_by(Operation.completed_at.desc())
                .limit(1)
            )
            latest_op_result = await self.db_session.execute(latest_op_query)
            latest_operation = latest_op_result.scalar_one_or_none()

            processing_time = 0.0
            strategy = collection.chunking_strategy or "fixed_size"
            last_updated = collection.updated_at

            if latest_operation:
                if latest_operation.started_at and latest_operation.completed_at:
                    processing_time = (latest_operation.completed_at - latest_operation.started_at).total_seconds()
                if latest_operation.config and "strategy" in latest_operation.config:
                    strategy = latest_operation.config["strategy"]
                last_updated = latest_operation.completed_at or collection.updated_at

            from .dtos.chunking_dtos import ServiceChunkingStats

            return ServiceChunkingStats(
                total_chunks=stats.total_chunks or 0,
                total_documents=total_documents,
                avg_chunk_size=float(stats.avg_chunk_size) if stats.avg_chunk_size else 0.0,
                min_chunk_size=stats.min_chunk_size or 0,
                max_chunk_size=stats.max_chunk_size or 0,
                size_variance=float(stats.size_variance) if stats.size_variance else 0.0,
                strategy_used=strategy,
                last_updated=last_updated,
                processing_time_seconds=processing_time,
                quality_metrics={
                    "quality_score": 0.85,  # Placeholder
                    "overlap_ratio": 0.2,  # Placeholder
                },
            )

        except Exception as e:
            logger.error(f"Failed to get chunk statistics: {e}")
            # Return empty stats on error
            from .dtos.chunking_dtos import ServiceChunkingStats

            return ServiceChunkingStats(
                total_chunks=0,
                total_documents=0,
                avg_chunk_size=0.0,
                min_chunk_size=0,
                max_chunk_size=0,
                size_variance=0.0,
                strategy_used="fixed_size",
                last_updated=datetime.now(UTC),
                processing_time_seconds=0.0,
                quality_metrics={},
            )

    async def get_quality_scores(self, collection_id: str | None = None) -> ServiceQualityAnalysis:
        """Analyze chunk quality across collections or a specific collection."""
        from packages.shared.database.models import Chunk, Document

        filters = []
        document_filters = []
        if collection_id:
            filters.append(Chunk.collection_id == collection_id)
            document_filters.append(Document.collection_id == collection_id)

        stats_query = select(
            func.count(Chunk.id).label("total_chunks"),
            func.avg(func.length(Chunk.content)).label("avg_length"),
            func.var_pop(func.length(Chunk.content)).label("variance"),
        ).where(*filters)
        stats_result = await self.db_session.execute(stats_query)
        stats = stats_result.one()

        total_chunks = int(stats.total_chunks or 0)
        avg_length = float(stats.avg_length or 0.0)
        variance = float(stats.variance or 0.0)
        std_dev = math.sqrt(max(variance, 0.0)) if total_chunks > 1 else 0.0

        # Size consistency penalises high variance relative to average chunk length
        size_consistency = 1.0 - min(1.0, std_dev / (avg_length + 1e-6)) if avg_length else 0.0
        coherence_score = size_consistency

        # Document completeness: proportion of documents with at least one chunk
        doc_query = select(
            func.count(Document.id).label("total_docs"),
            func.count(func.nullif(Document.chunk_count, 0)).label("docs_with_chunks"),
        ).where(*document_filters)
        doc_result = await self.db_session.execute(doc_query)
        doc_stats = doc_result.one()

        total_docs = int(doc_stats.total_docs or 0)
        docs_with_chunks = int(doc_stats.docs_with_chunks or 0)
        completeness_score = docs_with_chunks / total_docs if total_docs else 1.0

        # Combine metrics for overall quality
        quality_components = [size_consistency, coherence_score, completeness_score]
        quality_score = sum(quality_components) / len(quality_components) if quality_components else 0.0
        quality_score = max(0.0, min(quality_score, 1.0))

        if quality_score >= 0.85:
            overall_quality = "excellent"
        elif quality_score >= 0.7:
            overall_quality = "good"
        elif quality_score >= 0.5:
            overall_quality = "fair"
        else:
            overall_quality = "poor"

        recommendations: list[str] = []
        issues_detected: list[str] = []

        if size_consistency < 0.75:
            issues_detected.append("High variance in chunk sizes detected")
            recommendations.append("Consider lowering chunk_size or increasing chunk_overlap for more consistency")

        if completeness_score < 0.85:
            issues_detected.append("Some documents lack generated chunks")
            recommendations.append("Re-run chunking on documents without chunks or adjust filters")

        if total_chunks > 50000:
            recommendations.append("Large chunk volume detected; ensure storage and retrieval are tuned appropriately")

        if not recommendations:
            recommendations.append("Current configuration appears healthy. Continue monitoring periodically.")

        return ServiceQualityAnalysis(
            overall_quality=overall_quality,
            quality_score=quality_score,
            coherence_score=coherence_score,
            completeness_score=completeness_score,
            size_consistency=size_consistency,
            recommendations=recommendations,
            issues_detected=issues_detected,
        )

    async def analyze_document(
        self,
        *,
        content: str | None,
        document_id: str | None,
        file_type: str | None,
        user_id: int | None = None,  # noqa: ARG002 - reserved for personalised recommendations
        deep_analysis: bool = False,
    ) -> ServiceDocumentAnalysis:
        """Analyze document content and produce strategy recommendations."""

        document_content = content
        document_type = file_type or "unknown"
        document_metadata: dict[str, Any] = {}

        if not document_content and document_id:
            document = await self.document_repo.get_by_id(document_id)
            if document:
                document_type = (
                    file_type or document.mime_type or Path(document.file_name).suffix.lstrip(".") or "unknown"
                )
                document_metadata = {
                    "file_name": document.file_name,
                    "file_size": document.file_size,
                }
                file_path = getattr(document, "file_path", None)
                if file_path:
                    path = Path(file_path)
                    if path.exists() and path.is_file():
                        try:
                            document_content = await asyncio.to_thread(
                                path.read_text, encoding="utf-8", errors="ignore"
                            )
                        except Exception as exc:  # pragma: no cover - file system errors handled gracefully
                            logger.debug("Unable to read document %s for analysis: %s", document_id, exc)

        text = document_content or ""
        paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        words = re.findall(r"\b\w+\b", text)

        lexical_diversity = len(set(words)) / len(words) if words else 0.0
        sentence_density = len(sentences) / max(len(paragraphs), 1)
        complexity_score = min(1.0, (lexical_diversity + min(sentence_density / 5, 1.0)) / 2)
        if deep_analysis:
            complexity_score = min(1.0, complexity_score + 0.1)

        content_structure: dict[str, Any] = {
            "paragraphs": len(paragraphs),
            "sentences": len(sentences),
            "words": len(words),
            "characters": len(text),
        }
        if document_metadata:
            content_structure["metadata"] = document_metadata

        base_length = len(text)
        if base_length == 0 and document_metadata.get("file_size"):
            base_length = int(document_metadata["file_size"]) // 6  # rough approximation of characters

        estimated_chunks: dict[str | ChunkingStrategy, int] = {}
        for strategy in ChunkingStrategy:
            divisor = 900
            if strategy == ChunkingStrategy.FIXED_SIZE:
                divisor = 800
            elif strategy == ChunkingStrategy.RECURSIVE:
                divisor = 950
            elif strategy == ChunkingStrategy.MARKDOWN or strategy == ChunkingStrategy.DOCUMENT_STRUCTURE:
                divisor = 1100
            elif strategy == ChunkingStrategy.SEMANTIC:
                divisor = 700
            elif strategy == ChunkingStrategy.HYBRID:
                divisor = 750

            estimated = max(1, base_length // max(divisor, 1)) if base_length else max(len(paragraphs), 1)
            estimated_chunks[strategy] = estimated

        recommendation = await self.recommend_strategy(
            file_types=[document_type] if document_type and document_type != "unknown" else []
        )

        special_considerations: list[str] = []
        if lexical_diversity > 0.4:
            special_considerations.append("High lexical diversity detected; semantic chunking may excel")
        if len(sentences) > 0 and sentence_density > 8:
            special_considerations.append("Dense paragraphs detected; consider higher overlap for context retention")
        if len(words) < 200:
            special_considerations.append("Short document; chunking strategies may not significantly differ")

        return ServiceDocumentAnalysis(
            document_type=document_type,
            content_structure=content_structure,
            recommended_strategy=recommendation,
            estimated_chunks=estimated_chunks,
            complexity_score=complexity_score,
            special_considerations=special_considerations,
        )

    async def save_configuration(
        self,
        *,
        name: str,
        description: str | None,
        strategy: str,
        config: dict[str, Any],
        is_default: bool,
        tags: list[str],
        user_id: int,
    ) -> ServiceSavedConfiguration:
        """Persist a user-defined chunking configuration."""
        from packages.shared.chunking.infrastructure.exceptions import ValidationError as InfraValidationError

        builder_result = self.config_builder.build_config(strategy, config)
        if builder_result.validation_errors:
            summary = ", ".join(builder_result.validation_errors)
            raise InfraValidationError(
                field="config",
                value=config,
                reason=f"Invalid configuration: {summary}",
            )

        sanitized_tags = [tag.strip() for tag in tags if tag and isinstance(tag, str)]
        normalized_config = builder_result.config
        normalized_config["strategy"] = builder_result.strategy.value

        configs = await self._load_config_store()
        user_key = str(user_id)
        user_configs = configs.get(user_key, [])

        now = datetime.now(UTC)
        existing = next((item for item in user_configs if item.get("name", "").lower() == name.lower()), None)

        if is_default:
            for item in user_configs:
                item["is_default"] = False

        if existing:
            existing.update(
                {
                    "description": description,
                    "strategy": builder_result.strategy.value,
                    "config": normalized_config,
                    "updated_at": now.isoformat(),
                    "is_default": is_default,
                    "tags": sanitized_tags,
                }
            )
        else:
            existing = {
                "id": uuid.uuid4().hex,
                "name": name,
                "description": description,
                "strategy": builder_result.strategy.value,
                "config": normalized_config,
                "created_by": user_id,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "usage_count": 0,
                "is_default": is_default,
                "tags": sanitized_tags,
            }
            user_configs.append(existing)

        configs[user_key] = user_configs
        await self._write_config_store(configs)

        return self._deserialize_saved_config(existing)

    async def list_configurations(
        self,
        *,
        user_id: int,
        strategy: str | None = None,
        is_default: bool | None = None,
    ) -> list[ServiceSavedConfiguration]:
        """Return saved configurations for a user with optional filters."""

        configs = await self._load_config_store()
        user_configs = configs.get(str(user_id), [])

        results = []
        for entry in user_configs:
            if strategy and entry.get("strategy", "").lower() != strategy.lower():
                continue
            if is_default is not None and bool(entry.get("is_default", False)) != is_default:
                continue
            results.append(self._deserialize_saved_config(entry))

        # Return most recently updated first
        results.sort(key=lambda item: item.updated_at, reverse=True)
        return results

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
            cache_key = self.cache_manager._generate_cache_key("statistics", {"collection_id": collection_id})
            cached = await self.cache_manager.get(cache_key)
            if cached:
                return cast(dict[str, Any], cached)

        try:
            # Get collection
            collection = await self.collection_repo.get_by_uuid(collection_id)
            if not collection:
                raise ValueError(f"Collection {collection_id} not found")

            # Single aggregation query instead of loading all records
            stats_query = select(
                func.count(Operation.id).label("total_operations"),
                func.count(case((Operation.status == "completed", Operation.id), else_=None)).label(
                    "completed_operations"
                ),
                func.count(case((Operation.status == "failed", Operation.id), else_=None)).label("failed_operations"),
                func.count(case((Operation.status == "processing", Operation.id), else_=None)).label(
                    "processing_operations"
                ),
                func.avg(
                    case(
                        (
                            and_(
                                Operation.status == "completed",
                                Operation.completed_at.isnot(None),
                                Operation.started_at.isnot(None),
                            ),
                            func.extract("epoch", Operation.completed_at - Operation.started_at),
                        ),
                        else_=None,
                    )
                ).label("avg_processing_time"),
                func.max(Operation.created_at).label("last_operation_at"),
                func.min(Operation.created_at).label("first_operation_at"),
            ).where(and_(Operation.collection_id == collection_id, Operation.type == "chunking"))

            stats_result = await self.db_session.execute(stats_query)
            stats = stats_result.one()

            # Get latest strategy with a separate optimized query
            latest_strategy_query = (
                select(Operation.config["strategy"].label("strategy"))
                .where(
                    and_(
                        Operation.collection_id == collection_id,
                        Operation.type == "chunking",
                        Operation.config.isnot(None),
                    )
                )
                .order_by(Operation.created_at.desc())
                .limit(1)
            )

            strategy_result = await self.db_session.execute(latest_strategy_query)
            latest_strategy_row = strategy_result.one_or_none()
            latest_strategy = latest_strategy_row.strategy if latest_strategy_row else None

            result: dict[str, Any] = {
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
                cache_key = self.cache_manager._generate_cache_key("statistics", {"collection_id": collection_id})
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
        cache_key = f"preview:{preview_id}:{user_id}" if user_id else f"preview:{preview_id}"

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
        cache_key = f"preview:{preview_id}:{user_id}" if user_id else f"preview:{preview_id}"

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

    async def get_cached_preview_by_id(
        self,
        preview_id: str,
        user_id: int | None = None,  # noqa: ARG002
    ) -> ServicePreviewResponse | None:
        """Get cached preview by ID.

        Public method for retrieving cached preview results.

        Args:
            preview_id: Preview ID (cache key)
            user_id: Optional user ID for access control

        Returns:
            ServicePreviewResponse DTO or None if not found
        """
        cached_data = await self._get_cached_preview_by_key(preview_id)
        if not cached_data:
            return None

        # Convert cached dict to DTO
        chunks = self._transform_chunks_to_preview(cached_data.get("chunks", []))

        # Handle expires_at field
        expires_at = cached_data.get("expires_at")
        if expires_at:
            if isinstance(expires_at, str):
                try:
                    expires_at = datetime.fromisoformat(expires_at)
                except (ValueError, TypeError):
                    expires_at = datetime.now(UTC) + timedelta(minutes=15)
            elif not isinstance(expires_at, datetime):
                expires_at = datetime.now(UTC) + timedelta(minutes=15)
        else:
            expires_at = datetime.now(UTC) + timedelta(minutes=15)

        return ServicePreviewResponse(
            preview_id=cached_data.get("preview_id", preview_id),
            strategy=cached_data.get("strategy", "recursive"),
            config=cached_data.get("config", {}),
            chunks=cast(list[ServiceChunkPreview | dict[str, Any]], chunks),
            total_chunks=cached_data.get("total_chunks", len(chunks)),
            metrics=cached_data.get("performance_metrics"),
            processing_time_ms=cached_data.get("processing_time_ms", 0),
            cached=True,
            expires_at=expires_at,
            correlation_id=None,
        )

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

        chunk_sizes = [
            len(
                c.content
                if hasattr(c, "content") and isinstance(c.content, str) and c.content
                else c.text if hasattr(c, "text") and isinstance(c.text, str) and c.text else str(c)
            )
            for c in chunks
        ]
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
        chunk_sizes = [
            len(
                c.content
                if hasattr(c, "content") and isinstance(c.content, str) and c.content
                else c.text if hasattr(c, "text") and isinstance(c.text, str) and c.text else str(c)
            )
            for c in chunks
        ]
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

    def _build_strategy_info(self, strategy_data: dict[str, Any]) -> ServiceStrategyInfo:
        """Transform strategy data dictionary to ServiceStrategyInfo DTO.

        Args:
            strategy_data: Raw strategy data dictionary

        Returns:
            ServiceStrategyInfo DTO with all transformations applied
        """
        # Map internal ID to public ID if needed
        strategy_id = strategy_data.get("id", "")
        alias_map = {"character": "fixed_size", "markdown": "markdown", "hierarchical": "hierarchical"}
        public_id = alias_map.get(strategy_id, strategy_id)

        # Build default config with strategy field
        default_config = strategy_data.get("default_config", {}).copy()
        if "strategy" not in default_config:
            default_config["strategy"] = public_id

        return ServiceStrategyInfo(
            id=public_id,
            name=strategy_data.get("name", ""),
            description=strategy_data.get("description", ""),
            best_for=strategy_data.get("best_for", []),
            pros=strategy_data.get("pros", []),
            cons=strategy_data.get("cons", []),
            default_config=default_config,
            performance_characteristics=strategy_data.get("performance_characteristics", {}),
        )

    def _transform_chunks_to_preview(self, chunks: list[dict[str, Any]]) -> list[ServiceChunkPreview]:
        """Transform chunk dictionaries to ServiceChunkPreview DTOs.

        Args:
            chunks: List of chunk dictionaries from chunking operations

        Returns:
            List of ServiceChunkPreview DTOs with transformations applied
        """
        preview_chunks = []
        for chunk in chunks:
            # Handle both 'content' and 'text' keys for chunk content
            content = chunk.get("content") or chunk.get("text", "")
            char_count = len(content)
            # Approximate token count (rough estimate: ~4 chars per token) if not provided
            token_count = chunk.get("token_count")
            if token_count is None:
                token_count = char_count // 4

            preview_chunks.append(
                ServiceChunkPreview(
                    index=chunk.get("index", 0),
                    content=content,
                    text=None,  # We use content field
                    token_count=token_count,
                    char_count=char_count,
                    metadata=chunk.get("metadata", {}),
                    quality_score=chunk.get("quality_score", 0.8),
                    overlap_info=chunk.get("overlap_info"),
                )
            )
        return preview_chunks

    def _get_default_metrics(self) -> list[ServiceStrategyMetrics]:
        """Create default placeholder metrics for all primary strategies.

        Returns:
            List of ServiceStrategyMetrics with zero/default values
        """
        from packages.webui.api.v2.chunking_schemas import ChunkingStrategy

        strategies = [
            ChunkingStrategy.FIXED_SIZE,
            ChunkingStrategy.RECURSIVE,
            ChunkingStrategy.MARKDOWN,
            ChunkingStrategy.SEMANTIC,
            ChunkingStrategy.HIERARCHICAL,
            ChunkingStrategy.HYBRID,
        ]

        return [
            ServiceStrategyMetrics(
                strategy=s.value,
                usage_count=0,
                avg_chunk_size=0.0,
                avg_processing_time=0.0,
                success_rate=0.0,
                avg_quality_score=0.0,
                best_for_types=[],
            )
            for s in strategies
        ]

    # Additional methods for completeness

    async def execute_ingestion_chunking(
        self,
        text: str,
        document_id: str,
        collection: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        file_type: str | None = None,  # noqa: ARG002 - Reserved for future file-type-specific optimizations
        _from_segment: bool = False,  # Internal flag to prevent recursive segmentation
    ) -> dict[str, Any]:
        """Execute chunking for document ingestion with strategy resolution and fallback.

        This method provides a unified chunking interface for ingestion tasks (APPEND, REINDEX).
        It resolves the chunking strategy from collection configuration, executes it with proper
        error handling, and falls back to TokenChunker on recoverable errors.
        Tracks Prometheus metrics for observability.

        For large documents, it automatically uses progressive segmentation to maintain
        bounded memory usage.

        Args:
            text: The text content to chunk
            document_id: The document ID for chunk metadata
            collection: Collection dictionary containing chunking_strategy and chunking_config
            metadata: Optional metadata to include with chunks
            file_type: Optional file type for strategy optimization (reserved for future use)

        Returns:
            Dictionary containing:
                - chunks: List of chunk dictionaries in ingestion format
                - stats: Execution statistics including strategy used and timing

        Raises:
            Exception: For fatal errors that prevent chunking
        """
        import time

        from packages.webui.services.chunking_constants import SEGMENT_SIZE_THRESHOLD, STRATEGY_SEGMENT_THRESHOLDS

        # Check if document requires segmentation (skip if called from segment processing)
        if not _from_segment:
            text_size = len(text.encode("utf-8"))
            chunking_strategy = collection.get("chunking_strategy")

            # Get the threshold for this strategy - normalize the strategy name
            if chunking_strategy:
                # Use simple lowercase normalization
                strategy_key = chunking_strategy.lower().replace("_", "").replace("-", "")
                # Map common variations
                strategy_mapping = {
                    "fixedsize": "character",
                    "tokenchunker": "character",
                    "documentstructure": "markdown",
                    "slidingwindow": "sliding",
                }
                strategy_key = strategy_mapping.get(strategy_key, strategy_key)
            else:
                strategy_key = ""
            threshold = STRATEGY_SEGMENT_THRESHOLDS.get(strategy_key, SEGMENT_SIZE_THRESHOLD)

            if text_size > threshold:
                logger.info(
                    "Document exceeds size threshold, using progressive segmentation",
                    extra={
                        "document_id": document_id,
                        "text_size": text_size,
                        "threshold": threshold,
                        "strategy": strategy_key,
                    },
                )
                # Generate correlation ID for tracking
                correlation_id = str(uuid.uuid4())
                # Use segmented processing for large documents
                return await self.execute_ingestion_chunking_segmented(
                    text=text,
                    document_id=document_id,
                    collection=collection,
                    metadata=metadata,
                    file_type=file_type,
                    chunking_strategy=chunking_strategy or "recursive",
                    chunk_size=int(collection.get("chunk_size", 512)),
                    chunk_overlap=int(collection.get("chunk_overlap", 50)),
                    correlation_id=correlation_id,
                )

        start_time = time.time()
        strategy_used = None
        metrics_strategy_label = None  # For consistent metric labels
        fallback_used = False
        fallback_reason = None
        chunks = []
        correlation_id = str(uuid.uuid4())

        try:
            # Extract chunking configuration from collection
            chunking_strategy = collection.get("chunking_strategy")
            chunking_config = collection.get("chunking_config", {})

            # Get chunk_size and chunk_overlap for fallback (prefer chunking_config values)
            chunk_size = chunking_config.get("chunk_size", collection.get("chunk_size", 1000))
            chunk_overlap = chunking_config.get("chunk_overlap", collection.get("chunk_overlap", 200))

            # Ensure chunk_size and chunk_overlap are integers
            try:
                chunk_size = int(chunk_size)
            except (TypeError, ValueError):
                logger.warning(
                    f"Invalid chunk_size type: {type(chunk_size).__name__}, using default 1000",
                    extra={"correlation_id": correlation_id},
                )
                chunk_size = 1000

            try:
                chunk_overlap = int(chunk_overlap)
            except (TypeError, ValueError):
                logger.warning(
                    f"Invalid chunk_overlap type: {type(chunk_overlap).__name__}, using default 200",
                    extra={"correlation_id": correlation_id},
                )
                chunk_overlap = 200

            # Validate and sanitize chunk_size and chunk_overlap for fallback scenarios
            # If invalid values are provided, use safe defaults
            if chunk_size <= 0:
                logger.warning(
                    f"Invalid chunk_size {chunk_size}, using default 1000", extra={"correlation_id": correlation_id}
                )
                chunk_size = 1000
            if chunk_overlap < 0:
                logger.warning(
                    f"Invalid chunk_overlap {chunk_overlap}, using default 200",
                    extra={"correlation_id": correlation_id},
                )
                chunk_overlap = 200
            if chunk_overlap >= chunk_size:
                logger.warning(
                    f"chunk_overlap {chunk_overlap} >= chunk_size {chunk_size}, setting to chunk_size/2",
                    extra={"correlation_id": correlation_id},
                )
                chunk_overlap = chunk_size // 2

            # If no strategy specified, use TokenChunker directly
            if not chunking_strategy:
                logger.info(
                    "No chunking strategy specified for collection, using TokenChunker",
                    extra={
                        "collection_id": collection.get("id"),
                        "document_id": document_id,
                        "strategy_used": "TokenChunker",
                        "correlation_id": correlation_id,
                    },
                )
                try:
                    chunker = token_chunking.TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    # Execute in thread pool to avoid blocking event loop
                    chunks = await asyncio.to_thread(chunker.chunk_text, text, document_id, metadata or {})
                    strategy_used = "TokenChunker"
                    metrics_strategy_label = "character"  # Internal name for metrics

                    # Record metrics for direct TokenChunker usage
                    self._record_metric("record_chunks_produced", metrics_strategy_label, len(chunks))
                    self._record_metric("record_chunk_sizes", metrics_strategy_label, chunks)
                except (MemoryError, SystemError) as e:
                    # Fatal errors should be propagated
                    logger.error(f"Fatal error creating TokenChunker: {e}", extra={"correlation_id": correlation_id})
                    raise

            else:
                # Normalize strategy name using factory
                try:
                    # Build and validate configuration
                    config_result = self.config_builder.build_config(
                        strategy=chunking_strategy,
                        user_config=chunking_config,
                    )

                    strategy_value = (
                        config_result.strategy.value
                        if hasattr(config_result.strategy, "value")
                        else str(config_result.strategy)
                    )

                    # Fall back to the raw value if normalization fails; downstream logic will handle it.
                    with contextlib.suppress(Exception):
                        strategy_value = ChunkingStrategyFactory.normalize_strategy_name(strategy_value)

                    strategy_input_key = chunking_strategy or strategy_value
                    if isinstance(strategy_input_key, str):
                        normalized_input_key = strategy_input_key.lower().replace("-", "_")
                        strategy_value = self.STRATEGY_MAPPING.get(normalized_input_key, strategy_value)

                    if config_result.validation_errors:
                        logger.warning(
                            "Invalid chunking config, falling back to TokenChunker",
                            extra={
                                "collection_id": collection.get("id"),
                                "document_id": document_id,
                                "validation_errors": ", ".join(config_result.validation_errors),
                                "correlation_id": correlation_id,
                            },
                        )
                        chunker = token_chunking.TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        # Execute in thread pool to avoid blocking event loop
                        chunks = await asyncio.to_thread(chunker.chunk_text, text, document_id, metadata or {})
                        strategy_used = "TokenChunker"
                        metrics_strategy_label = "character"  # Internal name for metrics
                        fallback_used = True
                        fallback_reason = "invalid_config"

                        # Record fallback metrics with normalized label
                        try:
                            from packages.webui.services.chunking_strategy_factory import ChunkingStrategyFactory

                            fallback_label = chunking_strategy or strategy_value or "unknown"
                            normalized_strategy = ChunkingStrategyFactory.normalize_strategy_name(fallback_label)
                        except Exception:
                            normalized_strategy = chunking_strategy or strategy_value or "unknown"
                        self._record_metric("record_chunking_fallback", normalized_strategy, "invalid_config")
                        self._record_metric("record_chunks_produced", metrics_strategy_label, len(chunks))
                        self._record_metric("record_chunk_sizes", metrics_strategy_label, chunks)

                    else:
                        # Create strategy instance
                        try:
                            chunking_strategy_instance = self.strategy_factory.create_strategy(
                                strategy_name=config_result.strategy,
                                config=config_result.config,
                                correlation_id=str(uuid.uuid4()),
                            )

                            # Execute chunking with the strategy
                            from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig

                            config = config_result.config
                            chunk_size_from_config = int(config.get("chunk_size", chunk_size) or chunk_size)
                            chunk_overlap_from_config = int(config.get("chunk_overlap", chunk_overlap) or 0)

                            initial_min_tokens = self._calculate_min_tokens(chunk_size_from_config)

                            def build_chunk_config(min_tokens_override: int) -> ChunkConfig:
                                adjusted_min_tokens = max(10, min_tokens_override)
                                adjusted_max_tokens = max(chunk_size_from_config, adjusted_min_tokens + 1)
                                adjusted_overlap = min(
                                    chunk_overlap_from_config,
                                    max(0, adjusted_min_tokens - 1),
                                )
                                return ChunkConfig(
                                    strategy_name=strategy_value,
                                    min_tokens=adjusted_min_tokens,
                                    max_tokens=adjusted_max_tokens,
                                    overlap_tokens=adjusted_overlap,
                                    preserve_structure=config.get("preserve_structure", True),
                                    semantic_threshold=config.get("semantic_threshold", 0.7),
                                    hierarchy_levels=config.get("hierarchy_levels", 3),
                                )

                            chunk_config_obj = build_chunk_config(initial_min_tokens)

                            try:
                                # Execute chunking in thread pool to avoid blocking event loop
                                chunk_entities = await asyncio.to_thread(
                                    chunking_strategy_instance.chunk,
                                    content=text,
                                    config=chunk_config_obj,
                                )
                            except ChunkSizeViolationError as size_error:
                                relaxed_min_tokens = max(10, initial_min_tokens // 2)
                                if relaxed_min_tokens < initial_min_tokens:
                                    logger.info(
                                        "Retrying chunking with relaxed min_tokens due to size violation",
                                        extra={
                                            "correlation_id": correlation_id,
                                            "document_id": document_id,
                                            "collection_id": collection.get("id"),
                                            "initial_min_tokens": initial_min_tokens,
                                            "relaxed_min_tokens": relaxed_min_tokens,
                                            "chunk_size": chunk_size_from_config,
                                            "chunk_overlap": chunk_overlap_from_config,
                                            "error": str(size_error),
                                        },
                                    )
                                    chunk_config_obj = build_chunk_config(relaxed_min_tokens)
                                    chunk_entities = await asyncio.to_thread(
                                        chunking_strategy_instance.chunk,
                                        content=text,
                                        config=chunk_config_obj,
                                    )
                                else:
                                    raise

                            # Convert chunk entities to ingestion format
                            chunks = []
                            for idx, chunk_entity in enumerate(chunk_entities):
                                chunk_dict = {
                                    "chunk_id": f"{document_id}_{idx:04d}",  # Fixed format without '_chunk_' prefix
                                    "text": chunk_entity.content,
                                    "metadata": {
                                        **(metadata or {}),
                                        "index": idx,
                                        "strategy": strategy_value,
                                    },
                                }
                                chunks.append(chunk_dict)

                            strategy_used = strategy_value
                            # Normalize to internal strategy label for metrics
                            try:
                                from packages.webui.services.chunking_strategy_factory import ChunkingStrategyFactory

                                metrics_strategy_label = ChunkingStrategyFactory.normalize_strategy_name(strategy_used)
                            except Exception:
                                metrics_strategy_label = strategy_used
                            logger.info(
                                "Successfully chunked document using strategy",
                                extra={
                                    "document_id": document_id,
                                    "collection_id": collection.get("id"),
                                    "strategy_used": strategy_used,
                                    "chunk_count": len(chunks),
                                    "correlation_id": correlation_id,
                                },
                            )

                            # Record metrics for successful strategy execution
                            self._record_metric("record_chunks_produced", metrics_strategy_label, len(chunks))
                            self._record_metric("record_chunk_sizes", metrics_strategy_label, chunks)

                        except Exception as strategy_error:
                            # Strategy execution failed, fall back to TokenChunker
                            logger.warning(
                                "Strategy execution failed, falling back to TokenChunker",
                                extra={
                                    "document_id": document_id,
                                    "collection_id": collection.get("id"),
                                    "strategy": chunking_strategy,
                                    "error": str(strategy_error),
                                    "correlation_id": correlation_id,
                                },
                            )
                            chunker = token_chunking.TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                            # Execute in thread pool to avoid blocking event loop
                            chunks = await asyncio.to_thread(chunker.chunk_text, text, document_id, metadata or {})
                            strategy_used = "TokenChunker"
                            metrics_strategy_label = "character"  # Internal name for metrics
                            fallback_used = True
                            fallback_reason = "runtime_error"

                            # Record fallback metrics with normalized label
                            try:
                                from packages.webui.services.chunking_strategy_factory import ChunkingStrategyFactory

                                normalized_strategy = ChunkingStrategyFactory.normalize_strategy_name(
                                    chunking_strategy or strategy_value or "unknown"
                                )
                            except Exception:
                                normalized_strategy = chunking_strategy or strategy_value or "unknown"
                            self._record_metric("record_chunking_fallback", normalized_strategy, "runtime_error")
                            self._record_metric("record_chunks_produced", metrics_strategy_label, len(chunks))
                            self._record_metric("record_chunk_sizes", metrics_strategy_label, chunks)

                except Exception as config_error:
                    # Configuration building failed, fall back to TokenChunker
                    logger.warning(
                        "Failed to build config for strategy, falling back to TokenChunker",
                        extra={
                            "document_id": document_id,
                            "collection_id": collection.get("id"),
                            "strategy": chunking_strategy,
                            "error": str(config_error),
                            "correlation_id": correlation_id,
                        },
                    )
                    chunker = token_chunking.TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    # Execute in thread pool to avoid blocking event loop
                    chunks = await asyncio.to_thread(chunker.chunk_text, text, document_id, metadata or {})
                    strategy_used = "TokenChunker"
                    metrics_strategy_label = "character"  # Internal name for metrics
                    fallback_used = True
                    fallback_reason = "config_error"

                    # Record fallback metrics with normalized label
                    try:
                        from packages.webui.services.chunking_strategy_factory import ChunkingStrategyFactory

                        normalized_fallback = ChunkingStrategyFactory.normalize_strategy_name(
                            chunking_strategy or strategy_value or "unknown"
                        )
                    except Exception:
                        normalized_fallback = chunking_strategy or strategy_value or "unknown"
                    self._record_metric("record_chunking_fallback", normalized_fallback, "config_error")
                    self._record_metric("record_chunks_produced", metrics_strategy_label, len(chunks))
                    self._record_metric("record_chunk_sizes", metrics_strategy_label, chunks)

        except Exception as e:
            # Fatal error - cannot proceed with chunking
            logger.error(
                "Fatal error during chunking",
                extra={
                    "document_id": document_id,
                    "collection_id": collection.get("id"),
                    "error": str(e),
                    "correlation_id": correlation_id,
                },
            )
            raise

        # Calculate duration
        duration_seconds = time.time() - start_time
        duration_ms = int(duration_seconds * 1000)

        # Record duration metric for the strategy that was actually used
        if metrics_strategy_label:
            self._record_metric("record_chunking_duration", metrics_strategy_label, duration_seconds)

        # Log fallback event if occurred
        if fallback_used:
            logger.warning(
                "Chunking fallback occurred",
                extra={
                    "document_id": document_id,
                    "collection_id": collection.get("id"),
                    "original_strategy": chunking_strategy,
                    "strategy_used": strategy_used,
                    "fallback_reason": fallback_reason,
                    "correlation_id": correlation_id,
                },
            )

        return {
            "chunks": chunks,
            "stats": {
                "duration_ms": duration_ms,
                "strategy_used": strategy_used,
                "fallback": fallback_used,
                "fallback_reason": fallback_reason,
                "chunk_count": len(chunks),
            },
        }

    async def execute_ingestion_chunking_segmented(
        self,
        text: str,
        document_id: str,
        collection: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        file_type: str | None = None,
        chunking_strategy: str | None = None,
        chunk_size: int | None = None,  # noqa: ARG002 - Reserved for future use
        chunk_overlap: int | None = None,  # noqa: ARG002 - Reserved for future use
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute chunking for large documents using progressive segmentation.

        This method segments large documents into manageable pieces and processes
        each segment independently to maintain bounded memory usage.

        Args:
            text: The text content to chunk
            document_id: The document ID for chunk metadata
            collection: Collection dictionary containing chunking_strategy and chunking_config
            metadata: Optional metadata to include with chunks
            file_type: Optional file type for strategy optimization
            chunking_strategy: Resolved chunking strategy name
            chunk_size: Chunk size configuration
            chunk_overlap: Chunk overlap configuration

        Returns:
            Dictionary containing:
                - chunks: List of all chunks from all segments
                - stats: Execution statistics including segmentation info
        """
        import time

        from packages.webui.services.chunking_constants import (
            DEFAULT_SEGMENT_OVERLAP,
            DEFAULT_SEGMENT_SIZE,
            MAX_SEGMENTS_PER_DOCUMENT,
            STRATEGY_SEGMENT_THRESHOLDS,
        )
        from packages.webui.services.chunking_metrics import (
            record_document_segmented,
            record_segment_size,
            record_segments_created,
        )

        start_time = time.time()
        all_chunks = []
        chunk_id_counter = 0

        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Get segmentation configuration based on strategy - normalize the strategy name
        strategy = chunking_strategy or collection.get("chunking_strategy", "")
        if strategy:
            # Use simple lowercase normalization
            strategy_key = strategy.lower().replace("_", "").replace("-", "")
            # Map common variations
            strategy_mapping = {
                "fixedsize": "character",
                "tokenchunker": "character",
                "documentstructure": "markdown",
                "slidingwindow": "sliding",
            }
            strategy_key = strategy_mapping.get(strategy_key, strategy_key)
        else:
            strategy_key = "recursive"  # Default
        segment_threshold = STRATEGY_SEGMENT_THRESHOLDS.get(strategy_key, DEFAULT_SEGMENT_SIZE)
        segment_size = min(segment_threshold, DEFAULT_SEGMENT_SIZE)
        segment_overlap = DEFAULT_SEGMENT_OVERLAP

        # Calculate segments
        text_length = len(text.encode("utf-8"))
        segments = []

        if text_length <= segment_size:
            # Document is small enough to process as single segment
            segments = [text]
        else:
            # Create segments with overlap
            position = 0
            segment_count = 0

            while position < len(text) and segment_count < MAX_SEGMENTS_PER_DOCUMENT:
                # Calculate segment boundaries
                segment_start = max(0, position - segment_overlap if position > 0 else 0)
                segment_end = min(len(text), position + segment_size)

                # Extract segment ensuring we don't split in the middle of a character
                segment = text[segment_start:segment_end]

                # Find a good break point if not at the end
                if segment_end < len(text):
                    # Try to break at paragraph boundary
                    last_para = segment.rfind("\n\n")
                    if last_para > segment_size * 0.8:  # If we have a paragraph break in last 20%
                        segment = text[segment_start : segment_start + last_para]
                        segment_end = segment_start + last_para
                    else:
                        # Try to break at sentence boundary
                        last_sentence = max(segment.rfind(". "), segment.rfind("! "), segment.rfind("? "))
                        if last_sentence > segment_size * 0.8:
                            segment = text[segment_start : segment_start + last_sentence + 1]
                            segment_end = segment_start + last_sentence + 1

                segments.append(segment)
                segment_count += 1

                # Record segment metrics
                record_segment_size(strategy_key, len(segment.encode("utf-8")))

                # Move position forward (accounting for overlap)
                position = segment_end

        # Record segmentation metrics if document was segmented
        if len(segments) > 1:
            record_document_segmented(strategy_key)
            record_segments_created(strategy_key, len(segments))
            logger.info(
                "Document segmented for processing",
                extra={
                    "document_id": document_id,
                    "collection_id": collection.get("id"),
                    "segment_count": len(segments),
                    "text_length": text_length,
                    "strategy": strategy_key,
                    "correlation_id": correlation_id,
                },
            )

        # Process each segment
        for segment_idx, segment_text in enumerate(segments):
            try:
                # Process segment using the regular chunking logic
                segment_result = await self._process_segment(
                    segment_text,
                    document_id,
                    collection,
                    metadata,
                    file_type,
                    chunk_id_counter,
                    segment_idx,
                    len(segments),
                )

                # Add chunks from this segment
                all_chunks.extend(segment_result["chunks"])
                chunk_id_counter += len(segment_result["chunks"])

            except Exception as e:
                logger.error(
                    "Failed to process segment",
                    extra={
                        "document_id": document_id,
                        "segment_idx": segment_idx,
                        "error": str(e),
                        "correlation_id": correlation_id,
                    },
                )
                # Continue with other segments even if one fails
                continue

        # Calculate duration
        duration_seconds = time.time() - start_time
        duration_ms = int(duration_seconds * 1000)

        return {
            "chunks": all_chunks,
            "stats": {
                "duration_ms": duration_ms,
                "strategy_used": strategy_key,
                "chunk_count": len(all_chunks),
                "segment_count": len(segments),
                "segmented": len(segments) > 1,
            },
        }

    async def _process_segment(
        self,
        text: str,
        document_id: str,
        collection: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        file_type: str | None = None,
        chunk_id_start: int = 0,
        segment_idx: int = 0,
        total_segments: int = 1,
    ) -> dict[str, Any]:
        """Process a single segment of text.

        This is a helper method that processes a segment using the regular chunking logic
        but with adjusted chunk IDs to maintain continuity across segments.

        Args:
            text: The segment text to chunk
            document_id: The document ID for chunk metadata
            collection: Collection dictionary containing chunking configuration
            metadata: Optional metadata to include with chunks
            file_type: Optional file type for strategy optimization
            chunk_id_start: Starting index for chunk IDs
            segment_idx: Index of this segment
            total_segments: Total number of segments

        Returns:
            Dictionary containing chunks from this segment
        """
        # Use the regular chunking logic but adjust chunk IDs
        # Pass _from_segment=True to prevent recursive segmentation
        result = await self.execute_ingestion_chunking(
            text, document_id, collection, metadata, file_type, _from_segment=True
        )

        # Adjust chunk IDs to maintain continuity
        for idx, chunk in enumerate(result["chunks"]):
            chunk["chunk_id"] = f"{document_id}_{chunk_id_start + idx:04d}"  # Fixed format without '_chunk_' prefix
            # Add segment metadata
            if metadata is None:
                chunk["metadata"] = {}
            chunk["metadata"]["segment_idx"] = segment_idx
            chunk["metadata"]["total_segments"] = total_segments

        return result

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
