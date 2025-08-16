"""
Chunking orchestrator service.

Main coordinator that orchestrates chunking operations across all specialized services.
"""

import logging
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.chunking.infrastructure.exceptions import (
    ValidationError,
)
from packages.shared.database.repositories.collection_repository import (
    CollectionRepository,
)
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.webui.services.dtos import (
    ServiceChunkingStats,
    ServiceChunkPreview,
    ServiceCompareResponse,
    ServicePreviewResponse,
    ServiceStrategyComparison,
    ServiceStrategyInfo,
    ServiceStrategyMetrics,
    ServiceStrategyRecommendation,
)

from .cache import ChunkingCache
from .config_manager import ChunkingConfigManager
from .metrics import ChunkingMetrics
from .processor import ChunkingProcessor
from .validator import ChunkingValidator

logger = logging.getLogger(__name__)


class ChunkingOrchestrator:
    """Orchestrates chunking operations across specialized services."""

    def __init__(
        self,
        processor: ChunkingProcessor,
        cache: ChunkingCache,
        metrics: ChunkingMetrics,
        validator: ChunkingValidator,
        config_manager: ChunkingConfigManager,
        db_session: AsyncSession | None = None,
        collection_repo: CollectionRepository | None = None,
        document_repo: DocumentRepository | None = None,
    ):
        """
        Initialize the orchestrator with all required services.

        Args:
            processor: Core chunking processor
            cache: Caching service
            metrics: Metrics collection service
            validator: Validation service
            config_manager: Configuration management service
            db_session: Database session
            collection_repo: Collection repository
            document_repo: Document repository
        """
        self.processor = processor
        self.cache = cache
        self.metrics = metrics
        self.validator = validator
        self.config_manager = config_manager
        self.db_session = db_session
        self.collection_repo = collection_repo
        self.document_repo = document_repo

    async def preview_chunks(
        self,
        content: str | None = None,
        document_id: str | None = None,
        strategy: str = "recursive",
        config: dict[str, Any] | None = None,
        user_id: int | None = None,
        use_cache: bool = True,
    ) -> ServicePreviewResponse:
        """
        Preview chunking results for content or document.

        Args:
            content: Direct content to chunk
            document_id: Document ID to chunk
            strategy: Chunking strategy
            config: Strategy configuration
            user_id: User ID for access control
            use_cache: Whether to use caching

        Returns:
            Preview response with chunks and statistics
        """
        # Validate request
        await self.validator.validate_preview_request(content, document_id, strategy, config)

        # Validate access if document_id provided
        if document_id and user_id:
            await self.validator.validate_document_access(document_id, user_id)

        # Load content if document_id provided
        if document_id:
            content = await self._load_document_content(document_id)

        # Generate content hash for caching
        content_hash = self.cache.generate_content_hash(content)

        # Check cache
        if use_cache:
            cached = await self.cache.get_cached_preview(content_hash, strategy, config)
            if cached:
                self.metrics.record_cache_hit("preview")
                return self._build_preview_response(cached)
            self.metrics.record_cache_miss("preview")

        # Merge with default config
        merged_config = self.config_manager.merge_configs(strategy, config)

        # Execute chunking with metrics tracking
        async with self.metrics.measure_operation(strategy) as context:
            chunks = await self.processor.process_document(content, strategy, merged_config, use_fallback=True)
            context["chunks_produced"] = len(chunks)
            self.metrics.record_chunks_produced(strategy, chunks)

        # Calculate statistics
        statistics = self.processor.calculate_statistics(chunks)

        # Build response
        preview_data = {
            "chunks": chunks,
            "statistics": statistics,
            "strategy": strategy,
            "config": merged_config,
        }

        # Cache result
        if use_cache:
            await self.cache.cache_preview(content_hash, strategy, merged_config, preview_data)

        return self._build_preview_response(preview_data)

    async def compare_strategies(
        self,
        content: str,
        strategies: list[str] | None = None,
        base_config: dict[str, Any] | None = None,
        user_id: int | None = None,  # noqa: ARG002
    ) -> ServiceCompareResponse:
        """
        Compare multiple chunking strategies.

        Args:
            content: Content to chunk
            strategies: List of strategies to compare (default: all)
            base_config: Base configuration for all strategies
            user_id: User ID for tracking

        Returns:
            Comparison response with results for each strategy
        """
        # Validate content
        self.validator.validate_content(content)

        # Use all strategies if none specified
        if not strategies:
            strategies = self.config_manager.get_all_strategies()
            strategies = [s["id"] for s in strategies]

        comparisons = []

        for strategy in strategies:
            try:
                # Validate strategy
                self.validator.validate_strategy(strategy)

                # Get strategy-specific config
                config = self.config_manager.merge_configs(strategy, base_config)

                # Execute chunking
                async with self.metrics.measure_operation(strategy) as context:
                    chunks = await self.processor.process_document(content, strategy, config, use_fallback=False)
                    context["chunks_produced"] = len(chunks)

                # Calculate statistics
                stats = self.processor.calculate_statistics(chunks)

                # Build comparison entry
                comparison = ServiceStrategyComparison(
                    strategy=strategy,
                    config=config,
                    sample_chunks=[
                        ServiceChunkPreview(
                            content=chunk.get("content", ""),
                            index=chunk.get("index", i),
                            char_count=len(chunk.get("content", "")),
                            metadata=chunk.get("metadata", {}),
                        )
                        for i, chunk in enumerate(chunks[:3])
                    ],
                    total_chunks=len(chunks),
                    avg_chunk_size=stats["avg_chunk_size"],
                    size_variance=self._calculate_variance(stats),
                    quality_score=self._calculate_quality_score(stats),
                    processing_time_ms=0,  # Will be set from context after
                )
                comparisons.append(comparison)

            except Exception as e:
                logger.error("Error comparing strategy %s: %s", strategy, str(e))
                # Add failed comparison
                comparison = ServiceStrategyComparison(
                    strategy=strategy,
                    config=base_config or {},
                    sample_chunks=[],
                    total_chunks=0,
                    avg_chunk_size=0,
                    size_variance=0,
                    quality_score=0,
                    processing_time_ms=0,
                )
                comparisons.append(comparison)

        # Get recommendation
        recommendation = self._get_recommendation(comparisons, content)

        return ServiceCompareResponse(
            comparison_id=str(uuid.uuid4()),
            comparisons=comparisons,
            recommendation=recommendation,
            processing_time_ms=0,  # Could track actual time if needed
        )

    async def execute_ingestion_chunking(
        self,
        content: str,
        strategy: str,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Execute chunking for document ingestion.

        Args:
            content: Content to chunk
            strategy: Chunking strategy
            config: Strategy configuration
            metadata: Additional metadata

        Returns:
            List of chunks ready for ingestion
        """
        # Validate inputs
        self.validator.validate_content(content)
        self.validator.validate_strategy(strategy)
        if config:
            self.validator.validate_config(strategy, config)

        # Merge configuration
        merged_config = self.config_manager.merge_configs(strategy, config)

        # Execute chunking with fallback
        async with self.metrics.measure_operation(strategy) as context:
            try:
                chunks = await self.processor.process_document(content, strategy, merged_config, use_fallback=False)
            except Exception as e:
                logger.warning("Strategy %s failed, using fallback: %s", strategy, str(e))
                self.metrics.record_fallback(strategy)
                chunks = await self.processor.process_document(content, strategy, merged_config, use_fallback=True)

            context["chunks_produced"] = len(chunks)
            self.metrics.record_chunks_produced(strategy, chunks)

        # Add metadata to chunks
        if metadata:
            for chunk in chunks:
                chunk["metadata"] = {**chunk.get("metadata", {}), **metadata}

        return chunks

    async def get_available_strategies(self) -> list[ServiceStrategyInfo]:
        """
        Get all available chunking strategies with metadata.

        Returns:
            List of strategy information objects
        """
        strategies = []

        for strategy_data in self.config_manager.get_all_strategies():
            info = ServiceStrategyInfo(
                id=strategy_data["id"],
                name=strategy_data.get("name", strategy_data["id"]),
                description=strategy_data.get("description", ""),
                best_for=strategy_data.get("best_for", []),
                pros=strategy_data.get("pros", []),
                cons=strategy_data.get("cons", []),
                default_config=strategy_data.get("default_config", {}),
                supported_file_types=strategy_data.get("supported_file_types", []),
            )
            strategies.append(info)

        return strategies

    async def recommend_strategy(
        self,
        file_type: str | None = None,
        content_length: int | None = None,
        document_type: str | None = None,
        sample_content: str | None = None,
    ) -> ServiceStrategyRecommendation:
        """
        Recommend a chunking strategy based on document characteristics.

        Args:
            file_type: File extension or MIME type
            content_length: Length of content
            document_type: Type of document
            sample_content: Sample of content for analysis

        Returns:
            Strategy recommendation
        """
        # Get base recommendation from config manager
        rec_data = self.config_manager.recommend_strategy(file_type, content_length, document_type)

        # Enhance recommendation with sample analysis if provided
        if sample_content and ("```" in sample_content or "def " in sample_content or "class " in sample_content):
            if rec_data["strategy"] != "recursive":
                rec_data["reasoning"].append("Code patterns detected in sample")
                rec_data["alternatives"].append(rec_data["strategy"])
                rec_data["strategy"] = "recursive"

            elif sample_content.startswith("#") or "## " in sample_content:
                if rec_data["strategy"] != "markdown":
                    rec_data["reasoning"].append("Markdown headers detected in sample")
                    rec_data["alternatives"].append(rec_data["strategy"])
                    rec_data["strategy"] = "markdown"

        # Build recommendation object
        return ServiceStrategyRecommendation(
            strategy=rec_data["strategy"],
            confidence=rec_data["confidence"],
            reasoning="\n".join(rec_data["reasoning"]) if isinstance(rec_data["reasoning"], list) else rec_data["reasoning"],
            alternatives=[
                alt.get("strategy", alt) if isinstance(alt, dict) else alt 
                for alt in rec_data.get("alternatives", [])
            ],
        )

    async def get_collection_statistics(
        self,
        collection_id: str,
        user_id: int,
    ) -> ServiceChunkingStats:
        """
        Get chunking statistics for a collection.

        Args:
            collection_id: Collection ID
            user_id: User ID for access control

        Returns:
            Collection chunking statistics
        """
        # Validate access
        await self.validator.validate_collection_access(collection_id, user_id)

        # Get documents in collection
        if self.document_repo:
            documents = await self.document_repo.get_by_collection(collection_id)
            total_documents = len(documents)
            total_chunks = sum(doc.chunk_count or 0 for doc in documents)

            # Calculate strategy breakdown
            strategy_breakdown = {}
            for doc in documents:
                strategy = doc.chunking_strategy or "unknown"
                strategy_breakdown[strategy] = strategy_breakdown.get(strategy, 0) + 1

        else:
            total_documents = 0
            total_chunks = 0
            strategy_breakdown = {}

        # Get metrics from metrics service
        metrics_data = self.metrics.get_statistics()

        return ServiceChunkingStats(
            total_documents=total_documents,
            total_chunks=total_chunks,
            average_chunk_size=metrics_data.get("average_chunk_size", 0),
            strategy_breakdown=strategy_breakdown,
            last_updated=metrics_data.get("last_operation", {}).get("timestamp"),
        )

    async def _load_document_content(self, document_id: str) -> str:
        """Load document content from repository."""
        if not self.document_repo:
            raise ValidationError(
                field="document_repo",
                value=None,
                reason="Document repository not available"
            )

        document = await self.document_repo.get(document_id)
        if not document:
            raise ValidationError(
                field="document_id",
                value=document_id,
                reason="Document not found"
            )

        return document.content or ""

    def _build_preview_response(self, data: dict[str, Any]) -> ServicePreviewResponse:
        """Build preview response from data."""
        chunks = data.get("chunks", [])
        stats = data.get("statistics", {})

        # Convert chunks to preview format
        preview_chunks = [
            ServiceChunkPreview(
                content=chunk.get("content", ""),
                index=chunk.get("index", i),
                char_count=len(chunk.get("content", "")),
                metadata=chunk.get("metadata", {}),
            )
            for i, chunk in enumerate(chunks[:10])  # Limit to 10 for preview
        ]

        return ServicePreviewResponse(
            preview_id=data.get("cache_key", str(uuid.uuid4())),
            chunks=preview_chunks,
            total_chunks=len(chunks),
            metrics=stats,
            strategy=data.get("strategy", "unknown"),
            config=data.get("config", {}),
            cached=bool(data.get("cache_key")),
        )

    def _build_strategy_metrics(
        self,
        strategy: str,
        stats: dict[str, Any],
    ) -> ServiceStrategyMetrics:
        """Build strategy metrics object."""
        # Get metrics from metrics service
        strategy_metrics = self.metrics.get_strategy_metrics(strategy)

        return ServiceStrategyMetrics(
            strategy=strategy,
            avg_processing_time=strategy_metrics.get("average_duration", 0),
            avg_chunk_size=stats.get("avg_chunk_size", 0),
            success_rate=strategy_metrics.get("success_rate", 1.0),
            avg_quality_score=self._calculate_quality_score(stats),
        )

    def _calculate_quality_score(self, stats: dict[str, Any]) -> float:
        """Calculate quality score based on statistics."""
        # Simple quality heuristic
        avg_size = stats.get("avg_chunk_size", 0)
        min_size = stats.get("min_chunk_size", 0)
        max_size = stats.get("max_chunk_size", 0)

        if avg_size == 0:
            return 0

        # Penalize high variance
        variance = (max_size - min_size) / avg_size if avg_size > 0 else 1
        quality = max(0, 1 - variance / 10)  # Normalize to 0-1

        # Bonus for reasonable chunk sizes
        if 500 <= avg_size <= 1500:
            quality += 0.2

        return min(1.0, quality)

    def _calculate_variance(self, stats: dict[str, Any]) -> float:
        """Calculate chunk size variance."""
        min_size = stats.get("min_chunk_size", 0)
        max_size = stats.get("max_chunk_size", 0)
        avg_size = stats.get("avg_chunk_size", 1)

        return (max_size - min_size) / avg_size if avg_size > 0 else 0

    def _get_recommendation(
        self,
        comparisons: list[ServiceStrategyComparison],
        content: str,  # noqa: ARG002
    ) -> ServiceStrategyRecommendation:
        """Get strategy recommendation based on comparisons."""
        # Find best strategy based on quality score
        best_strategy = None
        best_score = 0

        for comp in comparisons:
            if comp.quality_score > best_score:
                best_score = comp.quality_score
                best_strategy = comp.strategy

        if not best_strategy:
            best_strategy = "recursive"  # Fallback

        return ServiceStrategyRecommendation(
            strategy=best_strategy,
            confidence=min(0.9, best_score),
            reasoning=f"Best quality score: {best_score:.2f}. Analyzed {len(comparisons)} strategies.",
            alternatives=[c.strategy for c in comparisons if c.strategy != best_strategy][:2],
        )
