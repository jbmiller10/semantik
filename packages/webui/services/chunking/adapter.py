"""
Adapter for backward compatibility with ChunkingService.

This adapter wraps the new ChunkingOrchestrator to provide the same
interface as the old monolithic ChunkingService.
"""

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository

from .orchestrator import ChunkingOrchestrator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from packages.webui.services.chunking_service import ChunkingService


class ChunkingServiceAdapter:
    """
    Adapter that provides ChunkingService interface using ChunkingOrchestrator.

    This class maintains backward compatibility while using the new
    refactored architecture under the hood.
    """

    def __init__(
        self,
        orchestrator: ChunkingOrchestrator,
        db_session: AsyncSession | None = None,
        collection_repo: CollectionRepository | None = None,
        document_repo: DocumentRepository | None = None,
    ) -> None:
        """Initialize adapter with orchestrator."""
        self.orchestrator = orchestrator
        self.db_session = db_session or orchestrator.db_session
        self.collection_repo = collection_repo or orchestrator.collection_repo
        self.document_repo = document_repo or orchestrator.document_repo
        self._legacy_service: ChunkingService | None = None

    async def preview_chunks(
        self,
        content: str | None = None,
        document_id: str | None = None,
        strategy: str = "recursive",
        config: dict[str, Any] | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        """Preview chunking results - delegates to orchestrator."""
        result = await self.orchestrator.preview_chunks(
            content=content,
            document_id=document_id,
            strategy=strategy,
            config=config,
            user_id=user_id,
            use_cache=True,
        )

        # Convert ServicePreviewResponse to dict for backward compatibility
        return {
            "chunks": [
                {
                    "content": chunk.content if hasattr(chunk, "content") else chunk.get("content", ""),
                    "index": chunk.index if hasattr(chunk, "index") else chunk.get("index", 0),
                    "size": (
                        chunk.char_count
                        if hasattr(chunk, "char_count")
                        else len(chunk.content if hasattr(chunk, "content") else chunk.get("content", ""))
                    ),
                    "metadata": chunk.metadata if hasattr(chunk, "metadata") else chunk.get("metadata", {}),
                }
                for chunk in result.chunks
            ],
            "total_chunks": result.total_chunks,
            "statistics": result.metrics,
            "strategy": result.strategy,
            "config": result.config,
            "cache_key": result.preview_id,
        }

    async def preview_chunking(
        self,
        content: str | None = None,
        document_id: str | None = None,
        strategy: str = "recursive",
        config: dict[str, Any] | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        """Alias for preview_chunks for backward compatibility."""
        return await self.preview_chunks(
            content=content,
            document_id=document_id,
            strategy=strategy,
            config=config,
            user_id=user_id,
        )

    async def compare_strategies(
        self,
        content: str,
        strategies: list[str] | None = None,
        base_config: dict[str, Any] | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        """Compare multiple strategies - delegates to orchestrator."""
        result = await self.orchestrator.compare_strategies(
            content=content,
            strategies=strategies,
            base_config=base_config,
            user_id=user_id,
        )

        # Convert ServiceCompareResponse to dict
        return {
            "comparisons": [
                {
                    "strategy": comp.strategy if hasattr(comp, "strategy") else comp.get("strategy", ""),
                    "chunk_count": comp.total_chunks if hasattr(comp, "total_chunks") else comp.get("total_chunks", 0),
                    "avg_chunk_size": (
                        comp.avg_chunk_size if hasattr(comp, "avg_chunk_size") else comp.get("avg_chunk_size", 0)
                    ),
                    "min_chunk_size": 0,  # Not available in ServiceStrategyComparison
                    "max_chunk_size": 0,  # Not available in ServiceStrategyComparison
                    "preview_chunks": [
                        {
                            "content": chunk.content if hasattr(chunk, "content") else chunk.get("content", ""),
                            "index": chunk.index if hasattr(chunk, "index") else chunk.get("index", 0),
                            "size": (
                                chunk.char_count
                                if hasattr(chunk, "char_count")
                                else len(chunk.content if hasattr(chunk, "content") else chunk.get("content", ""))
                            ),
                            "metadata": chunk.metadata if hasattr(chunk, "metadata") else chunk.get("metadata", {}),
                        }
                        for chunk in (
                            comp.sample_chunks if hasattr(comp, "sample_chunks") else comp.get("sample_chunks", [])
                        )
                    ],
                    "metrics": {
                        "processing_time": (
                            comp.processing_time_ms / 1000.0
                            if hasattr(comp, "processing_time_ms")
                            else comp.get("processing_time_ms", 0) / 1000.0
                        ),
                        "memory_usage": 0,
                        "quality_score": (
                            comp.quality_score if hasattr(comp, "quality_score") else comp.get("quality_score", 0)
                        ),
                        "chunk_variance": (
                            comp.size_variance if hasattr(comp, "size_variance") else comp.get("size_variance", 0)
                        ),
                        "error": None,
                    },
                }
                for comp in result.comparisons
            ],
            "recommendation": (
                {
                    "recommended_strategy": (
                        result.recommendation.strategy
                        if hasattr(result.recommendation, "strategy")
                        else result.recommendation.get("strategy", "")
                    ),
                    "confidence": (
                        result.recommendation.confidence
                        if hasattr(result.recommendation, "confidence")
                        else result.recommendation.get("confidence", 0)
                    ),
                    "reasoning": (
                        result.recommendation.reasoning
                        if hasattr(result.recommendation, "reasoning")
                        else result.recommendation.get("reasoning", "")
                    ),
                    "alternative_strategies": (
                        result.recommendation.alternatives
                        if hasattr(result.recommendation, "alternatives")
                        else result.recommendation.get("alternatives", [])
                    ),
                    "suggested_config": {},
                }
                if result.recommendation
                else None
            ),
            "metadata": {
                "comparison_id": result.comparison_id,
                "processing_time_ms": result.processing_time_ms,
                "strategies_compared": len(result.comparisons),
            },
        }

    async def compare_strategies_for_api(
        self,
        content: str,
        strategies: list[str] | None = None,
        base_config: dict[str, Any] | None = None,
        user_id: int | None = None,
    ) -> Any:
        """Compare strategies and return API response format."""
        return await self.orchestrator.compare_strategies(
            content=content,
            strategies=strategies,
            base_config=base_config,
            user_id=user_id,
        )

    async def execute_ingestion_chunking(
        self,
        content: str,
        strategy: str,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute chunking for ingestion - delegates to orchestrator."""
        return await self.orchestrator.execute_ingestion_chunking(
            content=content,
            strategy=strategy,
            config=config,
            metadata=metadata,
        )

    async def execute_ingestion_chunking_segmented(
        self,
        content: str,
        strategy: str,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        segment_size: int = 100000,  # noqa: ARG002
    ) -> list[dict[str, Any]]:
        """Execute chunking in segments for large documents."""
        # For now, just delegate to regular chunking
        # Segmentation can be added to processor if needed
        return await self.execute_ingestion_chunking(
            content=content,
            strategy=strategy,
            config=config,
            metadata=metadata,
        )

    async def get_available_strategies(self) -> list[dict[str, Any]]:
        """Get available strategies - converts from ServiceStrategyInfo."""
        strategies = await self.orchestrator.get_available_strategies()
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "best_for": s.best_for,
                "pros": s.pros,
                "cons": s.cons,
                "default_config": s.default_config,
                "supported_file_types": s.supported_file_types,
            }
            for s in strategies
        ]

    async def get_available_strategies_for_api(self) -> list[Any]:
        """Get available strategies in API format."""
        return await self.orchestrator.get_available_strategies()

    async def get_strategy_details(self, strategy_id: str) -> dict[str, Any] | None:
        """Get details for a specific strategy."""
        strategies = await self.get_available_strategies()
        for strategy in strategies:
            if strategy["id"] == strategy_id:
                return strategy
        return None

    async def recommend_strategy(
        self,
        file_type: str | None = None,
        content_length: int | None = None,
        document_type: str | None = None,
        sample_content: str | None = None,
    ) -> dict[str, Any]:
        """Get strategy recommendation."""
        result = await self.orchestrator.recommend_strategy(
            file_type=file_type,
            content_length=content_length,
            document_type=document_type,
            sample_content=sample_content,
        )

        return {
            "recommended_strategy": result.strategy,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "alternative_strategies": result.alternatives,
            "suggested_config": {},
        }

    async def get_collection_chunk_stats(
        self,
        collection_id: str,
        user_id: int | None = None,
    ) -> Any:
        """Get chunking statistics for a collection."""
        if user_id is not None:
            return await self.orchestrator.get_collection_statistics(
                collection_id=collection_id,
                user_id=user_id,
            )

        # FastAPI dependencies already validated permissions, so reuse legacy
        legacy = self._ensure_legacy_service()
        return await legacy.get_collection_chunk_stats(collection_id)

    async def get_chunking_statistics(
        self,
        collection_id: str,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        """Get chunking statistics as dict."""
        if user_id is None:
            user_id = 0

        stats = await self.orchestrator.get_collection_statistics(
            collection_id=collection_id,
            user_id=user_id,
        )

        return {
            "total_documents": stats.total_documents,
            "total_chunks": stats.total_chunks,
            "average_chunk_size": stats.avg_chunk_size,
            "strategy_breakdown": {},  # Not available in current ServiceChunkingStats
            "last_updated": stats.last_updated,
        }

    async def get_cached_preview(
        self,
        content: str,
        strategy: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Get cached preview if available."""
        content_hash = self.orchestrator.cache.generate_content_hash(content)
        return await self.orchestrator.cache.get_cached_preview(
            content_hash=content_hash,
            strategy=strategy,
            config=config,
        )

    async def cache_preview_with_user(
        self,
        preview_data: dict[str, Any],
        user_id: int,
        ttl: int = 1800,
    ) -> str:
        """Cache preview data with user context."""
        return await self.orchestrator.cache.cache_with_id(
            data={**preview_data, "user_id": user_id},
            ttl=ttl,
        )

    async def get_cached_preview_by_id(self, cache_id: str) -> dict[str, Any] | None:
        """Get cached preview by ID."""
        return await self.orchestrator.cache.get_cached_by_id(cache_id)

    async def clear_preview_cache(self, pattern: str | None = None) -> int:
        """Clear preview cache."""
        return await self.orchestrator.cache.clear_cache(pattern)

    async def validate_config_for_collection(
        self,
        collection_id: str,
        strategy: str,
        config: dict[str, Any],
        user_id: int,
    ) -> dict[str, Any]:
        """Validate configuration for a collection."""
        # Validate access
        await self.orchestrator.validator.validate_collection_access(collection_id, user_id)

        # Validate configuration
        self.orchestrator.validator.validate_strategy(strategy)
        self.orchestrator.validator.validate_config(strategy, config)

        return {"valid": True, "config": config}

    async def verify_collection_access(
        self,
        collection_id: str,
        user_id: int,
    ) -> bool:
        """Verify user has access to collection."""
        try:
            await self.orchestrator.validator.validate_collection_access(collection_id, user_id)
            return True
        except Exception:
            return False

    def _get_default_config(self, strategy: str) -> dict[str, Any]:
        """Get default configuration for a strategy."""
        return self.orchestrator.config_manager.get_default_config(strategy)

    def _get_alternative_strategies(self, primary_strategy: str) -> list[dict[str, str]]:
        """Get alternative strategies."""
        return self.orchestrator.config_manager.get_alternative_strategies(primary_strategy)

    def _calculate_metrics(
        self,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Calculate metrics for chunks."""
        return self.orchestrator.processor.calculate_statistics(chunks)

    def _ensure_legacy_service(self) -> "ChunkingService":
        """Instantiate the legacy ChunkingService lazily for fallback paths."""

        if self._legacy_service is not None:
            return self._legacy_service

        if self.db_session is None:
            raise RuntimeError("ChunkingServiceAdapter requires db_session to build legacy service")

        # Lazily import to avoid circular import at module load time
        from packages.webui.services.chunking_service import ChunkingService

        collection_repo = self.collection_repo or CollectionRepository(self.db_session)
        document_repo = self.document_repo or DocumentRepository(self.db_session)

        self._legacy_service = ChunkingService(
            db_session=self.db_session,
            collection_repo=collection_repo,
            document_repo=document_repo,
            redis_client=None,
        )
        return self._legacy_service

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the legacy ChunkingService."""

        legacy = self._ensure_legacy_service()
        return getattr(legacy, name)
