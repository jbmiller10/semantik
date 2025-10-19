"""
Chunking orchestrator service.

Main coordinator that orchestrates chunking operations across all specialized services.
"""

import contextlib
import logging
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.chunking.infrastructure.exceptions import (
    DocumentTooLargeError,
    ValidationError,
)
from packages.shared.chunking.infrastructure.exceptions import (
    PermissionDeniedError as InfraPermissionDeniedError,
)
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.webui.services.chunking_constants import MAX_PREVIEW_CONTENT_SIZE
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
    ) -> None:
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
        *,
        max_chunks: int | None = None,
        correlation_id: str | None = None,
    ) -> ServicePreviewResponse:
        """Preview chunking results for content or document."""
        correlation = correlation_id or str(uuid.uuid4())
        normalized_strategy = strategy or "recursive"
        incoming_config = config.copy() if config else None
        normalized_config = self._normalize_config(incoming_config)

        if content is not None and len(content) > MAX_PREVIEW_CONTENT_SIZE:
            raise DocumentTooLargeError(
                size=len(content),
                max_size=MAX_PREVIEW_CONTENT_SIZE,
                correlation_id=correlation,
            )

        if normalized_config and "chunk_size" in normalized_config:
            chunk_size_value = normalized_config["chunk_size"]
            try:
                chunk_size_int = int(chunk_size_value)
            except (TypeError, ValueError):
                chunk_size_int = None
            if chunk_size_int is None or chunk_size_int < 1 or chunk_size_int > 10_000:
                raise ValidationError(
                    field="chunk_size",
                    value=chunk_size_value,
                    reason="Must be between 1 and 10000",
                )

        try:
            await self.validator.validate_preview_request(
                content,
                document_id,
                normalized_strategy,
                normalized_config,
            )
        except ValidationError as exc:
            reason = (exc.reason or "").lower()
            if exc.field == "content" and "exceeds" in reason and content is not None:
                raise DocumentTooLargeError(
                    size=len(content),
                    max_size=MAX_PREVIEW_CONTENT_SIZE,
                    correlation_id=correlation,
                ) from exc
            raise

        if document_id:
            if user_id is None:
                raise InfraPermissionDeniedError(
                    user_id="anonymous",
                    resource=f"document:{document_id}",
                    action="read",
                    correlation_id=correlation,
                )
            await self.validator.validate_document_access(document_id, user_id)
            content = await self._load_document_content(document_id)

        if content is None:
            raise ValidationError(field="content", value=None, reason="Content is required for preview")

        if len(content) > MAX_PREVIEW_CONTENT_SIZE:
            raise DocumentTooLargeError(
                size=len(content),
                max_size=MAX_PREVIEW_CONTENT_SIZE,
                correlation_id=correlation,
            )

        merged_config = self.config_manager.merge_configs(normalized_strategy, normalized_config)
        config_for_response = merged_config.copy()
        config_for_response.setdefault("strategy", normalized_strategy)

        content_hash = self.cache.generate_content_hash(content)

        if use_cache:
            cached = await self.cache.get_cached_preview(content_hash, normalized_strategy, merged_config)
            if cached:
                self.metrics.record_cache_hit("preview")
                return self._build_preview_response_from_cache(
                    cached,
                    correlation_id=correlation,
                    max_chunks=max_chunks,
                )
            self.metrics.record_cache_miss("preview")

        async with self.metrics.measure_operation(normalized_strategy) as context:
            start_time = time.perf_counter()
            chunk_dicts = await self.processor.process_document(
                content,
                normalized_strategy,
                merged_config,
                use_fallback=True,
            )
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            context["chunks_produced"] = len(chunk_dicts)

        if any(chunk.get("strategy") == "fallback" for chunk in chunk_dicts):
            self.metrics.record_fallback(normalized_strategy)

        self.metrics.record_chunks_produced(normalized_strategy, chunk_dicts)

        preview_chunks = self._transform_chunks_to_preview(chunk_dicts)
        total_chunks = len(preview_chunks)
        display_chunks = preview_chunks if max_chunks is None else preview_chunks[:max_chunks]

        metrics = self._calculate_preview_metrics(
            preview_chunks,
            len(content),
            processing_time_ms / 1000 if processing_time_ms > 0 else 0,
        )

        expires_at = datetime.now(UTC) + timedelta(minutes=30)
        preview_id = str(uuid.uuid4())

        response = self._build_preview_response(
            preview_id=preview_id,
            strategy=normalized_strategy,
            config=config_for_response,
            chunks=display_chunks,
            total_chunks=total_chunks,
            metrics=metrics,
            processing_time_ms=processing_time_ms,
            cached=False,
            expires_at=expires_at,
            correlation_id=correlation,
        )

        if use_cache:
            cache_payload = self._serialize_preview_for_cache(
                preview_id=preview_id,
                strategy=normalized_strategy,
                config=config_for_response,
                chunks=preview_chunks,
                metrics=metrics,
                processing_time_ms=processing_time_ms,
                expires_at=expires_at,
                correlation_id=correlation,
            )
            await self.cache.cache_preview(content_hash, normalized_strategy, merged_config, cache_payload)

        return response

    async def compare_strategies(
        self,
        content: str,
        strategies: list[str] | None = None,
        base_config: dict[str, Any] | None = None,
        strategy_configs: dict[str, dict[str, Any]] | None = None,
        user_id: int | None = None,  # noqa: ARG002
        *,
        max_chunks_per_strategy: int | None = 5,
    ) -> ServiceCompareResponse:
        """
        Compare multiple chunking strategies.

        Args:
            content: Content to chunk
            strategies: List of strategies to compare (default: all)
            base_config: Base configuration for all strategies
            user_id: User ID for tracking
            max_chunks_per_strategy: Maximum number of preview chunks per strategy

        Returns:
            Comparison response with results for each strategy
        """
        self.validator.validate_content(content)

        selected_strategies = strategies or [s["id"] for s in self.config_manager.get_all_strategies()]
        comparisons: list[ServiceStrategyComparison] = []
        comparison_start = time.perf_counter()
        chunk_limit = max_chunks_per_strategy or 5
        normalized_base_config = self._normalize_config(base_config)
        normalized_strategy_configs = (
            {name: self._normalize_config(cfg) for name, cfg in strategy_configs.items()} if strategy_configs else None
        )

        for strategy in selected_strategies:
            try:
                self.validator.validate_strategy(strategy)
                config_override = None
                if normalized_strategy_configs and strategy in normalized_strategy_configs:
                    config_override = normalized_strategy_configs[strategy]
                merged_config = self.config_manager.merge_configs(strategy, config_override or normalized_base_config)

                preview = await self.preview_chunks(
                    content=content,
                    strategy=strategy,
                    config=merged_config,
                    use_cache=False,
                    max_chunks=chunk_limit,
                )

                preview_chunks = self._transform_chunks_to_preview(preview.chunks)
                metrics = (
                    dict(preview.metrics)
                    if preview.metrics
                    else self._calculate_preview_metrics(
                        preview_chunks,
                        len(content),
                        preview.processing_time_ms / 1000 if preview.processing_time_ms else 0,
                    )
                )

                strategy_info = self.config_manager.get_strategy_info(strategy)
                sample_chunks = preview_chunks[:chunk_limit]
                sample_chunks_union: list[ServiceChunkPreview | dict[str, Any]] = list(sample_chunks)

                comparison = ServiceStrategyComparison(
                    strategy=strategy,
                    config=preview.config,
                    sample_chunks=sample_chunks_union,
                    total_chunks=preview.total_chunks,
                    avg_chunk_size=metrics.get("avg_chunk_size", metrics.get("average_chunk_size", 0)),
                    size_variance=metrics.get("size_variance", 0),
                    quality_score=metrics.get("quality_score", 0),
                    processing_time_ms=preview.processing_time_ms,
                    pros=list(strategy_info.get("pros", [])),
                    cons=list(strategy_info.get("cons", [])),
                )
                comparisons.append(comparison)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Error comparing strategy %s: %s", strategy, exc)
                fallback_config = self.config_manager.get_default_config(strategy)
                comparisons.append(
                    ServiceStrategyComparison(
                        strategy=strategy,
                        config=fallback_config,
                        sample_chunks=[],
                        total_chunks=0,
                        avg_chunk_size=0,
                        size_variance=0,
                        quality_score=0,
                        processing_time_ms=0,
                        pros=[],
                        cons=[str(exc)],
                    )
                )

        recommendation = self._get_recommendation(comparisons, content)
        elapsed_ms = int((time.perf_counter() - comparison_start) * 1000)

        comparisons_union: list[ServiceStrategyComparison | dict[str, Any]] = list(comparisons)

        return ServiceCompareResponse(
            comparison_id=str(uuid.uuid4()),
            comparisons=comparisons_union,
            recommendation=recommendation,
            processing_time_ms=elapsed_ms,
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
        fallback_used = False
        fallback_reason: str | None = None

        async with self.metrics.measure_operation(strategy) as context:
            try:
                chunks = await self.processor.process_document(content, strategy, merged_config, use_fallback=False)
            except Exception as e:
                fallback_used = True
                fallback_reason = getattr(e, "reason", None) or type(e).__name__
                logger.warning("Strategy %s failed, using fallback: %s", strategy, str(e))
                self.metrics.record_fallback(strategy)
                chunks = await self.processor.process_document(content, strategy, merged_config, use_fallback=True)

            context["chunks_produced"] = len(chunks)
            if fallback_used:
                context["fallback"] = True
                context["fallback_reason"] = fallback_reason
            self.metrics.record_chunks_produced(strategy, chunks)

        if fallback_used:
            for chunk in chunks:
                chunk_metadata = chunk.setdefault("metadata", {})
                chunk_metadata.setdefault("fallback", True)
                chunk_metadata.setdefault("fallback_reason", fallback_reason)
                chunk_metadata.setdefault("original_strategy", strategy)

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

        # Normalize mutable fields for downstream updates
        reasoning_list = rec_data.get("reasoning")
        if not isinstance(reasoning_list, list):
            reasoning_list = [str(reasoning_list)] if reasoning_list else []
        rec_data["reasoning"] = reasoning_list

        alternatives_list = rec_data.get("alternatives")
        if not isinstance(alternatives_list, list):
            alternatives_list = [alternatives_list] if alternatives_list else []
        rec_data["alternatives"] = alternatives_list

        # Enhance recommendation with sample analysis if provided
        if sample_content and ("```" in sample_content or "def " in sample_content or "class " in sample_content):
            if rec_data["strategy"] != "recursive":
                reasoning_list.append("Code patterns detected in sample")
                alternatives_list.append(rec_data["strategy"])
                rec_data["strategy"] = "recursive"

        elif sample_content and (sample_content.startswith("#") or "## " in sample_content):
            if rec_data["strategy"] != "markdown":
                reasoning_list.append("Markdown headers detected in sample")
                alternatives_list.append(rec_data["strategy"])
                rec_data["strategy"] = "markdown"

        strategy_value = rec_data.get("strategy", "recursive")
        if not isinstance(strategy_value, str):
            strategy_value = str(strategy_value)

        suggested_config_raw = rec_data.get("suggested_config")
        if isinstance(suggested_config_raw, dict):
            suggested_config = suggested_config_raw.copy()
        else:
            suggested_config = {}

        # Ensure suggested config aligns with final strategy
        suggested_config["strategy"] = strategy_value

        def _safe_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        chunk_size = _safe_int(suggested_config.get("chunk_size"), 512)
        chunk_overlap = _safe_int(suggested_config.get("chunk_overlap"), 50)
        preserve_sentences_raw = suggested_config.get("preserve_sentences", True)
        preserve_sentences = bool(preserve_sentences_raw) if preserve_sentences_raw is not None else True

        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 2)

        metadata = {
            key: value
            for key, value in suggested_config.items()
            if key not in {"strategy", "chunk_size", "chunk_overlap", "preserve_sentences"}
        }
        metadata = metadata or None

        suggested_config["chunk_size"] = chunk_size
        suggested_config["chunk_overlap"] = chunk_overlap
        suggested_config["preserve_sentences"] = preserve_sentences
        rec_data["suggested_config"] = suggested_config

        try:
            confidence = float(rec_data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        # Build recommendation object
        return ServiceStrategyRecommendation(
            strategy=strategy_value,
            confidence=confidence,
            reasoning=("\n".join(reasoning_list) if reasoning_list else ""),
            alternatives=[
                alt.get("strategy", alt) if isinstance(alt, dict) else alt for alt in rec_data.get("alternatives", [])
            ],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_sentences=preserve_sentences,
            metadata=metadata,
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
            documents_tuple = await self.document_repo.list_by_collection(collection_id)
            documents = documents_tuple[0]  # Extract the list from the tuple
            total_documents = len(documents)
            total_chunks = sum(doc.chunk_count or 0 for doc in documents)

            # Calculate strategy breakdown
            strategy_breakdown: dict[str, int] = {}
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
            avg_chunk_size=metrics_data.get("average_chunk_size", 0),
            strategy_used="mixed",  # Collection has mixed strategies
            last_updated=metrics_data.get("last_operation", {}).get("timestamp"),
        )

    async def _load_document_content(self, document_id: str) -> str:
        """Load document content from repository."""
        if not self.document_repo:
            raise ValidationError(field="document_repo", value=None, reason="Document repository not available")

        document = await self.document_repo.get_by_id(document_id)
        if not document:
            raise ValidationError(field="document_id", value=document_id, reason="Document not found")

        return document.content or ""

    def _build_preview_response(
        self,
        *,
        preview_id: str,
        strategy: str,
        config: dict[str, Any],
        chunks: list[ServiceChunkPreview],
        total_chunks: int,
        metrics: dict[str, Any],
        processing_time_ms: int,
        cached: bool,
        expires_at: datetime | None,
        correlation_id: str | None,
    ) -> ServicePreviewResponse:
        preview_chunks_typed: list[ServiceChunkPreview | dict[str, Any]] = list(chunks)

        return ServicePreviewResponse(
            preview_id=preview_id,
            strategy=strategy,
            config=config,
            chunks=preview_chunks_typed,
            total_chunks=total_chunks,
            metrics=metrics,
            processing_time_ms=processing_time_ms,
            cached=cached,
            expires_at=expires_at,
            correlation_id=correlation_id,
        )

    def _build_preview_response_from_cache(
        self,
        cached: dict[str, Any],
        *,
        correlation_id: str | None,
        max_chunks: int | None = None,
    ) -> ServicePreviewResponse:
        preview_id = cached.get("preview_id") or cached.get("cache_id") or cached.get("cache_key") or str(uuid.uuid4())
        raw_expires = cached.get("expires_at")
        expires_at: datetime | None = None
        if isinstance(raw_expires, str):
            with contextlib.suppress(ValueError, TypeError):
                expires_at = datetime.fromisoformat(raw_expires)
        elif isinstance(raw_expires, datetime):
            expires_at = raw_expires

        preview_chunks_full = self._transform_chunks_to_preview(cached.get("chunks", []))
        display_chunks = (
            preview_chunks_full if max_chunks is None or max_chunks <= 0 else preview_chunks_full[:max_chunks]
        )
        metrics = cached.get("performance_metrics") or cached.get("statistics") or cached.get("metrics") or {}
        strategy = cached.get("strategy", "unknown")
        config = cached.get("config", {})
        total_chunks = cached.get("total_chunks", len(preview_chunks_full))
        processing_time_ms = cached.get("processing_time_ms", 0)

        return self._build_preview_response(
            preview_id=preview_id,
            strategy=strategy,
            config=config,
            chunks=display_chunks,
            total_chunks=total_chunks,
            metrics=metrics,
            processing_time_ms=processing_time_ms,
            cached=True,
            expires_at=expires_at,
            correlation_id=correlation_id or cached.get("correlation_id"),
        )

    def _serialize_preview_for_cache(
        self,
        *,
        preview_id: str,
        strategy: str,
        config: dict[str, Any],
        chunks: list[ServiceChunkPreview],
        metrics: dict[str, Any],
        processing_time_ms: int,
        expires_at: datetime,
        correlation_id: str | None,
    ) -> dict[str, Any]:
        return {
            "preview_id": preview_id,
            "strategy": strategy,
            "config": config,
            "chunks": [self._serialize_chunk_preview(chunk) for chunk in chunks],
            "total_chunks": len(chunks),
            "performance_metrics": metrics,
            "processing_time_ms": processing_time_ms,
            "expires_at": expires_at.isoformat(),
            "correlation_id": correlation_id,
        }

    def _serialize_chunk_preview(self, chunk: ServiceChunkPreview) -> dict[str, Any]:
        return {
            "index": chunk.index,
            "content": chunk.content,
            "text": chunk.text or chunk.content,
            "token_count": chunk.token_count,
            "char_count": chunk.char_count,
            "metadata": chunk.metadata,
            "quality_score": chunk.quality_score,
            "overlap_info": chunk.overlap_info,
        }

    def _normalize_config(self, config: dict[str, Any] | None) -> dict[str, Any] | None:
        """Flatten legacy config payloads (e.g., with nested params)."""

        if not config:
            return None

        normalized = config.copy()
        params = normalized.pop("params", None)
        if isinstance(params, dict):
            normalized.update(params)
        return normalized

    def _transform_chunks_to_preview(
        self,
        chunks: Sequence[ServiceChunkPreview | dict[str, Any]],
    ) -> list[ServiceChunkPreview]:
        preview_chunks: list[ServiceChunkPreview] = []

        for entry in chunks:
            if isinstance(entry, ServiceChunkPreview):
                preview_chunks.append(entry)
                continue

            content = entry.get("content") or entry.get("text") or ""
            char_count = entry.get("char_count") or len(content)
            token_count = entry.get("token_count") or (char_count // 4 if char_count else 0)

            preview_chunks.append(
                ServiceChunkPreview(
                    index=entry.get("index", len(preview_chunks)),
                    content=content,
                    text=None,
                    token_count=token_count,
                    char_count=char_count,
                    metadata=entry.get("metadata", {}),
                    quality_score=entry.get("quality_score", 0.8),
                    overlap_info=entry.get("overlap_info"),
                )
            )

        return preview_chunks

    def _calculate_preview_metrics(
        self,
        chunks: list[ServiceChunkPreview],
        text_length: int,
        processing_time: float,
    ) -> dict[str, Any]:
        if not chunks:
            return {
                "total_chunks": 0,
                "average_chunk_size": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "chunks_per_second": 0,
                "compression_ratio": 0,
                "size_variance": 0.0,
                "quality_score": 0.0,
            }

        sizes = [chunk.char_count or len(chunk.content or "") for chunk in chunks]
        total_size = sum(sizes)
        average = total_size / len(sizes) if sizes else 0
        variance = sum((s - average) ** 2 for s in sizes) / len(sizes) if len(sizes) > 1 else 0.0
        quality_score = 1.0 - min(1.0, variance / (average**2)) if average > 0 else 0.0

        chunks_per_second = (len(chunks) / processing_time) if processing_time > 0 else 0
        compression_ratio = (total_size / text_length) if text_length > 0 else 1

        return {
            "total_chunks": len(chunks),
            "average_chunk_size": average,
            "avg_chunk_size": average,
            "min_chunk_size": min(sizes) if sizes else 0,
            "max_chunk_size": max(sizes) if sizes else 0,
            "chunks_per_second": chunks_per_second,
            "compression_ratio": compression_ratio,
            "size_variance": variance,
            "quality_score": quality_score,
        }

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
        best_score = 0.0

        for comp in comparisons:
            if comp.quality_score > best_score:
                best_score = float(comp.quality_score)
                best_strategy = comp.strategy

        if not best_strategy:
            best_strategy = "recursive"  # Fallback

        return ServiceStrategyRecommendation(
            strategy=best_strategy,
            confidence=min(0.9, best_score),
            reasoning=f"Best quality score: {best_score:.2f}. Analyzed {len(comparisons)} strategies.",
            alternatives=[c.strategy for c in comparisons if c.strategy != best_strategy][:2],
        )
