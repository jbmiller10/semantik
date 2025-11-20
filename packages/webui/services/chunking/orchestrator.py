"""
Chunking orchestrator service.

Main coordinator that orchestrates chunking operations across all specialized services.
"""

import contextlib
import json
import logging
import time
import uuid
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from statistics import fmean
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.chunking.infrastructure.exceptions import DocumentTooLargeError, ValidationError
from packages.shared.chunking.infrastructure.exceptions import PermissionDeniedError as InfraPermissionDeniedError
from packages.shared.database.exceptions import AccessDeniedError
from packages.shared.database.models import OperationStatus
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.webui.services.chunking_constants import MAX_PREVIEW_CONTENT_SIZE
from packages.webui.services.dtos import (
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
            await self.cache.cache_preview(
                content_hash,
                normalized_strategy,
                merged_config,
                cache_payload,
                preview_id=preview_id,
            )

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

        # Extract document identifier early for deterministic chunk ids
        document_id = None
        if metadata:
            document_id = metadata.get("document_id") or metadata.get("doc_id") or metadata.get("id")

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

        # Ensure each chunk has a deterministic chunk_id expected by downstream tasks
        chunk_id_map: dict[str, str] = {}
        for idx, chunk in enumerate(chunks):
            chunk_index_candidate = chunk.get("chunk_index", chunk.get("index", idx))
            try:
                chunk_index = int(chunk_index_candidate)
            except (TypeError, ValueError):
                chunk_index = idx

            existing_chunk_id = chunk.get("chunk_id") or (chunk.get("metadata") or {}).get("chunk_id")
            if document_id:
                new_chunk_id = f"{document_id}_{chunk_index:04d}"
            else:
                new_chunk_id = existing_chunk_id or f"chunk_{uuid.uuid4().hex}"

            if existing_chunk_id and existing_chunk_id != new_chunk_id:
                chunk_id_map[existing_chunk_id] = new_chunk_id

            chunk["chunk_id"] = new_chunk_id
            chunk["chunk_index"] = chunk_index

            chunk_metadata = chunk.setdefault("metadata", {})
            chunk_metadata["chunk_id"] = new_chunk_id
            chunk_metadata.setdefault("chunk_index", chunk_index)
            if document_id:
                chunk_metadata.setdefault("document_id", document_id)

        if chunk_id_map:
            for chunk in chunks:
                chunk_metadata = chunk.get("metadata") or {}
                parent_id = chunk_metadata.get("parent_chunk_id")
                if isinstance(parent_id, str) and parent_id in chunk_id_map:
                    chunk_metadata["parent_chunk_id"] = chunk_id_map[parent_id]

                child_ids = chunk_metadata.get("child_chunk_ids")
                if isinstance(child_ids, list):
                    chunk_metadata["child_chunk_ids"] = [chunk_id_map.get(child_id, child_id) for child_id in child_ids]

                custom_attrs = chunk_metadata.get("custom_attributes")
                if isinstance(custom_attrs, dict):
                    parent_attr = custom_attrs.get("parent_chunk_id")
                    if isinstance(parent_attr, str) and parent_attr in chunk_id_map:
                        custom_attrs["parent_chunk_id"] = chunk_id_map[parent_attr]
                    child_attr_ids = custom_attrs.get("child_chunk_ids")
                    if isinstance(child_attr_ids, list):
                        custom_attrs["child_chunk_ids"] = [
                            chunk_id_map.get(child_id, child_id) for child_id in child_attr_ids
                        ]

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
        if (
            sample_content
            and ("```" in sample_content or "def " in sample_content or "class " in sample_content)
            and rec_data["strategy"] != "recursive"
        ):
            reasoning_list.append("Code patterns detected in sample")
            alternatives_list.append(rec_data["strategy"])
            rec_data["strategy"] = "recursive"

        elif (
            sample_content
            and (sample_content.startswith("#") or "## " in sample_content)
            and rec_data["strategy"] != "markdown"
        ):
            reasoning_list.append("Markdown headers detected in sample")
            alternatives_list.append(rec_data["strategy"])
            rec_data["strategy"] = "markdown"

        strategy_value = rec_data.get("strategy", "recursive")
        if not isinstance(strategy_value, str):
            strategy_value = str(strategy_value)

        suggested_config_raw = rec_data.get("suggested_config")
        suggested_config = suggested_config_raw.copy() if isinstance(suggested_config_raw, dict) else {}

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

        metadata_dict = {
            key: value
            for key, value in suggested_config.items()
            if key not in {"strategy", "chunk_size", "chunk_overlap", "preserve_sentences"}
        }
        metadata: dict[Any, Any] | None = metadata_dict or None

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
        await self.validator.validate_collection_access(collection_id, user_id)

        if not self.db_session:
            raise ValidationError(field="db_session", value=None, reason="Database session unavailable")

        from packages.shared.database.models import (
            Chunk,
            Document,
            Operation,
            OperationStatus,
            OperationType,
        )

        collection = await (self.collection_repo.get_by_uuid(collection_id) if self.collection_repo else None)
        if not collection:
            from packages.shared.chunking.infrastructure.exceptions import ResourceNotFoundError

            raise ResourceNotFoundError("Collection", str(collection_id))

        chunk_filters = [Chunk.collection_id == collection.id]
        chunk_stats_query = select(
            func.count(Chunk.id).label("total_chunks"),
            func.avg(func.length(Chunk.content)).label("avg_chunk_size"),
            func.min(func.length(Chunk.content)).label("min_chunk_size"),
            func.max(func.length(Chunk.content)).label("max_chunk_size"),
            func.var_pop(func.length(Chunk.content)).label("size_variance"),
        ).where(*chunk_filters)
        stats_row = (await self.db_session.execute(chunk_stats_query)).one()

        doc_count_query = select(func.count(Document.id)).where(Document.collection_id == collection.id)
        total_documents = int((await self.db_session.execute(doc_count_query)).scalar() or 0)

        latest_op_query = (
            select(Operation)
            .where(
                and_(
                    Operation.collection_id == collection.id,
                    Operation.type == OperationType.INDEX,
                    Operation.status == OperationStatus.COMPLETED,
                )
            )
            .order_by(Operation.completed_at.desc())
            .limit(1)
        )
        latest_operation = (await self.db_session.execute(latest_op_query)).scalar_one_or_none()

        processing_time = 0.0
        strategy_used = collection.chunking_strategy or "fixed_size"
        last_updated = collection.updated_at

        if latest_operation:
            if latest_operation.started_at and latest_operation.completed_at:
                processing_time = (latest_operation.completed_at - latest_operation.started_at).total_seconds()
            if latest_operation.config and "strategy" in latest_operation.config:
                strategy_used = latest_operation.config.get("strategy", strategy_used)
            last_updated = latest_operation.completed_at or collection.updated_at

        return ServiceChunkingStats(
            total_chunks=stats_row.total_chunks or 0,
            total_documents=total_documents,
            avg_chunk_size=float(stats_row.avg_chunk_size) if stats_row.avg_chunk_size else 0.0,
            min_chunk_size=stats_row.min_chunk_size or 0,
            max_chunk_size=stats_row.max_chunk_size or 0,
            size_variance=float(stats_row.size_variance) if stats_row.size_variance else 0.0,
            strategy_used=strategy_used,
            last_updated=last_updated,
            processing_time_seconds=processing_time,
            quality_metrics={},
        )

    async def get_collection_chunks(
        self,
        collection_id: str,
        *,
        page: int = 1,
        page_size: int = 20,
        document_id: str | None = None,
    ) -> ServiceChunkList:
        """Return paginated chunk rows for a collection."""

        if not self.db_session:
            raise ValidationError(field="db_session", value=None, reason="Database session unavailable")

        from packages.shared.chunking.infrastructure.exceptions import ResourceNotFoundError
        from packages.shared.database.models import Chunk

        collection = await (self.collection_repo.get_by_uuid(collection_id) if self.collection_repo else None)
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
            metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
            records.append(
                ServiceChunkRecord(
                    id=int(chunk.id),
                    collection_id=str(chunk.collection_id),
                    document_id=str(chunk.document_id) if chunk.document_id is not None else None,
                    chunk_index=int(chunk.chunk_index),
                    content=chunk.content or "",
                    token_count=getattr(chunk, "token_count", None),
                    metadata=metadata,
                    created_at=getattr(chunk, "created_at", None),
                    updated_at=getattr(chunk, "updated_at", None),
                )
            )

        return ServiceChunkList(chunks=records, total=total_chunks, page=safe_page, page_size=safe_page_size)

    async def get_global_metrics(
        self,
        *,
        period_days: int = 30,
        user_id: int | None = None,  # noqa: ARG002
    ) -> ServiceGlobalMetrics:
        """Compute global chunking metrics for the requested period."""

        if not self.db_session:
            raise ValidationError(field="db_session", value=None, reason="Database session unavailable")

        from packages.shared.database.models import Chunk, Collection, Document, Operation, OperationStatus

        period_end = datetime.now(UTC)
        safe_days = max(period_days, 1)
        period_start = period_end - timedelta(days=safe_days)

        chunk_count_result = await self.db_session.execute(
            select(func.count(Chunk.id)).where(Chunk.created_at >= period_start)
        )
        total_chunks_created = int(chunk_count_result.scalar() or 0)

        documents_processed_result = await self.db_session.execute(
            select(func.count(Document.id)).where(
                Document.chunking_completed_at.isnot(None),
                Document.chunking_completed_at >= period_start,
            )
        )
        total_documents_processed = int(documents_processed_result.scalar() or 0)

        avg_chunks_per_document = total_chunks_created / total_documents_processed if total_documents_processed else 0.0

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

        strategy_rows = await self.db_session.execute(
            select(Collection.chunking_strategy).where(
                Collection.chunking_strategy.isnot(None), Collection.updated_at >= period_start
            )
        )
        strategies = [row[0] for row in strategy_rows if row[0]]
        if not strategies and processed_collection_ids:
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

    async def get_metrics_by_strategy(
        self,
        period_days: int = 30,
        user_id: int | None = None,  # noqa: ARG002
    ) -> list[ServiceStrategyMetrics]:
        """Return metrics grouped by strategy (placeholder with sensible defaults)."""

        try:
            return await self.metrics.get_metrics_by_strategy(period_days=period_days)
        except Exception:  # pragma: no cover - defensive
            return ServiceStrategyMetrics.create_default_metrics()

    async def get_quality_scores(
        self,
        collection_id: str | None = None,
        user_id: int | None = None,
    ) -> ServiceQualityAnalysis:
        """Provide a lightweight quality analysis."""

        # Simple heuristic based on available metrics
        base_quality = 0.8
        issues: list[str] = []
        if collection_id:
            if user_id is None:
                raise ValidationError(field="user_id", value=user_id, reason="User required for collection metrics")
            try:
                stats = await self.get_collection_statistics(collection_id, user_id=user_id)
                if stats.avg_chunk_size < 200:
                    base_quality -= 0.1
                    issues.append("Average chunk size is very small")
            except Exception:
                base_quality = 0.5
                issues.append("Unable to compute collection-specific metrics")

        return ServiceQualityAnalysis(
            overall_quality="good" if base_quality >= 0.7 else "fair",
            quality_score=max(min(base_quality, 1.0), 0.0),
            coherence_score=base_quality,
            completeness_score=base_quality,
            size_consistency=base_quality,
            recommendations=["Adjust chunk_size if chunks feel too small"],
            issues_detected=issues,
        )

    async def analyze_document(
        self,
        *,
        content: str | None = None,
        document_id: str | None = None,  # noqa: ARG002 - document_id preserved for API parity
        file_type: str | None = None,
        user_id: int | None = None,  # noqa: ARG002
        deep_analysis: bool | None = None,  # noqa: ARG002
    ) -> ServiceDocumentAnalysis:
        """Provide a lightweight document analysis for strategy suggestion."""

        text = content or ""
        length = len(text)
        doc_type = file_type or ("markdown" if text.strip().startswith("#") else "plain_text")
        chunk_estimate = {
            "recursive": max(1, length // 800),
            "fixed_size": max(1, length // 1000),
        }
        recommendation = await self.recommend_strategy(
            file_type=file_type,
            content_length=length,
            document_type=doc_type,
            sample_content=text[:2000],
        )

        return ServiceDocumentAnalysis(
            document_type=doc_type,
            content_structure={"length": length},
            recommended_strategy=recommendation,
            estimated_chunks=chunk_estimate,
            complexity_score=0.5 if length < 10_000 else 0.8,
            special_considerations=[],
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
        """Persist a user-defined configuration using the DB-backed store."""

        self.validator.validate_strategy(strategy)
        if config:
            self.validator.validate_config(strategy, config)

        merged = self.config_manager.merge_configs(strategy, config)
        merged["strategy"] = strategy

        # In dev / DISABLE_AUTH mode the stub user_id is 0; avoid FK violations by returning
        # an ephemeral DTO instead of persisting.
        if user_id <= 0 or self.config_manager.profile_repo is None:
            now = datetime.now(UTC)
            return ServiceSavedConfiguration(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                strategy=strategy,
                config=merged,
                created_by=user_id,
                created_at=now,
                updated_at=now,
                usage_count=0,
                is_default=is_default,
                tags=tags,
            )

        return await self.config_manager.save_user_config(
            user_id=user_id,
            name=name,
            strategy=strategy,
            config=merged,
            description=description,
            is_default=is_default,
            tags=tags,
        )

    async def list_configurations(
        self,
        *,
        user_id: int,
        strategy: str | None = None,
        is_default: bool | None = None,
    ) -> list[ServiceSavedConfiguration]:
        """List persisted configurations for the user."""

        if user_id <= 0 or self.config_manager.profile_repo is None:
            return []

        return await self.config_manager.list_user_configs(
            user_id=user_id,
            strategy=strategy,
            is_default=is_default,
        )

    async def get_cached_preview_by_id(self, cache_id: str) -> ServicePreviewResponse | None:
        """Retrieve preview payload by cache id."""

        cached = await self.cache.get_cached_by_id(cache_id)
        if not cached:
            return None
        return self._build_preview_response_from_cache(cached, correlation_id=cached.get("correlation_id"))

    async def clear_preview_cache(self, preview_id: str | None = None) -> int:
        """Clear preview cache entries for a specific preview ID."""

        if preview_id:
            deleted = await self.cache.clear_preview_by_id(preview_id)
            if deleted:
                return deleted

        return await self.cache.clear_cache(preview_id)

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

    async def get_chunking_progress(self, operation_id: str, *, user_id: int | None = None) -> dict[str, Any] | None:
        """Return best-effort progress for a chunking operation."""

        if self.db_session is None:
            raise RuntimeError("Progress tracking requires a database session")

        operation_repo = OperationRepository(self.db_session)

        operation = await operation_repo.get_by_uuid(operation_id)
        if not operation:
            return None

        if user_id is not None:
            # Basic ownership/visibility checks to align with REST permissions
            await self.db_session.refresh(operation, ["collection"])
            owner_id = getattr(operation, "user_id", None)
            collection_owner = getattr(getattr(operation, "collection", None), "owner_id", None)
            collection_public = getattr(getattr(operation, "collection", None), "is_public", False)

            if owner_id not in (None, user_id) and collection_owner not in (None, user_id) and not collection_public:
                raise AccessDeniedError(str(user_id), "operation", operation_id)

        status_value = self._map_operation_status(getattr(operation, "status", None))
        meta = getattr(operation, "meta", {}) or {}

        documents_processed = self._coerce_int(
            meta.get("documents_processed")
            or meta.get("processed")
            or meta.get("documents_added")
            or meta.get("removed"),
            default=0,
        )
        total_documents = self._coerce_int(
            meta.get("total_documents")
            or meta.get("total")
            or meta.get("total_files_scanned")
            or meta.get("documents_total"),
            default=0,
        )
        chunks_created = self._coerce_int(
            meta.get("chunks_created")
            or meta.get("total_chunks")
            or meta.get("vector_count")
            or meta.get("chunks_total"),
            default=0,
        )

        progress_percentage: float | None = 100.0 if status_value == "completed" else None
        current_document = meta.get("current_document")
        errors = meta.get("errors") or meta.get("failed_documents") or []
        estimated_time_remaining = meta.get("estimated_time_remaining")

        stream_progress = await self._read_stream_progress(operation_id)
        if stream_progress:
            status_from_stream = stream_progress.get("status")
            if status_from_stream:
                status_value = status_from_stream

            progress_percentage = stream_progress.get("progress_percentage", progress_percentage)
            documents_processed = stream_progress.get("documents_processed", documents_processed)
            total_documents = stream_progress.get("total_documents", total_documents)
            chunks_created = stream_progress.get("chunks_created", chunks_created)
            current_document = stream_progress.get("current_document", current_document)
            errors = stream_progress.get("errors", errors)
            estimated_time_remaining = stream_progress.get("estimated_time_remaining", estimated_time_remaining)

        if progress_percentage is None:
            # Fallback best guess
            if total_documents and documents_processed:
                progress_percentage = min(100.0, (documents_processed / max(total_documents, 1)) * 100)
            elif status_value in {"failed", "cancelled"}:
                progress_percentage = 100.0
            else:
                progress_percentage = 0.0

        return {
            "operation_id": operation_id,
            "status": status_value,
            "progress_percentage": float(progress_percentage),
            "documents_processed": int(documents_processed or 0),
            "total_documents": int(total_documents or 0),
            "chunks_created": int(chunks_created or 0),
            "current_document": current_document,
            "estimated_time_remaining": estimated_time_remaining,
            "errors": errors or [],
        }

    async def _read_stream_progress(self, operation_id: str) -> dict[str, Any] | None:
        """Fetch the latest progress message from Redis if available."""

        redis_client = getattr(self.cache, "redis", None)
        if redis_client is None:
            return None

        try:
            entries = await redis_client.xrevrange(f"operation-progress:{operation_id}", max="+", min="-", count=1)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to read progress stream for %s: %s", operation_id, exc)
            return None

        if not entries:
            return None

        _, fields = entries[0]
        raw_message = fields.get("message")
        if not raw_message:
            return None

        try:
            message = json.loads(raw_message)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to parse progress message for %s: %s", operation_id, exc)
            return None

        return self._normalise_progress_message(message)

    def _normalise_progress_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Convert Redis progress messages to a unified structure."""

        payload = message.get("data") or {}
        msg_type = str(message.get("type") or "").lower()

        processed = payload.get("processed") or payload.get("documents_added") or payload.get("removed")
        failed = payload.get("failed") or payload.get("documents_failed") or 0
        total = payload.get("total") or payload.get("total_files_scanned") or payload.get("total_documents")

        progress_pct = payload.get("progress_percent")
        if progress_pct is None and processed is not None and total:
            try:
                progress_pct = (self._coerce_float(processed) + self._coerce_float(failed)) / self._coerce_float(total)
                progress_pct *= 100
            except Exception:  # pragma: no cover - defensive
                progress_pct = None

        status_value = self._status_from_message_type(msg_type)

        documents_processed = None
        if processed is not None:
            try:
                documents_processed = int(self._coerce_float(processed) + self._coerce_float(failed))
            except Exception:
                documents_processed = processed

        total_documents = self._coerce_int(total)
        chunks_created = self._coerce_int(
            payload.get("chunks_created") or payload.get("vectors_created") or payload.get("total_vectors_created"),
        )

        return {
            "status": status_value,
            "progress_percentage": progress_pct,
            "documents_processed": documents_processed,
            "total_documents": total_documents,
            "chunks_created": chunks_created,
            "current_document": payload.get("current_document"),
            "estimated_time_remaining": payload.get("estimated_time_remaining"),
            "errors": payload.get("errors") or payload.get("failed_documents") or [],
        }

    def _status_from_message_type(self, msg_type: str) -> str | None:
        """Map Redis message types to chunking status values."""

        mapping = {
            "operation_completed": "completed",
            "append_completed": "completed",
            "index_completed": "completed",
            "operation_failed": "failed",
            "operation_started": "in_progress",
            "document_processed": "in_progress",
            "scanning_documents": "in_progress",
            "scanning_completed": "in_progress",
        }
        return mapping.get(msg_type)

    def _map_operation_status(self, status: OperationStatus | str | None) -> str:
        """Normalize operation status to ChunkingStatus-compatible string."""

        status_value = status.value if isinstance(status, OperationStatus) else str(status or "pending").lower()

        mapping = {
            "pending": "pending",
            "processing": "in_progress",
            "in_progress": "in_progress",
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
        }

        return mapping.get(status_value, "pending")

    def _coerce_int(self, value: Any, default: int | None = None) -> int | None:
        """Safely cast a value to int when possible."""

        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _coerce_float(self, value: Any) -> float:
        """Safely convert values to float."""
        return float(value) if value is not None else 0.0

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
