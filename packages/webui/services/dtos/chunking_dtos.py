"""
Service layer DTOs for chunking operations.

These DTOs encapsulate business logic and provide conversion methods to API schemas.
They handle all transformations that should not be in the API router layer.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import ValidationError

from .api_models import (
    ChunkingConfigBase,
    ChunkingStats,
    ChunkingStrategy,
    ChunkListResponse,
    ChunkPreview,
    CompareResponse,
    DocumentAnalysisResponse,
    GlobalMetrics,
    PreviewResponse,
    QualityAnalysis,
    QualityLevel,
    SavedConfiguration,
    StrategyComparison,
    StrategyInfo,
    StrategyMetrics,
    StrategyRecommendation,
)

logger = logging.getLogger(__name__)


@dataclass
class ServiceStrategyInfo:
    """Internal representation of strategy information."""

    id: str
    name: str
    description: str
    best_for: list[str] = field(default_factory=list)
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    default_config: dict[str, Any] = field(default_factory=dict)
    supported_file_types: list[str] = field(default_factory=list)
    performance_characteristics: dict[str, Any] = field(default_factory=dict)

    def to_api_model(self) -> StrategyInfo:
        """Convert to API response model."""
        # Ensure default_config has a strategy field
        config_data = self.default_config.copy()
        if "strategy" not in config_data:
            config_data["strategy"] = self.id

        # Create config with error handling
        try:
            config = ChunkingConfigBase(**config_data)
        except ValidationError as e:
            logger.warning(f"Invalid config for strategy {self.id}: {e}")
            # Use minimal valid config as fallback
            config = ChunkingConfigBase(strategy=self.id)

        return StrategyInfo(
            id=self.id,
            name=self.name,
            description=self.description,
            best_for=self.best_for,
            pros=self.pros,
            cons=self.cons,
            default_config=config,
            performance_characteristics=self.performance_characteristics,
        )


@dataclass
class ServiceChunkPreview:
    """Internal representation of a chunk preview."""

    index: int
    content: str | None = None
    text: str | None = None  # Alternative field for content
    token_count: int | None = None
    char_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.8
    overlap_info: dict[str, Any] | None = None

    def to_api_model(self) -> ChunkPreview:
        """Convert to API response model with all transformation logic."""
        # Handle both 'content' and 'text' keys for chunk content
        content = self.content or self.text or ""

        # Always recalculate char_count to match router behavior
        char_count = len(content)

        # Approximate token count (rough estimate: ~4 chars per token) if not provided
        token_count = self.token_count if self.token_count is not None else char_count // 4

        return ChunkPreview(
            index=self.index,
            content=content,
            token_count=token_count,
            char_count=char_count,
            metadata=self.metadata,
            quality_score=self.quality_score,
            overlap_info=self.overlap_info,
        )


@dataclass
class ServicePreviewResponse:
    """Internal representation of preview response."""

    preview_id: str
    strategy: str | ChunkingStrategy
    config: dict[str, Any]
    chunks: list[ServiceChunkPreview | dict[str, Any]]
    total_chunks: int
    metrics: dict[str, Any] | None = None
    processing_time_ms: int = 0
    cached: bool = False
    expires_at: datetime | None = None
    correlation_id: str | None = None

    def to_api_model(self) -> PreviewResponse:
        """Convert to API response model with all transformation logic."""
        # Ensure config has strategy field
        config_data = self.config.copy() if isinstance(self.config, dict) else self.config
        if isinstance(config_data, dict) and "strategy" not in config_data:
            config_data["strategy"] = self.strategy

        # Transform chunks
        api_chunks = []
        for chunk in self.chunks:
            if isinstance(chunk, ServiceChunkPreview):
                api_chunks.append(chunk.to_api_model())
            elif isinstance(chunk, dict):
                # Handle dict chunks from service layer
                service_chunk = ServiceChunkPreview(
                    index=chunk.get("index", 0),
                    content=chunk.get("content"),
                    text=chunk.get("text"),
                    token_count=chunk.get("token_count"),
                    char_count=chunk.get("char_count"),
                    metadata=chunk.get("metadata", {}),
                    quality_score=chunk.get("quality_score", 0.8),
                    overlap_info=chunk.get("overlap_info"),
                )
                api_chunks.append(service_chunk.to_api_model())

        # Handle expires_at default
        expires_at = self.expires_at or (datetime.now(UTC) + timedelta(minutes=15))

        # Ensure strategy is ChunkingStrategy enum
        try:
            strategy = ChunkingStrategy(self.strategy) if isinstance(self.strategy, str) else self.strategy
        except ValueError as e:
            logger.warning(f"Invalid strategy value {self.strategy}: {e}")
            strategy = ChunkingStrategy.FIXED_SIZE  # Default fallback

        # Create config with error handling
        try:
            config = ChunkingConfigBase(**config_data)
        except ValidationError as e:
            logger.warning(f"Invalid config for preview {self.preview_id}: {e}")
            # Use minimal valid config as fallback
            config = ChunkingConfigBase(strategy=strategy)

        return PreviewResponse(
            preview_id=self.preview_id,
            strategy=strategy,
            config=config,
            chunks=api_chunks,
            total_chunks=self.total_chunks,
            metrics=self.metrics,
            processing_time_ms=self.processing_time_ms,
            cached=self.cached,
            expires_at=expires_at,
            correlation_id=self.correlation_id,
        )


@dataclass
class ServiceStrategyRecommendation:
    """Internal representation of strategy recommendation."""

    strategy: str | ChunkingStrategy
    confidence: float = 0.8
    reasoning: str = ""
    alternatives: list[str | ChunkingStrategy] = field(default_factory=list)
    chunk_size: int = 512
    chunk_overlap: int = 50
    preserve_sentences: bool = True
    metadata: dict[str, Any] | None = None

    def to_api_model(self) -> StrategyRecommendation:
        """Convert to API response model."""
        # Ensure strategy is ChunkingStrategy enum
        try:
            strategy = ChunkingStrategy(self.strategy) if isinstance(self.strategy, str) else self.strategy
        except ValueError as e:
            logger.warning(f"Invalid strategy value {self.strategy}: {e}")
            strategy = ChunkingStrategy.FIXED_SIZE  # Default fallback

        # Convert alternatives to enums
        alternative_strategies = []
        for alt in self.alternatives:
            try:
                if isinstance(alt, str):
                    alternative_strategies.append(ChunkingStrategy(alt))
                else:
                    alternative_strategies.append(alt)
            except ValueError as e:
                logger.warning(f"Invalid alternative strategy {alt}: {e}")
                # Skip invalid alternatives

        # Build suggested config
        metadata = self.metadata if isinstance(self.metadata, dict) else None

        try:
            suggested_config = ChunkingConfigBase(
                strategy=strategy,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                preserve_sentences=self.preserve_sentences,
                metadata=metadata,
            )
        except ValidationError as e:
            logger.warning(f"Invalid suggested config: {e}")
            # Use minimal valid config as fallback
            suggested_config = ChunkingConfigBase(strategy=strategy, metadata=metadata)

        return StrategyRecommendation(
            recommended_strategy=strategy,
            confidence=self.confidence,
            reasoning=self.reasoning,
            alternative_strategies=alternative_strategies,
            suggested_config=suggested_config,
        )


@dataclass
class ServiceStrategyComparison:
    """Internal representation of strategy comparison."""

    strategy: str | ChunkingStrategy
    config: dict[str, Any]
    sample_chunks: list[ServiceChunkPreview | dict[str, Any]]
    total_chunks: int
    avg_chunk_size: float
    size_variance: float
    quality_score: float
    processing_time_ms: int
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)

    def to_api_model(self) -> StrategyComparison:
        """Convert to API response model."""
        # Ensure strategy is ChunkingStrategy enum
        try:
            strategy = ChunkingStrategy(self.strategy) if isinstance(self.strategy, str) else self.strategy
        except ValueError as e:
            logger.warning(f"Invalid strategy value {self.strategy}: {e}")
            strategy = ChunkingStrategy.FIXED_SIZE  # Default fallback

        # Ensure config has strategy field
        config_data = self.config.copy() if isinstance(self.config, dict) else self.config
        if isinstance(config_data, dict) and "strategy" not in config_data:
            config_data["strategy"] = strategy

        # Transform chunks
        api_chunks = []
        for chunk in self.sample_chunks:
            if isinstance(chunk, ServiceChunkPreview):
                api_chunks.append(chunk.to_api_model())
            elif isinstance(chunk, dict):
                service_chunk = ServiceChunkPreview(
                    index=chunk.get("index", 0),
                    content=chunk.get("content"),
                    text=chunk.get("text"),
                    token_count=chunk.get("token_count"),
                    char_count=chunk.get("char_count"),
                    metadata=chunk.get("metadata", {}),
                    quality_score=chunk.get("quality_score", 0.8),
                    overlap_info=chunk.get("overlap_info"),
                )
                api_chunks.append(service_chunk.to_api_model())

        # Create config with error handling
        try:
            config = ChunkingConfigBase(**config_data)
        except ValidationError as e:
            logger.warning(f"Invalid config for comparison: {e}")
            # Use minimal valid config as fallback
            config = ChunkingConfigBase(strategy=strategy)

        return StrategyComparison(
            strategy=strategy,
            config=config,
            sample_chunks=api_chunks,
            total_chunks=self.total_chunks,
            avg_chunk_size=self.avg_chunk_size,
            size_variance=self.size_variance,
            quality_score=self.quality_score,
            processing_time_ms=self.processing_time_ms,
            pros=self.pros,
            cons=self.cons,
        )


@dataclass
class ServiceCompareResponse:
    """Internal representation of comparison response."""

    comparison_id: str
    comparisons: list[ServiceStrategyComparison | dict[str, Any]]
    recommendation: ServiceStrategyRecommendation | dict[str, Any]
    processing_time_ms: int = 0

    def to_api_model(self) -> CompareResponse:
        """Convert to API response model."""
        # Transform comparisons
        api_comparisons = []
        for comp in self.comparisons:
            if isinstance(comp, ServiceStrategyComparison):
                api_comparisons.append(comp.to_api_model())
            elif isinstance(comp, dict):
                # Handle dict comparisons from service layer
                service_comp = ServiceStrategyComparison(
                    strategy=comp["strategy"],
                    config=comp.get("config", {}),
                    sample_chunks=comp.get("sample_chunks", []),
                    total_chunks=comp.get("total_chunks", 0),
                    avg_chunk_size=comp.get("avg_chunk_size", 0.0),
                    size_variance=comp.get("size_variance", 0.0),
                    quality_score=comp.get("quality_score", 0.8),
                    processing_time_ms=comp.get("processing_time_ms", 0),
                    pros=comp.get("pros", []),
                    cons=comp.get("cons", []),
                )
                api_comparisons.append(service_comp.to_api_model())

        # Transform recommendation
        if isinstance(self.recommendation, ServiceStrategyRecommendation):
            api_recommendation = self.recommendation.to_api_model()
        elif isinstance(self.recommendation, dict):
            # Handle dict recommendation from service layer
            service_rec = ServiceStrategyRecommendation(
                strategy=self.recommendation["strategy"],
                confidence=self.recommendation.get("confidence", 0.8),
                reasoning=self.recommendation.get("reasoning", ""),
                alternatives=self.recommendation.get("alternatives", []),
                chunk_size=self.recommendation.get("chunk_size", 512),
                chunk_overlap=self.recommendation.get("chunk_overlap", 50),
                preserve_sentences=self.recommendation.get("preserve_sentences", True),
            )
            api_recommendation = service_rec.to_api_model()
        else:
            # Fallback
            api_recommendation = StrategyRecommendation(
                recommended_strategy=ChunkingStrategy.FIXED_SIZE,
                confidence=0.5,
                reasoning="Unable to determine best strategy",
                alternative_strategies=[],
                suggested_config=ChunkingConfigBase(
                    strategy=ChunkingStrategy.FIXED_SIZE,
                    chunk_size=512,
                    chunk_overlap=50,
                    preserve_sentences=True,
                ),
            )

        return CompareResponse(
            comparison_id=self.comparison_id,
            comparisons=api_comparisons,
            recommendation=api_recommendation,
            processing_time_ms=self.processing_time_ms,
        )


@dataclass
class ServiceStrategyMetrics:
    """Internal representation of strategy metrics."""

    strategy: str | ChunkingStrategy
    usage_count: int = 0
    avg_chunk_size: float = 0.0
    avg_processing_time: float = 0.0
    success_rate: float = 0.0
    avg_quality_score: float = 0.0
    best_for_types: list[str] = field(default_factory=list)

    def to_api_model(self) -> StrategyMetrics:
        """Convert to API response model."""
        # Ensure strategy is ChunkingStrategy enum
        try:
            strategy = ChunkingStrategy(self.strategy) if isinstance(self.strategy, str) else self.strategy
        except ValueError as e:
            logger.warning(f"Invalid strategy value {self.strategy}: {e}")
            strategy = ChunkingStrategy.FIXED_SIZE  # Default fallback

        return StrategyMetrics(
            strategy=strategy,
            usage_count=self.usage_count,
            avg_chunk_size=self.avg_chunk_size,
            avg_processing_time=self.avg_processing_time,
            success_rate=self.success_rate,
            avg_quality_score=self.avg_quality_score,
            best_for_types=self.best_for_types,
        )

    @classmethod
    def create_default_metrics(cls) -> list["ServiceStrategyMetrics"]:
        """Create default placeholder metrics for all primary strategies."""
        strategies = [
            ChunkingStrategy.FIXED_SIZE,
            ChunkingStrategy.RECURSIVE,
            ChunkingStrategy.MARKDOWN,
            ChunkingStrategy.SEMANTIC,
            ChunkingStrategy.HIERARCHICAL,
            ChunkingStrategy.HYBRID,
        ]
        return [
            cls(
                strategy=s,
                usage_count=0,
                avg_chunk_size=0,
                avg_processing_time=0.0,
                success_rate=0.0,
                avg_quality_score=0.0,
                best_for_types=[],
            )
            for s in strategies
        ]


@dataclass
class ServiceChunkingStats:
    """Internal representation of chunking statistics."""

    total_chunks: int = 0
    total_documents: int = 0
    avg_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0
    size_variance: float = 0.0
    strategy_used: str | ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    processing_time_seconds: float = 0.0
    quality_metrics: dict[str, Any] = field(default_factory=dict)

    def to_api_model(self) -> ChunkingStats:
        """Convert to API response model."""
        # Handle strategy conversion
        try:
            strategy = (
                ChunkingStrategy(self.strategy_used) if isinstance(self.strategy_used, str) else self.strategy_used
            )
        except ValueError as e:
            logger.warning(f"Invalid strategy value {self.strategy_used}: {e}")
            strategy = ChunkingStrategy.FIXED_SIZE  # Default fallback

        return ChunkingStats(
            total_chunks=self.total_chunks,
            total_documents=self.total_documents,
            avg_chunk_size=self.avg_chunk_size,
            min_chunk_size=self.min_chunk_size,
            max_chunk_size=self.max_chunk_size,
            size_variance=self.size_variance,
            strategy_used=strategy,
            last_updated=self.last_updated,
            processing_time_seconds=self.processing_time_seconds,
            quality_metrics=self.quality_metrics or {},
        )


@dataclass
class ServiceChunkRecord:
    """Representation of a single chunk entry."""

    id: int
    collection_id: str
    document_id: str | None
    chunk_index: int
    content: str
    token_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_api_dict(self) -> dict[str, Any]:
        """Convert to dictionary suitable for API response."""
        char_count = len(self.content or "")
        token_estimate = self.token_count if self.token_count is not None else max(char_count // 4, 1)
        return {
            "id": self.id,
            "collection_id": self.collection_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "token_count": token_estimate,
            "char_count": char_count,
            "metadata": self.metadata or {},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ServiceChunkList:
    """Paginated chunk list."""

    chunks: list[ServiceChunkRecord]
    total: int
    page: int
    page_size: int

    def to_api_model(self) -> ChunkListResponse:
        """Convert to API response model."""
        has_next = (self.page - 1) * self.page_size + len(self.chunks) < self.total
        return ChunkListResponse(
            chunks=[chunk.to_api_dict() for chunk in self.chunks],
            total=self.total,
            page=self.page,
            page_size=self.page_size,
            has_next=has_next,
        )


@dataclass
class ServiceGlobalMetrics:
    """Global chunking metrics across collections."""

    total_collections_processed: int
    total_chunks_created: int
    total_documents_processed: int
    avg_chunks_per_document: float
    most_used_strategy: str | ChunkingStrategy
    avg_processing_time: float
    success_rate: float
    period_start: datetime
    period_end: datetime

    def to_api_model(self) -> GlobalMetrics:
        """Convert to API response model."""
        try:
            strategy = (
                ChunkingStrategy(self.most_used_strategy)
                if isinstance(self.most_used_strategy, str)
                else self.most_used_strategy
            )
        except ValueError:
            strategy = ChunkingStrategy.FIXED_SIZE

        return GlobalMetrics(
            total_collections_processed=self.total_collections_processed,
            total_chunks_created=self.total_chunks_created,
            total_documents_processed=self.total_documents_processed,
            avg_chunks_per_document=self.avg_chunks_per_document,
            most_used_strategy=strategy,
            avg_processing_time=self.avg_processing_time,
            success_rate=min(max(self.success_rate, 0.0), 1.0),
            period_start=self.period_start,
            period_end=self.period_end,
        )


@dataclass
class ServiceQualityAnalysis:
    """Quality analysis summary."""

    overall_quality: str | QualityLevel
    quality_score: float
    coherence_score: float
    completeness_score: float
    size_consistency: float
    recommendations: list[str] = field(default_factory=list)
    issues_detected: list[str] = field(default_factory=list)

    def to_api_model(self) -> QualityAnalysis:
        """Convert to API response model."""
        try:
            quality_level = (
                QualityLevel(self.overall_quality) if isinstance(self.overall_quality, str) else self.overall_quality
            )
        except ValueError:
            quality_level = QualityLevel.GOOD

        return QualityAnalysis(
            overall_quality=quality_level,
            quality_score=min(max(self.quality_score, 0.0), 1.0),
            coherence_score=min(max(self.coherence_score, 0.0), 1.0),
            completeness_score=min(max(self.completeness_score, 0.0), 1.0),
            size_consistency=min(max(self.size_consistency, 0.0), 1.0),
            recommendations=self.recommendations,
            issues_detected=self.issues_detected,
        )


@dataclass
class ServiceDocumentAnalysis:
    """Document analysis result."""

    document_type: str
    content_structure: dict[str, Any]
    recommended_strategy: ServiceStrategyRecommendation
    estimated_chunks: dict[str | ChunkingStrategy, int]
    complexity_score: float
    special_considerations: list[str] = field(default_factory=list)

    def to_api_model(self) -> DocumentAnalysisResponse:
        """Convert to API response model."""
        estimated: dict[ChunkingStrategy, int] = {}
        for strategy, count in self.estimated_chunks.items():
            try:
                strategy_enum = ChunkingStrategy(strategy) if isinstance(strategy, str) else strategy
            except ValueError:
                strategy_enum = ChunkingStrategy.FIXED_SIZE
            estimated[strategy_enum] = max(int(count), 0)

        return DocumentAnalysisResponse(
            document_type=self.document_type,
            content_structure=self.content_structure,
            recommended_strategy=self.recommended_strategy.to_api_model(),
            estimated_chunks=estimated,
            complexity_score=min(max(self.complexity_score, 0.0), 1.0),
            special_considerations=self.special_considerations,
        )


@dataclass
class ServiceSavedConfiguration:
    """Saved chunking configuration with metadata."""

    id: str
    name: str
    description: str | None
    strategy: str | ChunkingStrategy
    config: dict[str, Any]
    created_by: int
    created_at: datetime
    updated_at: datetime
    usage_count: int = 0
    is_default: bool = False
    tags: list[str] = field(default_factory=list)

    def to_api_model(self) -> SavedConfiguration:
        """Convert to API response model."""
        try:
            strategy = ChunkingStrategy(self.strategy) if isinstance(self.strategy, str) else self.strategy
        except ValueError:
            strategy = ChunkingStrategy.FIXED_SIZE

        try:
            config_model = ChunkingConfigBase(**self.config)
        except ValidationError:
            config_model = ChunkingConfigBase(strategy=strategy)

        return SavedConfiguration(
            id=self.id,
            name=self.name,
            description=self.description,
            strategy=strategy,
            config=config_model,
            created_by=self.created_by,
            created_at=self.created_at,
            updated_at=self.updated_at,
            usage_count=self.usage_count,
            is_default=self.is_default,
            tags=self.tags,
        )
