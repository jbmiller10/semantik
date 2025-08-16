"""
API response models for the service layer.

These models define the contract between the service and API layers.
They are owned by the service layer to avoid circular dependencies.
The API layer can use these models directly in its responses.

This separation ensures that:
1. The service layer doesn't depend on the API layer
2. The API layer can use service-defined response models
3. There's a clear contract for data transformation

NOTE: These models are copies of the essential parts from the API schemas,
maintained here to preserve proper separation of concerns.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"
    # Legacy/aliases retained for backward compatibility
    SLIDING_WINDOW = "sliding_window"
    DOCUMENT_STRUCTURE = "document_structure"


class ChunkingConfigBase(BaseModel):
    """Base configuration for chunking strategies."""

    model_config = ConfigDict(from_attributes=True)

    strategy: ChunkingStrategy = Field(..., description="The chunking strategy to use")
    chunk_size: int = Field(default=512, ge=100, le=4096, description="Target size for chunks in tokens")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="Number of overlapping tokens between chunks")
    preserve_sentences: bool = Field(default=True, description="Whether to preserve sentence boundaries")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional configuration metadata")

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info: ValidationInfo) -> int:  # noqa: N805
        """Ensure overlap is less than chunk size."""
        if info.data and "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("Chunk overlap must be less than chunk size")
        return v


class StrategyInfo(BaseModel):
    """Information about a chunking strategy."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Unique strategy identifier")
    name: str = Field(..., description="Human-readable strategy name")
    description: str = Field(..., description="Detailed strategy description")
    best_for: list[str] = Field(..., description="File types this strategy works best with")
    pros: list[str] = Field(..., description="Strategy advantages")
    cons: list[str] = Field(..., description="Strategy limitations")
    default_config: ChunkingConfigBase = Field(..., description="Default configuration")
    performance_characteristics: dict[str, Any] = Field(..., description="Performance metrics and characteristics")


class ChunkPreview(BaseModel):
    """Preview of a single chunk."""

    model_config = ConfigDict(from_attributes=True)

    index: int = Field(..., description="Chunk index")
    content: str = Field(..., description="Chunk content")
    token_count: int = Field(..., description="Number of tokens")
    char_count: int = Field(..., description="Number of characters")
    metadata: dict[str, Any] = Field(default={}, description="Chunk metadata")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Quality score for this chunk")
    overlap_info: dict[str, Any] | None = Field(
        default=None, description="Information about overlaps with adjacent chunks"
    )


class PreviewResponse(BaseModel):
    """Response containing chunk preview results."""

    model_config = ConfigDict(from_attributes=True)

    preview_id: str = Field(..., description="Unique preview identifier for caching")
    strategy: ChunkingStrategy = Field(..., description="Strategy used")
    config: ChunkingConfigBase = Field(..., description="Configuration used")
    chunks: list[ChunkPreview] = Field(..., description="Preview chunks")
    total_chunks: int = Field(..., description="Total number of chunks")
    metrics: dict[str, Any] | None = Field(default=None, description="Preview metrics and statistics")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    cached: bool = Field(default=False, description="Whether result was cached")
    expires_at: datetime = Field(..., description="Cache expiration time")
    correlation_id: str | None = Field(default=None, description="Request correlation ID for tracing")


class StrategyRecommendation(BaseModel):
    """Recommendation for chunking strategy based on content analysis."""

    model_config = ConfigDict(from_attributes=True)

    recommended_strategy: ChunkingStrategy = Field(..., description="Recommended strategy")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for recommendation")
    reasoning: str = Field(..., description="Explanation for the recommendation")
    alternative_strategies: list[ChunkingStrategy] = Field(default=[], description="Alternative strategies to consider")
    suggested_config: ChunkingConfigBase = Field(..., description="Suggested configuration")


class StrategyComparison(BaseModel):
    """Comparison results for a single strategy."""

    model_config = ConfigDict(from_attributes=True)

    strategy: ChunkingStrategy = Field(..., description="Strategy name")
    config: ChunkingConfigBase = Field(..., description="Configuration used")
    sample_chunks: list[ChunkPreview] = Field(..., description="Sample chunks")
    total_chunks: int = Field(..., description="Total number of chunks")
    avg_chunk_size: float = Field(..., description="Average chunk size")
    size_variance: float = Field(..., description="Variance in chunk sizes")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    processing_time_ms: int = Field(..., description="Processing time")
    pros: list[str] = Field(..., description="Advantages for this content")
    cons: list[str] = Field(..., description="Disadvantages for this content")


class CompareResponse(BaseModel):
    """Response containing strategy comparison results."""

    model_config = ConfigDict(from_attributes=True)

    comparison_id: str = Field(..., description="Unique comparison identifier")
    comparisons: list[StrategyComparison] = Field(..., description="Strategy comparisons")
    recommendation: StrategyRecommendation = Field(..., description="Recommended strategy based on comparison")
    processing_time_ms: int = Field(..., description="Total processing time")


class StrategyMetrics(BaseModel):
    """Metrics for a specific chunking strategy."""

    model_config = ConfigDict(from_attributes=True)

    strategy: ChunkingStrategy = Field(..., description="Strategy name")
    usage_count: int = Field(..., description="Number of times used")
    avg_chunk_size: float = Field(..., description="Average chunk size")
    avg_processing_time: float = Field(..., description="Average processing time")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate")
    avg_quality_score: float = Field(..., ge=0.0, le=1.0, description="Average quality score")
    best_for_types: list[str] = Field(..., description="Best performing file types")


class ChunkingStats(BaseModel):
    """Statistics for collection chunking."""

    model_config = ConfigDict(from_attributes=True)

    total_chunks: int = Field(..., description="Total number of chunks")
    total_documents: int = Field(..., description="Total documents processed")
    avg_chunk_size: float = Field(..., description="Average chunk size in tokens")
    min_chunk_size: int = Field(..., description="Minimum chunk size")
    max_chunk_size: int = Field(..., description="Maximum chunk size")
    size_variance: float = Field(..., description="Variance in chunk sizes")
    strategy_used: ChunkingStrategy = Field(..., description="Current strategy")
    last_updated: datetime = Field(..., description="Last update time")
    processing_time_seconds: float = Field(..., description="Total processing time")
    quality_metrics: dict[str, Any] = Field(..., description="Quality assessment metrics")
