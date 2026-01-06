"""
Pydantic schemas for chunking API v2 endpoints.

This module provides comprehensive schemas for all chunking-related operations
including strategy management, preview operations, and analytics.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

# Import enums from shared location and re-export for backward compatibility
from shared.chunking.types import ChunkingStatus, ChunkingStrategy, QualityLevel

__all__ = ["ChunkingStrategy", "ChunkingStatus", "QualityLevel"]


# Base configuration schemas
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


class SemanticChunkingConfig(ChunkingConfigBase):
    """Configuration specific to semantic chunking."""

    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Similarity threshold for semantic grouping"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", description="Model to use for semantic similarity"
    )
    max_chunk_size: int = Field(default=1024, ge=100, le=8192, description="Maximum allowed chunk size")


class RecursiveChunkingConfig(ChunkingConfigBase):
    """Configuration for recursive chunking."""

    separators: list[str] = Field(
        default=["\n\n", "\n", " ", ""], description="Separators to use in order of precedence"
    )
    recursion_depth: int = Field(default=3, ge=1, le=10, description="Maximum recursion depth")


class HybridChunkingConfig(ChunkingConfigBase):
    """Configuration for hybrid chunking strategy."""

    primary_strategy: ChunkingStrategy = Field(..., description="Primary chunking strategy")
    secondary_strategy: ChunkingStrategy = Field(..., description="Secondary/fallback strategy")
    switch_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Threshold for switching strategies")


# Strategy information responses
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


class StrategyRecommendation(BaseModel):
    """Recommendation for chunking strategy based on content analysis."""

    model_config = ConfigDict(from_attributes=True)

    recommended_strategy: ChunkingStrategy = Field(..., description="Recommended strategy")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for recommendation")
    reasoning: str = Field(..., description="Explanation for the recommendation")
    alternative_strategies: list[ChunkingStrategy] = Field(default=[], description="Alternative strategies to consider")
    suggested_config: ChunkingConfigBase = Field(..., description="Suggested configuration")


# Preview operation schemas
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


class PreviewRequest(BaseModel):
    """Request for generating chunk preview."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str | None = Field(default=None, description="Document ID for preview (if from collection)")
    content: str | None = Field(default=None, description="Raw content for preview (if not from collection)")
    strategy: ChunkingStrategy = Field(..., description="Strategy to use")
    config: ChunkingConfigBase | None = Field(default=None, description="Custom configuration")
    max_chunks: int = Field(default=10, ge=1, le=50, description="Maximum number of chunks to preview")
    include_metrics: bool = Field(default=True, description="Include quality metrics in response")


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


class CompareRequest(BaseModel):
    """Request for comparing multiple chunking strategies."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str | None = Field(default=None, description="Document ID for comparison")
    content: str | None = Field(default=None, description="Raw content for comparison")
    strategies: list[ChunkingStrategy] = Field(..., min_length=2, max_length=6, description="Strategies to compare")
    configs: dict[str, ChunkingConfigBase] | None = Field(default=None, description="Custom configs per strategy")
    max_chunks_per_strategy: int = Field(default=5, ge=1, le=20, description="Max chunks to show per strategy")


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


# Collection processing schemas
class ChunkingOperationRequest(BaseModel):
    """Request to start chunking operation on a collection."""

    model_config = ConfigDict(from_attributes=True)

    strategy: ChunkingStrategy = Field(..., description="Strategy to use")
    config: ChunkingConfigBase | None = Field(default=None, description="Custom configuration")
    document_ids: list[str] | None = Field(default=None, description="Specific documents to chunk (if not all)")
    priority: int | str = Field(default=5, description="Operation priority (1-10 or 'low'/'normal'/'high')")
    notify_on_completion: bool = Field(default=True, description="Send notification when complete")

    @field_validator("priority", mode="before")
    @classmethod
    def validate_priority(cls, v: int | str) -> int:
        """Convert string priority to integer."""
        if isinstance(v, str):
            priority_map = {"low": 3, "normal": 5, "high": 8}
            return priority_map.get(v.lower(), 5)
        if isinstance(v, int):
            return max(1, min(10, v))  # Clamp to 1-10 range
        return 5  # Default


class ChunkingOperationResponse(BaseModel):
    """Response for chunking operation initiation."""

    model_config = ConfigDict(from_attributes=True)

    operation_id: str = Field(..., description="Operation identifier")
    collection_id: str = Field(..., description="Collection being processed")
    status: ChunkingStatus = Field(..., description="Current status")
    strategy: ChunkingStrategy = Field(..., description="Strategy being used")
    estimated_time_seconds: int | None = Field(default=None, description="Estimated completion time")
    queued_position: int | None = Field(default=None, description="Position in processing queue")
    websocket_channel: str = Field(..., description="WebSocket channel for progress updates")


class ChunkingProgress(BaseModel):
    """Progress information for ongoing chunking operation."""

    model_config = ConfigDict(from_attributes=True)

    operation_id: str = Field(..., description="Operation identifier")
    status: ChunkingStatus = Field(..., description="Current status")
    progress_percentage: float = Field(..., ge=0.0, le=100.0, description="Completion percentage")
    documents_processed: int = Field(..., description="Documents processed")
    total_documents: int = Field(..., description="Total documents to process")
    chunks_created: int = Field(..., description="Chunks created so far")
    current_document: str | None = Field(default=None, description="Currently processing document")
    estimated_time_remaining: int | None = Field(default=None, description="Estimated seconds remaining")
    errors: list[dict[str, Any]] = Field(default=[], description="Any errors encountered")


class ChunkingStrategyUpdate(BaseModel):
    """Request to update collection chunking strategy."""

    model_config = ConfigDict(from_attributes=True)

    strategy: ChunkingStrategy = Field(..., description="New strategy")
    config: ChunkingConfigBase | None = Field(default=None, description="New configuration")
    reprocess_existing: bool = Field(default=True, description="Reprocess existing documents")


class ChunkListResponse(BaseModel):
    """Response containing paginated chunks."""

    model_config = ConfigDict(from_attributes=True)

    chunks: list[dict[str, Any]] = Field(..., description="Chunk data")
    total: int = Field(..., description="Total number of chunks")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Items per page")
    has_next: bool = Field(..., description="Whether there are more pages")


class ChunkResponse(BaseModel):
    """Response containing a single chunk."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Chunk primary key (database)")
    collection_id: str = Field(..., description="Collection UUID")
    document_id: str | None = Field(default=None, description="Document UUID (may be null)")
    chunk_index: int = Field(..., description="Chunk ordering index within document")
    content: str = Field(..., description="Full chunk content")
    token_count: int | None = Field(default=None, description="Token count if available")
    start_offset: int | None = Field(default=None, description="Start character offset in source document")
    end_offset: int | None = Field(default=None, description="End character offset in source document")
    metadata: dict[str, Any] | None = Field(default=None, description="Chunk metadata")
    created_at: datetime = Field(..., description="Chunk creation timestamp")
    updated_at: datetime = Field(..., description="Chunk update timestamp")


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


# Analytics schemas
class GlobalMetrics(BaseModel):
    """Global chunking metrics across all collections."""

    model_config = ConfigDict(from_attributes=True)

    total_collections_processed: int = Field(..., description="Total collections processed")
    total_chunks_created: int = Field(..., description="Total chunks created")
    total_documents_processed: int = Field(..., description="Total documents processed")
    avg_chunks_per_document: float = Field(..., description="Average chunks per document")
    most_used_strategy: ChunkingStrategy = Field(..., description="Most popular strategy")
    avg_processing_time: float = Field(..., description="Average processing time")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate of operations")
    period_start: datetime = Field(..., description="Metrics period start")
    period_end: datetime = Field(..., description="Metrics period end")


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


class QualityAnalysis(BaseModel):
    """Quality analysis for chunks."""

    model_config = ConfigDict(from_attributes=True)

    overall_quality: QualityLevel = Field(..., description="Overall quality level")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Numeric quality score")
    coherence_score: float = Field(..., ge=0.0, le=1.0, description="Semantic coherence")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Information completeness")
    size_consistency: float = Field(..., ge=0.0, le=1.0, description="Size consistency across chunks")
    recommendations: list[str] = Field(..., description="Improvement recommendations")
    issues_detected: list[str] = Field(..., description="Quality issues found")


class DocumentAnalysisRequest(BaseModel):
    """Request for document analysis."""

    model_config = ConfigDict(from_attributes=True)

    document_id: str | None = Field(default=None, description="Document ID to analyze")
    content: str | None = Field(default=None, description="Raw content to analyze")
    file_type: str | None = Field(default=None, description="File type hint")
    deep_analysis: bool = Field(default=False, description="Perform deep content analysis")


class DocumentAnalysisResponse(BaseModel):
    """Response containing document analysis results."""

    model_config = ConfigDict(from_attributes=True)

    document_type: str = Field(..., description="Detected document type")
    content_structure: dict[str, Any] = Field(..., description="Document structure analysis")
    recommended_strategy: StrategyRecommendation = Field(..., description="Strategy recommendation")
    estimated_chunks: dict[ChunkingStrategy, int] = Field(..., description="Estimated chunks per strategy")
    complexity_score: float = Field(..., ge=0.0, le=1.0, description="Document complexity")
    special_considerations: list[str] = Field(default=[], description="Special handling considerations")


# Configuration management schemas
class SavedConfiguration(BaseModel):
    """Saved chunking configuration."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Configuration ID")
    name: str = Field(..., description="Configuration name")
    description: str | None = Field(default=None, description="Description")
    strategy: ChunkingStrategy = Field(..., description="Strategy")
    config: ChunkingConfigBase = Field(..., description="Configuration details")
    created_by: int = Field(..., description="User ID who created")
    created_at: datetime = Field(..., description="Creation time")
    updated_at: datetime = Field(..., description="Last update time")
    usage_count: int = Field(default=0, description="Times used")
    is_default: bool = Field(default=False, description="Is default config")
    tags: list[str] = Field(default=[], description="Configuration tags")


class CreateConfigurationRequest(BaseModel):
    """Request to save a configuration."""

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(..., min_length=1, max_length=100, description="Configuration name")
    description: str | None = Field(default=None, max_length=500, description="Description")
    strategy: ChunkingStrategy = Field(..., description="Strategy")
    config: ChunkingConfigBase = Field(..., description="Configuration details")
    is_default: bool = Field(default=False, description="Set as default")
    tags: list[str] = Field(default=[], max_length=10, description="Tags")


# Error response schemas
class ChunkingErrorResponse(BaseModel):
    """Error response for chunking operations."""

    model_config = ConfigDict(from_attributes=True)

    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type classification")
    correlation_id: str = Field(..., description="Correlation ID for tracking")
    details: dict[str, Any] | None = Field(default=None, description="Additional details")
    suggestions: list[str] = Field(default=[], description="Suggested remedies")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
