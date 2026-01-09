"""
V2 API schemas for collection-based search and operations.

This module extends the base schemas with v2-specific features like
multi-collection search and enhanced result metadata.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from webui.api.schemas import SearchResult as BaseSearchResult

# Type alias for search mode
SearchMode = Literal["dense", "sparse", "hybrid"]


class CollectionSearchRequest(BaseModel):
    """Multi-collection search request schema."""

    collection_uuids: list[str] = Field(
        ..., min_length=1, max_length=10, description="List of collection UUIDs to search"
    )
    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    search_type: str = Field(default="semantic", description="Type of search: semantic, question, code")
    # New search_mode for sparse/hybrid search
    search_mode: SearchMode = Field(
        default="dense",
        description="Search mode: 'dense' (vector only), 'sparse' (BM25/SPLADE only), 'hybrid' (dense + sparse with RRF)",
    )
    rrf_k: int = Field(default=60, ge=1, le=1000, description="RRF constant k for hybrid mode ranking")
    use_reranker: bool = Field(default=True, description="Enable cross-encoder reranking")
    rerank_model: str | None = Field(None, description="Override reranker model")
    reranker_id: str | None = Field(
        None,
        description="Reranker plugin ID (alternative to rerank_model). If both set, reranker_id takes precedence.",
    )
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum score threshold")
    metadata_filter: dict[str, Any] | None = Field(None, description="Metadata filters for search")
    include_content: bool = Field(default=True, description="Include chunk content in results")
    # Legacy hybrid search parameters (deprecated, use search_mode instead)
    hybrid_alpha: float | None = Field(
        None, ge=0.0, le=1.0, description="[DEPRECATED] Use search_mode='hybrid' instead"
    )
    hybrid_mode: str | None = Field(None, description="[DEPRECATED] Use search_mode='hybrid' instead")
    keyword_mode: str | None = Field(None, description="[DEPRECATED] Use search_mode='hybrid' instead")

    @field_validator("collection_uuids")
    @classmethod
    def validate_collection_uuids(cls, v: list[str]) -> list[str]:
        """Validate collection UUIDs format."""
        import uuid

        for uuid_str in v:
            try:
                uuid.UUID(uuid_str)
            except ValueError as e:
                raise ValueError(f"Invalid UUID format: {uuid_str}") from e
        return v

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "collection_uuids": [
                    "123e4567-e89b-12d3-a456-426614174000",
                    "456e7890-e89b-12d3-a456-426614174001",
                ],
                "query": "How to implement authentication?",
                "k": 20,
                "search_type": "semantic",
                "use_reranker": True,
                "score_threshold": 0.5,
            }
        },
    )


class CollectionSearchResult(BaseSearchResult):
    """Extended search result with collection information."""

    collection_id: str = Field(..., description="UUID of the collection this result belongs to")
    collection_name: str = Field(..., description="Name of the collection")
    original_score: float = Field(..., description="Original score before re-ranking")
    reranked_score: float | None = Field(None, description="Score after re-ranking")
    embedding_model: str = Field(..., description="Embedding model used for this result")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_123",
                "chunk_id": "chunk_456",
                "score": 0.95,
                "original_score": 0.85,
                "reranked_score": 0.95,
                "text": "To implement authentication, you can use JWT tokens...",
                "metadata": {"page": 1, "section": "Authentication"},
                "file_name": "auth_guide.md",
                "file_path": "/docs/auth_guide.md",
                "collection_id": "123e4567-e89b-12d3-a456-426614174000",
                "collection_name": "Documentation",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            }
        }
    )


class CollectionSearchResponse(BaseModel):
    """Multi-collection search response schema."""

    query: str
    results: list[CollectionSearchResult]
    total_results: int
    collections_searched: list[dict[str, Any]]  # Collection info including id, name, model
    search_type: str
    # New search_mode fields
    search_mode_used: SearchMode = Field(
        default="dense",
        description="Actual search mode used (may differ from requested if sparse not available)",
    )
    warnings: list[str] = Field(default_factory=list, description="Warnings about search mode fallback or other issues")
    reranking_used: bool
    reranker_model: str | None = None
    # Timing metrics
    embedding_time_ms: float | None = None
    search_time_ms: float
    reranking_time_ms: float | None = None
    total_time_ms: float
    # Failure information
    partial_failure: bool = Field(default=False, description="Whether some collections failed to search")
    failed_collections: list[dict[str, str]] | None = Field(
        None, description="Collections that failed with error messages"
    )
    api_version: str = Field(default="2.0", description="API version")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How to implement authentication?",
                "results": [
                    {
                        "document_id": "doc_123",
                        "chunk_id": "chunk_456",
                        "score": 0.95,
                        "original_score": 0.85,
                        "reranked_score": 0.95,
                        "text": "To implement authentication, you can use JWT tokens...",
                        "metadata": {"page": 1, "section": "Authentication"},
                        "file_name": "auth_guide.md",
                        "file_path": "/docs/auth_guide.md",
                        "collection_id": "123e4567-e89b-12d3-a456-426614174000",
                        "collection_name": "Documentation",
                        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                    }
                ],
                "total_results": 1,
                "collections_searched": [
                    {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "name": "Documentation",
                        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                        "document_count": 150,
                    }
                ],
                "search_type": "semantic",
                "reranking_used": True,
                "reranker_model": "Qwen/Qwen3-Reranker",
                "search_time_ms": 125.5,
                "reranking_time_ms": 50.3,
                "total_time_ms": 175.8,
                "partial_failure": False,
                "api_version": "2.0",
            }
        }
    )


class SingleCollectionSearchRequest(BaseModel):
    """Single collection search request (backward compatibility)."""

    collection_id: str = Field(..., description="Collection UUID to search")
    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    search_type: str = Field(default="semantic", description="Type of search: semantic, question, code")
    # New search_mode for sparse/hybrid search
    search_mode: SearchMode = Field(
        default="dense",
        description="Search mode: 'dense' (vector only), 'sparse' (BM25/SPLADE only), 'hybrid' (dense + sparse with RRF)",
    )
    rrf_k: int = Field(default=60, ge=1, le=1000, description="RRF constant k for hybrid mode ranking")
    use_reranker: bool = Field(default=False, description="Enable cross-encoder reranking")
    rerank_model: str | None = Field(None, description="Override reranker model")
    reranker_id: str | None = Field(
        None,
        description="Reranker plugin ID (alternative to rerank_model). If both set, reranker_id takes precedence.",
    )
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum score threshold")
    metadata_filter: dict[str, Any] | None = Field(None, description="Metadata filters for search")
    include_content: bool = Field(default=True, description="Include chunk content in results")
    # Legacy hybrid search parameters (deprecated, use search_mode instead)
    hybrid_alpha: float | None = Field(
        None, ge=0.0, le=1.0, description="[DEPRECATED] Use search_mode='hybrid' instead"
    )
    hybrid_mode: str | None = Field(None, description="[DEPRECATED] Use search_mode='hybrid' instead")
    keyword_mode: str | None = Field(None, description="[DEPRECATED] Use search_mode='hybrid' instead")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "collection_id": "123e4567-e89b-12d3-a456-426614174000",
                "query": "How to implement authentication?",
                "k": 10,
                "search_type": "semantic",
                "use_reranker": False,
                "score_threshold": 0.7,
            }
        },
    )


class ProjectionBuildRequest(BaseModel):
    """Request payload to start a projection build."""

    reducer: str = Field(default="umap", description="Dimensionality reduction algorithm to use")
    dimensionality: int = Field(
        default=2,
        ge=2,
        le=3,
        description="Target dimensionality for visualization output",
    )
    config: dict[str, Any] | None = Field(default=None, description="Reducer-specific configuration overrides")
    color_by: str = Field(
        default="document_id",
        description="Attribute used to colour points in the projection",
    )
    sample_size: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap on the number of vectors sampled when building the projection",
    )
    sample_n: int | None = Field(
        default=None,
        ge=1,
        description="Alias for sample_size; kept for compatibility with earlier clients",
    )
    metadata_hash: str | None = Field(
        default=None,
        description=(
            "Optional deterministic hash of reducer/config/color_by/sampling inputs and collection vector state. "
            "If omitted, the backend will compute a stable hash for idempotent recompute."
        ),
    )

    @field_validator("color_by")
    @classmethod
    def _validate_color_by(cls, value: str) -> str:
        allowed = {"source_dir", "document_id", "filetype", "age_bucket"}
        colour = value.lower() if isinstance(value, str) else value
        if colour not in allowed:
            raise ValueError("color_by must be one of: source_dir, document_id, filetype, age_bucket")
        return colour

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reducer": "umap",
                "dimensionality": 2,
                "config": {"n_neighbors": 15, "min_dist": 0.1},
                "color_by": "document_id",
                "sample_size": 5000,
            }
        }
    )


class ProjectionMetadataResponse(BaseModel):
    """Minimal metadata payload describing a projection run.

    Progress Tracking Fields:
        operation_id: UUID of the associated background operation tracking this projection build
        operation_status: Real-time status from the operations table (pending/processing/completed/failed/cancelled)
        status: ProjectionRun status (pending/running/completed/failed/cancelled)

    For accurate progress tracking, prefer operation_status over status when available, as operation_status
    reflects the latest state from the background task while status may lag during processing.
    """

    id: str = Field(description="Projection run UUID")
    collection_id: str = Field(description="Collection UUID")
    status: str = Field(
        description="ProjectionRun lifecycle status (pending/running/completed/failed/cancelled). "
        "May lag behind operation_status during processing. Prefer operation_status when available."
    )
    reducer: str = Field(description="Algorithm used for the projection")
    dimensionality: int = Field(description="Target dimensionality of the projection output")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")
    operation_id: str | None = Field(
        default=None,
        description="UUID of the associated Operation tracking this projection build. "
        "Use with WebSocket channel 'operation-progress:{operation_id}' for real-time updates.",
    )
    operation_status: str | None = Field(
        default=None,
        description="Current status from the operations table (pending/processing/completed/failed/cancelled). "
        "This reflects the most recent state from the background task and should be preferred over 'status' "
        "for accurate progress indication. Null if no operation is associated.",
    )
    message: str | None = Field(
        default=None,
        description="Optional status or error message. Automatically populated with error details for failed operations.",
    )
    config: dict[str, Any] | None = Field(default=None, description="Reducer configuration parameters")
    meta: dict[str, Any] | None = Field(default=None, description="Latest metadata captured for the run")
    idempotent_reuse: bool | None = Field(
        default=None,
        description=(
            "True when the projection build request was satisfied by reusing an existing completed run with an "
            "identical metadata_hash instead of creating a new run."
        ),
    )


class ProjectionListResponse(BaseModel):
    """Wrapper for listing projection runs belonging to a collection."""

    projections: list[ProjectionMetadataResponse]


class ProjectionSelectionRequest(BaseModel):
    """Selection request expressed as a set of projection point identifiers."""

    ids: list[int] = Field(min_length=1, description="Int32 identifiers from ids.i32.bin")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ids": [12, 45, 78],
            }
        }
    )


class ProjectionSelectionItem(BaseModel):
    """Metadata for a selected projection point."""

    selected_id: int = Field(description="Int32 identifier from ids.i32.bin")
    index: int = Field(description="Zero-based index within the projection arrays")
    original_id: str | None = Field(default=None, description="Original vector identifier from Qdrant")
    chunk_id: int | None = Field(default=None, description="Chunk primary key if available")
    document_id: str | None = Field(default=None, description="Document UUID if resolved")
    chunk_index: int | None = Field(default=None, description="Chunk index within the document")
    content_preview: str | None = Field(default=None, description="Snippet of the chunk content")
    document: dict[str, Any] | None = Field(default=None, description="Resolved document metadata")


class ProjectionSelectionResponse(BaseModel):
    """Response detailing resolved metadata for selected projection points."""

    projection_id: str
    items: list[ProjectionSelectionItem]
    missing_ids: list[int] = Field(default_factory=list, description="IDs not found in the projection artifact")
    degraded: bool = Field(
        default=False,
        description=(
            "True when the underlying projection run is degraded (e.g. "
            "artifacts are stale, incomplete or produced via a fallback reducer)."
        ),
    )
