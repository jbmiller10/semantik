"""
V2 API schemas for collection-based search and operations.

This module extends the base schemas with v2-specific features like
multi-collection search and enhanced result metadata.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from packages.shared.contracts.search import normalize_hybrid_mode, normalize_keyword_mode
from packages.webui.api.schemas import SearchResult as BaseSearchResult


class CollectionSearchRequest(BaseModel):
    """Multi-collection search request schema."""

    collection_uuids: list[str] = Field(
        ..., min_length=1, max_length=10, description="List of collection UUIDs to search"
    )
    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    search_type: str = Field(default="semantic", description="Type of search: semantic, question, code, hybrid")
    use_reranker: bool = Field(default=True, description="Enable cross-encoder reranking")
    rerank_model: str | None = Field(None, description="Override reranker model")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum score threshold")
    metadata_filter: dict[str, Any] | None = Field(None, description="Metadata filters for search")
    include_content: bool = Field(default=True, description="Include chunk content in results")
    # Hybrid search parameters
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for hybrid search (vector vs keyword)")
    hybrid_mode: str = Field(default="weighted", description="Hybrid search mode: 'filter' or 'weighted'")
    keyword_mode: str = Field(default="any", description="Keyword matching: 'any' or 'all'")

    @field_validator("keyword_mode", mode="before")
    @classmethod
    def normalize_keyword_mode(cls, value: str) -> str:
        return normalize_keyword_mode(value)

    @field_validator("hybrid_mode", mode="before")
    @classmethod
    def normalize_hybrid_mode(cls, value: str) -> str:
        return normalize_hybrid_mode(value)

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
        }
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
    search_type: str = Field(default="semantic", description="Type of search: semantic, question, code, hybrid")
    use_reranker: bool = Field(default=False, description="Enable cross-encoder reranking")
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum score threshold")
    metadata_filter: dict[str, Any] | None = Field(None, description="Metadata filters for search")
    include_content: bool = Field(default=True, description="Include chunk content in results")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "collection_id": "123e4567-e89b-12d3-a456-426614174000",
                "query": "How to implement authentication?",
                "k": 10,
                "search_type": "semantic",
                "use_reranker": False,
                "score_threshold": 0.7,
            }
        }
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

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "reducer": "umap",
                "dimensionality": 2,
                "config": {"n_neighbors": 15, "min_dist": 0.1},
            }
        }
    )


class ProjectionMetadataResponse(BaseModel):
    """Minimal metadata payload describing a projection run."""

    id: str = Field(description="Projection run UUID")
    collection_id: str = Field(description="Collection UUID")
    status: str = Field(description="Lifecycle status for the projection run")
    reducer: str = Field(description="Algorithm used for the projection")
    dimensionality: int = Field(description="Target dimensionality of the projection output")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")
    message: str | None = Field(default=None, description="Optional status message")


class ProjectionListResponse(BaseModel):
    """Wrapper for listing projection runs belonging to a collection."""

    projections: list[ProjectionMetadataResponse]


class ProjectionArrayResponse(BaseModel):
    """Placeholder payload for coordinate responses until streaming is implemented."""

    projection_id: str
    message: str = "Projection array streaming not yet implemented"


class ProjectionSelectionRequest(BaseModel):
    """Selection request expressed as a bounding box in screen coordinates."""

    x: float
    y: float
    width: float
    height: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "x": 100,
                "y": 150,
                "width": 250,
                "height": 200,
            }
        }
    )


class ProjectionSelectionResponse(BaseModel):
    """Placeholder response for selection requests."""

    projection_id: str
    chunks: list[str]
    message: str = Field(default="Projection selection not yet implemented")
