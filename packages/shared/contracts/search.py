"""Search API contracts for unified search functionality."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


def normalize_hybrid_mode(value: str | None) -> str:
    """Map legacy hybrid modes to supported values."""

    if value is None:
        return "weighted"

    value_normalized = value.strip().lower()
    legacy_map = {
        "rerank": "weighted",
        "reciprocal_rank": "weighted",
        "relative_score": "weighted",
    }
    return legacy_map.get(value_normalized, value_normalized)


def normalize_keyword_mode(value: str | None) -> str:
    """Map legacy keyword modes to supported values."""

    if value is None:
        return "any"

    value_normalized = value.strip().lower()
    legacy_map = {
        "bm25": "any",
    }
    return legacy_map.get(value_normalized, value_normalized)


class SearchRequest(BaseModel):
    """Unified search API request model."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    # Use 'k' as the canonical field, but accept 'top_k' as an alias for backward compatibility
    k: int = Field(default=10, ge=1, le=100, description="Number of results", alias="top_k")
    search_type: str = Field("semantic", description="Type of search: semantic, question, code, hybrid, vector")
    model_name: str | None = Field(None, description="Override embedding model")
    quantization: str | None = Field(None, description="Override quantization: float32, float16, int8")
    filters: dict[str, Any] | None = Field(None, description="Metadata filters for search")
    include_content: bool = Field(False, description="Include chunk content in results")
    collection: str | None = Field(None, description="Collection name (e.g., operation_123)")
    operation_uuid: str | None = Field(None, description="Operation UUID for collection inference")
    use_reranker: bool = Field(False, description="Enable cross-encoder reranking")
    rerank_model: str | None = Field(None, description="Override reranker model")
    rerank_quantization: str | None = Field(None, description="Override reranker quantization: float32, float16, int8")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum score threshold")
    # Hybrid search specific parameters
    hybrid_alpha: float = Field(0.7, ge=0.0, le=1.0, description="Weight for hybrid search (vector vs keyword)")
    hybrid_mode: str = Field("weighted", description="Hybrid search mode: 'filter' or 'weighted'")
    keyword_mode: str = Field("any", description="Keyword matching: 'any' or 'all'")

    class Config:
        populate_by_name = True  # Allow both 'k' and 'top_k'
        extra = "forbid"  # Reject extra fields

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        """Clean and validate query string."""
        return v.strip()

    @field_validator("search_type")
    @classmethod
    def validate_search_type(cls, v: str) -> str:
        """Validate search type and map 'vector' to 'semantic'."""
        # Map 'vector' to 'semantic' for backward compatibility
        if v == "vector":
            return "semantic"
        valid_types = {"semantic", "question", "code", "hybrid", "vector"}
        if v not in valid_types:
            raise ValueError(f"Invalid search_type: {v}. Must be one of {valid_types}")
        return v

    @field_validator("hybrid_mode")
    @classmethod
    def validate_hybrid_mode(cls, v: str) -> str:
        """Validate hybrid mode."""
        v = normalize_hybrid_mode(v)
        valid_modes = {"filter", "weighted"}
        if v not in valid_modes:
            raise ValueError(f"Invalid hybrid_mode: {v}. Must be one of {valid_modes}")
        return v

    @field_validator("keyword_mode")
    @classmethod
    def validate_keyword_mode(cls, v: str) -> str:
        """Validate keyword mode."""
        v = normalize_keyword_mode(v)
        valid_modes = {"any", "all"}
        if v not in valid_modes:
            raise ValueError(f"Invalid keyword_mode: {v}. Must be one of {valid_modes}")
        return v


class SearchResult(BaseModel):
    """Individual search result."""

    doc_id: str = Field(..., max_length=200)
    chunk_id: str = Field(..., max_length=200)
    score: float
    path: str = Field(..., max_length=4096, description="File path")
    content: str | None = Field(None, max_length=10000, description="Chunk content (if include_content=True)")
    metadata: dict[str, Any] | None = Field(default_factory=dict)
    highlights: list[str] | None = None
    # Additional fields for frontend compatibility
    file_path: str | None = Field(None, max_length=4096)  # Duplicate of path for frontend
    file_name: str | None = Field(None, max_length=255)
    chunk_index: int | None = None
    total_chunks: int | None = None
    operation_uuid: str | None = Field(None, max_length=200)


class SearchResponse(BaseModel):
    """Search API response model."""

    query: str
    results: list[SearchResult]
    num_results: int
    search_type: str | None = None
    model_used: str | None = None
    embedding_time_ms: float | None = None
    search_time_ms: float | None = None
    reranking_used: bool | None = None
    reranker_model: str | None = None
    reranking_time_ms: float | None = None
    collection: str | None = None
    api_version: str = Field(default="1.0", description="API version")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "quantum computing",
                "results": [
                    {
                        "doc_id": "doc_123",
                        "chunk_id": "chunk_456",
                        "score": 0.95,
                        "path": "/data/quantum_intro.pdf",
                        "content": "Quantum computing leverages quantum mechanics...",
                        "metadata": {"source": "quantum_intro.pdf", "page": 1, "operation_uuid": "operation_456"},
                    }
                ],
                "num_results": 1,
                "search_type": "semantic",
                "model_used": "BAAI/bge-small-en-v1.5",
                "embedding_time_ms": 23.5,
                "search_time_ms": 45.2,
                "collection": "work_docs",
            }
        }


class BatchSearchRequest(BaseModel):
    """Batch search request for multiple queries."""

    queries: list[str] = Field(..., min_length=1, max_length=100, description="List of search queries (max 100)")
    k: int = Field(10, ge=1, le=100, description="Number of results per query")
    search_type: str = Field("semantic", max_length=50, description="Type of search")
    model_name: str | None = Field(None, max_length=500, description="Override embedding model")
    quantization: str | None = Field(None, max_length=20, description="Override quantization")
    collection: str | None = Field(None, max_length=200, description="Collection name")

    @field_validator("queries")
    @classmethod
    def validate_query_length(cls, v: list[str]) -> list[str]:
        """Validate each query doesn't exceed max length."""
        for query in v:
            if len(query) > 1000:
                raise ValueError("Each query must not exceed 1000 characters")
            if len(query) < 1:
                raise ValueError("Each query must have at least 1 character")
        return v


class BatchSearchResponse(BaseModel):
    """Batch search response."""

    responses: list[SearchResponse]
    total_time_ms: float
    api_version: str = Field(default="1.0", description="API version")


class HybridSearchResult(BaseModel):
    """Hybrid search result with keyword matching information."""

    path: str = Field(..., max_length=4096)
    chunk_id: str = Field(..., max_length=200)
    score: float
    doc_id: str = Field(..., max_length=200)
    matched_keywords: list[str] = Field(default_factory=list)
    keyword_score: float | None = None
    combined_score: float | None = None
    metadata: dict[str, Any] | None = None
    content: str | None = Field(None, max_length=10000)


class HybridSearchResponse(BaseModel):
    """Hybrid search response."""

    query: str
    results: list[HybridSearchResult]
    num_results: int
    keywords_extracted: list[str]
    search_mode: str  # "filter" or "weighted"
    api_version: str = Field(default="1.0", description="API version")


class HybridSearchRequest(BaseModel):
    """Hybrid search request model (simplified version for backward compatibility)."""

    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(default=10, ge=1, le=100)
    operation_uuid: str | None = Field(None, max_length=200)
    mode: str = Field(default="filter", max_length=20, description="Hybrid search mode: 'filter' or 'weighted'")
    keyword_mode: str = Field(default="any", max_length=20, description="Keyword matching: 'any' or 'all'")
    score_threshold: float | None = None
    collection: str | None = Field(None, max_length=200)
    model_name: str | None = Field(None, max_length=500)
    quantization: str | None = Field(None, max_length=20)

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, value: str) -> str:
        return normalize_hybrid_mode(value)


# Additional models for specialized endpoints


class PreloadModelRequest(BaseModel):
    """Request to preload a model for faster initial searches."""

    model_name: str = Field(..., max_length=500, description="Model name to preload")
    quantization: str = Field(default="float16", max_length=20, description="Quantization type")


class PreloadModelResponse(BaseModel):
    """Response for model preload request."""

    status: str = Field(..., max_length=50)
    message: str = Field(..., max_length=500)
    api_version: str = Field(default="1.0", description="API version")
