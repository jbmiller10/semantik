"""Search API contracts for unified search functionality."""

from typing import Any

from pydantic import BaseModel, Field, validator


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
    collection: str | None = Field(None, description="Collection name (e.g., job_123)")
    job_id: str | None = Field(None, description="Job ID for collection inference")
    use_reranker: bool = Field(False, description="Enable cross-encoder reranking")
    rerank_model: str | None = Field(None, description="Override reranker model")
    rerank_quantization: str | None = Field(None, description="Override reranker quantization: float32, float16, int8")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum score threshold")
    # Hybrid search specific parameters
    hybrid_alpha: float = Field(0.7, ge=0.0, le=1.0, description="Weight for hybrid search (vector vs keyword)")
    hybrid_mode: str = Field("rerank", description="Hybrid search mode: 'filter' or 'rerank'")
    keyword_mode: str = Field("any", description="Keyword matching: 'any' or 'all'")

    class Config:
        populate_by_name = True  # Allow both 'k' and 'top_k'

    @validator("query")
    def clean_query(cls, v: str) -> str:  # noqa: N805
        """Clean and validate query string."""
        return v.strip()

    @validator("search_type")
    def validate_search_type(cls, v: str) -> str:  # noqa: N805
        """Validate search type and map 'vector' to 'semantic'."""
        # Map 'vector' to 'semantic' for backward compatibility
        if v == "vector":
            return "semantic"
        valid_types = {"semantic", "question", "code", "hybrid", "vector"}
        if v not in valid_types:
            raise ValueError(f"Invalid search_type: {v}. Must be one of {valid_types}")
        return v

    @validator("hybrid_mode")
    def validate_hybrid_mode(cls, v: str) -> str:  # noqa: N805
        """Validate hybrid mode."""
        valid_modes = {"filter", "rerank"}
        if v not in valid_modes:
            raise ValueError(f"Invalid hybrid_mode: {v}. Must be one of {valid_modes}")
        return v

    @validator("keyword_mode")
    def validate_keyword_mode(cls, v: str) -> str:  # noqa: N805
        """Validate keyword mode."""
        valid_modes = {"any", "all"}
        if v not in valid_modes:
            raise ValueError(f"Invalid keyword_mode: {v}. Must be one of {valid_modes}")
        return v


class SearchResult(BaseModel):
    """Individual search result."""

    doc_id: str
    chunk_id: str
    score: float
    path: str = Field(description="File path")
    content: str | None = Field(None, description="Chunk content (if include_content=True)")
    metadata: dict[str, Any] | None = Field(default_factory=dict)
    highlights: list[str] | None = None
    # Additional fields for frontend compatibility
    file_path: str | None = None  # Duplicate of path for frontend
    file_name: str | None = None
    chunk_index: int | None = None
    total_chunks: int | None = None
    job_id: str | None = None


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
                        "metadata": {"source": "quantum_intro.pdf", "page": 1, "job_id": "job_456"},
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

    queries: list[str] = Field(..., description="List of search queries")
    k: int = Field(10, ge=1, le=100, description="Number of results per query")
    search_type: str = Field("semantic", description="Type of search")
    model_name: str | None = Field(None, description="Override embedding model")
    quantization: str | None = Field(None, description="Override quantization")
    collection: str | None = Field(None, description="Collection name")


class BatchSearchResponse(BaseModel):
    """Batch search response."""

    responses: list[SearchResponse]
    total_time_ms: float


class HybridSearchResult(BaseModel):
    """Hybrid search result with keyword matching information."""

    path: str
    chunk_id: str
    score: float
    doc_id: str | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    keyword_score: float | None = None
    combined_score: float | None = None
    metadata: dict[str, Any] | None = None
    content: str | None = None


class HybridSearchResponse(BaseModel):
    """Hybrid search response."""

    query: str
    results: list[HybridSearchResult]
    num_results: int
    keywords_extracted: list[str]
    search_mode: str  # "filter" or "rerank"


class HybridSearchRequest(BaseModel):
    """Hybrid search request model (simplified version for backward compatibility)."""

    query: str
    k: int = Field(default=10, ge=1, le=100)
    job_id: str | None = None
    mode: str = Field(default="filter", description="Hybrid search mode: 'filter' or 'rerank'")
    keyword_mode: str = Field(default="any", description="Keyword matching: 'any' or 'all'")
    score_threshold: float | None = None
    collection: str | None = None
    model_name: str | None = None
    quantization: str | None = None


# Additional models for specialized endpoints


class PreloadModelRequest(BaseModel):
    """Request to preload a model for faster initial searches."""

    model_name: str = Field(..., description="Model name to preload")
    quantization: str = Field(default="float16", description="Quantization type")


class PreloadModelResponse(BaseModel):
    """Response for model preload request."""

    status: str
    message: str
