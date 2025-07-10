"""Search API contracts for unified search functionality."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, validator


class SearchRequest(BaseModel):
    """Unified search API request model."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    # Use 'k' as the canonical field, but accept 'top_k' as an alias for backward compatibility
    k: int = Field(default=10, ge=1, le=100, description="Number of results", alias="top_k")
    search_type: str = Field("semantic", description="Type of search: semantic, question, code, hybrid, vector")
    model_name: Optional[str] = Field(None, description="Override embedding model")
    quantization: Optional[str] = Field(None, description="Override quantization: float32, float16, int8")
    filters: Optional[dict[str, Any]] = Field(None, description="Metadata filters for search")
    include_content: bool = Field(False, description="Include chunk content in results")
    collection: Optional[str] = Field(None, description="Collection name (e.g., job_123)")
    job_id: Optional[str] = Field(None, description="Job ID for collection inference")
    use_reranker: bool = Field(False, description="Enable cross-encoder reranking")
    rerank_model: Optional[str] = Field(None, description="Override reranker model")
    rerank_quantization: Optional[str] = Field(
        None, description="Override reranker quantization: float32, float16, int8"
    )
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum score threshold")
    # Hybrid search specific parameters
    hybrid_alpha: float = Field(0.7, ge=0.0, le=1.0, description="Weight for hybrid search (vector vs keyword)")
    hybrid_mode: str = Field("rerank", description="Hybrid search mode: 'filter' or 'rerank'")
    keyword_mode: str = Field("any", description="Keyword matching: 'any' or 'all'")

    class Config:
        populate_by_name = True  # Allow both 'k' and 'top_k'

    @validator("query")
    def clean_query(cls, v: str) -> str:
        """Clean and validate query string."""
        return v.strip()

    @validator("search_type")
    def validate_search_type(cls, v: str) -> str:
        """Validate search type and map 'vector' to 'semantic'."""
        # Map 'vector' to 'semantic' for backward compatibility
        if v == "vector":
            return "semantic"
        valid_types = {"semantic", "question", "code", "hybrid", "vector"}
        if v not in valid_types:
            raise ValueError(f"Invalid search_type: {v}. Must be one of {valid_types}")
        return v

    @validator("hybrid_mode")
    def validate_hybrid_mode(cls, v: str) -> str:
        """Validate hybrid mode."""
        valid_modes = {"filter", "rerank"}
        if v not in valid_modes:
            raise ValueError(f"Invalid hybrid_mode: {v}. Must be one of {valid_modes}")
        return v

    @validator("keyword_mode")
    def validate_keyword_mode(cls, v: str) -> str:
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
    content: Optional[str] = Field(None, description="Chunk content (if include_content=True)")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)
    highlights: Optional[list[str]] = None
    # Additional fields for frontend compatibility
    file_path: Optional[str] = None  # Duplicate of path for frontend
    file_name: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    job_id: Optional[str] = None


class SearchResponse(BaseModel):
    """Search API response model."""

    query: str
    results: list[SearchResult]
    num_results: int
    search_type: Optional[str] = None
    model_used: Optional[str] = None
    embedding_time_ms: Optional[float] = None
    search_time_ms: Optional[float] = None
    reranking_used: Optional[bool] = None
    reranker_model: Optional[str] = None
    reranking_time_ms: Optional[float] = None
    collection: Optional[str] = None

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
    model_name: Optional[str] = Field(None, description="Override embedding model")
    quantization: Optional[str] = Field(None, description="Override quantization")
    collection: Optional[str] = Field(None, description="Collection name")


class BatchSearchResponse(BaseModel):
    """Batch search response."""

    responses: list[SearchResponse]
    total_time_ms: float


class HybridSearchResult(BaseModel):
    """Hybrid search result with keyword matching information."""

    path: str
    chunk_id: str
    score: float
    doc_id: Optional[str] = None
    matched_keywords: list[str] = Field(default_factory=list)
    keyword_score: Optional[float] = None
    combined_score: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None
    content: Optional[str] = None


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
    job_id: Optional[str] = None
    mode: str = Field(default="filter", description="Hybrid search mode: 'filter' or 'rerank'")
    keyword_mode: str = Field(default="any", description="Keyword matching: 'any' or 'all'")
    score_threshold: Optional[float] = None
    collection: Optional[str] = None
    model_name: Optional[str] = None
    quantization: Optional[str] = None


# Additional models for specialized endpoints


class PreloadModelRequest(BaseModel):
    """Request to preload a model for faster initial searches."""

    model_name: str = Field(..., description="Model name to preload")
    quantization: str = Field(default="float16", description="Quantization type")


class PreloadModelResponse(BaseModel):
    """Response for model preload request."""

    status: str
    message: str
