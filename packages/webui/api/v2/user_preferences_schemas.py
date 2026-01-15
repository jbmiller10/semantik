"""Pydantic schemas for user preferences API endpoints."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 - Required at runtime for Pydantic
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SearchPreferences(BaseModel):
    """User preferences for search behavior."""

    top_k: int = Field(default=10, ge=1, le=250, description="Number of results to return")
    mode: Literal["dense", "sparse", "hybrid"] = Field(default="dense", description="Search mode")
    use_reranker: bool = Field(default=False, description="Enable reranking")
    rrf_k: int = Field(default=60, ge=1, le=100, description="RRF constant for hybrid fusion")
    similarity_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum similarity score (null for no threshold)"
    )
    hyde_enabled_default: bool = Field(default=False, description="Enable HyDE by default")
    hyde_llm_tier: Literal["high", "low"] = Field(default="low", description="LLM tier for HyDE generation")

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "top_k": 10,
                "mode": "dense",
                "use_reranker": False,
                "rrf_k": 60,
                "similarity_threshold": None,
            }
        },
    )


class CollectionDefaults(BaseModel):
    """User defaults for new collection creation."""

    embedding_model: str | None = Field(
        default=None,
        max_length=128,
        description="Default embedding model (null for system default)",
    )
    quantization: Literal["float32", "float16", "int8"] = Field(default="float16", description="Model precision type")
    chunking_strategy: Literal["character", "recursive", "markdown", "semantic"] = Field(
        default="recursive", description="Text chunking strategy"
    )
    chunk_size: int = Field(default=1024, ge=256, le=4096, description="Chunk size in characters")
    chunk_overlap: int = Field(default=200, ge=0, le=512, description="Chunk overlap in characters")
    enable_sparse: bool = Field(default=False, description="Enable sparse indexing")
    sparse_type: Literal["bm25", "splade"] = Field(default="bm25", description="Sparse indexer type")
    enable_hybrid: bool = Field(default=False, description="Enable hybrid search (requires sparse indexing)")

    @model_validator(mode="after")
    def hybrid_requires_sparse(self) -> Self:
        """Validate that hybrid search requires sparse indexing."""
        if self.enable_hybrid and not self.enable_sparse:
            msg = "enable_hybrid requires enable_sparse to be true"
            raise ValueError(msg)
        return self

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "embedding_model": None,
                "quantization": "float16",
                "chunking_strategy": "recursive",
                "chunk_size": 1024,
                "chunk_overlap": 200,
                "enable_sparse": False,
                "sparse_type": "bm25",
                "enable_hybrid": False,
            }
        },
    )


class InterfacePreferences(BaseModel):
    """User preferences for UI behavior."""

    data_refresh_interval_ms: int = Field(
        default=30000,
        ge=10000,
        le=60000,
        description="Data polling interval in milliseconds (10s-60s)",
    )
    visualization_sample_limit: int = Field(
        default=200000,
        ge=10000,
        le=500000,
        description="Maximum points for UMAP/PCA visualizations (10K-500K)",
    )
    animation_enabled: bool = Field(
        default=True,
        description="Enable UI animations",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "data_refresh_interval_ms": 30000,
                "visualization_sample_limit": 200000,
                "animation_enabled": True,
            }
        },
    )


class UserPreferencesResponse(BaseModel):
    """Response for GET /preferences."""

    search: SearchPreferences
    collection_defaults: CollectionDefaults
    interface: InterfacePreferences
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "search": {
                    "top_k": 10,
                    "mode": "dense",
                    "use_reranker": False,
                    "rrf_k": 60,
                    "similarity_threshold": None,
                },
                "collection_defaults": {
                    "embedding_model": None,
                    "quantization": "float16",
                    "chunking_strategy": "recursive",
                    "chunk_size": 1024,
                    "chunk_overlap": 200,
                    "enable_sparse": False,
                    "sparse_type": "bm25",
                    "enable_hybrid": False,
                },
                "interface": {
                    "data_refresh_interval_ms": 30000,
                    "visualization_sample_limit": 200000,
                    "animation_enabled": True,
                },
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            }
        },
    )


class UserPreferencesUpdate(BaseModel):
    """Request body for updating user preferences (partial update)."""

    search: SearchPreferences | None = None
    collection_defaults: CollectionDefaults | None = None
    interface: InterfacePreferences | None = None

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "search": {"top_k": 20, "mode": "hybrid"},
                "collection_defaults": {"chunk_size": 512},
                "interface": {"animation_enabled": False},
            }
        },
    )
