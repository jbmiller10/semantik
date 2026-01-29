"""Local Pydantic schemas used by the vecpipe search API."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    """Request model for batch embedding generation."""

    texts: list[str] = Field(..., min_length=1, max_length=1000, description="List of texts to embed")
    model_name: str = Field(..., description="Embedding model name")
    quantization: str = Field("float32", description="Model quantization type: float32, float16, int8")
    instruction: str | None = Field(None, description="Optional instruction for embedding generation")
    mode: Literal["query", "document"] | None = Field(
        None, description="Embedding mode: 'query' for search queries, 'document' for indexing. Defaults to 'query'."
    )
    batch_size: int = Field(32, ge=1, le=256, description="Batch size for processing")


class EmbedResponse(BaseModel):
    """Response model for batch embedding generation."""

    embeddings: list[list[float]] = Field(..., description="List of embedding vectors")
    model_used: str = Field(..., description="Model and quantization used")
    embedding_time_ms: float | None = Field(None, description="Time taken to generate embeddings in milliseconds")
    batch_count: int = Field(..., description="Number of batches processed")


class PointPayload(BaseModel):
    """Payload structure for Qdrant points."""

    doc_id: str
    chunk_id: str
    path: str
    content: str | None = None
    metadata: dict[str, Any] | None = None
    collection_id: str | None = None
    chunk_index: int | None = None
    total_chunks: int | None = None
    path_id: str | None = Field(
        None,
        description="Pipeline path that produced this chunk (for parallel fan-out)",
    )


class UpsertPoint(BaseModel):
    """Individual point for upserting to Qdrant."""

    id: str = Field(..., description="Unique point ID")
    vector: list[float] = Field(..., description="Embedding vector")
    payload: PointPayload = Field(..., description="Point metadata")


class UpsertRequest(BaseModel):
    """Request model for Qdrant upsert operation."""

    collection_name: str = Field(..., description="Target collection name")
    points: list[UpsertPoint] = Field(..., min_length=1, max_length=1000, description="Points to upsert")
    wait: bool = Field(True, description="Wait for operation to complete")


class UpsertResponse(BaseModel):
    """Response model for Qdrant upsert operation."""

    status: str = Field(..., description="Operation status")
    points_upserted: int = Field(..., description="Number of points successfully upserted")
    collection_name: str = Field(..., description="Target collection name")
    upsert_time_ms: float | None = Field(None, description="Time taken for upsert operation in milliseconds")


__all__ = [
    "EmbedRequest",
    "EmbedResponse",
    "PointPayload",
    "UpsertPoint",
    "UpsertRequest",
    "UpsertResponse",
]
