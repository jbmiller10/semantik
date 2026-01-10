"""HTTP router for sparse encoding endpoints.

This router provides endpoints for encoding documents and queries to sparse
vectors using sparse indexer plugins (BM25, SPLADE). The endpoints are
protected by the internal API key and integrate with the memory governor.

Endpoints:
- POST /sparse/encode: Encode documents to sparse vectors
- POST /sparse/query: Encode a query to sparse vector
- GET /sparse/plugins: List available sparse indexer plugins
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from shared.plugins.loader import load_plugins
from shared.plugins.registry import plugin_registry

from . import state as search_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sparse", tags=["sparse"])


def _get_internal_api_key() -> str:
    """Get the internal API key from settings."""
    from shared.config import settings

    return settings.INTERNAL_API_KEY


def require_internal_api_key(
    x_internal_api_key: str | None = Header(default=None, alias="X-Internal-Api-Key"),
) -> None:
    """Verify the internal API key for protected endpoints."""
    expected_key = _get_internal_api_key()
    if not expected_key:
        raise HTTPException(status_code=500, detail="Internal API key is not configured")
    if not x_internal_api_key or not secrets.compare_digest(x_internal_api_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing internal API key")


# === Request/Response Models ===


class SparseEncodeRequest(BaseModel):
    """Request to encode documents to sparse vectors."""

    texts: list[str] = Field(..., description="Document texts to encode")
    chunk_ids: list[str] = Field(..., description="Chunk IDs for each text (must match texts length)")
    plugin_id: str = Field(..., description="Sparse indexer plugin ID (e.g., 'splade-local', 'bm25-local')")
    model_config_data: dict[str, Any] | None = Field(
        default=None,
        description="Plugin-specific configuration (model_name, quantization, batch_size, etc.)",
    )


class SparseVectorResult(BaseModel):
    """Sparse vector representation for a single document."""

    chunk_id: str = Field(..., description="Chunk ID this vector belongs to")
    indices: list[int] = Field(..., description="Token/term indices in vocabulary")
    values: list[float] = Field(..., description="Weight/importance values for each index")


class SparseEncodeResponse(BaseModel):
    """Response with encoded sparse vectors."""

    vectors: list[SparseVectorResult] = Field(..., description="Encoded sparse vectors")
    plugin_id: str = Field(..., description="Plugin ID used for encoding")
    encoding_time_ms: float = Field(..., description="Time taken to encode in milliseconds")
    document_count: int = Field(..., description="Number of documents encoded")


class SparseQueryRequest(BaseModel):
    """Request to encode a query to sparse vector."""

    query: str = Field(..., description="Query text to encode")
    plugin_id: str = Field(..., description="Sparse indexer plugin ID")
    model_config_data: dict[str, Any] | None = Field(
        default=None,
        description="Plugin-specific configuration",
    )


class SparseQueryResponse(BaseModel):
    """Response with sparse query vector."""

    indices: list[int] = Field(..., description="Token/term indices in vocabulary")
    values: list[float] = Field(..., description="Weight/importance values for each index")
    encoding_time_ms: float = Field(..., description="Time taken to encode in milliseconds")


class SparsePluginInfo(BaseModel):
    """Information about an available sparse indexer plugin."""

    plugin_id: str = Field(..., description="Unique plugin identifier")
    plugin_type: str = Field(..., description="Plugin type (always 'sparse_indexer')")
    display_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Plugin description")
    sparse_type: str = Field(..., description="Sparse vector type ('bm25' or 'splade')")
    requires_gpu: bool = Field(..., description="Whether plugin requires GPU")


class SparsePluginsResponse(BaseModel):
    """Response listing available sparse indexer plugins."""

    plugins: list[SparsePluginInfo] = Field(..., description="Available plugins")


# === Endpoints ===


@router.post("/encode", response_model=SparseEncodeResponse, dependencies=[Depends(require_internal_api_key)])
async def encode_documents(request: SparseEncodeRequest) -> SparseEncodeResponse:
    """Encode documents to sparse vectors using specified plugin.

    This endpoint encodes a batch of documents to sparse vectors using the
    specified sparse indexer plugin. The plugin is loaded on demand and
    managed by the memory governor for GPU memory coordination.

    Args:
        request: Encode request with texts, chunk_ids, and plugin configuration.

    Returns:
        SparseEncodeResponse with encoded vectors and timing information.

    Raises:
        HTTPException 400: If texts and chunk_ids have different lengths.
        HTTPException 404: If plugin not found.
        HTTPException 500: If encoding fails.
        HTTPException 507: If insufficient GPU memory.
    """
    # Validate input
    if len(request.texts) != len(request.chunk_ids):
        raise HTTPException(
            status_code=400,
            detail=f"texts ({len(request.texts)}) and chunk_ids ({len(request.chunk_ids)}) must have same length",
        )

    # Get sparse manager
    sparse_manager = search_state.sparse_manager
    if sparse_manager is None:
        raise HTTPException(status_code=503, detail="Sparse model manager not initialized")

    # Encode documents
    start_time = time.time()
    try:
        vectors = await sparse_manager.encode_documents(
            plugin_id=request.plugin_id,
            texts=request.texts,
            chunk_ids=request.chunk_ids,
            config=request.model_config_data,
        )
    except ValueError as e:
        # Plugin not found or validation error
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        error_msg = str(e)
        if "Cannot allocate memory" in error_msg:
            raise HTTPException(status_code=507, detail=error_msg) from e
        raise HTTPException(status_code=500, detail=error_msg) from e

    encoding_time_ms = (time.time() - start_time) * 1000

    # Convert to response format
    vector_results = [
        SparseVectorResult(
            chunk_id=v.chunk_id,
            indices=list(v.indices),
            values=list(v.values),
        )
        for v in vectors
    ]

    return SparseEncodeResponse(
        vectors=vector_results,
        plugin_id=request.plugin_id,
        encoding_time_ms=encoding_time_ms,
        document_count=len(vectors),
    )


@router.post("/query", response_model=SparseQueryResponse, dependencies=[Depends(require_internal_api_key)])
async def encode_query(request: SparseQueryRequest) -> SparseQueryResponse:
    """Encode a search query to sparse vector.

    This endpoint encodes a single query to a sparse vector for use in
    sparse or hybrid search. The plugin is loaded on demand and managed
    by the memory governor.

    Args:
        request: Query encode request with query text and plugin configuration.

    Returns:
        SparseQueryResponse with sparse query vector and timing.

    Raises:
        HTTPException 404: If plugin not found.
        HTTPException 500: If encoding fails.
        HTTPException 507: If insufficient GPU memory.
    """
    # Get sparse manager
    sparse_manager = search_state.sparse_manager
    if sparse_manager is None:
        raise HTTPException(status_code=503, detail="Sparse model manager not initialized")

    # Encode query
    start_time = time.time()
    try:
        vector = await sparse_manager.encode_query(
            plugin_id=request.plugin_id,
            query=request.query,
            config=request.model_config_data,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        error_msg = str(e)
        if "Cannot allocate memory" in error_msg:
            raise HTTPException(status_code=507, detail=error_msg) from e
        raise HTTPException(status_code=500, detail=error_msg) from e

    encoding_time_ms = (time.time() - start_time) * 1000

    return SparseQueryResponse(
        indices=list(vector.indices),
        values=list(vector.values),
        encoding_time_ms=encoding_time_ms,
    )


@router.get("/plugins", response_model=SparsePluginsResponse)
async def list_sparse_plugins() -> SparsePluginsResponse:
    """List available sparse indexer plugins.

    Returns information about all registered sparse indexer plugins,
    including their capabilities and GPU requirements.

    Returns:
        SparsePluginsResponse with list of available plugins.
    """
    # Ensure plugins are loaded
    load_plugins(plugin_types={"sparse_indexer"})

    # Get all sparse indexer plugins
    plugins_info = []
    for record in plugin_registry.get_all("sparse_indexer"):
        plugin_cls = record.plugin_class

        # Get manifest for display info
        manifest = plugin_cls.get_manifest() if hasattr(plugin_cls, "get_manifest") else None

        # Determine if GPU required based on sparse_type
        sparse_type = getattr(plugin_cls, "SPARSE_TYPE", "unknown")
        requires_gpu = sparse_type == "splade"

        plugins_info.append(
            SparsePluginInfo(
                plugin_id=record.plugin_id,
                plugin_type=record.plugin_type,
                display_name=manifest.display_name if manifest else record.plugin_id,
                description=manifest.description if manifest else "",
                sparse_type=sparse_type,
                requires_gpu=requires_gpu,
            )
        )

    return SparsePluginsResponse(plugins=plugins_info)


@router.get("/status")
async def sparse_status() -> dict[str, Any]:
    """Get status of sparse model manager.

    Returns information about loaded sparse plugins and their memory usage.

    Returns:
        Dict with loaded plugins and manager status.
    """
    sparse_manager = search_state.sparse_manager
    if sparse_manager is None:
        return {"status": "not_initialized"}

    return {
        "status": "ready",
        "loaded_plugins": sparse_manager.get_loaded_plugins(),
    }
