"""Shared mutable state for the search service runtime."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

qdrant_client: httpx.AsyncClient | None = None
sdk_client: Any | None = None  # Will be AsyncQdrantClient for SDK operations (metadata scroll)
model_manager: Any | None = None  # Will be ModelManager once imported at runtime
embedding_service: Any | None = None  # Will be BaseEmbeddingService once initialized
executor: ThreadPoolExecutor | None = None
sparse_manager: Any | None = None  # Will be SparseModelManager once imported at runtime
llm_manager: Any | None = None  # Will be LLMModelManager once imported at runtime


def set_resources(
    *,
    qdrant: httpx.AsyncClient,
    model_mgr: Any,
    embed_service: Any,
    pool: ThreadPoolExecutor,
    qdrant_sdk: Any | None = None,
    sparse_mgr: Any | None = None,
    llm_mgr: Any | None = None,
) -> None:
    """Record the runtime resources for later retrieval."""
    global qdrant_client, sdk_client, model_manager, embedding_service, executor, sparse_manager, llm_manager
    qdrant_client = qdrant
    sdk_client = qdrant_sdk
    model_manager = model_mgr
    embedding_service = embed_service
    executor = pool
    sparse_manager = sparse_mgr
    llm_manager = llm_mgr


def clear_resources() -> None:
    """Reset all stored resources (used during shutdown/testing)."""
    global qdrant_client, sdk_client, model_manager, embedding_service, executor, sparse_manager, llm_manager
    qdrant_client = None
    sdk_client = None
    model_manager = None
    embedding_service = None
    executor = None
    sparse_manager = None
    llm_manager = None


def get_resources() -> dict[str, Any]:
    """Get all runtime resources as a dictionary.

    Returns:
        Dictionary with keys: qdrant, model_mgr, embed_service, pool, qdrant_sdk, sparse_mgr, llm_mgr
    """
    return {
        "qdrant": qdrant_client,
        "model_mgr": model_manager,
        "embed_service": embedding_service,
        "pool": executor,
        "qdrant_sdk": sdk_client,
        "sparse_mgr": sparse_manager,
        "llm_mgr": llm_manager,
    }


__all__ = [
    "qdrant_client",
    "sdk_client",
    "model_manager",
    "embedding_service",
    "executor",
    "sparse_manager",
    "llm_manager",
    "set_resources",
    "clear_resources",
    "get_resources",
]
