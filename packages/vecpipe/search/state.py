"""Shared mutable state for the search service runtime."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

qdrant_client: httpx.AsyncClient | None = None
sdk_client: Any | None = None  # Will be AsyncQdrantClient for SDK operations (metadata scroll)
model_manager: Any | None = None  # Will be ModelManager once imported at runtime
embedding_service: Any | None = None  # Will be BaseEmbeddingService once initialized
executor: ThreadPoolExecutor | None = None


def set_resources(
    *,
    qdrant: httpx.AsyncClient,
    model_mgr: Any,
    embed_service: Any,
    pool: ThreadPoolExecutor,
    qdrant_sdk: Any | None = None,
) -> None:
    """Record the runtime resources for later retrieval."""
    global qdrant_client, sdk_client, model_manager, embedding_service, executor
    qdrant_client = qdrant
    sdk_client = qdrant_sdk
    model_manager = model_mgr
    embedding_service = embed_service
    executor = pool


def clear_resources() -> None:
    """Reset all stored resources (used during shutdown/testing)."""
    global qdrant_client, sdk_client, model_manager, embedding_service, executor
    qdrant_client = None
    sdk_client = None
    model_manager = None
    embedding_service = None
    executor = None


__all__ = [
    "qdrant_client",
    "sdk_client",
    "model_manager",
    "embedding_service",
    "executor",
    "set_resources",
    "clear_resources",
]
