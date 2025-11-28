#!/usr/bin/env python3
"""Thin entrypoint for the vecpipe search API."""

import sys
import types
from typing import Any

from shared.config import settings
from shared.embedding.service import get_embedding_service
from shared.metrics.prometheus import start_metrics_server
from vecpipe.hybrid_search import HybridSearchEngine
from vecpipe.qwen3_search_config import get_reranker_for_embedding_model
from vecpipe.search import state as search_state
from vecpipe.search.app import app, create_app
from vecpipe.search.lifespan import lifespan
from vecpipe.search.metrics import (
    embedding_generation_latency,
    get_or_create_metric,
    search_errors,
    search_latency,
    search_requests,
)
from vecpipe.search.router import batch_search, hybrid_search, keyword_search, search_post
from vecpipe.search.schemas import EmbedRequest, EmbedResponse, PointPayload, UpsertPoint, UpsertRequest, UpsertResponse
from vecpipe.search.service import (
    embed_texts,
    generate_embedding_async,
    generate_mock_embedding,
    perform_batch_search,
    perform_hybrid_search,
    perform_keyword_search,
    perform_search,
    upsert_points,
)
from vecpipe.search_utils import search_qdrant

# Placeholder bindings to satisfy export contract; actual values live in ``vecpipe.search.state``.
model_manager = None
state_model_manager = None
qdrant_client = None
embedding_service = None
executor = None

__all__ = [
    "app",
    "create_app",
    "lifespan",
    "generate_mock_embedding",
    "generate_embedding_async",
    "perform_search",
    "perform_hybrid_search",
    "perform_keyword_search",
    "perform_batch_search",
    "embed_texts",
    "upsert_points",
    "EmbedRequest",
    "EmbedResponse",
    "PointPayload",
    "UpsertPoint",
    "UpsertRequest",
    "UpsertResponse",
    "search_latency",
    "search_errors",
    "search_requests",
    "embedding_generation_latency",
    "get_or_create_metric",
    "get_embedding_service",
    "start_metrics_server",
    "batch_search",
    "hybrid_search",
    "keyword_search",
    "search_post",
    "model_manager",
    "state_model_manager",
    "qdrant_client",
    "embedding_service",
    "executor",
    "search_qdrant",
    "HybridSearchEngine",
    "get_reranker_for_embedding_model",
    "search_state",
]

_FORWARDED_ATTRS = {
    "qdrant_client": "qdrant_client",
    "embedding_service": "embedding_service",
    "executor": "executor",
    "model_manager": "model_manager",
    # Kept for backward compatibility; mirrors ``model_manager``.
    "state_model_manager": "model_manager",
}


class _SearchApiModule(types.ModuleType):
    """Module wrapper that keeps public globals synced with ``vecpipe.search.state``."""

    def __getattribute__(self, name: str) -> Any:
        target = _FORWARDED_ATTRS.get(name)
        if target:
            return getattr(search_state, target)
        return super().__getattribute__(name)

    def __getattr__(self, name: str) -> Any:
        target = _FORWARDED_ATTRS.get(name)
        if target:
            return getattr(search_state, target)
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        target = _FORWARDED_ATTRS.get(name)
        if target:
            setattr(search_state, target, value)
            # Avoid shadowing forwarded attributes in the module dict
            if name in self.__dict__:
                super().__delattr__(name)
            return
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        target = _FORWARDED_ATTRS.get(name)
        if target:
            setattr(search_state, target, None)
            if name in self.__dict__:
                super().__delattr__(name)
            return
        super().__delattr__(name)


# Ensure attribute access on this module always mirrors the shared runtime state.
sys.modules[__name__].__class__ = _SearchApiModule


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.SEARCH_API_PORT,
        reload=False,
        log_level="info",
    )
