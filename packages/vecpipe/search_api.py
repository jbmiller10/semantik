#!/usr/bin/env python3
"""Thin entrypoint for the vecpipe search API."""

import vecpipe.model_manager as model_manager
from shared.config import settings
from shared.embedding.service import get_embedding_service
from shared.metrics.prometheus import start_metrics_server
from vecpipe.search.app import app, create_app
from vecpipe.search.lifespan import lifespan
from vecpipe.search.metrics import (
    embedding_generation_latency,
    get_or_create_metric,
    search_errors,
    search_latency,
    search_requests,
)
from vecpipe.search.schemas import (
    EmbedRequest,
    EmbedResponse,
    PointPayload,
    UpsertPoint,
    UpsertRequest,
    UpsertResponse,
)
from vecpipe.search.service import (
    generate_embedding_async,
    generate_mock_embedding,
    perform_batch_search,
    perform_hybrid_search,
    perform_keyword_search,
    perform_search,
    embed_texts,
    upsert_points,
)


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
    "model_manager",
    "get_embedding_service",
    "start_metrics_server",
]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=settings.SEARCH_API_PORT,
        reload=False,
        log_level="info",
    )
