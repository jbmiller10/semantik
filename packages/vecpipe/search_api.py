#!/usr/bin/env python3
"""Public entrypoint for the VecPipe search API.

This module is intentionally thin: it exposes the FastAPI app factory and a
small set of helpers for backwards compatibility. Runtime wiring happens via
FastAPI lifespan and `app.state.vecpipe_runtime`.
"""

# ruff: noqa: E402

import os
import sys
from pathlib import Path

# Add plugins directory to sys.path AFTER site-packages so app packages take precedence.
_plugins_dir = os.environ.get("SEMANTIK_PLUGINS_DIR", "/app/plugins")
if Path(_plugins_dir).is_dir() and _plugins_dir not in sys.path:
    sys.path.append(_plugins_dir)

from shared.config import settings
from shared.embedding.service import get_embedding_service
from shared.metrics.prometheus import start_metrics_server
from vecpipe.qwen3_search_config import get_reranker_for_embedding_model
from vecpipe.search.app import app, create_app
from vecpipe.search.lifespan import lifespan
from vecpipe.search.metrics import (
    embedding_generation_latency,
    get_or_create_metric,
    search_errors,
    search_latency,
    search_requests,
)
from vecpipe.search.router import batch_search, search_post
from vecpipe.search.schemas import EmbedRequest, EmbedResponse, PointPayload, UpsertPoint, UpsertRequest, UpsertResponse
from vecpipe.search.service import (
    embed_texts,
    generate_mock_embedding,
    perform_batch_search,
    perform_search,
    upsert_points,
)
from vecpipe.search_utils import search_qdrant

__all__ = [
    "app",
    "create_app",
    "lifespan",
    "generate_mock_embedding",
    "perform_search",
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
    "search_post",
    "search_qdrant",
    "get_reranker_for_embedding_model",
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
