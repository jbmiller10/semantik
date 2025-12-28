"""Application lifespan management for the search API."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import httpx
from qdrant_client import AsyncQdrantClient

if TYPE_CHECKING:
    from fastapi import FastAPI

from shared.config import settings
from shared.embedding.plugin_loader import ensure_providers_registered, load_embedding_plugins
from shared.metrics.prometheus import start_metrics_server as _base_start_metrics_server
from vecpipe.model_manager import ModelManager
from vecpipe.search.metrics import search_requests
from vecpipe.search.state import clear_resources, set_resources

logger = logging.getLogger(__name__)


def _resolve_start_metrics_server() -> Any:
    """Return a metrics starter function, honoring any patch on search_api."""
    # 1) If this module's symbol is patched (common in unit tests), use it.
    patched_local = globals().get("start_metrics_server")
    if patched_local and patched_local is not _base_start_metrics_server:
        return patched_local

    # 2) If the public entrypoint module has been patched, honor that.
    try:
        import vecpipe.search_api as search_api

        patched = getattr(search_api, "start_metrics_server", None)
        if patched and patched is not _base_start_metrics_server:
            return patched
    except Exception as e:
        logger.warning("Failed to check for patched start_metrics_server: %s", e, exc_info=True)

    # 3) Fall back to the base implementation
    return _base_start_metrics_server


# Expose a patchable reference for unit tests that mock vecpipe.search.lifespan.start_metrics_server
start_metrics_server = _base_start_metrics_server


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:  # noqa: ARG001
    """Manage application lifecycle.

    ModelManager now handles embedding provider lifecycle internally using the
    plugin-aware provider system. No separate embedding service initialization needed.
    """
    # Ensure embedding providers are registered before use
    ensure_providers_registered()
    logger.info("Built-in embedding providers registered")

    # Load any external embedding plugins
    registered_plugins = load_embedding_plugins()
    if registered_plugins:
        logger.info("Loaded embedding plugins: %s", ", ".join(registered_plugins))

    start_metrics = _resolve_start_metrics_server()
    start_metrics(settings.METRICS_PORT)
    logger.info("Metrics server started on port %s", settings.METRICS_PORT)

    qdrant = httpx.AsyncClient(
        base_url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}", timeout=httpx.Timeout(60.0)
    )
    qdrant_sdk = AsyncQdrantClient(url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    logger.info("Connected to Qdrant at %s:%s", settings.QDRANT_HOST, settings.QDRANT_PORT)

    unload_after = settings.MODEL_UNLOAD_AFTER_SECONDS
    model_mgr = ModelManager(unload_after_seconds=unload_after)
    logger.info(
        "Initialized model manager with %ss inactivity timeout (mock_mode=%s)",
        unload_after,
        settings.USE_MOCK_EMBEDDINGS,
    )

    pool = ThreadPoolExecutor(max_workers=4)

    # embed_service is None - ModelManager now manages providers internally
    set_resources(qdrant=qdrant, model_mgr=model_mgr, embed_service=None, pool=pool, qdrant_sdk=qdrant_sdk)

    # Touch metrics to ensure registered
    search_requests.labels(endpoint="startup", search_type="health").inc()

    try:
        yield
    finally:
        await qdrant.aclose()
        await qdrant_sdk.close()
        model_mgr.shutdown()
        pool.shutdown(wait=True)
        clear_resources()
        logger.info("Disconnected from Qdrant")
