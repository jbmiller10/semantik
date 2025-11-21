"""Application lifespan management for the search API."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx
from fastapi import FastAPI

from shared.config import settings
from shared.embedding.service import get_embedding_service
from shared.metrics.prometheus import start_metrics_server as _base_start_metrics_server
from vecpipe.model_manager import ModelManager
from vecpipe.search.metrics import search_requests
from vecpipe.search.state import clear_resources, set_resources

logger = logging.getLogger(__name__)


def _resolve_start_metrics_server() -> Any:
    """Return a metrics starter function, honoring any patch on search_api."""

    try:
        import vecpipe.search_api as search_api

        patched = getattr(search_api, "start_metrics_server", None)
        if patched:
            return patched
    except Exception:
        pass

    return _base_start_metrics_server


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:  # noqa: ARG001
    """Manage application lifecycle."""
    start_metrics = _resolve_start_metrics_server()
    start_metrics(settings.METRICS_PORT)
    logger.info("Metrics server started on port %s", settings.METRICS_PORT)

    qdrant = httpx.AsyncClient(
        base_url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}", timeout=httpx.Timeout(60.0)
    )
    logger.info("Connected to Qdrant at %s:%s", settings.QDRANT_HOST, settings.QDRANT_PORT)

    unload_after = settings.MODEL_UNLOAD_AFTER_SECONDS
    model_mgr = ModelManager(unload_after_seconds=unload_after)
    logger.info("Initialized model manager with %ss inactivity timeout", unload_after)

    base_service = await get_embedding_service(config=settings)
    if not settings.USE_MOCK_EMBEDDINGS:
        await base_service.initialize(
            model_name=settings.DEFAULT_EMBEDDING_MODEL,
            quantization=settings.DEFAULT_QUANTIZATION,
            allow_quantization_fallback=True,
        )
        logger.info("Initialized embedding service with model: %s", settings.DEFAULT_EMBEDDING_MODEL)
    else:
        await base_service.initialize(model_name="mock", mock_mode=True)
        logger.info("Initialized embedding service in mock mode")

    class LegacyEmbeddingServiceWrapper:
        def __init__(self, service: Any):
            self._service = service
            self.mock_mode = settings.USE_MOCK_EMBEDDINGS
            self.allow_quantization_fallback = True

        @property
        def current_model_name(self) -> str | None:
            return getattr(self._service, "model_name", None)

        @property
        def current_quantization(self) -> str:
            return getattr(self._service, "quantization", "float32")

        @property
        def current_model(self) -> Any:
            return getattr(self._service, "model", None)

        @property
        def current_tokenizer(self) -> Any:
            return getattr(self._service, "tokenizer", None)

        @property
        def device(self) -> str:
            return getattr(self._service, "device", "cpu")

        @property
        def is_initialized(self) -> bool:
            return getattr(self._service, "is_initialized", False)

        def get_model_info(self, model_name: str | None = None, quantization: str | None = None) -> dict[str, Any]:
            if hasattr(self._service, "get_model_info"):
                try:
                    if model_name is not None and quantization is not None:
                        return cast(dict[str, Any], self._service.get_model_info(model_name, quantization))
                    return cast(dict[str, Any], self._service.get_model_info())
                except TypeError:
                    if model_name is None and quantization is None:
                        return cast(
                            dict[str, Any],
                            self._service.get_model_info(
                                self.current_model_name or "unknown", self.current_quantization or "float32"
                            ),
                        )
                    return cast(dict[str, Any], self._service.get_model_info())
            return {"model_name": model_name or self.current_model_name, "dimension": 1024}

        def __getattr__(self, name: str) -> Any:
            return getattr(self._service, name)

    embed_service = LegacyEmbeddingServiceWrapper(base_service)
    logger.info("Created embedding service wrapper (mock_mode=%s)", settings.USE_MOCK_EMBEDDINGS)

    pool = ThreadPoolExecutor(max_workers=4)
    set_resources(qdrant=qdrant, model_mgr=model_mgr, embed_service=embed_service, pool=pool)

    # Touch metrics to ensure registered
    search_requests.labels(endpoint="startup", search_type="health").inc()

    try:
        yield
    finally:
        await qdrant.aclose()
        model_mgr.shutdown()
        pool.shutdown(wait=True)
        clear_resources()
        logger.info("Disconnected from Qdrant")
