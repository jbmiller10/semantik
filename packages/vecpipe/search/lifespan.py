"""Application lifespan management for the search API."""

from __future__ import annotations

import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import httpx
from qdrant_client import AsyncQdrantClient

if TYPE_CHECKING:
    from fastapi import FastAPI

from shared.config import settings
from shared.config.internal_api_key import ensure_internal_api_key
from shared.metrics.prometheus import start_metrics_server as _base_start_metrics_server
from shared.plugins.loader import load_plugins
from shared.plugins.registry import PluginSource
from shared.plugins.state import get_disabled_plugin_ids
from vecpipe.governed_model_manager import GovernedModelManager
from vecpipe.memory_governor import create_memory_budget
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

    Plugin state (enabled/disabled) is read from a shared state file written by WebUI.
    This allows VecPipe to respect plugin enable/disable without database access.
    """
    try:
        key = ensure_internal_api_key(settings)
        fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        logger.info("Internal API key configured (fingerprint=%s)", fingerprint)
    except RuntimeError as exc:
        logger.error("Internal API key configuration failed: %s", exc)
        raise

    # Read disabled plugins from state file (written by WebUI)
    disabled_ids = get_disabled_plugin_ids()
    if disabled_ids:
        logger.info("Disabled plugins (from state file): %s", ", ".join(sorted(disabled_ids)))

    # Load plugins, excluding disabled ones
    registry = load_plugins(
        plugin_types={"embedding"},
        disabled_plugin_ids=disabled_ids if disabled_ids else None,
    )
    external_plugins = registry.list_ids(plugin_type="embedding", source=PluginSource.EXTERNAL)
    if external_plugins:
        logger.info("Loaded embedding plugins: %s", ", ".join(external_plugins))

    start_metrics = _resolve_start_metrics_server()
    start_metrics(settings.METRICS_PORT)
    logger.info("Metrics server started on port %s", settings.METRICS_PORT)

    # Build Qdrant connection with optional API key authentication
    qdrant_headers = {}
    if settings.QDRANT_API_KEY:
        qdrant_headers["api-key"] = settings.QDRANT_API_KEY

    qdrant = httpx.AsyncClient(
        base_url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
        timeout=httpx.Timeout(60.0),
        headers=qdrant_headers,
    )
    qdrant_sdk = AsyncQdrantClient(
        url=f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
        api_key=settings.QDRANT_API_KEY,
    )
    logger.info("Connected to Qdrant at %s:%s", settings.QDRANT_HOST, settings.QDRANT_PORT)

    unload_after = settings.MODEL_UNLOAD_AFTER_SECONDS

    # Initialize model manager - use GovernedModelManager if memory governor is enabled
    if settings.ENABLE_MEMORY_GOVERNOR:
        # Build memory budget from settings using factory for auto-detection
        budget = create_memory_budget(
            total_gpu_mb=None,  # Auto-detect GPU memory via factory
            gpu_reserve_percent=settings.GPU_MEMORY_RESERVE_PERCENT,
            gpu_max_percent=settings.GPU_MEMORY_MAX_PERCENT,
            cpu_reserve_percent=settings.CPU_MEMORY_RESERVE_PERCENT,
            cpu_max_percent=settings.CPU_MEMORY_MAX_PERCENT,
        )

        model_mgr = GovernedModelManager(
            unload_after_seconds=unload_after,
            budget=budget,
            enable_cpu_offload=settings.ENABLE_CPU_OFFLOAD,
            enable_preemptive_eviction=True,
            eviction_idle_threshold_seconds=settings.EVICTION_IDLE_THRESHOLD_SECONDS,
        )

        # Start the governor's background monitor
        await model_mgr.start()

        logger.info(
            "Initialized GovernedModelManager with memory governor "
            "(gpu_budget=%dMB, cpu_offload=%s, eviction_threshold=%ds)",
            budget.usable_gpu_mb,
            settings.ENABLE_CPU_OFFLOAD,
            settings.EVICTION_IDLE_THRESHOLD_SECONDS,
        )
    else:
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
        # Use async shutdown for GovernedModelManager to avoid deadlock
        if hasattr(model_mgr, "shutdown_async"):
            await model_mgr.shutdown_async()
        else:
            model_mgr.shutdown()
        pool.shutdown(wait=True)
        clear_resources()
        logger.info("Disconnected from Qdrant")
