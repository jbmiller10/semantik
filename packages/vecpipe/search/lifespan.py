"""Application lifespan management for the search API."""

from __future__ import annotations

import hashlib
import logging
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
from vecpipe.llm_model_manager import LLMModelManager
from vecpipe.memory_governor import create_memory_budget
from vecpipe.model_manager import ModelManager
from vecpipe.search.metrics import search_requests
from vecpipe.search.runtime import VecpipeRuntime
from vecpipe.sparse_model_manager import SparseModelManager

logger = logging.getLogger(__name__)


# Expose a patchable reference for unit tests that mock vecpipe.search.lifespan.start_metrics_server
start_metrics_server = _base_start_metrics_server


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
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

    start_metrics_server(settings.METRICS_PORT)
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
            gpu_max_percent=settings.GPU_MEMORY_MAX_PERCENT,
            cpu_max_percent=settings.CPU_MEMORY_MAX_PERCENT,
        )

        model_mgr = GovernedModelManager(
            unload_after_seconds=unload_after,
            budget=budget,
            enable_cpu_offload=settings.ENABLE_CPU_OFFLOAD,
            enable_preemptive_eviction=True,
            eviction_idle_threshold_seconds=settings.EVICTION_IDLE_THRESHOLD_SECONDS,
            pressure_check_interval_seconds=settings.PRESSURE_CHECK_INTERVAL_SECONDS,
            probe_mode=settings.GPU_FREE_PROBE_MODE,
            probe_safe_threshold=settings.GPU_FREE_PROBE_SAFE_THRESHOLD_PERCENT,
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

    # Initialize sparse model manager with shared governor (if available)
    governor = model_mgr._governor if hasattr(model_mgr, "_governor") else None
    sparse_mgr = SparseModelManager(governor=governor)
    if governor:
        logger.info("Initialized SparseModelManager with shared memory governor")
    else:
        logger.info("Initialized SparseModelManager without memory governor")

    # Initialize LLM model manager (optional, controlled by ENABLE_LOCAL_LLM)
    llm_mgr: LLMModelManager | None = None
    if settings.ENABLE_LOCAL_LLM:
        llm_mgr = LLMModelManager(governor=governor)
        if governor:
            logger.info("Initialized LLMModelManager with shared memory governor")
        else:
            logger.info("Initialized LLMModelManager without memory governor")
    else:
        logger.info("Local LLM support disabled (ENABLE_LOCAL_LLM=false)")

    # Create runtime container and attach to app.state
    runtime = VecpipeRuntime(
        qdrant_http=qdrant,
        qdrant_sdk=qdrant_sdk,
        model_manager=model_mgr,
        sparse_manager=sparse_mgr,
        llm_manager=llm_mgr,
        executor=pool,
    )
    app.state.vecpipe_runtime = runtime

    # Touch metrics to ensure registered
    search_requests.labels(endpoint="startup", search_type="health").inc()

    try:
        yield
    finally:
        await runtime.aclose()
        logger.info("Disconnected from Qdrant")
