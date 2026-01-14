"""
System status and capabilities API endpoints.

This module provides endpoints for checking system capabilities like GPU availability,
supported features, and service health information.
"""

import asyncio
import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx
from fastapi import APIRouter, Depends

from shared.config import settings as shared_settings
from webui.api.health import (
    _check_database_health,
    _check_qdrant_health,
    _check_redis_health,
    _check_search_api_health,
)
from webui.auth import get_current_user
from webui.config.rate_limits import RateLimitConfig

logger = logging.getLogger(__name__)

# Version from pyproject.toml via package metadata
try:
    APP_VERSION = version("semantik")
except PackageNotFoundError:
    APP_VERSION = "0.0.0-dev"

router = APIRouter(prefix="/api/v2/system", tags=["system-v2"])

VECPIPE_BASE_URL = "http://vecpipe:8000"
AVAILABLE_RERANKING_MODELS = [
    "Qwen/Qwen3-Reranker-0.6B",
    "Qwen/Qwen3-Reranker-4B",
    "Qwen/Qwen3-Reranker-8B",
]


@router.get("/status")
async def get_system_status(
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> dict[str, Any]:
    """
    Get system status including GPU availability and reranking capabilities.

    Queries the vecpipe service to determine actual GPU availability since
    the webui container doesn't have direct GPU access.

    Returns information about:
    - GPU availability (from vecpipe)
    - Reranking model support
    - Available reranking models
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Use /models/suggest endpoint which checks actual GPU hardware availability
            # via memory detection, not just configuration flags
            response = await client.get(f"{VECPIPE_BASE_URL}/models/suggest")
            response.raise_for_status()
            suggest_response = response.json()

        # The suggest endpoint returns gpu_available based on actual hardware detection
        gpu_available = suggest_response.get("gpu_available", False)
        gpu_memory = suggest_response.get("gpu_memory", {})

        return {
            "gpu_available": gpu_available,
            "reranking_available": gpu_available,
            "available_reranking_models": AVAILABLE_RERANKING_MODELS if gpu_available else [],
            "cuda_device_count": 1 if gpu_available else 0,
            "cuda_device_name": f"GPU ({gpu_memory.get('total_mb', 0)}MB)" if gpu_available else None,
        }
    except httpx.RequestError as e:
        logger.warning("Could not reach vecpipe service: %s", e, exc_info=True)
        return {
            "gpu_available": False,
            "reranking_available": False,
            "available_reranking_models": [],
            "cuda_device_count": 0,
            "cuda_device_name": None,
        }
    except Exception as e:
        logger.error("Error checking system status: %s", e, exc_info=True)
        return {
            "gpu_available": False,
            "reranking_available": False,
            "available_reranking_models": [],
            "cuda_device_count": 0,
            "cuda_device_name": None,
        }


@router.get("/info")
async def get_system_info() -> dict[str, Any]:
    """
    Get system information including version, environment, and limits.

    This endpoint does not require authentication for transparency.
    Returns static configuration information about the system.
    """
    return {
        "version": APP_VERSION,
        "environment": shared_settings.ENVIRONMENT,
        "python_version": sys.version.split()[0],
        "limits": {
            "max_collections_per_user": shared_settings.MAX_COLLECTIONS_PER_USER,
            "max_storage_gb_per_user": shared_settings.MAX_STORAGE_GB_PER_USER,
        },
        "rate_limits": {
            "chunking_preview": RateLimitConfig.PREVIEW_RATE,
            "plugin_install": RateLimitConfig.PLUGIN_INSTALL_RATE,
            "llm_test": RateLimitConfig.LLM_TEST_RATE,
        },
    }


@router.get("/health")
async def get_system_health() -> dict[str, Any]:
    """
    Get health status for all backend services.

    Always returns 200 with per-service status. This allows the frontend
    to display partial results even when some services are unhealthy.

    Checks: PostgreSQL, Redis, Qdrant, VecPipe
    """
    # Run all health checks concurrently
    results = await asyncio.gather(
        _check_database_health(),
        _check_redis_health(),
        _check_qdrant_health(),
        _check_search_api_health(),
        return_exceptions=True,
    )

    service_names = ["postgres", "redis", "qdrant", "vecpipe"]
    services: dict[str, dict[str, Any]] = {}

    for name, result in zip(service_names, results, strict=True):
        if isinstance(result, BaseException):
            services[name] = {"status": "unhealthy", "message": f"Health check failed: {result!s}"}
        else:
            services[name] = result

    return services
