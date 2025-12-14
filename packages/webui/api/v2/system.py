"""
System status and capabilities API endpoints.

This module provides endpoints for checking system capabilities like GPU availability
and supported features by querying the vecpipe service.
"""

import logging
from typing import Any

import httpx
from fastapi import APIRouter, Depends

from webui.auth import get_current_user

logger = logging.getLogger(__name__)

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
            response = await client.get(f"{VECPIPE_BASE_URL}/model/status")
            response.raise_for_status()
            vecpipe_status = response.json()

        # Vecpipe has GPU access - if it responds, check if models can load
        # The is_mock_mode flag indicates if real embeddings are disabled
        is_mock_mode = vecpipe_status.get("is_mock_mode", False)
        gpu_available = not is_mock_mode

        return {
            "gpu_available": gpu_available,
            "reranking_available": gpu_available,
            "available_reranking_models": AVAILABLE_RERANKING_MODELS if gpu_available else [],
            "cuda_device_count": 1 if gpu_available else 0,
            "cuda_device_name": "GPU (via vecpipe)" if gpu_available else None,
        }
    except httpx.RequestError as e:
        logger.warning(f"Could not reach vecpipe service: {e}")
        return {
            "gpu_available": False,
            "reranking_available": False,
            "available_reranking_models": [],
            "cuda_device_count": 0,
            "cuda_device_name": None,
        }
    except Exception as e:
        logger.error(f"Error checking system status: {e}")
        return {
            "gpu_available": False,
            "reranking_available": False,
            "available_reranking_models": [],
            "cuda_device_count": 0,
            "cuda_device_name": None,
        }
