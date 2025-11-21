"""
System status and capabilities API endpoints.

This module provides endpoints for checking system capabilities like GPU availability
and supported features.
"""

import logging
from typing import Any

import torch
from fastapi import APIRouter, Depends
from webui.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/system", tags=["system-v2"])


@router.get("/status")
async def get_system_status(
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> dict[str, Any]:
    """
    Get system status including GPU availability and reranking capabilities.

    Returns information about:
    - GPU availability
    - Reranking model support
    - Available reranking models
    """
    try:
        gpu_available = torch.cuda.is_available()

        # List of supported reranking models
        available_reranking_models = []
        if gpu_available:
            available_reranking_models = [
                "Qwen/Qwen3-Reranker-0.6B",
                "Qwen/Qwen3-Reranker-4B",
                "Qwen/Qwen3-Reranker-8B",
            ]

        return {
            "gpu_available": gpu_available,
            "reranking_available": gpu_available,  # Reranking requires GPU
            "available_reranking_models": available_reranking_models,
            "cuda_device_count": torch.cuda.device_count() if gpu_available else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if gpu_available else None,
        }
    except Exception as e:
        logger.error(f"Error checking system status: {e}")
        # Return safe defaults on error
        return {
            "gpu_available": False,
            "reranking_available": False,
            "available_reranking_models": [],
            "cuda_device_count": 0,
            "cuda_device_name": None,
        }
