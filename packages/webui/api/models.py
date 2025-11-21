"""
Model management routes for the Web UI
"""

from typing import Any

from fastapi import APIRouter, Depends

from shared.embedding import POPULAR_MODELS, embedding_service
from webui.auth import get_current_user

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
async def get_models(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:  # noqa: ARG001
    """Get available embedding models"""
    return {
        "models": POPULAR_MODELS,
        "current_device": embedding_service.device,
        "using_real_embeddings": True,  # Always true with unified service
    }
