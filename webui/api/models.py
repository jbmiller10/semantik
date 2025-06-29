"""
Model management routes for the Web UI
"""

from typing import Dict, Any

from fastapi import APIRouter, Depends

from webui.auth import get_current_user
from webui.embedding_service import embedding_service, POPULAR_MODELS

router = APIRouter(prefix="/api", tags=["models"])

@router.get("/models")
async def get_models(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Get available embedding models"""
    return {
        "models": POPULAR_MODELS,
        "current_device": embedding_service.device,
        "using_real_embeddings": True  # Always true with unified service
    }