"""
Model management routes for the Web UI.

This module provides the /api/models endpoint for discovering available
embedding models, including both built-in and plugin-provided models.
"""

from typing import Any

from fastapi import APIRouter, Depends

from shared.embedding import embedding_service
from shared.embedding.factory import get_all_supported_models
from shared.plugins.loader import load_plugins
from webui.auth import get_current_user

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
async def get_models(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:  # noqa: ARG001
    """Get available embedding models including plugin models.

    Returns all registered embedding models from built-in providers and plugins.
    The response shape maintains backward compatibility with legacy consumers.

    Returns:
        dict with keys:
            - models: Dict of model configs keyed by model_name
            - current_device: Current compute device (e.g., "cuda:0", "cpu")
            - using_real_embeddings: Always True with unified service
    """
    # Ensure built-in providers and plugins are registered before querying
    load_plugins(plugin_types={"embedding"})

    all_models = get_all_supported_models()

    # Convert list to dict keyed by model_name for backward compatibility
    models_dict = {
        model.get("model_name", model.get("name", "")): model
        for model in all_models
        if model.get("model_name") or model.get("name")
    }

    return {
        "models": models_dict,
        "current_device": embedding_service.device,
        "using_real_embeddings": True,  # Always true with unified service
    }
