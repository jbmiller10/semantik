"""
Model management routes for the Web UI.

This module provides the /api/models endpoint for discovering available
embedding models, including both built-in and plugin-provided models.
"""

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from shared.config.vecpipe import VecpipeConfig
from shared.embedding import embedding_service
from shared.embedding.factory import get_all_supported_models
from shared.model_manager.hf_cache import scan_hf_cache
from shared.plugins.loader import load_plugins
from webui.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["models"])

# Providers that require local HuggingFace model installation
_LOCAL_HF_PROVIDERS = {"dense_local"}


def _is_model_available(
    model: dict[str, Any],
    installed_model_ids: set[str],
    default_model: str,
) -> bool:
    """Check if a model should be shown based on availability.

    Args:
        model: Model configuration dict with provider and model_name fields.
        installed_model_ids: Set of HuggingFace model IDs that are installed locally.
        default_model: The default embedding model ID (always shown).

    Returns:
        True if the model should be included in the list.
    """
    model_name = model.get("model_name", model.get("name", ""))
    provider = model.get("provider", "")

    # Always include the default model
    if model_name == default_model:
        return True

    # Only dense_local provider requires HF cache installation check
    # All other providers (plugins, mock, API-based) are always available
    if provider not in _LOCAL_HF_PROVIDERS:
        return True

    # For dense_local provider, check if installed in HF cache
    return model_name in installed_model_ids


@router.get("/models")
async def get_models(
    installed_only: bool = Query(
        default=True,
        description="Only return installed models (plus default model and API-based models)",
    ),
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> dict[str, Any]:
    """Get available embedding models including plugin models.

    Returns registered embedding models from built-in providers and plugins.
    By default, only returns models that are installed locally (downloaded to
    HuggingFace cache), plus the default model and API-based models (like OpenAI).

    Args:
        installed_only: If True (default), filter to installed models only.
            The default model and API-based models are always included.

    Returns:
        dict with keys:
            - models: Dict of model configs keyed by model_name
            - current_device: Current compute device (e.g., "cuda:0", "cpu")
            - using_real_embeddings: Always True with unified service
    """
    # Ensure built-in providers and plugins are registered before querying
    load_plugins(plugin_types={"embedding"})

    all_models = get_all_supported_models()

    # Filter for installed models if requested
    if installed_only:
        # Scan HF cache in thread pool to avoid blocking
        cache_info = await asyncio.to_thread(scan_hf_cache)
        if cache_info.scan_error:
            logger.warning(
                "HF cache scan encountered errors, model list may be incomplete: %s",
                cache_info.scan_error,
            )
        installed_model_ids = {repo_id for (repo_type, repo_id) in cache_info.repos if repo_type == "model"}

        # Get default model from config
        settings = VecpipeConfig()
        default_model = settings.DEFAULT_EMBEDDING_MODEL

        # Filter models
        all_models = [model for model in all_models if _is_model_available(model, installed_model_ids, default_model)]

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
