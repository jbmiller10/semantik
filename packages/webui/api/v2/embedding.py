"""Embedding provider discovery endpoints.

This module provides API endpoints for discovering available embedding
providers and models.
"""

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException

from shared.embedding.factory import EmbeddingProviderFactory, get_all_supported_models, get_model_config_from_providers
from shared.embedding.provider_registry import (
    get_provider_definition,
    list_provider_metadata_list,
)
from webui.auth import get_current_user

router = APIRouter(prefix="/embedding", tags=["embedding"])


@router.get("/providers")
async def list_embedding_providers(
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> list[dict[str, Any]]:
    """List all available embedding providers.

    Returns provider metadata including capabilities and supported models.
    This endpoint is useful for building model selection UIs.

    Returns:
        List of provider metadata dictionaries
    """
    return cast(list[dict[str, Any]], list_provider_metadata_list())


@router.get("/providers/{provider_id}")
async def get_provider_info(
    provider_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> dict[str, Any]:
    """Get detailed information for a specific provider.

    Args:
        provider_id: The provider API ID or internal ID

    Returns:
        Provider metadata dictionary

    Raises:
        HTTPException: 404 if provider not found
    """
    definition = get_provider_definition(provider_id)
    if not definition:
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_id}")
    return cast(dict[str, Any], definition.to_metadata_dict())


@router.get("/models")
async def list_embedding_models(
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> list[dict[str, Any]]:
    """List all available embedding models from all providers.

    Returns model configurations aggregated from all registered providers,
    including model dimensions, capabilities, and memory requirements.

    Returns:
        List of model configuration dictionaries with provider information
    """
    return cast(list[dict[str, Any]], get_all_supported_models())


@router.get("/models/{model_name:path}")
async def get_model_info(
    model_name: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> dict[str, Any]:
    """Get detailed information for a specific model.

    Args:
        model_name: The HuggingFace model name (e.g., "Qwen/Qwen3-Embedding-0.6B")

    Returns:
        Model configuration dictionary with provider information

    Raises:
        HTTPException: 404 if model not found
    """
    config = get_model_config_from_providers(model_name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

    # Convert to dict and add provider info
    result = config.to_dict() if hasattr(config, "to_dict") else {}
    result["model_name"] = getattr(config, "name", model_name)
    result["provider"] = EmbeddingProviderFactory.get_provider_for_model(model_name)

    return result


@router.get("/models/{model_name:path}/supported")
async def check_model_support(
    model_name: str,
    current_user: dict[str, Any] = Depends(get_current_user),  # noqa: ARG001
) -> dict[str, Any]:
    """Check if a model is supported by any provider.

    Args:
        model_name: The HuggingFace model name

    Returns:
        Dictionary with support status and provider name
    """
    provider = EmbeddingProviderFactory.get_provider_for_model(model_name)
    return {
        "model_name": model_name,
        "supported": provider is not None,
        "provider": provider,
    }
