"""Model registry for curated LLM models.

Loads model metadata from YAML file with LRU caching.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass(frozen=True)
class ModelInfo:
    """Information about an LLM model."""

    id: str
    name: str
    display_name: str
    provider: str
    tier_recommendation: str  # "high" or "low"
    context_window: int
    description: str


@lru_cache(maxsize=1)
def load_model_registry() -> dict[str, list[ModelInfo]]:
    """Load and cache model registry from YAML.

    Returns:
        Dictionary mapping provider name to list of ModelInfo objects.
    """
    yaml_path = Path(__file__).parent / "model_registry.yaml"
    with yaml_path.open() as f:
        data: dict[str, list[dict[str, Any]]] = yaml.safe_load(f)

    registry: dict[str, list[ModelInfo]] = {}
    for provider, models in data.items():
        registry[provider] = [
            ModelInfo(
                id=m["id"],
                name=m["name"],
                display_name=m["display_name"],
                provider=provider,
                tier_recommendation=m["tier_recommendation"],
                context_window=m["context_window"],
                description=m["description"],
            )
            for m in models
        ]
    return registry


def get_default_model(provider: str, tier: str) -> str:
    """Get default model ID for provider and tier.

    Args:
        provider: Provider name ("anthropic" or "openai")
        tier: Quality tier ("high" or "low")

    Returns:
        Model ID string

    Raises:
        ValueError: If no model found for provider/tier combination
    """
    registry = load_model_registry()
    for model in registry.get(provider, []):
        if model.tier_recommendation == tier:
            return model.id
    raise ValueError(f"No {tier} tier model for {provider}")


def get_all_models() -> list[ModelInfo]:
    """Get flat list of all models for API response.

    Returns:
        List of all ModelInfo objects across all providers.
    """
    registry = load_model_registry()
    return [m for models in registry.values() for m in models]


def get_model_by_id(model_id: str) -> ModelInfo | None:
    """Get model info by ID.

    Args:
        model_id: The model identifier to look up

    Returns:
        ModelInfo if found, None otherwise
    """
    registry = load_model_registry()
    for models in registry.values():
        for model in models:
            if model.id == model_id:
                return model
    return None


__all__ = [
    "ModelInfo",
    "load_model_registry",
    "get_default_model",
    "get_all_models",
    "get_model_by_id",
]
