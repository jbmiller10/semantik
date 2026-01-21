"""Model manager module for managing local ML models.

This module provides:
- Curated model registry aggregating models from multiple sources
- HuggingFace cache utilities for tracking installed models
"""

from shared.model_manager.curated_registry import (
    CuratedModel,
    ModelType,
    get_curated_model_ids,
    get_curated_models,
    get_model_by_id,
    get_models_by_type,
)
from shared.model_manager.hf_cache import (
    CacheSizeBreakdown,
    HFCacheInfo,
    InstalledModel,
    clear_cache,
    get_cache_size_info,
    get_installed_models,
    get_model_size_on_disk,
    is_model_installed,
    resolve_hf_cache_dir,
    scan_hf_cache,
)

__all__ = [
    # Curated registry
    "CuratedModel",
    "ModelType",
    "get_curated_model_ids",
    "get_curated_models",
    "get_model_by_id",
    "get_models_by_type",
    # HF cache utilities
    "CacheSizeBreakdown",
    "HFCacheInfo",
    "InstalledModel",
    "clear_cache",
    "get_cache_size_info",
    "get_installed_models",
    "get_model_size_on_disk",
    "is_model_installed",
    "resolve_hf_cache_dir",
    "scan_hf_cache",
]
