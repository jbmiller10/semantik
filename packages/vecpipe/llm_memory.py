"""Memory estimation utilities for local LLM models.

Provides memory requirement estimates for LLM models based on:
1. Curated values from model_registry.yaml (preferred)
2. Parameter-based estimation for unknown models (fallback)
"""

import logging
import re

from shared.config import settings

logger = logging.getLogger(__name__)


def get_llm_memory_requirement(model_name: str, quantization: str) -> int:
    """Get memory requirement for LLM model.

    First checks model_registry.yaml, then falls back to estimation.

    Args:
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
        quantization: Quantization type (e.g., "int4", "int8", "float16")

    Returns:
        Estimated memory requirement in MB
    """
    # Try registry first (single source of truth)
    from shared.llm.model_registry import get_model_by_id

    model_info = get_model_by_id(model_name, provider="local")
    if model_info and model_info.memory_mb and quantization in model_info.memory_mb:
        return int(model_info.memory_mb[quantization])

    # Fallback: estimate based on model name patterns + KV cache buffer
    weights_mb = _estimate_llm_weights_memory(model_name, quantization)
    kv_cache_buffer_mb: int = settings.LLM_KV_CACHE_BUFFER_MB
    total_mb: int = weights_mb + kv_cache_buffer_mb

    logger.debug(
        "Estimated memory for %s (%s): %dMB (weights=%dMB, kv_buffer=%dMB)",
        model_name,
        quantization,
        total_mb,
        weights_mb,
        kv_cache_buffer_mb,
    )

    return total_mb


def _estimate_llm_weights_memory(model_name: str, quantization: str) -> int:
    """Estimate weights memory for unknown models based on name patterns.

    Uses parameter count extracted from model name (e.g., "0.5B", "1.5B", "7B").

    Args:
        model_name: HuggingFace model ID
        quantization: Quantization type

    Returns:
        Estimated weights memory in MB
    """
    # Parse size from model name (e.g., "0.5B", "1.5B", "7B")
    match = re.search(r"(\d+\.?\d*)B", model_name, re.IGNORECASE)
    if not match:
        logger.warning(
            "Could not parse parameter count from model name '%s', using conservative default",
            model_name,
        )
        return 4000  # Conservative default for unknown models

    params_b = float(match.group(1))

    # Base memory: ~2 bytes per param for float16
    base_mb = int(params_b * 2000)

    # Adjust for quantization
    if quantization == "int8":
        base_mb = int(base_mb * 0.55)
    elif quantization == "int4":
        base_mb = int(base_mb * 0.35)
    elif quantization == "float32":
        base_mb = base_mb * 2

    # Add overhead for activations / fragmentation (30%)
    return int(base_mb * 1.3)


__all__ = [
    "get_llm_memory_requirement",
]
