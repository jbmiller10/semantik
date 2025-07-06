"""
Memory management utilities for model loading
"""

import logging

import torch

logger = logging.getLogger(__name__)

# Approximate memory requirements in MB (conservative estimates)
MODEL_MEMORY_REQUIREMENTS = {
    # Embedding models
    ("Qwen/Qwen3-Embedding-0.6B", "float32"): 2400,
    ("Qwen/Qwen3-Embedding-0.6B", "float16"): 1200,
    ("Qwen/Qwen3-Embedding-0.6B", "int8"): 600,
    ("Qwen/Qwen3-Embedding-4B", "float32"): 16000,
    ("Qwen/Qwen3-Embedding-4B", "float16"): 8000,
    ("Qwen/Qwen3-Embedding-4B", "int8"): 4000,
    ("Qwen/Qwen3-Embedding-8B", "float32"): 32000,
    ("Qwen/Qwen3-Embedding-8B", "float16"): 16000,
    ("Qwen/Qwen3-Embedding-8B", "int8"): 8000,
    # Reranker models
    ("Qwen/Qwen3-Reranker-0.6B", "float32"): 2400,
    ("Qwen/Qwen3-Reranker-0.6B", "float16"): 1200,
    ("Qwen/Qwen3-Reranker-0.6B", "int8"): 600,
    ("Qwen/Qwen3-Reranker-4B", "float32"): 16000,
    ("Qwen/Qwen3-Reranker-4B", "float16"): 8000,
    ("Qwen/Qwen3-Reranker-4B", "int8"): 4000,
    ("Qwen/Qwen3-Reranker-8B", "float32"): 32000,
    ("Qwen/Qwen3-Reranker-8B", "float16"): 16000,
    ("Qwen/Qwen3-Reranker-8B", "int8"): 8000,
}

# Add 20% overhead for activations and temporary buffers
MEMORY_OVERHEAD_FACTOR = 1.2


class InsufficientMemoryError(Exception):
    """Raised when there's not enough memory to load a model"""



def get_gpu_memory_info() -> tuple[int, int]:
    """
    Get GPU memory information

    Returns:
        Tuple of (free_memory_mb, total_memory_mb)
    """
    if not torch.cuda.is_available():
        return 0, 0

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return free_bytes // (1024 * 1024), total_bytes // (1024 * 1024)


def get_model_memory_requirement(model_name: str, quantization: str = "float32") -> int:
    """
    Get estimated memory requirement for a model

    Args:
        model_name: Model name
        quantization: Quantization type

    Returns:
        Memory requirement in MB
    """
    base_requirement = MODEL_MEMORY_REQUIREMENTS.get((model_name, quantization))

    if base_requirement is None:
        # Estimate based on model size in name
        if "0.6B" in model_name:
            base_requirement = {"float32": 2400, "float16": 1200, "int8": 600}[quantization]
        elif "4B" in model_name:
            base_requirement = {"float32": 16000, "float16": 8000, "int8": 4000}[quantization]
        elif "8B" in model_name:
            base_requirement = {"float32": 32000, "float16": 16000, "int8": 8000}[quantization]
        else:
            # Conservative default
            base_requirement = 16000

    return int(base_requirement * MEMORY_OVERHEAD_FACTOR)


def check_memory_availability(
    model_name: str, quantization: str = "float32", current_models: dict[str, tuple[str, str]] = None
) -> tuple[bool, str]:
    """
    Check if there's enough memory to load a model

    Args:
        model_name: Model to load
        quantization: Quantization type
        current_models: Dict of currently loaded models {role: (model_name, quantization)}

    Returns:
        Tuple of (can_load, message)
    """
    if not torch.cuda.is_available():
        return True, "CPU mode - memory checks skipped"

    free_mb, total_mb = get_gpu_memory_info()
    required_mb = get_model_memory_requirement(model_name, quantization)

    # Check if we have enough free memory
    if free_mb >= required_mb:
        return True, f"Sufficient memory: {free_mb}MB free, {required_mb}MB required"

    # Calculate memory that could be freed
    freeable_mb = 0
    models_to_unload = []

    if current_models:
        for role, (loaded_model, loaded_quant) in current_models.items():
            model_mb = get_model_memory_requirement(loaded_model, loaded_quant)
            freeable_mb += model_mb
            models_to_unload.append(f"{role}: {loaded_model}")

            if free_mb + freeable_mb >= required_mb:
                break

    if free_mb + freeable_mb >= required_mb:
        models_str = ", ".join(models_to_unload)
        return (
            False,
            f"Insufficient memory: {free_mb}MB free, {required_mb}MB required. Can free {freeable_mb}MB by unloading: {models_str}",
        )
    else:
        return (
            False,
            f"Insufficient memory: {free_mb}MB free, {required_mb}MB required. Even after unloading all models, only {free_mb + freeable_mb}MB would be available",
        )


def suggest_model_configuration(available_memory_mb: int) -> dict[str, str]:
    """
    Suggest optimal model configuration based on available memory

    Args:
        available_memory_mb: Available GPU memory in MB

    Returns:
        Dict with suggested configuration
    """
    suggestions = {
        "embedding_model": None,
        "embedding_quantization": None,
        "reranker_model": None,
        "reranker_quantization": None,
        "notes": [],
    }

    # Conservative approach - leave 20% free for operations
    usable_memory = int(available_memory_mb * 0.8)

    if usable_memory >= 32000:
        # Can fit both 8B models in float16
        suggestions.update(
            {
                "embedding_model": "Qwen/Qwen3-Embedding-8B",
                "embedding_quantization": "float16",
                "reranker_model": "Qwen/Qwen3-Reranker-8B",
                "reranker_quantization": "float16",
                "notes": ["Optimal configuration with both 8B models"],
            }
        )
    elif usable_memory >= 24000:
        # 8B embedding + 4B reranker in float16
        suggestions.update(
            {
                "embedding_model": "Qwen/Qwen3-Embedding-8B",
                "embedding_quantization": "float16",
                "reranker_model": "Qwen/Qwen3-Reranker-4B",
                "reranker_quantization": "float16",
                "notes": ["Good balance of embedding quality with faster reranking"],
            }
        )
    elif usable_memory >= 16000:
        # Both 4B models in float16
        suggestions.update(
            {
                "embedding_model": "Qwen/Qwen3-Embedding-4B",
                "embedding_quantization": "float16",
                "reranker_model": "Qwen/Qwen3-Reranker-4B",
                "reranker_quantization": "float16",
                "notes": ["Balanced configuration for mid-range GPUs"],
            }
        )
    elif usable_memory >= 12000:
        # 4B embedding float16 + 0.6B reranker
        suggestions.update(
            {
                "embedding_model": "Qwen/Qwen3-Embedding-4B",
                "embedding_quantization": "float16",
                "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
                "reranker_quantization": "float16",
                "notes": ["Good embedding quality with fast reranking"],
            }
        )
    elif usable_memory >= 8000:
        # 8B embedding int8 or dual 0.6B models
        suggestions.update(
            {
                "embedding_model": "Qwen/Qwen3-Embedding-8B",
                "embedding_quantization": "int8",
                "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
                "reranker_quantization": "float16",
                "notes": ["Quantized large embedding model with small reranker"],
            }
        )
    else:
        # Minimum configuration
        suggestions.update(
            {
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "embedding_quantization": "float16",
                "reranker_model": "Qwen/Qwen3-Reranker-0.6B",
                "reranker_quantization": "int8",
                "notes": ["Minimum configuration for limited memory", "Consider upgrading GPU for better performance"],
            }
        )

    return suggestions
