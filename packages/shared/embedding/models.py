"""
Centralized model configurations for embedding services.

This module contains all model-specific configurations, making it easy to:
- Add new models
- Update model properties
- Manage model-specific settings
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for an embedding model.

    Asymmetric Embedding Support:
        Many retrieval models need different handling for queries vs documents.
        Set `is_asymmetric=True` and configure the appropriate prefixes/instructions.

        For prefix-based models (E5, BGE):
            - query_prefix: Prefix added to queries (e.g., "query: ")
            - document_prefix: Prefix added to documents (e.g., "passage: ")

        For instruction-based models (Qwen):
            - default_query_instruction: Task instruction for queries
            - Documents typically use no instruction (raw text)
    """

    name: str
    dimension: int
    description: str
    max_sequence_length: int = 512
    supports_quantization: bool = True
    recommended_quantization: str = "float32"
    memory_estimate: dict[str, int] | None = None
    requires_instruction: bool = False
    pooling_method: str = "mean"  # mean, cls, last_token

    # Asymmetric embedding support
    is_asymmetric: bool = False
    query_prefix: str = ""  # For prefix-based models (e.g., "query: " for E5)
    document_prefix: str = ""  # For prefix-based models (e.g., "passage: " for E5)
    default_query_instruction: str = ""  # For instruction-based models (Qwen)

    def __post_init__(self) -> None:
        if self.memory_estimate is None:
            # Rough estimates based on dimension
            base_memory = self.dimension * 4  # Very rough estimate
            self.memory_estimate = {
                "float32": base_memory,
                "float16": base_memory // 2,
                "int8": base_memory // 4,
            }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backwards compatibility."""
        return {
            "dimension": self.dimension,
            "dim": self.dimension,  # Legacy compatibility
            "description": self.description,
            "supports_quantization": self.supports_quantization,
            "recommended_quantization": self.recommended_quantization,
            "memory_estimate": self.memory_estimate,
            "max_sequence_length": self.max_sequence_length,
            "is_asymmetric": self.is_asymmetric,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "default_query_instruction": self.default_query_instruction,
        }


# Default query instruction for Qwen models
_QWEN_DEFAULT_QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

# Model configurations
MODEL_CONFIGS = {
    # Qwen3 Embedding Models - instruction-based asymmetric
    "Qwen/Qwen3-Embedding-0.6B": ModelConfig(
        name="Qwen/Qwen3-Embedding-0.6B",
        dimension=1024,
        description="Qwen3 small model, instruction-aware (1024d)",
        max_sequence_length=32768,
        supports_quantization=True,
        recommended_quantization="float16",
        memory_estimate={"float32": 2400, "float16": 1200, "int8": 600},
        requires_instruction=True,
        pooling_method="last_token",
        is_asymmetric=True,
        default_query_instruction=_QWEN_DEFAULT_QUERY_INSTRUCTION,
    ),
    "Qwen/Qwen3-Embedding-4B": ModelConfig(
        name="Qwen/Qwen3-Embedding-4B",
        dimension=2560,
        description="Qwen3 medium model, MTEB top performer (2560d)",
        max_sequence_length=32768,
        supports_quantization=True,
        recommended_quantization="float16",
        memory_estimate={"float32": 16000, "float16": 8000, "int8": 4000},
        requires_instruction=True,
        pooling_method="last_token",
        is_asymmetric=True,
        default_query_instruction=_QWEN_DEFAULT_QUERY_INSTRUCTION,
    ),
    "Qwen/Qwen3-Embedding-8B": ModelConfig(
        name="Qwen/Qwen3-Embedding-8B",
        dimension=4096,
        description="Qwen3 large model, MTEB #1 (4096d)",
        max_sequence_length=32768,
        supports_quantization=True,
        recommended_quantization="int8",
        memory_estimate={"float32": 32000, "float16": 16000, "int8": 8000},
        requires_instruction=True,
        pooling_method="last_token",
        is_asymmetric=True,
        default_query_instruction=_QWEN_DEFAULT_QUERY_INSTRUCTION,
    ),
    # Popular sentence-transformers models - symmetric (no prefix needed)
    "sentence-transformers/all-MiniLM-L6-v2": ModelConfig(
        name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        description="Fast, lightweight model for general use",
        max_sequence_length=256,
        memory_estimate={"float32": 90, "float16": 45, "int8": 23},
        is_asymmetric=False,  # Symmetric model
    ),
    "sentence-transformers/all-mpnet-base-v2": ModelConfig(
        name="sentence-transformers/all-mpnet-base-v2",
        dimension=768,
        description="High-quality general-purpose embeddings",
        max_sequence_length=384,
        memory_estimate={"float32": 420, "float16": 210, "int8": 105},
        is_asymmetric=False,  # Symmetric model
    ),
    # BGE models - prefix-based asymmetric
    "BAAI/bge-large-en-v1.5": ModelConfig(
        name="BAAI/bge-large-en-v1.5",
        dimension=1024,
        description="State-of-the-art English embeddings",
        max_sequence_length=512,
        memory_estimate={"float32": 1300, "float16": 650, "int8": 325},
        is_asymmetric=True,
        query_prefix="Represent this sentence for searching relevant passages: ",
        document_prefix="",  # No prefix for documents
    ),
}


def get_model_config(model_name: str) -> ModelConfig | None:
    """Get configuration for a model."""
    return MODEL_CONFIGS.get(model_name)


def list_available_models() -> dict[str, ModelConfig]:
    """List all available models."""
    return MODEL_CONFIGS.copy()


def add_model_config(config: ModelConfig) -> None:
    """Add or update a model configuration."""
    MODEL_CONFIGS[config.name] = config


# Export for backwards compatibility
QUANTIZED_MODEL_INFO = {name: config.to_dict() for name, config in MODEL_CONFIGS.items()}
POPULAR_MODELS = QUANTIZED_MODEL_INFO
