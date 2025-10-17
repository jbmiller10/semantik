"""Embedding service module.

This module provides embedding generation capabilities for the system.
"""

from .base import BaseEmbeddingService
from .batch_manager import AdaptiveBatchSizeManager
from .context import ManagedEmbeddingService, embedding_service_context, temporary_embedding_service
from .dense import (
    DenseEmbeddingService,
    EmbeddingService,
    EmbeddingServiceProtocol,
    configure_global_embedding_service,
    embedding_service,
    enhanced_embedding_service,
)
from .models import (
    POPULAR_MODELS,
    QUANTIZED_MODEL_INFO,
    ModelConfig,
    add_model_config,
    get_model_config,
    list_available_models,
)
from .service import (
    cleanup,
    embed_single,
    embed_texts,
    get_embedding_service,
    get_embedding_service_sync,
    initialize_embedding_service,
)
from .validation import (
    adjust_embeddings_dimension,
    get_collection_dimension,
    get_model_dimension,
    validate_collection_model_compatibility,
    validate_dimension_compatibility,
    validate_embedding_dimensions,
)

__all__ = [
    # Core classes
    "BaseEmbeddingService",
    "DenseEmbeddingService",
    "EmbeddingService",
    "EmbeddingServiceProtocol",
    # Batch management
    "AdaptiveBatchSizeManager",
    # Service functions
    "get_embedding_service",
    "get_embedding_service_sync",
    "initialize_embedding_service",
    "embed_texts",
    "embed_single",
    "cleanup",
    "configure_global_embedding_service",
    # Context managers
    "embedding_service_context",
    "temporary_embedding_service",
    "ManagedEmbeddingService",
    # Model configuration
    "ModelConfig",
    "get_model_config",
    "list_available_models",
    "add_model_config",
    # Compatibility exports
    "embedding_service",
    "enhanced_embedding_service",
    "POPULAR_MODELS",
    "QUANTIZED_MODEL_INFO",
    # Validation functions
    "get_collection_dimension",
    "get_model_dimension",
    "validate_dimension_compatibility",
    "validate_embedding_dimensions",
    "validate_collection_model_compatibility",
    "adjust_embeddings_dimension",
]
