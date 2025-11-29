"""Embedding service module.

This module provides embedding generation capabilities for the system.

The embedding system now uses a plugin-based architecture, allowing new
embedding providers to be added without modifying core code.

For new code, prefer using the factory:
    from shared.embedding.factory import EmbeddingProviderFactory
    provider = EmbeddingProviderFactory.create_provider("model-name")

For backward compatibility, the existing APIs remain available:
    from shared.embedding import EmbeddingService, embedding_service
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
from .factory import EmbeddingProviderFactory, get_all_supported_models, get_model_config_from_providers
from .models import (
    POPULAR_MODELS,
    QUANTIZED_MODEL_INFO,
    ModelConfig,
    add_model_config,
    get_model_config,
    list_available_models,
)
from .plugin_base import BaseEmbeddingPlugin, EmbeddingProviderDefinition
from .plugin_loader import ensure_providers_registered, load_embedding_plugins
from .provider_registry import (
    get_provider_definition,
    get_provider_metadata,
    is_provider_registered,
    list_provider_definitions,
    list_provider_metadata,
    list_provider_metadata_list,
    register_provider_definition,
)
from .service import (
    cleanup,
    embed_single,
    embed_texts,
    get_embedding_service,
    get_embedding_service_sync,
    initialize_embedding_service,
)
from .types import EmbeddingMode
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
    # Plugin architecture
    "BaseEmbeddingPlugin",
    "EmbeddingProviderDefinition",
    "EmbeddingProviderFactory",
    # Types
    "EmbeddingMode",
    # Plugin loading
    "load_embedding_plugins",
    "ensure_providers_registered",
    # Provider registry
    "register_provider_definition",
    "get_provider_definition",
    "get_provider_metadata",
    "list_provider_definitions",
    "list_provider_metadata",
    "list_provider_metadata_list",
    "is_provider_registered",
    # Factory utilities
    "get_all_supported_models",
    "get_model_config_from_providers",
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
