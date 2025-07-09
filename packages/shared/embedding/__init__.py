"""Embedding service module.

This module provides embedding generation capabilities for the system.
"""

from .base import BaseEmbeddingService
from .dense import (
    POPULAR_MODELS,
    QUANTIZED_MODEL_INFO,
    DenseEmbeddingService,
    EmbeddingService,
    embedding_service,
    enhanced_embedding_service,
)
from .service import (
    cleanup,
    embed_single,
    embed_texts,
    get_embedding_service,
    get_embedding_service_sync,
    initialize_embedding_service,
)

__all__ = [
    # Core classes
    "BaseEmbeddingService",
    "DenseEmbeddingService",
    "EmbeddingService",
    # Service functions
    "get_embedding_service",
    "get_embedding_service_sync",
    "initialize_embedding_service",
    "embed_texts",
    "embed_single",
    "cleanup",
    # Compatibility exports
    "embedding_service",
    "enhanced_embedding_service",
    "POPULAR_MODELS",
    "QUANTIZED_MODEL_INFO",
]
