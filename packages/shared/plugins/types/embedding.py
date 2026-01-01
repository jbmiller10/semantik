"""Embedding plugin base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from shared.embedding.plugin_base import EmbeddingProviderDefinition

from ..base import SemanticPlugin


class EmbeddingPlugin(SemanticPlugin, ABC):
    """Base class for embedding provider plugins."""

    PLUGIN_TYPE = "embedding"

    @classmethod
    @abstractmethod
    def get_definition(cls) -> EmbeddingProviderDefinition:
        """Return the embedding provider definition for this plugin."""

    @classmethod
    @abstractmethod
    def supports_model(cls, model_name: str) -> bool:
        """Return True if this plugin supports the given model name."""
