"""Base class and definitions for embedding provider plugins.

This module defines the plugin interface that all embedding providers must implement.
Plugins can be built-in (like DenseLocalEmbeddingProvider) or external (loaded via entry points).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from shared.plugins.base import SemanticPlugin
from shared.plugins.manifest import PluginManifest

from .base import BaseEmbeddingService

if TYPE_CHECKING:
    from .models import ModelConfig


def _copy_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    """Return a defensive copy of the provided mapping."""
    if not mapping:
        return {}
    from copy import deepcopy

    return deepcopy(dict(mapping))


@dataclass(frozen=True)
class EmbeddingProviderDefinition:
    """Canonical description of an embedding model provider.

    This dataclass holds metadata about a provider for API exposure and UI display.

    Asymmetric Embedding Support:
        Set `supports_asymmetric=True` if this provider handles query and document
        embeddings differently. When True, the provider's embed_texts method will
        respect the `mode` parameter (EmbeddingMode.QUERY vs EmbeddingMode.DOCUMENT).
    """

    api_id: str  # API-facing identifier (e.g., "dense_local")
    internal_id: str  # Internal name for factory registration
    display_name: str  # Human-readable display name
    description: str  # Description for UI
    provider_type: str  # "local", "remote", or "hybrid"

    # Capability flags
    supports_quantization: bool = True
    supports_instruction: bool = False
    supports_batch_processing: bool = True
    supports_asymmetric: bool = False  # Query vs document mode handling

    # Performance characteristics
    performance_characteristics: dict[str, Any] = field(default_factory=dict)

    # Supported models (for providers with fixed model lists)
    supported_models: tuple[str, ...] = ()

    # Default configuration
    default_config: dict[str, Any] = field(default_factory=dict)

    # Is this a plugin (external) vs built-in?
    is_plugin: bool = False

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return provider metadata in dictionary form for API consumers."""
        return {
            "id": self.api_id,
            "internal_id": self.internal_id,
            "name": self.display_name,
            "description": self.description,
            "provider_type": self.provider_type,
            "supports_quantization": self.supports_quantization,
            "supports_instruction": self.supports_instruction,
            "supports_batch_processing": self.supports_batch_processing,
            "supports_asymmetric": self.supports_asymmetric,
            "supported_models": list(self.supported_models),
            "default_config": _copy_mapping(dict(self.default_config)),
            "is_plugin": self.is_plugin,
            "performance": _copy_mapping(dict(self.performance_characteristics)),
        }


class BaseEmbeddingPlugin(SemanticPlugin, BaseEmbeddingService):
    """Abstract base class for embedding model plugins.

    This extends both SemanticPlugin (for unified plugin system integration) and
    BaseEmbeddingService (for embedding operations). The dual inheritance enables
    embedding plugins to work with both the plugin registry and embedding factory.

    Class Attributes:
        PLUGIN_TYPE: Always "embedding" for this base class
        INTERNAL_NAME: Internal identifier for factory registration (required)
        API_ID: API-facing identifier, also serves as PLUGIN_ID (required)
        PROVIDER_TYPE: Provider type - "local", "remote", or "hybrid" (required)
        PLUGIN_VERSION: Semantic version string (default "0.0.0")
        METADATA: Additional metadata dict (optional)

    Example:
        class MyEmbeddingProvider(BaseEmbeddingPlugin):
            INTERNAL_NAME = "my_provider"
            API_ID = "my-provider"
            PROVIDER_TYPE = "local"
            METADATA = {
                "display_name": "My Custom Embeddings",
                "description": "Custom embedding provider",
            }

            @classmethod
            def get_definition(cls) -> EmbeddingProviderDefinition:
                return EmbeddingProviderDefinition(...)

            @classmethod
            def supports_model(cls, model_name: str) -> bool:
                return model_name in ("my-model-1", "my-model-2")
    """

    # Plugin type - always "embedding" for embedding plugins
    PLUGIN_TYPE: ClassVar[str] = "embedding"

    # Required class attributes - subclasses must override these
    INTERNAL_NAME: ClassVar[str] = ""  # Internal identifier
    API_ID: ClassVar[str] = ""  # API-facing identifier (also serves as PLUGIN_ID)
    PROVIDER_TYPE: ClassVar[str] = "local"  # "local", "remote", or "hybrid"
    PLUGIN_VERSION: ClassVar[str] = "0.0.0"

    # Optional metadata for UI/API exposure
    METADATA: ClassVar[dict[str, Any]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Set PLUGIN_ID from API_ID when subclass is defined."""
        super().__init_subclass__(**kwargs)
        # Bridge API_ID to PLUGIN_ID at class level for SemanticPlugin compatibility
        if cls.API_ID and not getattr(cls, "_plugin_id_set", False):
            cls.PLUGIN_ID = cls.API_ID
            cls._plugin_id_set = True

    def __init__(self, config: Any | None = None, **_kwargs: Any) -> None:
        """Initialize the plugin with optional configuration.

        Args:
            config: Optional configuration object (VecpipeConfig or similar)
            **_kwargs: Additional initialization options (unused in base class)
        """
        # Convert config to dict for SemanticPlugin base class
        if config is not None and hasattr(config, "__dict__"):
            config_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {}

        SemanticPlugin.__init__(self, config_dict)
        self.config = config  # Keep original config for backwards compatibility

    @classmethod
    def get_manifest(cls) -> PluginManifest:
        """Generate plugin manifest from get_definition().

        This method bridges the embedding-specific get_definition() to the
        unified plugin manifest system.

        Returns:
            PluginManifest with provider metadata
        """
        definition = cls.get_definition()
        metadata = cls.METADATA or {}
        return PluginManifest(
            id=definition.api_id,
            type=cls.PLUGIN_TYPE,
            version=cls.PLUGIN_VERSION,
            display_name=definition.display_name,
            description=definition.description,
            author=metadata.get("author"),
            license=metadata.get("license"),
            homepage=metadata.get("homepage"),
            capabilities={
                "internal_id": definition.internal_id,
                "provider_type": definition.provider_type,
                "supports_quantization": definition.supports_quantization,
                "supports_instruction": definition.supports_instruction,
                "supports_batch_processing": definition.supports_batch_processing,
                "supports_asymmetric": definition.supports_asymmetric,
                "supported_models": list(definition.supported_models),
            },
        )

    @classmethod
    @abstractmethod
    def get_definition(cls) -> EmbeddingProviderDefinition:
        """Return the canonical definition for this plugin.

        This method must return an EmbeddingProviderDefinition instance
        that describes the provider's capabilities and metadata.

        Returns:
            EmbeddingProviderDefinition with provider metadata
        """

    @classmethod
    @abstractmethod
    def supports_model(cls, model_name: str) -> bool:
        """Check if this plugin supports the given model name.

        This is used by the factory to auto-detect the appropriate provider
        for a given model name.

        Args:
            model_name: HuggingFace model name or other model identifier

        Returns:
            True if this provider can handle the model, False otherwise
        """

    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig | None:  # noqa: ARG003
        """Get configuration for a specific model.

        Override this method in subclasses to provide model-specific configuration.

        Args:
            model_name: Model identifier

        Returns:
            ModelConfig if model is known, None otherwise
        """
        return None

    @classmethod
    def list_supported_models(cls) -> list[ModelConfig]:
        """List all models supported by this plugin.

        Override this method in subclasses to enumerate available models.

        Returns:
            List of ModelConfig objects for supported models
        """
        return []

    @classmethod
    def validate_plugin_contract(cls) -> tuple[bool, str | None]:
        """Validate that this class meets the plugin contract.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required class attributes
        if not cls.INTERNAL_NAME:
            return False, f"Plugin {cls.__name__} missing INTERNAL_NAME"
        if not cls.API_ID:
            return False, f"Plugin {cls.__name__} missing API_ID"
        if cls.PROVIDER_TYPE not in ("local", "remote", "hybrid"):
            return False, f"Plugin {cls.__name__} has invalid PROVIDER_TYPE: {cls.PROVIDER_TYPE}"

        # Check required methods are implemented (not abstract)
        try:
            # These should be implemented, not raise NotImplementedError
            _ = cls.get_definition
            _ = cls.supports_model
        except AttributeError as e:
            return False, f"Plugin {cls.__name__} missing required method: {e}"

        return True, None
