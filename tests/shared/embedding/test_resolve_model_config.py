"""Tests for resolve_model_config function.

This module tests the unified model configuration resolution that checks
providers (including plugins) first, then falls back to built-in configs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from shared.embedding.factory import EmbeddingProviderFactory, resolve_model_config
from shared.embedding.models import MODEL_CONFIGS, ModelConfig
from shared.embedding.plugin_base import BaseEmbeddingPlugin, EmbeddingProviderDefinition
from shared.plugins.loader import load_plugins

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ConfigAwarePlugin(BaseEmbeddingPlugin):
    """Plugin that provides its own model configuration."""

    INTERNAL_NAME = "config_aware"
    API_ID = "config_aware"
    PROVIDER_TYPE = "local"

    # Custom dimension for testing
    PLUGIN_DIMENSION = 512

    def __init__(self, config: Any | None = None, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)
        self._initialized = False
        self.model_name: str | None = None

    @classmethod
    def get_definition(cls) -> EmbeddingProviderDefinition:
        return EmbeddingProviderDefinition(
            api_id=cls.API_ID,
            internal_id=cls.INTERNAL_NAME,
            display_name="Config Aware Plugin",
            description="Plugin that provides model config",
            provider_type="local",
            supported_models=("config_aware/test-model",),
        )

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("config_aware/")

    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig | None:
        if cls.supports_model(model_name):
            return ModelConfig(
                name=model_name,
                dimension=cls.PLUGIN_DIMENSION,
                description="Plugin model with custom dimension",
                max_sequence_length=8192,
            )
        return None

    @classmethod
    def list_supported_models(cls) -> list[ModelConfig]:
        return [
            ModelConfig(
                name="config_aware/test-model",
                dimension=cls.PLUGIN_DIMENSION,
                description="Plugin test model",
            )
        ]

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self._initialized = True

    async def embed_texts(self, texts: list[str], batch_size: int = 32, **kwargs: Any) -> NDArray[np.float32]:
        if not self._initialized:
            raise RuntimeError("Not initialized")
        return np.zeros((len(texts), self.PLUGIN_DIMENSION), dtype=np.float32)

    async def embed_single(self, text: str, **kwargs: Any) -> NDArray[np.float32]:
        embeddings = await self.embed_texts([text], **kwargs)
        return embeddings[0]

    def get_dimension(self) -> int:
        return self.PLUGIN_DIMENSION

    def get_model_info(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "dimension": self.PLUGIN_DIMENSION}

    async def cleanup(self) -> None:
        self._initialized = False


class TestResolveModelConfigBuiltIn:
    """Tests for resolving built-in model configurations."""

    def test_resolve_qwen_model(self, clean_registry: None) -> None:
        """Test resolving a built-in Qwen model config.

        Note: This tests the fallback to MODEL_CONFIGS, since Qwen models
        are in MODEL_CONFIGS but may not be registered with providers.
        """
        config = resolve_model_config("Qwen/Qwen3-Embedding-0.6B")

        assert config is not None
        assert config.name == "Qwen/Qwen3-Embedding-0.6B"
        assert config.dimension == 1024

    def test_resolve_sentence_transformer_model(self, clean_registry: None) -> None:
        """Test resolving a built-in sentence-transformers model config."""
        config = resolve_model_config("sentence-transformers/all-MiniLM-L6-v2")

        assert config is not None
        assert config.dimension == 384

    def test_resolve_bge_model(self, clean_registry: None) -> None:
        """Test resolving a built-in BGE model config."""
        config = resolve_model_config("BAAI/bge-large-en-v1.5")

        assert config is not None
        assert config.dimension == 1024

    def test_resolve_mock_model_via_provider(self, empty_registry: None) -> None:
        """Test resolving the mock provider's model config via providers."""
        # Ensure mock provider is registered
        load_plugins(plugin_types={"embedding"})

        config = resolve_model_config("mock")

        assert config is not None
        assert config.name == "mock"
        assert config.dimension == 384


class TestResolveModelConfigPlugin:
    """Tests for resolving plugin model configurations."""

    def test_resolve_plugin_model_config(self, empty_registry: None) -> None:
        """Test resolving a plugin model that provides its own config."""
        EmbeddingProviderFactory.register_provider("config_aware", ConfigAwarePlugin)

        config = resolve_model_config("config_aware/test-model")

        assert config is not None
        assert config.name == "config_aware/test-model"
        assert config.dimension == ConfigAwarePlugin.PLUGIN_DIMENSION

    def test_resolve_plugin_model_dimension_flows_correctly(self, empty_registry: None) -> None:
        """Test that plugin model dimension is correctly resolved."""
        EmbeddingProviderFactory.register_provider("config_aware", ConfigAwarePlugin)

        config = resolve_model_config("config_aware/any-model-name")

        # Plugin supports any model starting with "config_aware/"
        assert config is not None
        assert config.dimension == 512  # The plugin's PLUGIN_DIMENSION


class TestResolveModelConfigFallback:
    """Tests for fallback behavior when model is unknown."""

    def test_resolve_unknown_model_returns_none(self, clean_registry: None) -> None:
        """Test that unknown models return None."""
        config = resolve_model_config("unknown/nonexistent-model-xyz")

        assert config is None

    def test_resolve_empty_string_returns_none(self, clean_registry: None) -> None:
        """Test that empty string returns None."""
        config = resolve_model_config("")

        assert config is None


class TestResolveModelConfigPrecedence:
    """Tests for provider precedence over built-in configs."""

    def test_provider_config_takes_precedence_over_builtin(self, empty_registry: None) -> None:
        """Test that provider config is checked before built-in configs.

        If a plugin claims to support a built-in model name but returns
        a different config, the plugin's config should be used.
        """

        class OverridePlugin(BaseEmbeddingPlugin):
            """Plugin that overrides a built-in model's config."""

            INTERNAL_NAME = "override"
            API_ID = "override"
            PROVIDER_TYPE = "local"
            OVERRIDE_DIMENSION = 999

            def __init__(self, config: Any | None = None, **kwargs: Any) -> None:
                super().__init__(config=config, **kwargs)
                self._initialized = False

            @classmethod
            def get_definition(cls) -> EmbeddingProviderDefinition:
                return EmbeddingProviderDefinition(
                    api_id=cls.API_ID,
                    internal_id=cls.INTERNAL_NAME,
                    display_name="Override Plugin",
                    description="Plugin that overrides built-in config",
                    provider_type="local",
                )

            @classmethod
            def supports_model(cls, model_name: str) -> bool:
                # Claim support for a built-in model
                return model_name == "sentence-transformers/all-MiniLM-L6-v2"

            @classmethod
            def get_model_config(cls, model_name: str) -> ModelConfig | None:
                if cls.supports_model(model_name):
                    return ModelConfig(
                        name=model_name,
                        dimension=cls.OVERRIDE_DIMENSION,
                        description="Overridden config from plugin",
                    )
                return None

            @property
            def is_initialized(self) -> bool:
                return self._initialized

            async def initialize(self, model_name: str, **kwargs: Any) -> None:
                self._initialized = True

            async def embed_texts(self, texts: list[str], batch_size: int = 32, **kwargs: Any) -> NDArray[np.float32]:
                return np.zeros((len(texts), self.OVERRIDE_DIMENSION), dtype=np.float32)

            async def embed_single(self, text: str, **kwargs: Any) -> NDArray[np.float32]:
                embeddings = await self.embed_texts([text], **kwargs)
                return embeddings[0]

            def get_dimension(self) -> int:
                return self.OVERRIDE_DIMENSION

            def get_model_info(self) -> dict[str, Any]:
                return {}

            async def cleanup(self) -> None:
                self._initialized = False

        EmbeddingProviderFactory.register_provider("override", OverridePlugin)

        # Resolve the model that has both a built-in config and a plugin config
        config = resolve_model_config("sentence-transformers/all-MiniLM-L6-v2")

        # Plugin's config should take precedence
        assert config is not None
        assert config.dimension == OverridePlugin.OVERRIDE_DIMENSION

    def test_builtin_used_when_no_provider_supports(self, empty_registry: None) -> None:
        """Test that built-in config is used when no provider supports the model."""
        # Empty registry - no providers registered
        # But MODEL_CONFIGS still has built-in models

        config = resolve_model_config("Qwen/Qwen3-Embedding-0.6B")

        # Should fall back to built-in config
        assert config is not None
        assert config.dimension == MODEL_CONFIGS["Qwen/Qwen3-Embedding-0.6B"].dimension


class TestResolveModelConfigIntegration:
    """Integration tests for resolve_model_config with real call sites."""

    def test_validation_uses_resolved_config(self, empty_registry: None) -> None:
        """Test that validation module uses resolved config for plugin models."""
        from shared.embedding.validation import get_model_dimension

        EmbeddingProviderFactory.register_provider("config_aware", ConfigAwarePlugin)

        # get_model_dimension now uses resolve_model_config internally
        dimension = get_model_dimension("config_aware/test-model")

        assert dimension == ConfigAwarePlugin.PLUGIN_DIMENSION

    def test_batch_manager_uses_resolved_config(self, empty_registry: None) -> None:
        """Test that batch manager can resolve plugin model configs.

        Note: This doesn't test the actual batch size calculation
        (which requires GPU), just that the model config is resolved.
        """
        EmbeddingProviderFactory.register_provider("config_aware", ConfigAwarePlugin)

        # Verify the config resolution works
        config = resolve_model_config("config_aware/test-model")

        assert config is not None
        assert hasattr(config, "memory_estimate")  # batch_manager accesses this
        assert hasattr(config, "max_sequence_length")  # batch_manager accesses this


class TestDenseEmbeddingServicePluginConfig:
    """Tests for DenseEmbeddingService using plugin-resolved config."""

    @pytest.mark.asyncio()
    async def test_dense_service_uses_plugin_dimension_in_mock_mode(self, empty_registry: None) -> None:
        """Test that DenseEmbeddingService mock mode uses plugin's dimension."""
        from shared.embedding.dense import DenseEmbeddingService

        EmbeddingProviderFactory.register_provider("config_aware", ConfigAwarePlugin)

        service = DenseEmbeddingService(mock_mode=True)
        await service.initialize("config_aware/test-model")

        assert service.dimension == 512  # ConfigAwarePlugin.PLUGIN_DIMENSION
        await service.cleanup()

    @pytest.mark.asyncio()
    async def test_dense_service_raises_for_unknown_model_in_mock_mode(self, empty_registry: None) -> None:
        """Test that DenseEmbeddingService raises for unknown models.

        Note: The ValueError is wrapped in RuntimeError by the initialize method's
        exception handler.
        """
        from shared.embedding.dense import DenseEmbeddingService

        service = DenseEmbeddingService(mock_mode=True)

        with pytest.raises(RuntimeError, match="No model configuration found"):
            await service.initialize("completely/unknown-model-xyz")

    @pytest.mark.asyncio()
    async def test_embedding_service_generate_embeddings_uses_plugin_dimension(self, empty_registry: None) -> None:
        """Test EmbeddingService.generate_embeddings uses plugin dimension."""
        from shared.embedding.dense import EmbeddingService

        EmbeddingProviderFactory.register_provider("config_aware", ConfigAwarePlugin)

        service = EmbeddingService(mock_mode=True)
        embeddings = service.generate_embeddings(
            texts=["test"],
            model_name="config_aware/test-model",
        )

        assert embeddings is not None
        assert embeddings.shape[1] == 512
        service.shutdown()

    def test_dense_local_provider_get_model_config_returns_builtin(self, clean_registry: None) -> None:
        """Test DenseLocalEmbeddingProvider.get_model_config returns built-in configs.

        Note: The provider's get_model_config returns configs it owns (built-in).
        It does NOT call resolve_model_config (that would cause infinite recursion).
        Plugin configs are resolved at the factory level, not provider level.
        """
        from shared.embedding.providers.dense_local import DenseLocalEmbeddingProvider

        # Should return built-in config
        config = DenseLocalEmbeddingProvider.get_model_config("Qwen/Qwen3-Embedding-0.6B")

        assert config is not None
        assert config.dimension == 1024
