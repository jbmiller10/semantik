"""Tests for the EmbeddingProviderFactory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import shared.embedding.providers  # noqa: F401  # Ensure built-in providers are registered for these tests.
from shared.embedding.factory import (
    _PROVIDER_CLASSES,
    EmbeddingProviderFactory,
    get_all_supported_models,
    get_model_config_from_providers,
)

if TYPE_CHECKING:
    from shared.embedding.plugin_base import BaseEmbeddingPlugin


class TestProviderRegistration:
    """Tests for provider registration in the factory."""

    def test_register_provider(self, empty_registry: None, dummy_plugin_class: type[BaseEmbeddingPlugin]) -> None:
        """Test registering a provider class."""
        EmbeddingProviderFactory.register_provider("dummy", dummy_plugin_class)

        assert "dummy" in _PROVIDER_CLASSES
        assert _PROVIDER_CLASSES["dummy"] == dummy_plugin_class

    def test_unregister_provider(self, empty_registry: None, dummy_plugin_class: type[BaseEmbeddingPlugin]) -> None:
        """Test unregistering a provider class."""
        EmbeddingProviderFactory.register_provider("dummy", dummy_plugin_class)
        assert "dummy" in _PROVIDER_CLASSES

        EmbeddingProviderFactory.unregister_provider("dummy")
        assert "dummy" not in _PROVIDER_CLASSES

    def test_unregister_nonexistent_provider(self, empty_registry: None) -> None:
        """Test unregistering a provider that doesn't exist (should be no-op)."""
        # Should not raise an exception
        EmbeddingProviderFactory.unregister_provider("nonexistent")

    def test_list_available_providers(
        self, empty_registry: None, dummy_plugin_class: type[BaseEmbeddingPlugin]
    ) -> None:
        """Test listing available providers."""
        EmbeddingProviderFactory.register_provider("dummy", dummy_plugin_class)

        providers = EmbeddingProviderFactory.list_available_providers()

        assert "dummy" in providers

    def test_list_available_providers_empty(self, empty_registry: None) -> None:
        """Test listing available providers when registry is empty."""
        providers = EmbeddingProviderFactory.list_available_providers()
        assert len(providers) == 0

    def test_get_provider_class(self, empty_registry: None, dummy_plugin_class: type[BaseEmbeddingPlugin]) -> None:
        """Test getting a provider class by name."""
        EmbeddingProviderFactory.register_provider("dummy", dummy_plugin_class)

        result = EmbeddingProviderFactory.get_provider_class("dummy")
        assert result == dummy_plugin_class

    def test_get_provider_class_not_found(self, empty_registry: None) -> None:
        """Test getting a provider class that doesn't exist."""
        result = EmbeddingProviderFactory.get_provider_class("nonexistent")
        assert result is None

    def test_clear_providers(self, empty_registry: None, dummy_plugin_class: type[BaseEmbeddingPlugin]) -> None:
        """Test clearing all providers."""
        EmbeddingProviderFactory.register_provider("dummy", dummy_plugin_class)
        assert len(_PROVIDER_CLASSES) == 1

        EmbeddingProviderFactory.clear_providers()
        assert len(_PROVIDER_CLASSES) == 0


class TestProviderCreation:
    """Tests for creating provider instances."""

    def test_create_provider_auto_detection(
        self, empty_registry: None, dummy_plugin_class: type[BaseEmbeddingPlugin]
    ) -> None:
        """Test creating a provider via auto-detection."""
        EmbeddingProviderFactory.register_provider("dummy", dummy_plugin_class)

        provider = EmbeddingProviderFactory.create_provider("dummy/my-model")

        assert isinstance(provider, dummy_plugin_class)

    def test_create_provider_by_name(self, empty_registry: None, dummy_plugin_class: type[BaseEmbeddingPlugin]) -> None:
        """Test creating a provider by explicit name."""
        EmbeddingProviderFactory.register_provider("dummy", dummy_plugin_class)

        provider = EmbeddingProviderFactory.create_provider_by_name("dummy")

        assert isinstance(provider, dummy_plugin_class)

    def test_create_provider_unknown_model_raises(self, empty_registry: None) -> None:
        """Test that creating a provider for an unknown model raises ValueError."""
        with pytest.raises(ValueError, match="No provider found for model"):
            EmbeddingProviderFactory.create_provider("unknown/model")

    def test_create_provider_by_name_unknown_raises(self, empty_registry: None) -> None:
        """Test that creating a provider by unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            EmbeddingProviderFactory.create_provider_by_name("unknown")

    def test_create_provider_passes_config(
        self, empty_registry: None, dummy_plugin_class: type[BaseEmbeddingPlugin]
    ) -> None:
        """Test that config is passed to the provider constructor."""
        EmbeddingProviderFactory.register_provider("dummy", dummy_plugin_class)

        config = {"test_key": "test_value"}
        provider = EmbeddingProviderFactory.create_provider("dummy/my-model", config=config)

        assert provider.config == config

    def test_create_provider_passes_kwargs(
        self, empty_registry: None, dummy_plugin_class: type[BaseEmbeddingPlugin]
    ) -> None:
        """Test that kwargs are passed to the provider constructor."""
        EmbeddingProviderFactory.register_provider("dummy", dummy_plugin_class)

        provider = EmbeddingProviderFactory.create_provider("dummy/my-model", dimension=256)

        assert provider.dimension == 256

    def test_protocol_provider_receives_state_config_via_config(
        self, empty_registry: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Protocol providers should not receive plugin_config kwarg from state.

        The EmbeddingProtocol __init__ signature is (config: dict | None) and does
        not accept plugin_config. The factory should merge plugin state into config.
        """
        import shared.embedding.factory as embedding_factory

        class DummyProtocolProvider:
            API_ID = "proto"
            INTERNAL_NAME = "proto"
            PLUGIN_ID = "proto"
            PLUGIN_TYPE = "embedding"
            PLUGIN_VERSION = "1.0.0"
            PROVIDER_TYPE = "remote"
            METADATA: dict[str, Any] = {}

            def __init__(self, config: dict[str, Any] | None = None) -> None:
                self.config = config

            @classmethod
            def supports_model(cls, model_name: str) -> bool:
                return model_name == "proto-model"

            async def embed_texts(self, texts: list[str], mode: str = "document") -> list[list[float]]:  # noqa: ARG002
                return [[0.0] for _ in texts]

            @classmethod
            def get_definition(cls) -> dict[str, Any]:
                return {}

        EmbeddingProviderFactory.register_provider("proto", DummyProtocolProvider)
        monkeypatch.setattr(
            embedding_factory, "get_plugin_config", lambda _plugin_id, resolve_secrets=True: {"api_key": "x"}
        )

        provider = EmbeddingProviderFactory.create_provider("proto-model")
        assert provider.config == {"api_key": "x"}

    def test_protocol_provider_merges_explicit_config_over_state(
        self, empty_registry: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit config should override values from plugin state."""
        import shared.embedding.factory as embedding_factory

        class DummyProtocolProvider:
            API_ID = "proto"
            INTERNAL_NAME = "proto"
            PLUGIN_ID = "proto"
            PLUGIN_TYPE = "embedding"
            PLUGIN_VERSION = "1.0.0"
            PROVIDER_TYPE = "remote"
            METADATA: dict[str, Any] = {}

            def __init__(self, config: dict[str, Any] | None = None) -> None:
                self.config = config

            @classmethod
            def supports_model(cls, model_name: str) -> bool:
                return model_name == "proto-model"

            async def embed_texts(self, texts: list[str], mode: str = "document") -> list[list[float]]:  # noqa: ARG002
                return [[0.0] for _ in texts]

            @classmethod
            def get_definition(cls) -> dict[str, Any]:
                return {}

        EmbeddingProviderFactory.register_provider("proto", DummyProtocolProvider)
        monkeypatch.setattr(
            embedding_factory,
            "get_plugin_config",
            lambda _plugin_id, resolve_secrets=True: {"timeout": 1, "api_key": "from_state"},
        )

        provider = EmbeddingProviderFactory.create_provider("proto-model", config={"timeout": 2})
        assert provider.config == {"timeout": 2, "api_key": "from_state"}

    def test_protocol_provider_by_name_receives_state_config_via_config(
        self, empty_registry: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """create_provider_by_name should follow the same rules as auto-detection."""
        import shared.embedding.factory as embedding_factory

        class DummyProtocolProvider:
            API_ID = "proto"
            INTERNAL_NAME = "proto"
            PLUGIN_ID = "proto"
            PLUGIN_TYPE = "embedding"
            PLUGIN_VERSION = "1.0.0"
            PROVIDER_TYPE = "remote"
            METADATA: dict[str, Any] = {}

            def __init__(self, config: dict[str, Any] | None = None) -> None:
                self.config = config

            @classmethod
            def supports_model(cls, model_name: str) -> bool:  # noqa: ARG003
                return False

            async def embed_texts(self, texts: list[str], mode: str = "document") -> list[list[float]]:  # noqa: ARG002
                return [[0.0] for _ in texts]

            @classmethod
            def get_definition(cls) -> dict[str, Any]:
                return {}

        EmbeddingProviderFactory.register_provider("proto", DummyProtocolProvider)
        monkeypatch.setattr(
            embedding_factory, "get_plugin_config", lambda _plugin_id, resolve_secrets=True: {"api_key": "x"}
        )

        provider = EmbeddingProviderFactory.create_provider_by_name("proto")
        assert provider.config == {"api_key": "x"}


class TestModelSupport:
    """Tests for model support checking."""

    def test_is_model_supported_true(self, empty_registry: None, dummy_plugin_class: type[BaseEmbeddingPlugin]) -> None:
        """Test checking if a model is supported (positive case)."""
        EmbeddingProviderFactory.register_provider("dummy", dummy_plugin_class)

        assert EmbeddingProviderFactory.is_model_supported("dummy/my-model") is True

    def test_is_model_supported_false(self, empty_registry: None) -> None:
        """Test checking if a model is supported (negative case)."""
        assert EmbeddingProviderFactory.is_model_supported("unknown/model") is False

    def test_get_provider_for_model(self, empty_registry: None, dummy_plugin_class: type[BaseEmbeddingPlugin]) -> None:
        """Test getting the provider name for a model."""
        EmbeddingProviderFactory.register_provider("dummy", dummy_plugin_class)

        result = EmbeddingProviderFactory.get_provider_for_model("dummy/my-model")
        assert result == "dummy"

    def test_get_provider_for_model_not_found(self, empty_registry: None) -> None:
        """Test getting the provider for an unsupported model."""
        result = EmbeddingProviderFactory.get_provider_for_model("unknown/model")
        assert result is None


class TestBuiltinProviders:
    """Tests for built-in provider auto-detection.

    These tests use the clean_registry fixture which preserves built-in providers.
    """

    def test_create_provider_mock(self, clean_registry: None) -> None:
        """Test auto-detection of the mock provider."""
        provider = EmbeddingProviderFactory.create_provider("mock")

        # Should create a MockEmbeddingProvider
        assert provider.INTERNAL_NAME == "mock"

    def test_create_provider_sentence_transformer(self, clean_registry: None) -> None:
        """Test auto-detection of sentence-transformer models."""
        provider = EmbeddingProviderFactory.create_provider("sentence-transformers/all-MiniLM-L6-v2")

        # Should create a DenseLocalEmbeddingProvider
        assert provider.INTERNAL_NAME == "dense_local"

    def test_create_provider_qwen(self, clean_registry: None) -> None:
        """Test auto-detection of Qwen models."""
        provider = EmbeddingProviderFactory.create_provider("Qwen/Qwen3-Embedding-0.6B")

        # Should create a DenseLocalEmbeddingProvider
        assert provider.INTERNAL_NAME == "dense_local"

    def test_create_provider_bge(self, clean_registry: None) -> None:
        """Test auto-detection of BGE models."""
        provider = EmbeddingProviderFactory.create_provider("BAAI/bge-large-en-v1.5")

        # Should create a DenseLocalEmbeddingProvider
        assert provider.INTERNAL_NAME == "dense_local"


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_all_supported_models(
        self, empty_registry: None, dummy_plugin_class: type[BaseEmbeddingPlugin]
    ) -> None:
        """Test getting all supported models from all providers."""
        EmbeddingProviderFactory.register_provider("dummy", dummy_plugin_class)

        models = get_all_supported_models()

        # Should contain models from the dummy plugin
        # Note: The dummy plugin doesn't override list_supported_models,
        # so it returns an empty list by default
        assert isinstance(models, list)

    def test_get_model_config_from_providers(
        self, empty_registry: None, dummy_plugin_class: type[BaseEmbeddingPlugin]
    ) -> None:
        """Test getting model config from providers."""
        EmbeddingProviderFactory.register_provider("dummy", dummy_plugin_class)

        # Dummy plugin doesn't override get_model_config, so returns None
        result = get_model_config_from_providers("dummy/my-model")
        assert result is None

    def test_get_model_config_from_providers_builtin(self, clean_registry: None) -> None:
        """Test getting model config for built-in models."""
        result = get_model_config_from_providers("mock")

        # Mock provider should return a config
        assert result is not None
        assert result.name == "mock"
