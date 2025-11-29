"""Tests for the embedding plugin loader."""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from shared.embedding import plugin_loader
from shared.embedding.factory import _PROVIDER_CLASSES
from shared.embedding.plugin_base import BaseEmbeddingPlugin, EmbeddingProviderDefinition
from shared.embedding.provider_registry import (
    _PROVIDERS,
    _clear_caches,
    get_provider_definition,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ValidPlugin(BaseEmbeddingPlugin):
    """A valid plugin class for testing."""

    INTERNAL_NAME = "test_plugin"
    API_ID = "test_plugin"
    PROVIDER_TYPE = "local"
    METADATA = {
        "display_name": "Test Plugin",
        "description": "A test plugin",
    }

    def __init__(self, config: Any | None = None, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)
        self._initialized = False
        self.dimension = 128
        self.model_name: str | None = None

    @classmethod
    def get_definition(cls) -> EmbeddingProviderDefinition:
        return EmbeddingProviderDefinition(
            api_id=cls.API_ID,
            internal_id=cls.INTERNAL_NAME,
            display_name="Test Plugin",
            description="A test plugin for testing",
            provider_type="local",
            is_plugin=False,  # Plugin loader should mark this as True
        )

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("test/")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self._initialized = True

    async def embed_texts(self, texts: list[str], batch_size: int = 32, **kwargs: Any) -> NDArray[np.float32]:
        return np.zeros((len(texts), self.dimension), dtype=np.float32)

    async def embed_single(self, text: str, **kwargs: Any) -> NDArray[np.float32]:
        return np.zeros(self.dimension, dtype=np.float32)

    def get_dimension(self) -> int:
        return self.dimension

    def get_model_info(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "dimension": self.dimension}

    async def cleanup(self) -> None:
        self._initialized = False


class TestPluginLoader:
    """Tests for the plugin loading functionality."""

    def test_plugin_loader_registers_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the plugin loader registers a valid provider."""

        class DummyEntryPoint:
            name = "test_plugin"

            def load(self) -> type:
                return ValidPlugin

        class DummyEntryPoints:
            def select(self, group: str) -> list:
                assert group == plugin_loader.ENTRYPOINT_GROUP
                return [DummyEntryPoint()]

        # Enable plugins
        monkeypatch.setenv(plugin_loader.ENV_FLAG, "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        # Save original state
        original_classes = dict(_PROVIDER_CLASSES)
        original_providers = dict(_PROVIDERS)
        _clear_caches()
        plugin_loader._reset_plugin_loader_state()  # Reset for fresh test

        try:
            registered = plugin_loader.load_embedding_plugins()

            assert "test_plugin" in registered

            # Verify registered in factory
            assert "test_plugin" in _PROVIDER_CLASSES

            # Verify definition registered
            definition = get_provider_definition("test_plugin")
            assert definition is not None
            assert definition.is_plugin is True
            assert definition.display_name == "Test Plugin"

        finally:
            # Restore state
            _PROVIDER_CLASSES.clear()
            _PROVIDER_CLASSES.update(original_classes)
            _PROVIDERS.clear()
            _PROVIDERS.update(original_providers)
            _clear_caches()
            plugin_loader._reset_plugin_loader_state()

    def test_plugin_loader_disabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that plugins can be disabled via environment variable."""
        plugin_loader._reset_plugin_loader_state()  # Reset for fresh test
        monkeypatch.setenv(plugin_loader.ENV_FLAG, "false")

        registered = plugin_loader.load_embedding_plugins()

        assert registered == []

    def test_plugin_loader_disabled_via_env_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that plugins can be disabled with '0'."""
        plugin_loader._reset_plugin_loader_state()  # Reset for fresh test
        monkeypatch.setenv(plugin_loader.ENV_FLAG, "0")

        registered = plugin_loader.load_embedding_plugins()

        assert registered == []

    def test_plugin_loader_marks_as_plugin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the loader marks definitions as plugins."""

        class DummyEntryPoint:
            name = "test_plugin"

            def load(self) -> type:
                return ValidPlugin

        class DummyEntryPoints:
            def select(self, group: str) -> list:
                return [DummyEntryPoint()]

        monkeypatch.setenv(plugin_loader.ENV_FLAG, "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        original_classes = dict(_PROVIDER_CLASSES)
        original_providers = dict(_PROVIDERS)
        _clear_caches()
        plugin_loader._reset_plugin_loader_state()  # Reset for fresh test

        try:
            plugin_loader.load_embedding_plugins()

            definition = get_provider_definition("test_plugin")
            assert definition is not None
            assert definition.is_plugin is True

        finally:
            _PROVIDER_CLASSES.clear()
            _PROVIDER_CLASSES.update(original_classes)
            _PROVIDERS.clear()
            _PROVIDERS.update(original_providers)
            _clear_caches()
            plugin_loader._reset_plugin_loader_state()

    def test_plugin_loader_skips_invalid_contract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that invalid plugins are skipped."""

        class InvalidPlugin:
            # Missing INTERNAL_NAME, API_ID, PROVIDER_TYPE
            pass

        class DummyEntryPoint:
            name = "invalid_plugin"

            def load(self) -> type:
                return InvalidPlugin

        class DummyEntryPoints:
            def select(self, group: str) -> list:
                return [DummyEntryPoint()]

        monkeypatch.setenv(plugin_loader.ENV_FLAG, "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        original_classes = dict(_PROVIDER_CLASSES)
        original_providers = dict(_PROVIDERS)
        _clear_caches()
        plugin_loader._reset_plugin_loader_state()  # Reset for fresh test

        try:
            registered = plugin_loader.load_embedding_plugins()

            assert "invalid_plugin" not in registered
            assert "invalid_plugin" not in _PROVIDER_CLASSES

        finally:
            _PROVIDER_CLASSES.clear()
            _PROVIDER_CLASSES.update(original_classes)
            _PROVIDERS.clear()
            _PROVIDERS.update(original_providers)
            _clear_caches()
            plugin_loader._reset_plugin_loader_state()

    def test_plugin_loader_handles_entry_point_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that errors during entry point loading are handled."""
        plugin_loader._reset_plugin_loader_state()  # Reset for fresh test

        class FailingEntryPoint:
            name = "failing_plugin"

            def load(self) -> None:
                raise ImportError("Failed to import")

        class DummyEntryPoints:
            def select(self, group: str) -> list:
                return [FailingEntryPoint()]

        monkeypatch.setenv(plugin_loader.ENV_FLAG, "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        # Should not raise, just return empty list
        registered = plugin_loader.load_embedding_plugins()
        assert registered == []

    def test_load_embedding_plugins_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that load_embedding_plugins only loads once (idempotent)."""
        call_count = 0

        class DummyEntryPoint:
            name = "test_plugin"

            def load(self) -> type:
                return ValidPlugin

        class CountingEntryPoints:
            def select(self, group: str) -> list:
                nonlocal call_count
                call_count += 1
                return [DummyEntryPoint()]

        monkeypatch.setenv(plugin_loader.ENV_FLAG, "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: CountingEntryPoints())

        # Save original state and reset
        original_classes = dict(_PROVIDER_CLASSES)
        original_providers = dict(_PROVIDERS)
        _clear_caches()
        plugin_loader._reset_plugin_loader_state()

        try:
            # First call should query entry points
            result1 = plugin_loader.load_embedding_plugins()
            assert call_count == 1
            assert "test_plugin" in result1

            # Second call should be idempotent - no new entry_points query
            result2 = plugin_loader.load_embedding_plugins()
            assert call_count == 1  # Still 1, not 2
            assert result2 == result1

        finally:
            # Restore state
            _PROVIDER_CLASSES.clear()
            _PROVIDER_CLASSES.update(original_classes)
            _PROVIDERS.clear()
            _PROVIDERS.update(original_providers)
            _clear_caches()
            plugin_loader._reset_plugin_loader_state()


class TestValidatePluginContract:
    """Tests for the plugin contract validation."""

    def test_validate_contract_valid(self) -> None:
        """Test validation of a valid plugin."""
        is_valid, error = plugin_loader._validate_plugin_contract(ValidPlugin)

        assert is_valid is True
        assert error is None

    def test_validate_contract_missing_internal_name(self) -> None:
        """Test validation fails when INTERNAL_NAME is missing."""

        class MissingInternalName(BaseEmbeddingPlugin):
            INTERNAL_NAME = ""  # Empty
            API_ID = "test"
            PROVIDER_TYPE = "local"

            @classmethod
            def get_definition(cls) -> EmbeddingProviderDefinition:
                return EmbeddingProviderDefinition(
                    api_id="test", internal_id="test", display_name="Test", description="Test", provider_type="local"
                )

            @classmethod
            def supports_model(cls, _model_name: str) -> bool:
                return False

            @property
            def is_initialized(self) -> bool:
                return False

            async def initialize(self, model_name: str, **kwargs: Any) -> None:
                pass

            async def embed_texts(self, texts: list[str], **kwargs: Any) -> NDArray[np.float32]:
                return np.zeros((len(texts), 128), dtype=np.float32)

            async def embed_single(self, text: str, **kwargs: Any) -> NDArray[np.float32]:
                return np.zeros(128, dtype=np.float32)

            def get_dimension(self) -> int:
                return 128

            def get_model_info(self) -> dict[str, Any]:
                return {}

            async def cleanup(self) -> None:
                pass

        is_valid, error = plugin_loader._validate_plugin_contract(MissingInternalName)

        assert is_valid is False
        assert error is not None
        assert "INTERNAL_NAME" in error

    def test_validate_contract_missing_api_id(self) -> None:
        """Test validation fails when API_ID is missing."""

        class MissingApiId(BaseEmbeddingPlugin):
            INTERNAL_NAME = "test"
            API_ID = ""  # Empty
            PROVIDER_TYPE = "local"

            @classmethod
            def get_definition(cls) -> EmbeddingProviderDefinition:
                return EmbeddingProviderDefinition(
                    api_id="test", internal_id="test", display_name="Test", description="Test", provider_type="local"
                )

            @classmethod
            def supports_model(cls, _model_name: str) -> bool:
                return False

            @property
            def is_initialized(self) -> bool:
                return False

            async def initialize(self, model_name: str, **kwargs: Any) -> None:
                pass

            async def embed_texts(self, texts: list[str], **kwargs: Any) -> NDArray[np.float32]:
                return np.zeros((len(texts), 128), dtype=np.float32)

            async def embed_single(self, text: str, **kwargs: Any) -> NDArray[np.float32]:
                return np.zeros(128, dtype=np.float32)

            def get_dimension(self) -> int:
                return 128

            def get_model_info(self) -> dict[str, Any]:
                return {}

            async def cleanup(self) -> None:
                pass

        is_valid, error = plugin_loader._validate_plugin_contract(MissingApiId)

        assert is_valid is False
        assert error is not None
        assert "API_ID" in error

    def test_validate_contract_invalid_provider_type(self) -> None:
        """Test validation fails when PROVIDER_TYPE is invalid."""

        class InvalidProviderType(BaseEmbeddingPlugin):
            INTERNAL_NAME = "test"
            API_ID = "test"
            PROVIDER_TYPE = "invalid"  # Not local/remote/hybrid

            @classmethod
            def get_definition(cls) -> EmbeddingProviderDefinition:
                return EmbeddingProviderDefinition(
                    api_id="test", internal_id="test", display_name="Test", description="Test", provider_type="local"
                )

            @classmethod
            def supports_model(cls, _model_name: str) -> bool:
                return False

            @property
            def is_initialized(self) -> bool:
                return False

            async def initialize(self, model_name: str, **kwargs: Any) -> None:
                pass

            async def embed_texts(self, texts: list[str], **kwargs: Any) -> NDArray[np.float32]:
                return np.zeros((len(texts), 128), dtype=np.float32)

            async def embed_single(self, text: str, **kwargs: Any) -> NDArray[np.float32]:
                return np.zeros(128, dtype=np.float32)

            def get_dimension(self) -> int:
                return 128

            def get_model_info(self) -> dict[str, Any]:
                return {}

            async def cleanup(self) -> None:
                pass

        is_valid, error = plugin_loader._validate_plugin_contract(InvalidProviderType)

        assert is_valid is False
        assert error is not None
        assert "PROVIDER_TYPE" in error

    def test_validate_contract_missing_method(self) -> None:
        """Test validation fails when a required method is missing."""

        class MissingMethod:
            INTERNAL_NAME = "test"
            API_ID = "test"
            PROVIDER_TYPE = "local"

            @classmethod
            def get_definition(cls) -> EmbeddingProviderDefinition:
                return EmbeddingProviderDefinition(
                    api_id="test", internal_id="test", display_name="Test", description="Test", provider_type="local"
                )

            @classmethod
            def supports_model(cls, _model_name: str) -> bool:
                return False

            # Missing: initialize, embed_texts, embed_single, cleanup

        is_valid, error = plugin_loader._validate_plugin_contract(MissingMethod)

        assert is_valid is False
        assert error is not None
        assert "missing required method" in error


class TestEnsureProvidersRegistered:
    """Tests for ensure_providers_registered."""

    def test_ensure_providers_registered_idempotent(self, clean_registry: None) -> None:
        """Test that ensure_providers_registered is idempotent."""
        # First call
        plugin_loader.ensure_providers_registered()
        first_count = len(_PROVIDER_CLASSES)

        # Second call should not duplicate registrations
        plugin_loader.ensure_providers_registered()
        second_count = len(_PROVIDER_CLASSES)

        assert first_count == second_count

    def test_ensure_providers_registered_registers_builtins(self, clean_registry: None) -> None:
        """Test that ensure_providers_registered has builtins available.

        Note: We use clean_registry (not empty_registry) because the built-in
        providers are registered on module import and cannot be unregistered
        without unloading the module. This test verifies the builtins exist.
        """
        plugin_loader.ensure_providers_registered()

        # Should have at least mock and dense_local
        assert "mock" in _PROVIDER_CLASSES or "dense_local" in _PROVIDER_CLASSES


class TestCoerceClass:
    """Tests for the _coerce_class helper."""

    def test_coerce_class_with_class(self) -> None:
        """Test coercing a class directly."""

        class MyClass:
            pass

        result = plugin_loader._coerce_class(MyClass)
        assert result is MyClass

    def test_coerce_class_with_callable_returning_class(self) -> None:
        """Test coercing a callable that returns a class."""

        class MyClass:
            pass

        def factory() -> type:
            return MyClass

        result = plugin_loader._coerce_class(factory)
        assert result is MyClass

    def test_coerce_class_with_callable_returning_instance(self) -> None:
        """Test coercing a callable that returns an instance."""

        class MyClass:
            pass

        def factory() -> MyClass:
            return MyClass()

        result = plugin_loader._coerce_class(factory)
        assert result is MyClass

    def test_coerce_class_with_non_callable(self) -> None:
        """Test coercing a non-callable non-class returns None."""
        result = plugin_loader._coerce_class("not a class")
        assert result is None

    def test_coerce_class_with_callable_returning_none(self) -> None:
        """Test coercing a callable that returns None."""

        def factory() -> None:
            return None

        result = plugin_loader._coerce_class(factory)
        assert result is None


class TestServiceIntegration:
    """Tests for plugin loading integration with embedding service."""

    @pytest.mark.asyncio()
    async def test_get_embedding_service_registers_plugins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that get_embedding_service loads plugins via _ensure_providers_registered."""
        from shared.embedding import service
        from shared.embedding.service import _ensure_providers_registered

        class DummyEntryPoint:
            name = "test_plugin"

            def load(self) -> type:
                return ValidPlugin

        class DummyEntryPoints:
            def select(self, group: str) -> list:
                return [DummyEntryPoint()]

        monkeypatch.setenv(plugin_loader.ENV_FLAG, "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        # Save original state and reset
        original_classes = dict(_PROVIDER_CLASSES)
        original_providers = dict(_PROVIDERS)
        _clear_caches()
        plugin_loader._reset_plugin_loader_state()

        # Also reset service state
        original_service = service._embedding_service
        original_model = service._current_model_name
        service._embedding_service = None
        service._current_model_name = None

        try:
            # Call _ensure_providers_registered (what get_embedding_service calls)
            _ensure_providers_registered()

            # Verify plugin is registered in factory
            assert "test_plugin" in _PROVIDER_CLASSES

            # Verify definition is registered
            definition = get_provider_definition("test_plugin")
            assert definition is not None
            assert definition.is_plugin is True

        finally:
            # Restore state
            _PROVIDER_CLASSES.clear()
            _PROVIDER_CLASSES.update(original_classes)
            _PROVIDERS.clear()
            _PROVIDERS.update(original_providers)
            _clear_caches()
            plugin_loader._reset_plugin_loader_state()
            service._embedding_service = original_service
            service._current_model_name = original_model
