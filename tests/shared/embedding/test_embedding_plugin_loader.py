"""Tests for the unified plugin loader (embedding)."""

from __future__ import annotations

from importlib import metadata
from typing import TYPE_CHECKING, Any

import numpy as np

from shared.embedding.factory import _PROVIDER_CLASSES
from shared.embedding.plugin_base import BaseEmbeddingPlugin, EmbeddingProviderDefinition
from shared.embedding.provider_registry import _PROVIDERS, _clear_caches, get_provider_definition
from shared.plugins.loader import load_plugins
from shared.plugins.registry import PluginSource, plugin_registry

if TYPE_CHECKING:
    import pytest
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
            is_plugin=False,
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


class ValidProtocolPlugin:
    """A valid protocol-only embedding plugin returning a dict definition."""

    PLUGIN_ID = "protocol_plugin"
    PLUGIN_TYPE = "embedding"
    PLUGIN_VERSION = "1.0.0"
    INTERNAL_NAME = "protocol_plugin"
    API_ID = "protocol_plugin"
    PROVIDER_TYPE = "remote"
    METADATA: dict[str, Any] = {
        "display_name": "Protocol Plugin",
        "description": "A protocol-only plugin",
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:  # noqa: ARG002
        pass

    @classmethod
    def get_definition(cls) -> dict[str, Any]:
        return {
            "api_id": cls.API_ID,
            "internal_id": cls.INTERNAL_NAME,
            "display_name": "Protocol Plugin",
            "description": "A protocol-only plugin for testing",
            "provider_type": cls.PROVIDER_TYPE,
        }

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("proto/")

    async def embed_texts(self, texts: list[str], mode: str = "document") -> list[list[float]]:  # noqa: ARG002
        return [[0.0] for _ in texts]

    @classmethod
    def get_manifest(cls) -> dict[str, Any]:
        return {"id": cls.API_ID, "type": cls.PLUGIN_TYPE, "version": cls.PLUGIN_VERSION}


class TestPluginLoader:
    def test_plugin_loader_registers_provider(self, monkeypatch: pytest.MonkeyPatch, clean_registry) -> None:
        class DummyEntryPoint:
            name = "test_plugin"

            def load(self) -> type:
                return ValidPlugin

        class DummyEntryPoints:
            def select(self, group: str) -> list:
                assert group == "semantik.plugins"
                return [DummyEntryPoint()]

        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
        monkeypatch.setenv("SEMANTIK_ENABLE_EMBEDDING_PLUGINS", "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        # Save original state
        original_classes = dict(_PROVIDER_CLASSES)
        original_providers = dict(_PROVIDERS)
        _clear_caches()
        plugin_registry.reset()

        try:
            load_plugins(plugin_types={"embedding"}, include_builtins=False)

            assert "test_plugin" in _PROVIDER_CLASSES
            definition = get_provider_definition("test_plugin")
            assert definition is not None
            assert definition.is_plugin is True

            record = plugin_registry.get("embedding", "test_plugin")
            assert record is not None
            assert record.source == PluginSource.EXTERNAL
        finally:
            _PROVIDER_CLASSES.clear()
            _PROVIDER_CLASSES.update(original_classes)
            _PROVIDERS.clear()
            _PROVIDERS.update(original_providers)
            _clear_caches()
            plugin_registry.reset()

    def test_plugin_loader_registers_protocol_provider(self, monkeypatch: pytest.MonkeyPatch, clean_registry) -> None:
        class DummyEntryPoint:
            name = "protocol_plugin"

            def load(self) -> type:
                return ValidProtocolPlugin

        class DummyEntryPoints:
            def select(self, group: str) -> list:
                assert group == "semantik.plugins"
                return [DummyEntryPoint()]

        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
        monkeypatch.setenv("SEMANTIK_ENABLE_EMBEDDING_PLUGINS", "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        load_plugins(plugin_types={"embedding"}, include_builtins=False)

        assert "protocol_plugin" in _PROVIDER_CLASSES
        definition = get_provider_definition("protocol_plugin")
        assert definition is not None
        assert definition.api_id == "protocol_plugin"
        assert definition.internal_id == "protocol_plugin"
        assert definition.is_plugin is True

        record = plugin_registry.get("embedding", "protocol_plugin")
        assert record is not None
        assert record.source == PluginSource.EXTERNAL

    def test_plugin_loader_disabled_via_env(self, monkeypatch: pytest.MonkeyPatch, clean_registry) -> None:
        class DummyEntryPoint:
            name = "test_plugin"

            def load(self) -> type:
                return ValidPlugin

        class DummyEntryPoints:
            def select(self, group: str) -> list:
                return [DummyEntryPoint()]

        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "false")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        plugin_registry.reset()
        load_plugins(plugin_types={"embedding"}, include_builtins=False)

        assert plugin_registry.get("embedding", "test_plugin") is None

    def test_plugin_loader_marks_as_plugin(self, monkeypatch: pytest.MonkeyPatch, clean_registry) -> None:
        class DummyEntryPoint:
            name = "test_plugin"

            def load(self) -> type:
                return ValidPlugin

        class DummyEntryPoints:
            def select(self, group: str) -> list:
                return [DummyEntryPoint()]

        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
        monkeypatch.setenv("SEMANTIK_ENABLE_EMBEDDING_PLUGINS", "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        plugin_registry.reset()
        load_plugins(plugin_types={"embedding"}, include_builtins=False)

        definition = get_provider_definition("test_plugin")
        assert definition is not None
        assert definition.is_plugin is True

    def test_plugin_loader_skips_invalid_contract(self, monkeypatch: pytest.MonkeyPatch, clean_registry) -> None:
        class InvalidPlugin:
            pass

        class DummyEntryPoint:
            name = "invalid_plugin"

            def load(self) -> type:
                return InvalidPlugin

        class DummyEntryPoints:
            def select(self, group: str) -> list:
                return [DummyEntryPoint()]

        monkeypatch.setenv("SEMANTIK_ENABLE_PLUGINS", "true")
        monkeypatch.setenv("SEMANTIK_ENABLE_EMBEDDING_PLUGINS", "true")
        monkeypatch.setattr(metadata, "entry_points", lambda: DummyEntryPoints())

        original_classes = dict(_PROVIDER_CLASSES)
        original_providers = dict(_PROVIDERS)
        _clear_caches()
        plugin_registry.reset()

        try:
            load_plugins(plugin_types={"embedding"}, include_builtins=False)
            assert "invalid_plugin" not in _PROVIDER_CLASSES
            assert plugin_registry.get("embedding", "invalid_plugin") is None
        finally:
            _PROVIDER_CLASSES.clear()
            _PROVIDER_CLASSES.update(original_classes)
            _PROVIDERS.clear()
            _PROVIDERS.update(original_providers)
            _clear_caches()
            plugin_registry.reset()
