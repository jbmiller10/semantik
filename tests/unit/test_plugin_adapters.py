"""Unit tests for plugin adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.plugins.adapters import (
    _metadata_value,
    get_config_schema,
    manifest_from_chunking_plugin,
    manifest_from_connector_plugin,
    manifest_from_embedding_plugin,
)


class TestMetadataValue:
    """Tests for _metadata_value helper."""

    def test_none_metadata(self):
        """Test with None metadata."""
        assert _metadata_value(None, "key") is None
        assert _metadata_value(None, "key", "default") == "default"

    def test_missing_key(self):
        """Test with missing key."""
        assert _metadata_value({"other": "value"}, "key") is None
        assert _metadata_value({"other": "value"}, "key", "default") == "default"

    def test_present_key(self):
        """Test with present key."""
        assert _metadata_value({"key": "value"}, "key") == "value"
        assert _metadata_value({"key": "value"}, "key", "default") == "value"

    def test_empty_dict(self):
        """Test with empty dict."""
        assert _metadata_value({}, "key") is None


class TestManifestFromEmbeddingPlugin:
    """Tests for manifest_from_embedding_plugin."""

    def test_basic_manifest(self):
        """Test creating manifest with basic definition."""

        @dataclass
        class MockDefinition:
            api_id: str = "test-embed"
            internal_id: str = "test_embed"
            display_name: str = "Test Embedding"
            description: str = "Test description"
            provider_type: str = "local"
            supports_quantization: bool = True
            supports_instruction: bool = False
            supports_batch_processing: bool = True
            supports_asymmetric: bool = False
            supported_models: tuple = ("model1", "model2")
            default_config: dict = None
            performance_characteristics: dict = None

            def __post_init__(self):
                if self.default_config is None:
                    self.default_config = {}
                if self.performance_characteristics is None:
                    self.performance_characteristics = {}

        class MockPlugin:
            PLUGIN_VERSION = "2.0.0"
            METADATA = {
                "author": "Test Author",
                "license": "MIT",
                "homepage": "https://example.com",
                "requires": ["dep1"],
                "semantik_version": ">=1.0.0",
            }

        definition = MockDefinition()
        manifest = manifest_from_embedding_plugin(MockPlugin, definition)

        assert manifest.id == "test-embed"
        assert manifest.type == "embedding"
        assert manifest.version == "2.0.0"
        assert manifest.display_name == "Test Embedding"
        assert manifest.description == "Test description"
        assert manifest.author == "Test Author"
        assert manifest.license == "MIT"
        assert manifest.homepage == "https://example.com"
        assert manifest.requires == ["dep1"]
        assert manifest.semantik_version == ">=1.0.0"
        assert manifest.capabilities["internal_id"] == "test_embed"
        assert manifest.capabilities["supports_quantization"] is True

    def test_manifest_with_metadata_override(self):
        """Test manifest uses METADATA for display_name if available."""

        @dataclass
        class MockDefinition:
            api_id: str = "test"
            internal_id: str = "test"
            display_name: str = "Definition Name"
            description: str = "Def description"
            provider_type: str = "local"
            supports_quantization: bool = False
            supports_instruction: bool = False
            supports_batch_processing: bool = False
            supports_asymmetric: bool = False
            supported_models: tuple = ()
            default_config: dict = None
            performance_characteristics: dict = None

            def __post_init__(self):
                if self.default_config is None:
                    self.default_config = {}
                if self.performance_characteristics is None:
                    self.performance_characteristics = {}

        class MockPlugin:
            METADATA = {
                "display_name": "Metadata Name",
                "description": "Metadata description",
            }

        definition = MockDefinition()
        manifest = manifest_from_embedding_plugin(MockPlugin, definition)

        assert manifest.display_name == "Metadata Name"
        assert manifest.description == "Metadata description"


class TestManifestFromChunkingPlugin:
    """Tests for manifest_from_chunking_plugin."""

    def test_basic_manifest(self):
        """Test creating manifest for chunking plugin."""

        class MockChunkingPlugin:
            PLUGIN_VERSION = "1.5.0"
            METADATA = {
                "display_name": "Custom Chunker",
                "description": "Chunks text",
                "author": "Author",
                "license": "Apache-2.0",
                "best_for": ["code", "markdown"],
                "pros": ["Fast"],
                "cons": ["Simple"],
            }

        manifest = manifest_from_chunking_plugin(
            MockChunkingPlugin,
            api_id="custom_chunker",
            internal_id="custom_chunker_internal",
        )

        assert manifest.id == "custom_chunker"
        assert manifest.type == "chunking"
        assert manifest.version == "1.5.0"
        assert manifest.display_name == "Custom Chunker"
        assert manifest.description == "Chunks text"
        assert manifest.capabilities["internal_id"] == "custom_chunker_internal"
        assert manifest.capabilities["best_for"] == ["code", "markdown"]
        assert manifest.capabilities["pros"] == ["Fast"]

    def test_manifest_default_display_name(self):
        """Test manifest uses formatted api_id when no display_name."""

        class MockChunkingPlugin:
            pass

        manifest = manifest_from_chunking_plugin(
            MockChunkingPlugin,
            api_id="my_custom_strategy",
            internal_id="my_custom",
        )

        assert manifest.display_name == "My Custom Strategy"

    def test_manifest_with_visual_example(self):
        """Test manifest includes visual_example capability."""

        class MockChunkingPlugin:
            METADATA = {
                "visual_example": {"url": "https://example.com/image.png", "caption": "Example"},
            }

        manifest = manifest_from_chunking_plugin(
            MockChunkingPlugin,
            api_id="visual",
            internal_id="visual",
        )

        assert manifest.capabilities["visual_example"]["url"] == "https://example.com/image.png"


class TestManifestFromConnectorPlugin:
    """Tests for manifest_from_connector_plugin."""

    def test_basic_manifest(self):
        """Test creating manifest for connector plugin."""

        class MockConnector:
            PLUGIN_VERSION = "3.0.0"
            METADATA = {
                "name": "Test Connector",
                "description": "Connects to things",
                "author": "Connector Author",
                "icon": "plug",
                "supports_sync": True,
                "preview_endpoint": "/api/preview",
            }

        manifest = manifest_from_connector_plugin(MockConnector, "test_connector")

        assert manifest.id == "test_connector"
        assert manifest.type == "connector"
        assert manifest.version == "3.0.0"
        assert manifest.display_name == "Test Connector"
        assert manifest.description == "Connects to things"
        assert manifest.capabilities["icon"] == "plug"
        assert manifest.capabilities["supports_sync"] is True

    def test_manifest_default_display_name(self):
        """Test manifest uses formatted plugin_id when no name."""

        class MockConnector:
            pass

        manifest = manifest_from_connector_plugin(MockConnector, "my_test_connector")

        assert manifest.display_name == "My Test Connector"


class TestGetConfigSchema:
    """Tests for get_config_schema function."""

    def test_get_schema_from_method(self):
        """Test getting schema from get_config_schema method."""

        class PluginWithMethod:
            @classmethod
            def get_config_schema(cls):
                return {"type": "object", "properties": {"key": {"type": "string"}}}

        schema = get_config_schema(PluginWithMethod)
        assert schema is not None
        assert schema["type"] == "object"

    def test_get_schema_from_attribute(self):
        """Test getting schema from CONFIG_SCHEMA attribute."""

        class PluginWithAttribute:
            CONFIG_SCHEMA = {"type": "object", "properties": {}}

        schema = get_config_schema(PluginWithAttribute)
        assert schema is not None
        assert schema["type"] == "object"

    def test_get_schema_method_priority(self):
        """Test method takes priority over attribute."""

        class PluginWithBoth:
            CONFIG_SCHEMA = {"type": "object", "from": "attribute"}

            @classmethod
            def get_config_schema(cls):
                return {"type": "object", "from": "method"}

        schema = get_config_schema(PluginWithBoth)
        assert schema["from"] == "method"

    def test_get_schema_none(self):
        """Test getting schema when none defined."""

        class PluginWithoutSchema:
            pass

        schema = get_config_schema(PluginWithoutSchema)
        assert schema is None

    def test_get_schema_method_exception(self):
        """Test getting schema when method raises exception."""

        class PluginWithBrokenMethod:
            @classmethod
            def get_config_schema(cls):
                raise RuntimeError("Schema error")

            CONFIG_SCHEMA = {"type": "object", "fallback": True}

        schema = get_config_schema(PluginWithBrokenMethod)
        # Falls back to CONFIG_SCHEMA
        assert schema is not None
        assert schema["fallback"] is True

    def test_get_schema_method_returns_none(self):
        """Test getting schema when method returns None."""

        class PluginMethodReturnsNone:
            @classmethod
            def get_config_schema(cls):
                return None

            CONFIG_SCHEMA = {"type": "object"}

        schema = get_config_schema(PluginMethodReturnsNone)
        # Falls back to CONFIG_SCHEMA
        assert schema is not None
        assert schema["type"] == "object"

    def test_get_schema_non_dict_attribute(self):
        """Test getting schema when attribute is not a dict."""

        class PluginWithNonDictSchema:
            CONFIG_SCHEMA = "not a dict"

        schema = get_config_schema(PluginWithNonDictSchema)
        assert schema is None
