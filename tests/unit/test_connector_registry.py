"""Unit tests for ConnectorRegistry."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from shared.connectors.base import BaseConnector
from shared.plugins.manifest import PluginManifest
from shared.plugins.registry import PluginRecord, PluginSource, plugin_registry
from webui.services.connector_registry import (
    _build_definition,
    get_connector_catalog,
    get_connector_definition,
    invalidate_connector_cache,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from shared.dtos.ingestion import IngestedDocument


class DummyConnector(BaseConnector):
    """Dummy connector for testing."""

    PLUGIN_ID = "dummy"
    METADATA = {
        "name": "Dummy Connector",
        "description": "A test connector",
        "icon": "test-icon",
        "supports_sync": True,
        "preview_endpoint": "/preview/dummy",
    }

    @classmethod
    def get_config_fields(cls):
        return [{"name": "field1", "type": "text", "label": "Field 1", "required": True}]

    @classmethod
    def get_secret_fields(cls):
        return [{"name": "secret1", "label": "Secret 1", "required": True}]

    async def authenticate(self) -> bool:
        return True

    async def load_documents(self) -> AsyncIterator[IngestedDocument]:
        return
        yield


class MinimalConnector(BaseConnector):
    """Minimal connector with no metadata."""

    PLUGIN_ID = "minimal"

    @classmethod
    def get_config_fields(cls):
        return []

    @classmethod
    def get_secret_fields(cls):
        return []

    async def authenticate(self) -> bool:
        return True

    async def load_documents(self) -> AsyncIterator[IngestedDocument]:
        return
        yield


@pytest.fixture(autouse=True)
def _clear_registry():
    """Clear plugin registry and connector cache before and after each test."""
    plugin_registry.reset()
    invalidate_connector_cache()
    yield
    plugin_registry.reset()
    invalidate_connector_cache()


def _register_connector(connector_cls: type[BaseConnector], source: PluginSource = PluginSource.EXTERNAL):
    """Register a connector in the plugin registry."""
    plugin_id = getattr(connector_cls, "PLUGIN_ID", connector_cls.__name__)
    manifest = PluginManifest(
        id=plugin_id,
        type="connector",
        version="1.0.0",
        display_name=plugin_id,
        description="",
    )
    plugin_registry.register(
        PluginRecord(
            plugin_type="connector",
            plugin_id=plugin_id,
            plugin_version="1.0.0",
            manifest=manifest,
            plugin_class=connector_cls,
            source=source,
        )
    )


class TestBuildDefinition:
    """Tests for _build_definition function."""

    def test_build_definition_full_metadata(self):
        """Test building definition with full metadata."""
        definition = _build_definition(DummyConnector)

        assert definition["name"] == "Dummy Connector"
        assert definition["description"] == "A test connector"
        assert definition["icon"] == "test-icon"
        assert definition["supports_sync"] is True
        assert definition["preview_endpoint"] == "/preview/dummy"
        assert len(definition["fields"]) == 1
        assert len(definition["secrets"]) == 1

    def test_build_definition_minimal_metadata(self):
        """Test building definition with minimal metadata."""
        definition = _build_definition(MinimalConnector)

        assert definition["name"] == "MinimalConnector"
        assert definition["description"] == ""
        assert definition["icon"] == "plug"
        assert definition["supports_sync"] is True
        assert definition["preview_endpoint"] == ""
        assert definition["fields"] == []
        assert definition["secrets"] == []

    def test_build_definition_display_name_fallback(self):
        """Test building definition falls back to display_name."""

        class DisplayNameConnector(MinimalConnector):
            PLUGIN_ID = "display-name-test"
            METADATA = {"display_name": "Display Name Test"}

        definition = _build_definition(DisplayNameConnector)
        assert definition["name"] == "Display Name Test"


class TestGetConnectorCatalog:
    """Tests for get_connector_catalog function."""

    def test_get_catalog_empty(self):
        """Test getting empty catalog."""
        with patch("webui.services.connector_registry.load_plugins"):
            catalog = get_connector_catalog()
            assert catalog == {}

    def test_get_catalog_with_connectors(self):
        """Test getting catalog with registered connectors."""
        _register_connector(DummyConnector)
        _register_connector(MinimalConnector)

        with patch("webui.services.connector_registry.load_plugins"):
            catalog = get_connector_catalog()

            assert "dummy" in catalog
            assert "minimal" in catalog
            assert catalog["dummy"]["name"] == "Dummy Connector"

    def test_get_catalog_excludes_disabled_external(self):
        """Test catalog excludes disabled external plugins."""
        _register_connector(DummyConnector, PluginSource.EXTERNAL)
        plugin_registry.set_disabled({"dummy"})

        with patch("webui.services.connector_registry.load_plugins"):
            catalog = get_connector_catalog()

            assert "dummy" not in catalog

    def test_get_catalog_includes_disabled_builtin(self):
        """Test catalog includes disabled builtin plugins."""
        _register_connector(DummyConnector, PluginSource.BUILTIN)
        plugin_registry.set_disabled({"dummy"})

        with patch("webui.services.connector_registry.load_plugins"):
            catalog = get_connector_catalog()

            assert "dummy" in catalog


class TestGetConnectorDefinition:
    """Tests for get_connector_definition function."""

    def test_get_definition_found(self):
        """Test getting existing connector definition."""
        _register_connector(DummyConnector)

        with patch("webui.services.connector_registry.load_plugins"):
            definition = get_connector_definition("dummy")

            assert definition is not None
            assert definition["name"] == "Dummy Connector"

    def test_get_definition_not_found(self):
        """Test getting nonexistent connector returns None."""
        with patch("webui.services.connector_registry.load_plugins"):
            definition = get_connector_definition("nonexistent")

            assert definition is None

    def test_get_definition_disabled_external(self):
        """Test getting disabled external connector returns None."""
        _register_connector(DummyConnector, PluginSource.EXTERNAL)
        plugin_registry.set_disabled({"dummy"})

        with patch("webui.services.connector_registry.load_plugins"):
            definition = get_connector_definition("dummy")

            assert definition is None

    def test_get_definition_disabled_builtin(self):
        """Test getting disabled builtin connector returns definition."""
        _register_connector(DummyConnector, PluginSource.BUILTIN)
        plugin_registry.set_disabled({"dummy"})

        with patch("webui.services.connector_registry.load_plugins"):
            definition = get_connector_definition("dummy")

            # Builtin plugins are not filtered by disabled status
            assert definition is not None
