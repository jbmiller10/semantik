"""Unit tests for ConnectorFactory."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from shared.connectors.base import BaseConnector
from shared.dtos.ingestion import IngestedDocument
from shared.plugins.exceptions import PluginDuplicateError
from shared.plugins.manifest import PluginManifest
from shared.plugins.registry import PluginRecord, PluginSource, plugin_registry
from webui.services.connector_factory import ConnectorFactory


def _register_connector(connector_type: str, connector_cls: type[BaseConnector]) -> None:
    plugin_id = connector_type.strip().lower()
    connector_cls.PLUGIN_ID = plugin_id  # type: ignore[assignment]
    manifest = PluginManifest(
        id=plugin_id,
        type="connector",
        version=getattr(connector_cls, "PLUGIN_VERSION", "0.0.0"),
        display_name=plugin_id,
        description="",
    )
    plugin_registry.register(
        PluginRecord(
            plugin_type="connector",
            plugin_id=plugin_id,
            plugin_version=getattr(connector_cls, "PLUGIN_VERSION", "0.0.0"),
            manifest=manifest,
            plugin_class=connector_cls,
            source=PluginSource.EXTERNAL,
        )
    )


class DummyConnector(BaseConnector):
    """Dummy connector for factory tests."""

    async def authenticate(self) -> bool:
        return True

    async def load_documents(self) -> AsyncIterator[IngestedDocument]:
        return
        yield  # Make this a generator


class AnotherConnector(BaseConnector):
    """Another dummy connector for factory tests."""

    async def authenticate(self) -> bool:
        return False

    async def load_documents(self) -> AsyncIterator[IngestedDocument]:
        return
        yield  # Make this a generator


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    """Clear registry before and after each test."""
    plugin_registry.reset()
    yield
    plugin_registry.reset()


class TestConnectorFactory:
    """Tests for ConnectorFactory."""

    def test_register_and_get_connector(self) -> None:
        """Test registering and retrieving a connector."""
        _register_connector("dummy", DummyConnector)

        connector = ConnectorFactory.get_connector("dummy", {"key": "value"})

        assert isinstance(connector, DummyConnector)
        assert connector.config == {"key": "value"}

    def test_get_connector_unknown_type_raises(self) -> None:
        """Test get_connector raises ValueError for unknown type."""
        with pytest.raises(ValueError, match="Unknown source type: 'unknown'"):
            ConnectorFactory.get_connector("unknown", {})

    def test_get_connector_unknown_type_shows_available_types(self) -> None:
        """Test error message includes available types."""
        with pytest.raises(ValueError, match="Available types") as excinfo:
            ConnectorFactory.get_connector("unknown", {})

        message = str(excinfo.value)
        assert "Available types" in message
        assert "directory" in message

    def test_source_type_normalized_lowercase(self) -> None:
        """Test source_type is case-insensitive."""
        _register_connector("MyType", DummyConnector)

        # Should find with different cases
        connector1 = ConnectorFactory.get_connector("mytype", {})
        connector2 = ConnectorFactory.get_connector("MYTYPE", {})
        connector3 = ConnectorFactory.get_connector("MyType", {})

        assert isinstance(connector1, DummyConnector)
        assert isinstance(connector2, DummyConnector)
        assert isinstance(connector3, DummyConnector)

    def test_source_type_normalized_stripped(self) -> None:
        """Test source_type is stripped of whitespace."""
        _register_connector("test", DummyConnector)

        connector = ConnectorFactory.get_connector("  test  ", {})
        assert isinstance(connector, DummyConnector)

    def test_list_available_types_includes_builtins(self) -> None:
        """Test listing includes built-in connector types."""
        types = ConnectorFactory.list_available_types()
        assert {"directory", "git", "imap"}.issubset(set(types))

    def test_list_available_types_multiple(self) -> None:
        """Test listing registered types."""
        _register_connector("type1", DummyConnector)
        _register_connector("type2", AnotherConnector)

        types = ConnectorFactory.list_available_types()
        assert {"type1", "type2"}.issubset(set(types))

    def test_error_message_includes_available_types(self) -> None:
        """Test error message shows available types."""
        _register_connector("web", AnotherConnector)
        _register_connector("repo", DummyConnector)

        with pytest.raises(ValueError, match="Unknown source type") as excinfo:
            ConnectorFactory.get_connector("slack", {})

        message = str(excinfo.value)
        assert "Unknown source type" in message
        assert "web" in message

    def test_register_duplicate_raises_error(self) -> None:
        """Test registering different class with same ID raises PluginDuplicateError."""
        _register_connector("test", DummyConnector)
        with pytest.raises(PluginDuplicateError) as exc_info:
            _register_connector("test", AnotherConnector)
        assert exc_info.value.error_code == "PLUGIN_CLASS_CONFLICT"
        assert exc_info.value.plugin_id == "test"

        # First connector should still be available
        connector = ConnectorFactory.get_connector("test", {})
        assert isinstance(connector, DummyConnector)

    def test_config_passed_to_connector(self) -> None:
        """Test config dictionary is passed to connector constructor."""
        _register_connector("test", DummyConnector)

        config: dict[str, Any] = {"source_path": "/data", "recursive": True, "count": 42}
        connector = ConnectorFactory.get_connector("test", config)

        assert connector.config == config
        assert connector.config["source_path"] == "/data"
        assert connector.config["recursive"] is True
        assert connector.config["count"] == 42

    def test_register_connector_normalizes_type(self) -> None:
        """Test register_connector normalizes type to lowercase."""
        _register_connector("UPPERCASE", DummyConnector)

        # Should be stored as lowercase
        assert plugin_registry.get("connector", "uppercase") is not None
        assert plugin_registry.get("connector", "UPPERCASE") is None
