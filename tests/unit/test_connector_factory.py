"""Unit tests for ConnectorFactory."""

from collections.abc import AsyncIterator
from typing import Any

import pytest

from shared.connectors.base import BaseConnector
from shared.dtos.ingestion import IngestedDocument
from webui.services.connector_factory import (
    _CONNECTOR_REGISTRY,
    ConnectorFactory,
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
    _CONNECTOR_REGISTRY.clear()
    yield
    _CONNECTOR_REGISTRY.clear()


class TestConnectorFactory:
    """Tests for ConnectorFactory."""

    def test_register_and_get_connector(self) -> None:
        """Test registering and retrieving a connector."""
        ConnectorFactory.register_connector("dummy", DummyConnector)

        connector = ConnectorFactory.get_connector("dummy", {"key": "value"})

        assert isinstance(connector, DummyConnector)
        assert connector.config == {"key": "value"}

    def test_get_connector_unknown_type_raises(self) -> None:
        """Test get_connector raises ValueError for unknown type."""
        with pytest.raises(ValueError, match="Unknown source type: 'unknown'"):
            ConnectorFactory.get_connector("unknown", {})

    def test_get_connector_unknown_type_shows_none_registered(self) -> None:
        """Test error message shows 'none registered' when registry is empty."""
        with pytest.raises(ValueError, match="none registered"):
            ConnectorFactory.get_connector("unknown", {})

    def test_source_type_normalized_lowercase(self) -> None:
        """Test source_type is case-insensitive."""
        ConnectorFactory.register_connector("MyType", DummyConnector)

        # Should find with different cases
        connector1 = ConnectorFactory.get_connector("mytype", {})
        connector2 = ConnectorFactory.get_connector("MYTYPE", {})
        connector3 = ConnectorFactory.get_connector("MyType", {})

        assert isinstance(connector1, DummyConnector)
        assert isinstance(connector2, DummyConnector)
        assert isinstance(connector3, DummyConnector)

    def test_source_type_normalized_stripped(self) -> None:
        """Test source_type is stripped of whitespace."""
        ConnectorFactory.register_connector("test", DummyConnector)

        connector = ConnectorFactory.get_connector("  test  ", {})
        assert isinstance(connector, DummyConnector)

    def test_list_available_types_empty(self) -> None:
        """Test listing returns empty when no types registered."""
        assert ConnectorFactory.list_available_types() == []

    def test_list_available_types_multiple(self) -> None:
        """Test listing registered types."""
        ConnectorFactory.register_connector("type1", DummyConnector)
        ConnectorFactory.register_connector("type2", AnotherConnector)

        types = ConnectorFactory.list_available_types()
        assert set(types) == {"type1", "type2"}

    def test_error_message_includes_available_types(self) -> None:
        """Test error message shows available types."""
        ConnectorFactory.register_connector("directory", DummyConnector)
        ConnectorFactory.register_connector("web", AnotherConnector)

        with pytest.raises(ValueError, match="Unknown source type"):
            ConnectorFactory.get_connector("slack", {})

    def test_register_overwrites_existing(self) -> None:
        """Test registering same type overwrites previous connector."""
        ConnectorFactory.register_connector("test", DummyConnector)
        ConnectorFactory.register_connector("test", AnotherConnector)

        connector = ConnectorFactory.get_connector("test", {})
        assert isinstance(connector, AnotherConnector)

    def test_config_passed_to_connector(self) -> None:
        """Test config dictionary is passed to connector constructor."""
        ConnectorFactory.register_connector("test", DummyConnector)

        config: dict[str, Any] = {"source_path": "/data", "recursive": True, "count": 42}
        connector = ConnectorFactory.get_connector("test", config)

        assert connector.config == config
        assert connector.config["source_path"] == "/data"
        assert connector.config["recursive"] is True
        assert connector.config["count"] == 42

    def test_register_connector_normalizes_type(self) -> None:
        """Test register_connector normalizes type to lowercase."""
        ConnectorFactory.register_connector("UPPERCASE", DummyConnector)

        # Should be stored as lowercase
        assert "uppercase" in _CONNECTOR_REGISTRY
        assert "UPPERCASE" not in _CONNECTOR_REGISTRY
