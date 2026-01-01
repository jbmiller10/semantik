"""Integration tests for ConnectorFactory with LocalFileConnector."""

from pathlib import Path

import pytest

from shared.connectors.local import LocalFileConnector
from shared.plugins.loader import load_plugins
from shared.plugins.registry import plugin_registry
from webui.services.connector_factory import ConnectorFactory


@pytest.fixture(autouse=True)
def _ensure_directory_registered():
    """Ensure 'directory' connector is registered before each test."""
    plugin_registry.reset()
    load_plugins(plugin_types={"connector"}, include_external=False)
    yield
    plugin_registry.reset()


class TestConnectorFactoryIntegration:
    """Test factory returns correct connector for 'directory' type."""

    def test_get_directory_connector(self, tmp_path: Path) -> None:
        """Test ConnectorFactory returns LocalFileConnector for 'directory'."""
        connector = ConnectorFactory.get_connector(
            "directory",
            {"path": str(tmp_path)},
        )
        assert isinstance(connector, LocalFileConnector)
        assert connector.config["path"] == str(tmp_path)

    def test_directory_type_registered(self) -> None:
        """Test 'directory' is in available types."""
        types = ConnectorFactory.list_available_types()
        assert "directory" in types

    def test_directory_type_case_insensitive(self, tmp_path: Path) -> None:
        """Test 'DIRECTORY' works case-insensitively."""
        connector = ConnectorFactory.get_connector(
            "DIRECTORY",
            {"path": str(tmp_path)},
        )
        assert isinstance(connector, LocalFileConnector)

    def test_directory_connector_validates_config(self) -> None:
        """Test factory-created connector validates config."""
        # This should raise because 'path' is required
        with pytest.raises(ValueError, match="requires 'path'"):
            ConnectorFactory.get_connector("directory", {})
