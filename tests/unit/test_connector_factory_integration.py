"""Integration tests for ConnectorFactory with LocalFileConnector."""

from pathlib import Path

import pytest

from shared.connectors.local import LocalFileConnector
from webui.services.connector_factory import ConnectorFactory, register_connector


@pytest.fixture(autouse=True)
def _ensure_directory_registered() -> None:
    """Ensure 'directory' connector is registered before each test.

    This is needed because test_connector_factory.py clears the registry
    with an autouse fixture.
    """
    register_connector("directory", LocalFileConnector)


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
