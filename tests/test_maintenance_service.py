"""Tests for the maintenance service."""

from unittest.mock import Mock, patch

import pytest
from vecpipe.maintenance import QdrantMaintenanceService

# Mock settings before importing maintenance module
with patch("vecpipe.maintenance.settings") as mock_settings:
    mock_settings.DEFAULT_COLLECTION = "work_docs"
    mock_settings.QDRANT_HOST = "localhost"
    mock_settings.QDRANT_PORT = 6333
    mock_settings.INTERNAL_API_KEY = None


@pytest.fixture()
def mock_qdrant_client() -> Mock:
    """Create a mock Qdrant client."""
    return Mock()


@pytest.fixture()
def maintenance_service(mock_qdrant_client, monkeypatch) -> QdrantMaintenanceService:
    """Create a maintenance service instance with mocked dependencies."""
    monkeypatch.setattr("vecpipe.maintenance.settings.DEFAULT_COLLECTION", "work_docs")
    monkeypatch.setattr("vecpipe.maintenance.settings.QDRANT_HOST", "localhost")
    monkeypatch.setattr("vecpipe.maintenance.settings.QDRANT_PORT", 6333)
    monkeypatch.setattr("vecpipe.maintenance.settings.INTERNAL_API_KEY", None)

    with patch("vecpipe.maintenance.QdrantClient", return_value=mock_qdrant_client):
        return QdrantMaintenanceService(
            qdrant_host="localhost", qdrant_port=6333, webui_host="localhost", webui_port=8080
        )


class TestQdrantMaintenanceService:
    """Test the maintenance service functionality."""

    def test_get_operation_collections_success(self, maintenance_service) -> None:
        """Test successful retrieval of operation collections from API."""
        mock_response = Mock()
        mock_response.json.return_value = ["operation_1", "operation_2", "operation_3"]
        mock_response.raise_for_status = Mock()

        # No need to mock HTTP calls as get_operation_collections now only returns default collection
        collections = maintenance_service.get_operation_collections()

        # Should only include DEFAULT_COLLECTION in new architecture
        assert collections == ["work_docs"]

    def test_get_operation_collections_with_retry(self, maintenance_service) -> None:
        """Test that get_operation_collections now only returns default collection without retry."""
        # In new architecture, no API calls are made
        collections = maintenance_service.get_operation_collections()

        assert collections == ["work_docs"]

    def test_collection_not_found_handling(self, maintenance_service) -> None:
        """Test handling of collection not found errors during deletion."""
        # In new architecture, all operation_* collections are orphaned
        # Mock Qdrant collections
        qdrant_collections = []
        for name in ["work_docs", "operation_operation_1", "operation_operation_2"]:
            mock_col = Mock()
            mock_col.name = name
            qdrant_collections.append(mock_col)
        maintenance_service.client.get_collections.return_value = Mock(collections=qdrant_collections)

        # Mock delete to raise an exception
        maintenance_service.client.delete_collection.side_effect = Exception("Collection not found")

        # Run cleanup - should handle the error gracefully
        result = maintenance_service.cleanup_orphaned_collections(dry_run=False)

        # In new architecture, both operation collections are orphaned
        assert set(result["orphaned_collections"]) == {"operation_operation_1", "operation_operation_2"}
        assert result["deleted_collections"] == []
        # Both collections should have attempted deletion
        assert maintenance_service.client.delete_collection.call_count == 2

    def test_internal_api_key_configuration(self, monkeypatch) -> None:
        """Test that maintenance service is properly configured."""
        # Test that service can be created with API key configured
        monkeypatch.setattr("vecpipe.maintenance.settings.INTERNAL_API_KEY", "test-api-key")
        monkeypatch.setattr("vecpipe.maintenance.settings.DEFAULT_COLLECTION", "work_docs")
        monkeypatch.setattr("vecpipe.maintenance.settings.QDRANT_HOST", "localhost")
        monkeypatch.setattr("vecpipe.maintenance.settings.QDRANT_PORT", 6333)

        with patch("vecpipe.maintenance.QdrantClient"):
            service = QdrantMaintenanceService(webui_host="localhost", webui_port=8080)
            collections = service.get_operation_collections()

            # In new architecture, only returns default collection
            assert collections == ["work_docs"]
