"""Tests for the maintenance service."""

from unittest.mock import Mock, patch

import httpx
import pytest

# Mock settings before importing maintenance module
with patch("packages.vecpipe.maintenance.settings") as mock_settings:
    mock_settings.DEFAULT_COLLECTION = "work_docs"
    mock_settings.QDRANT_HOST = "localhost"
    mock_settings.QDRANT_PORT = 6333
    mock_settings.INTERNAL_API_KEY = None
    from packages.vecpipe.maintenance import QdrantMaintenanceService


@pytest.fixture()
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    return Mock()


@pytest.fixture()
def maintenance_service(mock_qdrant_client, monkeypatch):
    """Create a maintenance service instance with mocked dependencies."""
    monkeypatch.setattr("packages.vecpipe.maintenance.settings.DEFAULT_COLLECTION", "work_docs")
    monkeypatch.setattr("packages.vecpipe.maintenance.settings.QDRANT_HOST", "localhost")
    monkeypatch.setattr("packages.vecpipe.maintenance.settings.QDRANT_PORT", 6333)
    monkeypatch.setattr("packages.vecpipe.maintenance.settings.INTERNAL_API_KEY", None)

    with patch("packages.vecpipe.maintenance.QdrantClient", return_value=mock_qdrant_client):
        return QdrantMaintenanceService(
            qdrant_host="localhost", qdrant_port=6333, webui_host="localhost", webui_port=8080
        )


class TestQdrantMaintenanceService:
    """Test the maintenance service functionality."""

    def test_get_job_collections_success(self, maintenance_service):
        """Test successful retrieval of job collections from API."""
        mock_response = Mock()
        mock_response.json.return_value = ["job_1", "job_2", "job_3"]
        mock_response.raise_for_status = Mock()

        with patch("packages.vecpipe.maintenance.httpx.get", return_value=mock_response) as mock_get:
            collections = maintenance_service.get_job_collections()

            # Should include DEFAULT_COLLECTION and job collections
            assert collections == ["work_docs", "job_job_1", "job_job_2", "job_job_3"]
            mock_get.assert_called_once()

            # Check that API was called with correct URL
            call_args = mock_get.call_args
            assert call_args[0][0] == "http://localhost:8080/api/internal/jobs/all-ids"

    def test_get_job_collections_with_retry(self, maintenance_service):
        """Test retry logic for API calls."""
        # First two calls fail, third succeeds
        mock_response = Mock()
        mock_response.json.return_value = ["job_1"]
        mock_response.raise_for_status = Mock()

        def side_effect(*_args, **_kwargs):
            if side_effect.call_count < 2:
                side_effect.call_count += 1
                raise httpx.HTTPStatusError("Server error", request=Mock(), response=Mock(status_code=500))
            return mock_response

        side_effect.call_count = 0

        with patch("packages.vecpipe.maintenance.httpx.get", side_effect=side_effect) as mock_get:
            collections = maintenance_service.get_job_collections()

            assert collections == ["work_docs", "job_job_1"]
            assert mock_get.call_count == 3

    def test_cleanup_orphaned_collections(self, maintenance_service):
        """Test cleanup of orphaned collections."""
        # Mock get_job_collections to return valid jobs
        mock_response = Mock()
        mock_response.json.return_value = ["job_1", "job_2"]
        mock_response.raise_for_status = Mock()

        with patch("packages.vecpipe.maintenance.httpx.get", return_value=mock_response):
            # Mock Qdrant collections
            qdrant_collections = []
            for name in ["work_docs", "job_job_1", "job_job_2", "job_job_3", "job_orphaned", "_collection_metadata"]:
                mock_col = Mock()
                mock_col.name = name  # Set name as an attribute, not Mock's name parameter
                qdrant_collections.append(mock_col)
            maintenance_service.client.get_collections.return_value = Mock(collections=qdrant_collections)
            maintenance_service.client.delete_collection = Mock()

            # Run cleanup
            result = maintenance_service.cleanup_orphaned_collections(dry_run=False)

            # Check results
            assert set(result["orphaned_collections"]) == {"job_job_3", "job_orphaned"}
            assert set(result["deleted_collections"]) == {"job_job_3", "job_orphaned"}
            assert result["dry_run"] is False

            # Verify delete was called for orphaned collections
            assert maintenance_service.client.delete_collection.call_count == 2
            maintenance_service.client.delete_collection.assert_any_call("job_job_3")
            maintenance_service.client.delete_collection.assert_any_call("job_orphaned")

    def test_cleanup_orphaned_collections_dry_run(self, maintenance_service):
        """Test dry run mode for cleanup."""
        # Mock get_job_collections to return valid jobs
        mock_response = Mock()
        mock_response.json.return_value = ["job_1", "job_2"]
        mock_response.raise_for_status = Mock()

        with patch("packages.vecpipe.maintenance.httpx.get", return_value=mock_response):
            # Mock Qdrant collections
            qdrant_collections = []
            for name in ["work_docs", "job_job_1", "job_job_2", "job_job_3"]:
                mock_col = Mock()
                mock_col.name = name
                qdrant_collections.append(mock_col)
            maintenance_service.client.get_collections.return_value = Mock(collections=qdrant_collections)
            maintenance_service.client.delete_collection = Mock()

            # Run cleanup in dry run mode
            result = maintenance_service.cleanup_orphaned_collections(dry_run=True)

            # Check results
            assert result["orphaned_collections"] == ["job_job_3"]
            assert result["deleted_collections"] == []
            assert result["dry_run"] is True

            # Verify delete was NOT called
            maintenance_service.client.delete_collection.assert_not_called()

    def test_cleanup_orphaned_collections_no_orphans(self, maintenance_service):
        """Test cleanup when no orphaned collections exist."""
        # Mock get_job_collections to return valid jobs
        mock_response = Mock()
        mock_response.json.return_value = ["job_1", "job_2"]
        mock_response.raise_for_status = Mock()

        with patch("packages.vecpipe.maintenance.httpx.get", return_value=mock_response):
            # Mock Qdrant collections - all are valid
            qdrant_collections = []
            for name in ["work_docs", "job_job_1", "job_job_2", "_collection_metadata"]:
                mock_col = Mock()
                mock_col.name = name
                qdrant_collections.append(mock_col)
            maintenance_service.client.get_collections.return_value = Mock(collections=qdrant_collections)
            maintenance_service.client.delete_collection = Mock()

            # Run cleanup
            result = maintenance_service.cleanup_orphaned_collections(dry_run=False)

            # Check results
            assert result["orphaned_collections"] == []
            assert result["deleted_collections"] == []
            assert result["dry_run"] is False

            # Verify delete was NOT called
            maintenance_service.client.delete_collection.assert_not_called()

    def test_get_job_collections_error_handling(self, maintenance_service):
        """Test that get_job_collections handles errors gracefully."""
        # All calls fail
        with patch("packages.vecpipe.maintenance.httpx.get") as mock_get:
            mock_get.side_effect = httpx.HTTPStatusError("Server error", request=Mock(), response=Mock(status_code=500))

            # Should return only DEFAULT_COLLECTION when API fails
            collections = maintenance_service.get_job_collections()
            assert collections == ["work_docs"]

            # Should have tried 3 times
            assert mock_get.call_count == 3

    def test_collection_not_found_handling(self, maintenance_service):
        """Test handling of collection not found errors during deletion."""
        # Mock get_job_collections to return valid jobs
        mock_response = Mock()
        mock_response.json.return_value = ["job_1"]
        mock_response.raise_for_status = Mock()

        with patch("packages.vecpipe.maintenance.httpx.get", return_value=mock_response):
            # Mock Qdrant collections
            qdrant_collections = []
            for name in ["work_docs", "job_job_1", "job_job_2"]:
                mock_col = Mock()
                mock_col.name = name
                qdrant_collections.append(mock_col)
            maintenance_service.client.get_collections.return_value = Mock(collections=qdrant_collections)

            # Mock delete to raise an exception
            maintenance_service.client.delete_collection.side_effect = Exception("Collection not found")

            # Run cleanup - should handle the error gracefully
            result = maintenance_service.cleanup_orphaned_collections(dry_run=False)

            # Should have identified the orphaned collection but not deleted it
            assert result["orphaned_collections"] == ["job_job_2"]
            assert result["deleted_collections"] == []
            maintenance_service.client.delete_collection.assert_called_once_with("job_job_2")

    def test_internal_api_key_configuration(self, monkeypatch):
        """Test that internal API key is properly configured when making requests."""
        # Test that API key is included in headers when configured
        monkeypatch.setattr("packages.vecpipe.maintenance.settings.INTERNAL_API_KEY", "test-api-key")
        monkeypatch.setattr("packages.vecpipe.maintenance.settings.DEFAULT_COLLECTION", "work_docs")
        monkeypatch.setattr("packages.vecpipe.maintenance.settings.QDRANT_HOST", "localhost")
        monkeypatch.setattr("packages.vecpipe.maintenance.settings.QDRANT_PORT", 6333)

        with (
            patch("packages.vecpipe.maintenance.QdrantClient"),
            patch("packages.vecpipe.maintenance.httpx.get") as mock_get,
        ):
            mock_response = Mock()
            mock_response.json.return_value = ["job_123"]
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            service = QdrantMaintenanceService(webui_host="localhost", webui_port=8080)
            collections = service.get_job_collections()

            assert collections == ["work_docs", "job_job_123"]
            mock_get.assert_called_with(
                "http://localhost:8080/api/internal/jobs/all-ids",
                headers={"X-Internal-Api-Key": "test-api-key"},
                timeout=30.0,
            )
