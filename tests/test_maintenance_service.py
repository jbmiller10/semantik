"""Tests for the maintenance service."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from qdrant_client.models import CollectionInfo, CollectionsResponse

from packages.vecpipe.maintenance import MaintenanceService


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = AsyncMock()
    return client


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client."""
    client = MagicMock()
    return client


@pytest.fixture
def maintenance_service(mock_qdrant_client):
    """Create a maintenance service instance with mocked dependencies."""
    service = MaintenanceService(qdrant_client=mock_qdrant_client, webui_base_url="http://localhost:8080")
    return service


class TestMaintenanceService:
    """Test the maintenance service functionality."""

    @pytest.mark.asyncio
    async def test_get_job_collections_success(self, maintenance_service):
        """Test successful retrieval of job collections from API."""
        mock_response = Mock()
        mock_response.json.return_value = ["job_1", "job_2", "job_3"]
        mock_response.raise_for_status = Mock()

        with patch("httpx.get", return_value=mock_response) as mock_get:
            collections = maintenance_service.get_job_collections()

            assert collections == ["job_1", "job_2", "job_3"]
            mock_get.assert_called_once()

            # Check that API key header is included if configured
            call_args = mock_get.call_args
            if maintenance_service.internal_api_key:
                assert "headers" in call_args.kwargs
                assert "X-Internal-Api-Key" in call_args.kwargs["headers"]

    @pytest.mark.asyncio
    async def test_get_job_collections_with_retry(self, maintenance_service):
        """Test retry logic for API calls."""
        # First two calls fail, third succeeds
        mock_responses = [
            httpx.HTTPStatusError("Server error", request=Mock(), response=Mock(status_code=500)),
            httpx.HTTPStatusError("Server error", request=Mock(), response=Mock(status_code=500)),
            Mock(json=Mock(return_value=["job_1"]), raise_for_status=Mock()),
        ]

        with patch("httpx.get", side_effect=mock_responses) as mock_get:
            with patch("time.sleep"):  # Don't actually sleep in tests
                collections = maintenance_service.get_job_collections()

                assert collections == ["job_1"]
                assert mock_get.call_count == 3

    @pytest.mark.asyncio
    async def test_identify_orphaned_collections(self, maintenance_service, mock_qdrant_client):
        """Test identification of orphaned Qdrant collections."""
        # Mock Qdrant collections
        qdrant_collections = [
            Mock(name="job_1"),
            Mock(name="job_2"),
            Mock(name="job_3"),
            Mock(name="orphaned_collection"),
            Mock(name="_collection_metadata"),  # Should be ignored
        ]
        mock_qdrant_client.get_collections.return_value = Mock(collections=qdrant_collections)

        # Mock job collections from API
        job_collections = ["job_1", "job_2"]

        # Identify orphaned collections
        orphaned = await maintenance_service.identify_orphaned_collections(job_collections)

        # Should identify job_3 and orphaned_collection as orphaned
        assert set(orphaned) == {"job_3", "orphaned_collection"}
        mock_qdrant_client.get_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_orphaned_collections(self, maintenance_service, mock_qdrant_client):
        """Test cleanup of orphaned collections."""
        orphaned_collections = ["orphaned_1", "orphaned_2"]

        # Run cleanup
        await maintenance_service.cleanup_orphaned_collections(orphaned_collections)

        # Verify delete was called for each orphaned collection
        assert mock_qdrant_client.delete_collection.call_count == 2
        mock_qdrant_client.delete_collection.assert_any_call("orphaned_1")
        mock_qdrant_client.delete_collection.assert_any_call("orphaned_2")

    @pytest.mark.asyncio
    async def test_cleanup_orphaned_collections_handles_errors(self, maintenance_service, mock_qdrant_client):
        """Test that cleanup continues even if some deletions fail."""
        orphaned_collections = ["orphaned_1", "orphaned_2", "orphaned_3"]

        # Make the second deletion fail
        mock_qdrant_client.delete_collection.side_effect = [
            None,  # First succeeds
            Exception("Delete failed"),  # Second fails
            None,  # Third succeeds
        ]

        # Run cleanup (should not raise exception)
        await maintenance_service.cleanup_orphaned_collections(orphaned_collections)

        # All deletions should be attempted
        assert mock_qdrant_client.delete_collection.call_count == 3

    @pytest.mark.asyncio
    async def test_run_maintenance_full_flow(self, maintenance_service, mock_qdrant_client):
        """Test the full maintenance flow."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = ["job_1", "job_2"]
        mock_response.raise_for_status = Mock()

        # Mock Qdrant collections
        qdrant_collections = [
            Mock(name="job_1"),
            Mock(name="job_2"),
            Mock(name="job_3"),  # This is orphaned
            Mock(name="_collection_metadata"),
        ]
        mock_qdrant_client.get_collections.return_value = Mock(collections=qdrant_collections)

        with patch("httpx.get", return_value=mock_response):
            cleaned_count = await maintenance_service.run_maintenance()

            # Should have cleaned 1 orphaned collection (job_3)
            assert cleaned_count == 1
            mock_qdrant_client.delete_collection.assert_called_once_with("job_3")

    @pytest.mark.asyncio
    async def test_run_maintenance_no_orphans(self, maintenance_service, mock_qdrant_client):
        """Test maintenance when there are no orphaned collections."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = ["job_1", "job_2"]
        mock_response.raise_for_status = Mock()

        # Mock Qdrant collections (all are valid)
        qdrant_collections = [
            Mock(name="job_1"),
            Mock(name="job_2"),
            Mock(name="_collection_metadata"),
        ]
        mock_qdrant_client.get_collections.return_value = Mock(collections=qdrant_collections)

        with patch("httpx.get", return_value=mock_response):
            cleaned_count = await maintenance_service.run_maintenance()

            # Should not have cleaned any collections
            assert cleaned_count == 0
            mock_qdrant_client.delete_collection.assert_not_called()

    def test_internal_api_key_configuration(self):
        """Test that internal API key is properly configured."""
        # Test with API key
        with patch("packages.vecpipe.maintenance.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "test-api-key"
            service = MaintenanceService(qdrant_client=AsyncMock(), webui_base_url="http://localhost:8080")
            assert service.internal_api_key == "test-api-key"

        # Test without API key
        with patch("packages.vecpipe.maintenance.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = None
            service = MaintenanceService(qdrant_client=AsyncMock(), webui_base_url="http://localhost:8080")
            assert service.internal_api_key is None
