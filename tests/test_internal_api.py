"""Tests for internal API endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from packages.webui.api.internal import router, verify_internal_api_key


class TestInternalAPIAuth:
    """Test internal API authentication."""

    def test_verify_internal_api_key_valid(self):
        """Test successful API key verification."""
        with patch("packages.webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "valid-key"

            # Should not raise exception
            verify_internal_api_key("valid-key")

    def test_verify_internal_api_key_invalid(self):
        """Test API key verification with invalid key."""
        with patch("packages.webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "valid-key"

            # Should raise HTTPException
            with pytest.raises(HTTPException) as exc_info:
                verify_internal_api_key("invalid-key")

            assert exc_info.value.status_code == 401
            assert "Invalid or missing internal API key" in str(exc_info.value.detail)

    def test_verify_internal_api_key_missing(self):
        """Test API key verification with missing key."""
        with patch("packages.webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "valid-key"

            # Should raise HTTPException
            with pytest.raises(HTTPException) as exc_info:
                verify_internal_api_key(None)

            assert exc_info.value.status_code == 401


class TestInternalAPIEndpoints:
    """Test internal API endpoints."""

    @pytest.fixture
    def mock_database(self):
        """Mock the database module."""
        with patch("packages.webui.api.internal.database") as mock_db:
            yield mock_db

    @pytest.fixture
    def client(self):
        """Create test client with the router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_get_all_job_ids_success(self, client, mock_database):
        """Test successful retrieval of all job IDs."""
        # Mock database response
        mock_database.list_jobs.return_value = [
            {"id": "job_1", "name": "Test Job 1"},
            {"id": "job_2", "name": "Test Job 2"},
            {"id": "job_3", "name": "Test Job 3"},
        ]

        # Mock settings for API key
        with patch("packages.webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "test-key"

            response = client.get("/api/internal/jobs/all-ids", headers={"X-Internal-Api-Key": "test-key"})

            assert response.status_code == 200
            assert response.json() == ["job_1", "job_2", "job_3"]
            mock_database.list_jobs.assert_called_once()

    def test_get_all_job_ids_unauthorized(self, client, mock_database):
        """Test unauthorized access to job IDs endpoint."""
        with patch("packages.webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "test-key"

            # Test without API key
            response = client.get("/api/internal/jobs/all-ids")
            assert response.status_code == 401

            # Test with wrong API key
            response = client.get("/api/internal/jobs/all-ids", headers={"X-Internal-Api-Key": "wrong-key"})
            assert response.status_code == 401

            # Database should not be called
            mock_database.list_jobs.assert_not_called()

    def test_get_all_job_ids_empty_list(self, client, mock_database):
        """Test retrieval when no jobs exist."""
        mock_database.list_jobs.return_value = []

        with patch("packages.webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "test-key"

            response = client.get("/api/internal/jobs/all-ids", headers={"X-Internal-Api-Key": "test-key"})

            assert response.status_code == 200
            assert response.json() == []


class TestInternalAPIIntegration:
    """Integration tests for internal API with maintenance service."""

    @pytest.mark.asyncio
    async def test_maintenance_service_api_integration(self):
        """Test that maintenance service can successfully call internal API."""
        from packages.vecpipe.maintenance import MaintenanceService

        # Create maintenance service
        service = MaintenanceService(qdrant_client=MagicMock(), webui_base_url="http://test-server")

        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = ["job_1", "job_2"]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response) as mock_get:
            with patch.object(service, "internal_api_key", "test-key"):
                result = service.get_job_collections()

                # Verify correct API call
                mock_get.assert_called_once_with(
                    "http://test-server/api/internal/jobs/all-ids",
                    headers={"X-Internal-Api-Key": "test-key"},
                    timeout=30.0,
                )

                assert result == ["job_1", "job_2"]
