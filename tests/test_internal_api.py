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

    @pytest.fixture()
    def client_with_mocked_repos(self, mock_collection_repository):
        """Create test client with mocked repositories."""
        from fastapi import FastAPI
        from packages.shared.database.factory import create_collection_repository
        from packages.webui.dependencies import get_collection_repository, get_db
        from unittest.mock import AsyncMock

        app = FastAPI()
        app.include_router(router)

        # Mock database session
        mock_db = AsyncMock()
        
        # Override dependencies
        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_collection_repository] = lambda: mock_collection_repository
        app.dependency_overrides[create_collection_repository] = lambda: mock_collection_repository

        client = TestClient(app)
        yield client

        app.dependency_overrides.clear()

    def test_get_all_vector_store_names_success(self, client_with_mocked_repos, mock_collection_repository):
        """Test successful retrieval of all vector store names."""
        # Mock repository response
        from unittest.mock import AsyncMock, MagicMock

        # Create mock collections with vector_store_name attribute
        mock_collection1 = MagicMock()
        mock_collection1.vector_store_name = "collection_1"
        mock_collection2 = MagicMock()
        mock_collection2.vector_store_name = "collection_2"
        mock_collection3 = MagicMock()
        mock_collection3.vector_store_name = None  # This should be filtered out

        mock_collection_repository.list_all = AsyncMock(
            return_value=[mock_collection1, mock_collection2, mock_collection3]
        )

        # Mock settings for API key
        with patch("packages.webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "test-key"

            response = client_with_mocked_repos.get(
                "/api/internal/collections/vector-store-names", headers={"X-Internal-Api-Key": "test-key"}
            )

            assert response.status_code == 200
            assert response.json() == ["collection_1", "collection_2"]
            mock_collection_repository.list_all.assert_called_once()

    def test_get_all_vector_store_names_unauthorized(self, client_with_mocked_repos, mock_collection_repository):
        """Test unauthorized access to vector store names endpoint."""
        with patch("packages.webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "test-key"

            # Test without API key
            response = client_with_mocked_repos.get("/api/internal/collections/vector-store-names")
            assert response.status_code == 401

            # Test with wrong API key
            response = client_with_mocked_repos.get(
                "/api/internal/collections/vector-store-names", headers={"X-Internal-Api-Key": "wrong-key"}
            )
            assert response.status_code == 401

            # Repository should not be called
            if hasattr(mock_collection_repository, "list_all"):
                mock_collection_repository.list_all.assert_not_called()

    def test_get_all_vector_store_names_empty_list(self, client_with_mocked_repos, mock_collection_repository):
        """Test retrieval when no collections exist."""
        # Mock repository response
        from unittest.mock import AsyncMock

        mock_collection_repository.list_all = AsyncMock(return_value=[])

        with patch("packages.webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "test-key"

            response = client_with_mocked_repos.get(
                "/api/internal/collections/vector-store-names", headers={"X-Internal-Api-Key": "test-key"}
            )

            assert response.status_code == 200
            assert response.json() == []


class TestInternalAPIIntegration:
    """Integration tests for internal API with maintenance service."""

    @pytest.mark.asyncio()
    async def test_maintenance_service_api_integration(self):
        """Test that maintenance service can successfully call internal API."""
        from packages.vecpipe.maintenance import QdrantMaintenanceService
        
        # Create maintenance service
        with patch("packages.vecpipe.maintenance.QdrantClient", return_value=MagicMock()):
            service = QdrantMaintenanceService(webui_host="test-server", webui_port=80)
        
        with patch("packages.vecpipe.maintenance.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "test-key"
            mock_settings.DEFAULT_COLLECTION = "work_docs"
            result = service.get_operation_collections()
            
            # In the new architecture, this returns only the default collection
            assert result == ["work_docs"]
