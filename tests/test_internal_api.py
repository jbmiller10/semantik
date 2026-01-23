"""Tests for internal API endpoints."""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from shared.database.factory import create_collection_repository
from vecpipe.maintenance import QdrantMaintenanceService
from webui.api.internal import router, verify_internal_api_key
from webui.dependencies import get_collection_repository, get_db


class TestInternalAPIAuth:
    """Test internal API authentication."""

    def test_verify_internal_api_key_valid(self) -> None:
        """Test successful API key verification."""
        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "valid-key"

            # Should not raise exception
            verify_internal_api_key("valid-key")

    def test_verify_internal_api_key_invalid(self) -> None:
        """Test API key verification with invalid key."""
        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "valid-key"

            # Should raise HTTPException
            with pytest.raises(HTTPException) as exc_info:
                verify_internal_api_key("invalid-key")

            assert exc_info.value.status_code == 401
            assert "Invalid or missing internal API key" in str(exc_info.value.detail)

    def test_verify_internal_api_key_missing(self) -> None:
        """Test API key verification with missing key."""
        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "valid-key"

            # Should raise HTTPException
            with pytest.raises(HTTPException) as exc_info:
                verify_internal_api_key(None)

            assert exc_info.value.status_code == 401

    def test_verify_internal_api_key_not_configured(self) -> None:
        """Test API key verification when key is not configured."""
        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = ""

            with pytest.raises(HTTPException) as exc_info:
                verify_internal_api_key("any-key")

            assert exc_info.value.status_code == 500


class TestInternalAPIEndpoints:
    """Test internal API endpoints."""

    @pytest.fixture()
    def client_with_mocked_repos(self, mock_collection_repository) -> Generator[TestClient, None, None]:
        """Create test client with mocked repositories."""

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

    def test_get_all_vector_store_names_success(self, client_with_mocked_repos, mock_collection_repository) -> None:
        """Test successful retrieval of all vector store names."""
        # Mock repository response

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
        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "test-key"

            response = client_with_mocked_repos.get(
                "/api/internal/collections/vector-store-names", headers={"X-Internal-Api-Key": "test-key"}
            )

            assert response.status_code == 200
            assert response.json() == ["collection_1", "collection_2"]
            mock_collection_repository.list_all.assert_called_once()

    def test_get_all_vector_store_names_unauthorized(
        self, client_with_mocked_repos, mock_collection_repository
    ) -> None:
        """Test unauthorized access to vector store names endpoint."""
        with patch("webui.api.internal.settings") as mock_settings:
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

    def test_get_all_vector_store_names_empty_list(self, client_with_mocked_repos, mock_collection_repository) -> None:
        """Test retrieval when no collections exist."""
        # Mock repository response

        mock_collection_repository.list_all = AsyncMock(return_value=[])

        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "test-key"

            response = client_with_mocked_repos.get(
                "/api/internal/collections/vector-store-names", headers={"X-Internal-Api-Key": "test-key"}
            )

            assert response.status_code == 200
            assert response.json() == []


class TestInternalAPIIntegration:
    """Integration tests for internal API with maintenance service."""

    @pytest.mark.asyncio()
    async def test_maintenance_service_api_integration(self) -> None:
        """Test that maintenance service can successfully call internal API."""

        # Create maintenance service
        with patch("vecpipe.maintenance.QdrantClient", return_value=MagicMock()):
            service = QdrantMaintenanceService(webui_host="test-server", webui_port=80)

        with patch("vecpipe.maintenance.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "test-key"
            mock_settings.DEFAULT_COLLECTION = "work_docs"
            result = service.get_operation_collections()

            # In the new architecture, this returns only the default collection
            assert result == ["work_docs"]


class TestValidateApiKeyEndpoint:
    """Tests for the /api/internal/validate-api-key endpoint."""

    @pytest.fixture()
    def client_with_mocked_auth(self) -> Generator[TestClient, None, None]:
        """Create test client with mocked auth verification."""
        app = FastAPI()
        app.include_router(router)

        mock_db = AsyncMock()
        app.dependency_overrides[get_db] = lambda: mock_db

        client = TestClient(app)
        yield client

        app.dependency_overrides.clear()

    def test_validate_api_key_empty_key_returns_error(self, client_with_mocked_auth) -> None:
        """Test that empty API key returns validation error."""
        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "internal-key"

            response = client_with_mocked_auth.post(
                "/api/internal/validate-api-key",
                json={"api_key": ""},
                headers={"X-Internal-Api-Key": "internal-key"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            assert data["error"] == "API key is required"
            assert data["user_id"] is None
            assert data["username"] is None

    def test_validate_api_key_whitespace_only_returns_error(self, client_with_mocked_auth) -> None:
        """Test that whitespace-only API key returns validation error."""
        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "internal-key"

            response = client_with_mocked_auth.post(
                "/api/internal/validate-api-key",
                json={"api_key": "   "},
                headers={"X-Internal-Api-Key": "internal-key"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            assert data["error"] == "API key is required"

    def test_validate_api_key_invalid_key_returns_error(self, client_with_mocked_auth) -> None:
        """Test that invalid API key returns validation error."""
        with (
            patch("webui.api.internal.settings") as mock_settings,
            patch("webui.api.internal._verify_api_key") as mock_verify,
        ):
            mock_settings.INTERNAL_API_KEY = "internal-key"
            mock_verify.return_value = None  # Invalid key

            response = client_with_mocked_auth.post(
                "/api/internal/validate-api-key",
                json={"api_key": "invalid-user-key"},
                headers={"X-Internal-Api-Key": "internal-key"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            assert data["error"] == "Invalid or expired API key"
            mock_verify.assert_called_once_with("invalid-user-key", update_last_used=False)

    def test_validate_api_key_no_user_id_returns_error(self, client_with_mocked_auth) -> None:
        """Test that API key without user ID returns error."""
        with (
            patch("webui.api.internal.settings") as mock_settings,
            patch("webui.api.internal._verify_api_key") as mock_verify,
        ):
            mock_settings.INTERNAL_API_KEY = "internal-key"
            mock_verify.return_value = {"user": {}}  # No user ID

            response = client_with_mocked_auth.post(
                "/api/internal/validate-api-key",
                json={"api_key": "user-key-without-user"},
                headers={"X-Internal-Api-Key": "internal-key"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            assert data["error"] == "API key has no associated user"

    def test_validate_api_key_inactive_user_returns_error(self, client_with_mocked_auth) -> None:
        """Test that API key for inactive user returns error."""
        with (
            patch("webui.api.internal.settings") as mock_settings,
            patch("webui.api.internal._verify_api_key") as mock_verify,
        ):
            mock_settings.INTERNAL_API_KEY = "internal-key"
            mock_verify.return_value = {
                "user": {"id": 123, "username": "inactive_user", "is_active": False}
            }

            response = client_with_mocked_auth.post(
                "/api/internal/validate-api-key",
                json={"api_key": "inactive-user-key"},
                headers={"X-Internal-Api-Key": "internal-key"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            assert data["error"] == "User account is inactive"

    def test_validate_api_key_valid_returns_user_info(self, client_with_mocked_auth) -> None:
        """Test that valid API key returns user info."""
        with (
            patch("webui.api.internal.settings") as mock_settings,
            patch("webui.api.internal._verify_api_key") as mock_verify,
        ):
            mock_settings.INTERNAL_API_KEY = "internal-key"
            mock_verify.return_value = {
                "user": {"id": 42, "username": "testuser", "is_active": True}
            }

            response = client_with_mocked_auth.post(
                "/api/internal/validate-api-key",
                json={"api_key": "valid-user-key"},
                headers={"X-Internal-Api-Key": "internal-key"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["user_id"] == 42
            assert data["username"] == "testuser"
            assert data["error"] is None


class TestCompleteReindexEndpoint:
    """Tests for the /api/internal/complete-reindex endpoint."""

    @pytest.fixture()
    def client_with_mocked_repos(self, mock_collection_repository) -> Generator[TestClient, None, None]:
        """Create test client with mocked repositories."""
        app = FastAPI()
        app.include_router(router)

        mock_db = AsyncMock()
        mock_db.begin = MagicMock(return_value=AsyncMock())

        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides[get_collection_repository] = lambda: mock_collection_repository

        client = TestClient(app)
        yield client

        app.dependency_overrides.clear()

    def test_complete_reindex_collection_not_found(self, client_with_mocked_repos, mock_collection_repository) -> None:
        """Test 404 when collection does not exist."""
        mock_collection_repository.get_by_uuid = AsyncMock(return_value=None)

        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "internal-key"

            response = client_with_mocked_repos.post(
                "/api/internal/complete-reindex",
                json={
                    "collection_id": "00000000-0000-0000-0000-000000000001",
                    "operation_id": "00000000-0000-0000-0000-000000000002",
                    "staging_collection_name": "staging_collection",
                    "vector_count": 100,
                },
                headers={"X-Internal-Api-Key": "internal-key"},
            )

            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()

    def test_complete_reindex_wrong_status_returns_409(
        self, client_with_mocked_repos, mock_collection_repository
    ) -> None:
        """Test 409 when collection is not in PROCESSING status."""
        from shared.database.models import CollectionStatus

        mock_collection = MagicMock()
        mock_collection.status = CollectionStatus.READY  # Wrong status

        mock_collection_repository.get_by_uuid = AsyncMock(return_value=mock_collection)

        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "internal-key"

            response = client_with_mocked_repos.post(
                "/api/internal/complete-reindex",
                json={
                    "collection_id": "00000000-0000-0000-0000-000000000001",
                    "operation_id": "00000000-0000-0000-0000-000000000002",
                    "staging_collection_name": "staging_collection",
                    "vector_count": 100,
                },
                headers={"X-Internal-Api-Key": "internal-key"},
            )

            assert response.status_code == 409
            assert "PROCESSING" in response.json()["detail"]

    def test_complete_reindex_with_config_updates(
        self, client_with_mocked_repos, mock_collection_repository
    ) -> None:
        """Test that new_config is applied during reindex completion."""
        from shared.database.models import CollectionStatus

        mock_collection = MagicMock()
        mock_collection.status = CollectionStatus.PROCESSING
        mock_collection.qdrant_collections = ["old_collection_1"]
        mock_collection.embedding_model = "old-model"
        mock_collection.chunk_size = 500
        mock_collection.chunk_overlap = 50

        mock_collection_repository.get_by_uuid = AsyncMock(return_value=mock_collection)
        mock_collection_repository.update = AsyncMock()

        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "internal-key"

            response = client_with_mocked_repos.post(
                "/api/internal/complete-reindex",
                json={
                    "collection_id": "00000000-0000-0000-0000-000000000001",
                    "operation_id": "00000000-0000-0000-0000-000000000002",
                    "staging_collection_name": "new_staging_collection",
                    "vector_count": 250,
                    "new_config": {
                        "embedding_model": "new-model",
                        "chunk_size": 1000,
                        "chunk_overlap": 100,
                    },
                },
                headers={"X-Internal-Api-Key": "internal-key"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["old_collection_names"] == ["old_collection_1"]
            assert "successfully" in data["message"].lower()

            # Verify update was called with config values
            mock_collection_repository.update.assert_called_once()
            call_args = mock_collection_repository.update.call_args
            updates = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("updates", {})

            assert updates["qdrant_collections"] == ["new_staging_collection"]
            assert updates["status"] == CollectionStatus.READY
            assert updates["vector_count"] == 250
            assert updates["embedding_model"] == "new-model"
            assert updates["chunk_size"] == 1000
            assert updates["chunk_overlap"] == 100

    def test_complete_reindex_success_without_config(
        self, client_with_mocked_repos, mock_collection_repository
    ) -> None:
        """Test successful reindex completion without config updates."""
        from shared.database.models import CollectionStatus

        mock_collection = MagicMock()
        mock_collection.status = CollectionStatus.PROCESSING
        mock_collection.qdrant_collections = ["old_1", "old_2"]

        mock_collection_repository.get_by_uuid = AsyncMock(return_value=mock_collection)
        mock_collection_repository.update = AsyncMock()

        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "internal-key"

            response = client_with_mocked_repos.post(
                "/api/internal/complete-reindex",
                json={
                    "collection_id": "00000000-0000-0000-0000-000000000001",
                    "operation_id": "00000000-0000-0000-0000-000000000002",
                    "staging_collection_name": "new_staging",
                    "vector_count": 50,
                },
                headers={"X-Internal-Api-Key": "internal-key"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["old_collection_names"] == ["old_1", "old_2"]

    def test_complete_reindex_invalid_uuid_format(self, client_with_mocked_repos) -> None:
        """Test validation error for invalid UUID format."""
        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "internal-key"

            response = client_with_mocked_repos.post(
                "/api/internal/complete-reindex",
                json={
                    "collection_id": "not-a-uuid",
                    "operation_id": "also-not-a-uuid",
                    "staging_collection_name": "staging",
                    "vector_count": 10,
                },
                headers={"X-Internal-Api-Key": "internal-key"},
            )

            assert response.status_code == 422  # Pydantic validation error

    def test_complete_reindex_negative_vector_count(self, client_with_mocked_repos) -> None:
        """Test validation error for negative vector count."""
        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "internal-key"

            response = client_with_mocked_repos.post(
                "/api/internal/complete-reindex",
                json={
                    "collection_id": "00000000-0000-0000-0000-000000000001",
                    "operation_id": "00000000-0000-0000-0000-000000000002",
                    "staging_collection_name": "staging",
                    "vector_count": -1,
                },
                headers={"X-Internal-Api-Key": "internal-key"},
            )

            assert response.status_code == 422  # Pydantic validation error

    def test_complete_reindex_empty_staging_name(self, client_with_mocked_repos) -> None:
        """Test validation error for empty staging collection name."""
        with patch("webui.api.internal.settings") as mock_settings:
            mock_settings.INTERNAL_API_KEY = "internal-key"

            response = client_with_mocked_repos.post(
                "/api/internal/complete-reindex",
                json={
                    "collection_id": "00000000-0000-0000-0000-000000000001",
                    "operation_id": "00000000-0000-0000-0000-000000000002",
                    "staging_collection_name": "",
                    "vector_count": 10,
                },
                headers={"X-Internal-Api-Key": "internal-key"},
            )

            assert response.status_code == 422  # Pydantic validation error


class TestInternalApiKeyConfiguration:
    """Tests for internal API key configuration during app startup."""

    def test_configure_internal_api_key_success(self, monkeypatch) -> None:
        """Ensure helper invocation succeeds and logs fingerprint."""
        import webui.main as main_module

        monkeypatch.setattr(main_module, "ensure_internal_api_key", lambda _settings: "test-key", raising=False)

        with patch.object(main_module, "logger") as mock_logger:
            main_module._configure_internal_api_key()

        mock_logger.info.assert_called()

    def test_configure_internal_api_key_failure(self, monkeypatch) -> None:
        """Ensure failures propagate when helper raises RuntimeError."""
        import webui.main as main_module

        def fail(_settings: Any) -> str:
            raise RuntimeError("missing key")

        monkeypatch.setattr(main_module, "ensure_internal_api_key", fail, raising=False)

        with patch.object(main_module, "logger") as mock_logger, pytest.raises(RuntimeError):
            main_module._configure_internal_api_key()

        mock_logger.error.assert_called()
