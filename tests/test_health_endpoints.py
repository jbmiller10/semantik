"""Tests for health check endpoints"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def mock_embedding_service():
    """Create a mock embedding service"""
    service = Mock()
    service.is_initialized = True
    service.get_model_info.return_value = {"model_name": "test-model", "dimension": 384, "device": "cpu"}

    # Create an async version of embed_single
    async def async_embed_single(_text):
        return [0.1] * 384

    service.embed_single = async_embed_single

    return service


class TestWebuiHealthEndpoints:
    """Test health endpoints in webui"""

    def test_basic_health_check(self, test_client):
        """Test basic health check endpoint"""
        response = test_client.get("/api/health/")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_embedding_health_initialized(self, test_client, mock_embedding_service):
        """Test embedding health when service is initialized"""

        async def async_get_service(*_args, **_kwargs):
            return mock_embedding_service

        # Patch both the function and the singleton instance
        with (
            patch("packages.webui.api.health.get_embedding_service", side_effect=async_get_service),
            patch("shared.embedding.service.get_embedding_service", side_effect=async_get_service),
            patch("shared.embedding.service._embedding_service", mock_embedding_service),
        ):
            response = test_client.get("/api/health/embedding")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["initialized"] is True
            assert "model" in data
            assert data["model"]["model_name"] == "test-model"

    def test_embedding_health_not_initialized(self, test_client, mock_embedding_service):
        """Test embedding health when service is not initialized"""
        mock_embedding_service.is_initialized = False

        async def async_get_service(*_args, **_kwargs):
            return mock_embedding_service

        with (
            patch("packages.webui.api.health.get_embedding_service", side_effect=async_get_service),
            patch("shared.embedding.service.get_embedding_service", side_effect=async_get_service),
            patch("shared.embedding.service._embedding_service", mock_embedding_service),
        ):
            response = test_client.get("/api/health/embedding")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["initialized"] is False
            assert "message" in data

    def test_embedding_health_service_error(self, test_client):
        """Test embedding health when service throws error or is not initialized"""

        # In CI, the embedding service singleton might already be initialized
        # We need to ensure our mock is applied at all levels

        # Mock get_embedding_service to raise an exception
        async def async_error(*_args, **_kwargs):
            raise Exception("Service error")

        # Reset the singleton to ensure our mock takes effect
        import shared.embedding.service

        original_service = shared.embedding.service._embedding_service
        shared.embedding.service._embedding_service = None

        try:
            # Patch the function in the health module and the service module
            with (
                patch("packages.webui.api.health.get_embedding_service", side_effect=async_error),
                patch("shared.embedding.service.get_embedding_service", side_effect=async_error),
            ):
                response = test_client.get("/api/health/embedding")
                assert response.status_code == 200
                data = response.json()

                # The endpoint should return unhealthy status
                assert data["status"] == "unhealthy"

                # It should have either 'error' (exception case) or 'message' (uninitialized case)
                assert "error" in data or "message" in data

                # Verify the response indicates a problem
                if "error" in data:
                    # Exception was thrown
                    assert "Failed to access embedding service" in data["error"]
                elif "message" in data:
                    # Service created but not initialized
                    assert "not initialized" in data["message"].lower()
        finally:
            # Restore the original service
            shared.embedding.service._embedding_service = original_service

    def test_readiness_check_ready(self, test_client, mock_embedding_service):
        """Test readiness check when service is ready"""

        # embed_single is already set up as an async function in the fixture
        async def async_get_service(*_args, **_kwargs):
            return mock_embedding_service

        # Mock Redis connection
        mock_redis = Mock()
        async def async_ping():
            return True
        mock_redis.ping = async_ping

        with (
            patch("packages.webui.api.health.get_embedding_service", side_effect=async_get_service),
            patch("shared.embedding.service.get_embedding_service", side_effect=async_get_service),
            patch("shared.embedding.service._embedding_service", mock_embedding_service),
            patch("packages.webui.api.health.ws_manager.redis", mock_redis),
        ):
            response = test_client.get("/api/health/readyz")
            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is True

    def test_readiness_check_not_ready(self, test_client, mock_embedding_service):
        """Test readiness check when service is not ready"""
        mock_embedding_service.is_initialized = False

        async def async_get_service(*_args, **_kwargs):
            return mock_embedding_service

        # Mock Redis connection - make it healthy so we can test embedding failure
        mock_redis = Mock()
        async def async_ping():
            return True
        mock_redis.ping = async_ping

        with (
            patch("packages.webui.api.health.get_embedding_service", side_effect=async_get_service),
            patch("shared.embedding.service.get_embedding_service", side_effect=async_get_service),
            patch("shared.embedding.service._embedding_service", mock_embedding_service),
            patch("packages.webui.api.health.ws_manager.redis", mock_redis),
        ):
            response = test_client.get("/api/health/readyz")
            assert response.status_code == 503  # Should be 503 when not ready
            data = response.json()
            assert data["ready"] is False


class TestVecpipeHealthEndpoints:
    """Test health endpoints in vecpipe"""

    @pytest.fixture()
    def vecpipe_app(self):
        """Create vecpipe app for testing"""
        from packages.vecpipe.search_api import app

        return TestClient(app)

    def test_vecpipe_health_all_healthy(self, vecpipe_app):
        """Test vecpipe health when all components are healthy"""
        mock_qdrant = Mock()

        # Create async mock for qdrant_client.get
        async def mock_get(_path):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": {"collections": [{"name": "col1"}, {"name": "col2"}]}}
            return mock_response

        mock_qdrant.get = mock_get

        mock_embedding = Mock()
        mock_embedding.is_initialized = True
        mock_embedding.get_model_info.return_value = {"model_name": "test-model", "dimension": 384}

        with (
            patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant),
            patch("packages.vecpipe.search_api.embedding_service", mock_embedding),
        ):
            response = vecpipe_app.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["components"]["qdrant"]["status"] == "healthy"
            assert data["components"]["embedding"]["status"] == "healthy"

    def test_vecpipe_health_qdrant_unhealthy(self, vecpipe_app):
        """Test vecpipe health when Qdrant is unhealthy"""
        with patch("packages.vecpipe.search_api.qdrant_client", None):
            response = vecpipe_app.get("/health")
            assert response.status_code == 503
            data = response.json()["detail"]
            assert data["status"] == "unhealthy"
            assert data["components"]["qdrant"]["status"] == "unhealthy"

    def test_vecpipe_health_embedding_degraded(self, vecpipe_app):
        """Test vecpipe health when embedding service is degraded"""
        mock_qdrant = Mock()

        # Create async mock for qdrant_client.get
        async def mock_get(_path):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": {"collections": []}}
            return mock_response

        mock_qdrant.get = mock_get

        mock_embedding = Mock()
        mock_embedding.is_initialized = False

        with (
            patch("packages.vecpipe.search_api.qdrant_client", mock_qdrant),
            patch("packages.vecpipe.search_api.embedding_service", mock_embedding),
        ):
            response = vecpipe_app.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["components"]["qdrant"]["status"] == "healthy"
            assert data["components"]["embedding"]["status"] == "unhealthy"
