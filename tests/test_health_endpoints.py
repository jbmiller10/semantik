"""Tests for health check endpoints"""

from unittest.mock import Mock, patch

import httpx
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

    def test_search_api_health_healthy(self, test_client):
        """Test search API health when service is healthy"""
        
        class MockResponse:
            status_code = 200
            
            def json(self):
                return {"status": "healthy"}
        
        async def mock_get(*args, **kwargs):
            return MockResponse()
        
        with patch("httpx.AsyncClient.get", side_effect=mock_get):
            response = test_client.get("/api/health/search-api")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["message"] == "Search API is ready"
            assert "api_response" in data

    def test_search_api_health_unhealthy(self, test_client):
        """Test search API health when service is unhealthy"""
        
        class MockResponse:
            status_code = 503
            text = "Service unavailable"
        
        async def mock_get(*args, **kwargs):
            return MockResponse()
        
        with patch("httpx.AsyncClient.get", side_effect=mock_get):
            response = test_client.get("/api/health/search-api")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "Search API returned status 503" in data["message"]
            assert data["details"] == "Service unavailable"

    def test_search_api_health_connection_error(self, test_client):
        """Test search API health when connection fails"""
        
        async def mock_get(*args, **kwargs):
            raise Exception("Connection failed")
        
        with patch("httpx.AsyncClient.get", side_effect=mock_get):
            response = test_client.get("/api/health/search-api")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "error" in data
            assert "Failed to access Search API" in data["error"]
            assert "Connection failed" in data["details"]

    def test_readiness_check_ready(self, test_client):
        """Test readiness check when all services are ready"""

        # Mock Redis connection
        mock_redis = Mock()

        async def async_ping():
            return True

        mock_redis.ping = async_ping
        
        # Mock Search API response
        class MockResponse:
            status_code = 200
            
        async def mock_get(*args, **kwargs):
            return MockResponse()

        with (
            patch("packages.webui.api.health.ws_manager.redis", mock_redis),
            patch("httpx.AsyncClient.get", side_effect=mock_get),
        ):
            response = test_client.get("/api/health/readyz")
            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is True

    def test_readiness_check_not_ready(self, test_client):
        """Test readiness check when search API is not ready"""

        # Mock Redis connection - make it healthy so we can test search API failure
        mock_redis = Mock()

        async def async_ping():
            return True

        mock_redis.ping = async_ping
        
        # Mock Search API to return unhealthy
        class MockResponse:
            status_code = 503
            
        async def mock_get(*args, **kwargs):
            return MockResponse()

        with (
            patch("packages.webui.api.health.ws_manager.redis", mock_redis),
            patch("httpx.AsyncClient.get", side_effect=mock_get),
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
