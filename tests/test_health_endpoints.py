"""Tests for health check endpoints"""

from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from packages.vecpipe.search_api import app


@pytest.fixture()
def mock_embedding_service() -> Mock:
    """Create a mock embedding service"""
    service = Mock()
    service.is_initialized = True
    service.get_model_info.return_value = {"model_name": "test-model", "dimension": 384, "device": "cpu"}

    # Create an async version of embed_single
    async def async_embed_single(_text) -> None:
        return [0.1] * 384

    service.embed_single = async_embed_single

    return service


class TestWebuiHealthEndpoints:
    """Test health endpoints in webui"""

    def test_basic_health_check(self, test_client) -> None:
        """Test basic health check endpoint"""
        response = test_client.get("/api/health/")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_search_api_health_healthy(self, test_client) -> None:
        """Test search API health when service is healthy"""

        class MockResponse:
            status_code = 200

            def json(self) -> None:
                return {"status": "healthy"}

        async def mock_get(*_args, **_kwargs) -> None:
            return MockResponse()

        with patch("httpx.AsyncClient.get", side_effect=mock_get):
            response = test_client.get("/api/health/search-api")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["message"] == "Search API is ready"
            assert "api_response" in data

    def test_search_api_health_unhealthy(self, test_client) -> None:
        """Test search API health when service is unhealthy"""

        class MockResponse:
            status_code = 503
            text = "Service unavailable"

        async def mock_get(*_args, **_kwargs) -> None:
            return MockResponse()

        with patch("httpx.AsyncClient.get", side_effect=mock_get):
            response = test_client.get("/api/health/search-api")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "Search API returned status 503" in data["message"]
            assert data["details"] == "Service unavailable"

    def test_search_api_health_connection_error(self, test_client) -> None:
        """Test search API health when connection fails"""

        async def mock_get(*_args, **_kwargs) -> None:
            raise Exception("Connection failed")

        with patch("httpx.AsyncClient.get", side_effect=mock_get):
            response = test_client.get("/api/health/search-api")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "error" in data
            assert "Failed to access Search API" in data["error"]
            assert "Connection failed" in data["details"]

    def test_readiness_check_ready(self, test_client) -> None:
        """Test readiness check when all services are ready"""

        # Mock Redis connection
        mock_redis = Mock()

        async def async_ping() -> None:
            return True

        mock_redis.ping = async_ping

        # Mock Search API response
        class MockResponse:
            status_code = 200

            def json(self) -> None:
                return {"status": "healthy", "components": {}}

        async def mock_get(*_args, **_kwargs) -> None:
            return MockResponse()

        # Mock embedding service health check
        async def mock_check_embedding_service_health() -> None:
            return {"status": "unhealthy", "message": "Embedding service not initialized"}

        # Mock database health check
        async def mock_check_postgres() -> None:
            return True

        # Mock Qdrant connection
        mock_qdrant_client = Mock()
        mock_qdrant_client.get_collections = Mock(return_value=Mock(collections=[]))

        mock_qdrant_manager = Mock()
        mock_qdrant_manager.get_client = Mock(return_value=mock_qdrant_client)

        with (
            patch("packages.webui.api.health.ws_manager.redis", mock_redis),
            patch("httpx.AsyncClient.get", side_effect=mock_get),
            patch(
                "packages.webui.api.health._check_embedding_service_health",
                side_effect=mock_check_embedding_service_health,
            ),
            patch("packages.webui.api.health.check_postgres_connection", side_effect=mock_check_postgres),
            patch("packages.webui.api.health.qdrant_manager", mock_qdrant_manager),
        ):
            response = test_client.get("/api/health/readyz")
            assert response.status_code == 200
            data = response.json()
            assert data["ready"] is True

    def test_readiness_check_not_ready(self, test_client) -> None:
        """Test readiness check when search API is not ready"""

        # Mock Redis connection - make it healthy so we can test search API failure
        mock_redis = Mock()

        async def async_ping() -> None:
            return True

        mock_redis.ping = async_ping

        # Mock Search API to return unhealthy
        class MockResponse:
            status_code = 503

            def json(self) -> None:
                return {"status": "unhealthy", "components": {}}

        async def mock_get(*_args, **_kwargs) -> None:
            return MockResponse()

        # Mock embedding service health check
        async def mock_check_embedding_service_health() -> None:
            return {"status": "unhealthy", "message": "Embedding service not initialized"}

        # Mock database health check - make it fail
        async def mock_check_postgres() -> None:
            return False

        # Mock Qdrant connection
        mock_qdrant_client = Mock()
        mock_qdrant_client.get_collections = Mock(return_value=Mock(collections=[]))

        mock_qdrant_manager = Mock()
        mock_qdrant_manager.get_client = Mock(return_value=mock_qdrant_client)

        with (
            patch("packages.webui.api.health.ws_manager.redis", mock_redis),
            patch("httpx.AsyncClient.get", side_effect=mock_get),
            patch(
                "packages.webui.api.health._check_embedding_service_health",
                side_effect=mock_check_embedding_service_health,
            ),
            patch("packages.webui.api.health.check_postgres_connection", side_effect=mock_check_postgres),
            patch("packages.webui.api.health.qdrant_manager", mock_qdrant_manager),
        ):
            response = test_client.get("/api/health/readyz")
            assert response.status_code == 503  # Should be 503 when not ready
            data = response.json()
            assert data["ready"] is False


class TestVecpipeHealthEndpoints:
    """Test health endpoints in vecpipe"""

    @pytest.fixture()
    def vecpipe_app(self) -> TestClient:
        """Create vecpipe app for testing"""

        return TestClient(app)

    def test_vecpipe_health_all_healthy(self, vecpipe_app) -> None:
        """Test vecpipe health when all components are healthy"""
        mock_qdrant = Mock()

        # Create async mock for qdrant_client.get
        async def mock_get(_path) -> None:
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

    def test_vecpipe_health_qdrant_unhealthy(self, vecpipe_app) -> None:
        """Test vecpipe health when Qdrant is unhealthy"""
        with patch("packages.vecpipe.search_api.qdrant_client", None):
            response = vecpipe_app.get("/health")
            assert response.status_code == 503
            data = response.json()["detail"]
            assert data["status"] == "unhealthy"
            assert data["components"]["qdrant"]["status"] == "unhealthy"

    def test_vecpipe_health_embedding_degraded(self, vecpipe_app) -> None:
        """Test vecpipe health when embedding service is degraded"""
        mock_qdrant = Mock()

        # Create async mock for qdrant_client.get
        async def mock_get(_path) -> None:
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
