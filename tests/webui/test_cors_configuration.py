"""Tests for CORS configuration in the WebUI."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Import the module so we can patch it properly
import packages.webui.main
from packages.webui.main import _validate_cors_origins, create_app


class TestCORSOriginValidation:
    """Test CORS origin validation function."""

    def test_valid_origins(self) -> None:
        """Test that valid origins are accepted."""
        origins = [
            "http://localhost:3000",
            "https://example.com",
            "http://127.0.0.1:5173",
        ]
        result = _validate_cors_origins(origins)
        assert len(result) == 3
        assert all(origin in result for origin in origins)

    def test_invalid_origin_format(self) -> None:
        """Test that invalid origins are rejected."""
        origins = [
            "not-a-url",
            "ftp://example.com",  # Valid URL but not HTTP/HTTPS
            "",
            "localhost:3000",  # Missing scheme
        ]
        result = _validate_cors_origins(origins)
        # FTP URL should be accepted (has scheme and netloc)
        assert len(result) == 1
        assert "ftp://example.com" in result

    def test_wildcard_origin_development(self) -> None:
        """Test wildcard origin handling in development."""
        origins = ["*", "http://localhost:3000"]

        # Patch both the module-level import and the function's access
        with (
            patch.object(packages.webui.main.shared_settings, "ENVIRONMENT", "development"),
            patch("packages.webui.main.logger") as mock_logger,
        ):
            result = _validate_cors_origins(origins)

            # In development, wildcard should be allowed but warned
            assert len(result) == 2
            assert "*" in result
            mock_logger.warning.assert_called()

    @patch("packages.webui.main.logger")
    def test_wildcard_origin_production(self, mock_logger) -> None:
        """Test wildcard origin rejection in production."""
        origins = ["*", "http://localhost:3000", "null"]

        with patch("packages.webui.main.shared_settings") as mock_settings:
            mock_settings.ENVIRONMENT = "production"
            result = _validate_cors_origins(origins)

        # In production, wildcards should be rejected
        assert len(result) == 1
        assert "http://localhost:3000" in result
        assert "*" not in result
        assert "null" not in result
        mock_logger.error.assert_called()


class TestCORSConfiguration:
    """Test CORS configuration in the FastAPI app."""

    @patch("packages.webui.main.shared_settings")
    @patch("packages.webui.main.configure_global_embedding_service")
    def test_cors_middleware_configuration(self, mock_embed_service, mock_settings) -> None:
        """Test that CORS middleware is configured correctly."""
        # Mock settings
        mock_settings.CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:5173"
        mock_settings.USE_MOCK_EMBEDDINGS = True
        mock_settings.INTERNAL_API_KEY = "test-key"
        mock_settings.ENVIRONMENT = "development"

        app = create_app()

        # Check that middleware is added by looking at user_middleware
        cors_middleware_found = False

        for middleware in app.user_middleware:
            # Check if this is CORS middleware
            if hasattr(middleware, "cls") and "CORSMiddleware" in str(middleware.cls):
                cors_middleware_found = True
                # Verify it's the CORS middleware (FastAPI uses Starlette's implementation)
                assert "starlette.middleware.cors.CORSMiddleware" in str(middleware.cls)
                break

        assert cors_middleware_found, "CORS middleware not found in app.user_middleware"

    @patch("packages.webui.main.shared_settings")
    @patch("packages.webui.main.configure_global_embedding_service")
    @patch("packages.webui.main.logger")
    def test_empty_cors_origins_warning(self, mock_logger, mock_embed_service, mock_settings) -> None:
        """Test warning when no valid CORS origins are configured."""
        # Mock settings with invalid origins
        mock_settings.CORS_ORIGINS = "not-a-url,"
        mock_settings.USE_MOCK_EMBEDDINGS = True
        mock_settings.INTERNAL_API_KEY = "test-key"
        mock_settings.ENVIRONMENT = "development"

        create_app()

        # Check that warning was logged
        warning_calls = [
            call for call in mock_logger.warning.call_args_list if "No valid CORS origins configured" in str(call)
        ]
        assert len(warning_calls) > 0

    @patch("packages.webui.main.shared_settings")
    @patch("packages.webui.main.configure_global_embedding_service")
    def test_cors_headers_in_response(self, mock_embed_service, mock_settings) -> None:
        """Test that CORS headers are present in responses."""
        # Mock settings
        mock_settings.CORS_ORIGINS = "http://testclient"  # TestClient default origin
        mock_settings.USE_MOCK_EMBEDDINGS = True
        mock_settings.INTERNAL_API_KEY = "test-key"
        mock_settings.ENVIRONMENT = "development"
        mock_settings.WEBUI_PORT = 8080

        app = create_app()
        client = TestClient(app)

        # Make a request with origin header
        response = client.get("/health", headers={"Origin": "http://testclient"})

        assert response.status_code == 200
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "http://testclient"
        assert "access-control-allow-credentials" in response.headers
        assert response.headers["access-control-allow-credentials"] == "true"

    @patch("packages.webui.main.shared_settings")
    @patch("packages.webui.main.configure_global_embedding_service")
    def test_cors_preflight_request(self, mock_embed_service, mock_settings) -> None:
        """Test CORS preflight (OPTIONS) request handling."""
        # Mock settings
        mock_settings.CORS_ORIGINS = "http://testclient"
        mock_settings.USE_MOCK_EMBEDDINGS = True
        mock_settings.INTERNAL_API_KEY = "test-key"
        mock_settings.ENVIRONMENT = "development"
        mock_settings.WEBUI_PORT = 8080

        app = create_app()
        client = TestClient(app)

        # Make a preflight request
        response = client.options(
            "/api/auth/login",
            headers={
                "Origin": "http://testclient",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://testclient"
        assert "POST" in response.headers["access-control-allow-methods"]
        assert "content-type" in response.headers["access-control-allow-headers"].lower()


if __name__ == "__main__":
    pytest.main([__file__])
