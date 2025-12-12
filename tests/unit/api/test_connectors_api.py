"""Unit tests for Connectors API v2 endpoints."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from webui.api.v2.connectors import router

# Create test app
app = FastAPI()
app.include_router(router)


@pytest.fixture()
def mock_current_user() -> dict[str, Any]:
    """Mock authenticated user."""
    return {"id": "1", "username": "testuser"}


@pytest.fixture()
def test_client(mock_current_user):
    """Create test client with mocked dependencies."""
    from webui.api.v2 import connectors

    # Override auth dependency
    app.dependency_overrides[connectors.get_current_user] = lambda: mock_current_user

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


class TestListConnectorsEndpoint:
    """Tests for GET /api/v2/connectors."""

    def test_list_connectors_success(self, test_client):
        """Test successful connector catalog listing."""
        with patch("webui.api.v2.connectors.get_connector_catalog") as mock_catalog:
            mock_catalog.return_value = [
                {
                    "type": "directory",
                    "name": "Local Directory",
                    "icon": "folder",
                },
                {
                    "type": "git",
                    "name": "Git Repository",
                    "icon": "git-branch",
                },
            ]

            response = test_client.get("/api/v2/connectors")

            assert response.status_code == 200
            data = response.json()
            assert "connectors" in data
            assert len(data["connectors"]) == 2
            assert data["connectors"][0]["type"] == "directory"


class TestGetConnectorEndpoint:
    """Tests for GET /api/v2/connectors/{connector_type}."""

    def test_get_connector_success(self, test_client):
        """Test successful connector definition retrieval."""
        with patch("webui.api.v2.connectors.get_connector_definition") as mock_def:
            mock_def.return_value = {
                "name": "Local Directory",
                "icon": "folder",
                "fields": [
                    {"name": "path", "type": "text", "required": True},
                ],
                "secrets": [],
            }

            response = test_client.get("/api/v2/connectors/directory")

            assert response.status_code == 200
            data = response.json()
            assert data["type"] == "directory"
            assert data["definition"]["name"] == "Local Directory"

    def test_get_connector_not_found(self, test_client):
        """Test 404 when connector type doesn't exist."""
        with patch("webui.api.v2.connectors.get_connector_definition") as mock_def:
            mock_def.return_value = None

            response = test_client.get("/api/v2/connectors/nonexistent")

            assert response.status_code == 404
            assert "Unknown connector type" in response.json()["detail"]


class TestGitPreviewEndpoint:
    """Tests for POST /api/v2/connectors/preview/git."""

    def test_git_preview_success(self, test_client):
        """Test successful git repository preview."""
        with patch("webui.api.v2.connectors.GitConnector") as mock_connector_cls:
            mock_connector = AsyncMock()
            mock_connector.authenticate.return_value = True
            mock_connector.get_refs.return_value = ["main", "develop", "v1.0.0"]
            mock_connector_cls.return_value = mock_connector

            response = test_client.post(
                "/api/v2/connectors/preview/git",
                json={
                    "repo_url": "https://github.com/user/repo.git",
                    "ref": "main",
                    "auth_method": "none",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["repo_url"] == "https://github.com/user/repo.git"
            assert "main" in data["refs_found"]
            assert data["error"] is None

    def test_git_preview_with_token(self, test_client):
        """Test git preview with HTTPS token authentication."""
        with patch("webui.api.v2.connectors.GitConnector") as mock_connector_cls:
            mock_connector = AsyncMock()
            mock_connector.authenticate.return_value = True
            mock_connector.get_refs.return_value = ["main"]
            mock_connector_cls.return_value = mock_connector

            response = test_client.post(
                "/api/v2/connectors/preview/git",
                json={
                    "repo_url": "https://github.com/user/private-repo.git",
                    "ref": "main",
                    "auth_method": "https_token",
                    "token": "ghp_xxxxxxxxxxxx",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            mock_connector.set_credentials.assert_called_once_with(token="ghp_xxxxxxxxxxxx")

    def test_git_preview_with_ssh_key(self, test_client):
        """Test git preview with SSH key authentication."""
        with patch("webui.api.v2.connectors.GitConnector") as mock_connector_cls:
            mock_connector = AsyncMock()
            mock_connector.authenticate.return_value = True
            mock_connector.get_refs.return_value = ["main"]
            mock_connector_cls.return_value = mock_connector

            response = test_client.post(
                "/api/v2/connectors/preview/git",
                json={
                    "repo_url": "git@github.com:user/repo.git",
                    "ref": "main",
                    "auth_method": "ssh_key",
                    "ssh_key": "-----BEGIN OPENSSH PRIVATE KEY-----\n...",
                    "ssh_passphrase": "secret",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            mock_connector.set_credentials.assert_called_once_with(
                ssh_key="-----BEGIN OPENSSH PRIVATE KEY-----\n...",
                ssh_passphrase="secret",
            )

    def test_git_preview_auth_failed(self, test_client):
        """Test git preview when authentication fails."""
        with patch("webui.api.v2.connectors.GitConnector") as mock_connector_cls:
            mock_connector = AsyncMock()
            mock_connector.authenticate.return_value = False
            mock_connector_cls.return_value = mock_connector

            response = test_client.post(
                "/api/v2/connectors/preview/git",
                json={
                    "repo_url": "https://github.com/user/repo.git",
                    "ref": "main",
                    "auth_method": "none",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            assert data["refs_found"] == []
            assert "Authentication failed" in data["error"]

    def test_git_preview_config_error(self, test_client):
        """Test git preview with invalid configuration."""
        with patch("webui.api.v2.connectors.GitConnector") as mock_connector_cls:
            mock_connector_cls.side_effect = ValueError("Invalid repository URL")

            response = test_client.post(
                "/api/v2/connectors/preview/git",
                json={
                    "repo_url": "not-a-valid-url",
                    "ref": "main",
                    "auth_method": "none",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            assert "Invalid repository URL" in data["error"]

    def test_git_preview_connection_error(self, test_client):
        """Test git preview when connection fails."""
        with patch("webui.api.v2.connectors.GitConnector") as mock_connector_cls:
            mock_connector = AsyncMock()
            mock_connector.authenticate.side_effect = Exception("Connection timeout")
            mock_connector_cls.return_value = mock_connector

            response = test_client.post(
                "/api/v2/connectors/preview/git",
                json={
                    "repo_url": "https://github.com/user/repo.git",
                    "ref": "main",
                    "auth_method": "none",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            assert "Connection failed" in data["error"]


class TestImapPreviewEndpoint:
    """Tests for POST /api/v2/connectors/preview/imap."""

    def test_imap_preview_success(self, test_client):
        """Test successful IMAP connection preview."""
        with patch("webui.api.v2.connectors.asyncio") as mock_asyncio:
            mock_loop = AsyncMock()
            mock_asyncio.get_event_loop.return_value = mock_loop

            async def mock_executor(executor, func):
                # Simulate successful connection returning mailboxes
                return ["INBOX", "Sent", "Drafts"]

            mock_loop.run_in_executor = mock_executor

            response = test_client.post(
                "/api/v2/connectors/preview/imap",
                json={
                    "host": "imap.example.com",
                    "port": 993,
                    "use_ssl": True,
                    "username": "user@example.com",
                    "password": "secret",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["host"] == "imap.example.com"
            assert "INBOX" in data["mailboxes_found"]
            assert data["error"] is None

    def test_imap_preview_auth_failed(self, test_client):
        """Test IMAP preview when authentication fails."""
        import imaplib

        with patch("webui.api.v2.connectors.asyncio") as mock_asyncio:
            mock_loop = AsyncMock()
            mock_asyncio.get_event_loop.return_value = mock_loop

            async def mock_executor(executor, func):
                raise imaplib.IMAP4.error("LOGIN failed")

            mock_loop.run_in_executor = mock_executor

            response = test_client.post(
                "/api/v2/connectors/preview/imap",
                json={
                    "host": "imap.example.com",
                    "port": 993,
                    "use_ssl": True,
                    "username": "user@example.com",
                    "password": "wrong-password",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            assert "IMAP error" in data["error"]

    def test_imap_preview_connection_failed(self, test_client):
        """Test IMAP preview when connection fails."""
        with patch("webui.api.v2.connectors.asyncio") as mock_asyncio:
            mock_loop = AsyncMock()
            mock_asyncio.get_event_loop.return_value = mock_loop

            async def mock_executor(executor, func):
                raise OSError("Connection refused")

            mock_loop.run_in_executor = mock_executor

            response = test_client.post(
                "/api/v2/connectors/preview/imap",
                json={
                    "host": "invalid.example.com",
                    "port": 993,
                    "use_ssl": True,
                    "username": "user@example.com",
                    "password": "secret",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            assert "Connection failed" in data["error"]

    def test_imap_preview_timeout(self, test_client):
        """Test IMAP preview when connection times out."""
        with patch("webui.api.v2.connectors.asyncio") as mock_asyncio:
            mock_loop = AsyncMock()
            mock_asyncio.get_event_loop.return_value = mock_loop

            async def mock_executor(executor, func):
                raise TimeoutError("Connection timed out")

            mock_loop.run_in_executor = mock_executor

            response = test_client.post(
                "/api/v2/connectors/preview/imap",
                json={
                    "host": "slow.example.com",
                    "port": 993,
                    "use_ssl": True,
                    "username": "user@example.com",
                    "password": "secret",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            assert "Connection failed" in data["error"]

    def test_imap_preview_unexpected_error(self, test_client):
        """Test IMAP preview handles unexpected errors."""
        with patch("webui.api.v2.connectors.asyncio") as mock_asyncio:
            mock_loop = AsyncMock()
            mock_asyncio.get_event_loop.return_value = mock_loop

            async def mock_executor(executor, func):
                raise RuntimeError("Unexpected error")

            mock_loop.run_in_executor = mock_executor

            response = test_client.post(
                "/api/v2/connectors/preview/imap",
                json={
                    "host": "imap.example.com",
                    "port": 993,
                    "use_ssl": True,
                    "username": "user@example.com",
                    "password": "secret",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is False
            assert "Unexpected error" in data["error"]
