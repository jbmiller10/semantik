#!/usr/bin/env python3
"""
Integration tests for WebSocket with Redis Streams.

Tests the full integration of WebSocket connections receiving
real-time job updates via Redis streams.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from packages.webui.main import app
from packages.webui.redis_streams import publish_job_update


class TestWebSocketRedisIntegration:
    """Test WebSocket integration with Redis streams."""

    @pytest.fixture()
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture()
    def auth_token(self):
        """Mock authentication token."""
        return "test-jwt-token"

    @pytest.fixture()
    def mock_job(self):
        """Mock job data."""
        return {
            "id": "test-job-123",
            "name": "Test Job",
            "status": "processing",
            "total_files": 10,
            "processed_files": 0,
            "failed_files": 0,
            "current_file": None,
            "error": None,
            "user_id": 1,
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }

    @pytest.mark.asyncio()
    @patch("packages.webui.api.jobs.settings")
    @patch("packages.webui.api.jobs.create_job_repository")
    @patch("packages.webui.auth.verify_token")
    @patch("packages.webui.auth.database.get_user")
    async def test_websocket_authentication_required(
        self, mock_get_user, mock_verify_token, mock_create_repo, mock_settings, client
    ):
        """Test that WebSocket requires authentication when enabled."""
        mock_settings.DISABLE_AUTH = False

        # Try to connect without token
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/test-job-123"):
                pass

        assert exc_info.value.code == 1008
        assert "Missing authentication token" in str(exc_info.value.reason)

    @pytest.mark.asyncio()
    @patch("packages.webui.api.jobs.settings")
    @patch("packages.webui.api.jobs.create_job_repository")
    @patch("packages.webui.auth.verify_token")
    @patch("packages.webui.auth.database.get_user")
    async def test_websocket_authentication_success(
        self, mock_get_user, mock_verify_token, mock_create_repo, mock_settings, client, auth_token, mock_job
    ):
        """Test successful WebSocket authentication."""
        mock_settings.DISABLE_AUTH = False
        mock_verify_token.return_value = "testuser"
        mock_get_user.return_value = {"id": 1, "username": "testuser", "is_active": True}

        # Mock job repository
        mock_repo = AsyncMock()
        mock_repo.get_job = AsyncMock(return_value=mock_job)
        mock_create_repo.return_value = mock_repo

        # Mock WebSocket manager
        with patch("packages.webui.api.jobs.manager") as mock_manager:
            mock_manager.connect = AsyncMock()
            mock_manager.send_initial_state = AsyncMock()
            mock_manager.disconnect = AsyncMock()

            # Connect with token
            with client.websocket_connect(f"/ws/test-job-123?token={auth_token}") as websocket:
                # Verify authentication was checked
                mock_verify_token.assert_called_once_with(auth_token, "access")
                mock_get_user.assert_called_once_with("testuser")

                # Verify job was fetched and initial state sent
                mock_repo.get_job.assert_called_once_with("test-job-123")
                mock_manager.send_initial_state.assert_called_once()

    @pytest.mark.asyncio()
    @patch("packages.webui.api.jobs.settings")
    @patch("packages.webui.api.jobs.create_job_repository")
    async def test_websocket_job_access_control(self, mock_create_repo, mock_settings, client):
        """Test that users can only access their own jobs."""
        mock_settings.DISABLE_AUTH = False

        # Mock user with different ID than job owner
        with patch("packages.webui.auth.verify_token", return_value="otheruser"):
            with patch(
                "packages.webui.auth.database.get_user",
                return_value={"id": 2, "username": "otheruser", "is_active": True},
            ):

                # Mock job owned by different user
                mock_repo = AsyncMock()
                mock_repo.get_job = AsyncMock(return_value={"user_id": 1})  # Different user
                mock_create_repo.return_value = mock_repo

                # Try to connect
                with pytest.raises(WebSocketDisconnect) as exc_info:
                    with client.websocket_connect("/ws/test-job-123?token=valid-token"):
                        pass

                assert exc_info.value.code == 1008
                assert "Access denied" in str(exc_info.value.reason)

    @pytest.mark.asyncio()
    @patch("packages.webui.api.jobs.settings")
    @patch("packages.webui.api.jobs.create_job_repository")
    @patch("packages.webui.api.jobs.manager")
    async def test_websocket_receives_initial_state(
        self, mock_manager, mock_create_repo, mock_settings, client, mock_job
    ):
        """Test that WebSocket receives initial job state on connection."""
        mock_settings.DISABLE_AUTH = True  # Disable auth for simplicity

        # Mock job repository
        mock_repo = AsyncMock()
        mock_repo.get_job = AsyncMock(return_value=mock_job)
        mock_create_repo.return_value = mock_repo

        # Mock WebSocket manager
        mock_manager.connect = AsyncMock()
        mock_manager.send_initial_state = AsyncMock()
        mock_manager.disconnect = AsyncMock()

        with client.websocket_connect("/ws/test-job-123") as websocket:
            # Verify initial state was sent
            mock_manager.send_initial_state.assert_called_once()

            # Check the data sent
            call_args = mock_manager.send_initial_state.call_args[0]
            assert call_args[1] == "test-job-123"
            assert call_args[2]["id"] == mock_job["id"]
            assert call_args[2]["status"] == mock_job["status"]

    @pytest.mark.asyncio()
    async def test_redis_stream_publish_integration(self):
        """Test publishing updates to Redis streams."""
        redis_url = "redis://localhost:6379/0"
        job_id = "test-job-123"

        # Mock Redis client
        with patch("redis.from_url") as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client

            # Publish an update
            publish_job_update(job_id, "file_processing", {"file": "test.txt", "progress": 50}, redis_url)

            # Verify Redis operations
            mock_client.xadd.assert_called_once()
            mock_client.expire.assert_called_once()

            # Check message format
            xadd_args = mock_client.xadd.call_args[0]
            assert xadd_args[0] == f"job:stream:{job_id}"

            message = xadd_args[1]
            assert message["type"] == "file_processing"
            assert json.loads(message["data"]) == {"file": "test.txt", "progress": 50}

    @pytest.mark.asyncio()
    @patch("packages.webui.api.jobs.settings")
    async def test_websocket_no_auth_when_disabled(self, mock_settings, client):
        """Test that WebSocket works without auth when disabled."""
        mock_settings.DISABLE_AUTH = True

        with patch("packages.webui.api.jobs.create_job_repository") as mock_create_repo:
            # Mock job repository
            mock_repo = AsyncMock()
            mock_repo.get_job = AsyncMock(
                return_value={
                    "id": "test-job",
                    "name": "Test",
                    "status": "completed",
                    "user_id": None,  # No user when auth disabled
                }
            )
            mock_create_repo.return_value = mock_repo

            with patch("packages.webui.api.jobs.manager") as mock_manager:
                mock_manager.connect = AsyncMock()
                mock_manager.send_initial_state = AsyncMock()
                mock_manager.disconnect = AsyncMock()

                # Should connect without token
                with client.websocket_connect("/ws/test-job") as websocket:
                    # Verify connection was established
                    mock_manager.connect.assert_called_once()

                    # Verify no user check was performed
                    mock_repo.get_job.assert_called_once_with("test-job")
