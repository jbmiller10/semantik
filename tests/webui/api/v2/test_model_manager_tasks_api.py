"""API tests for model manager task progress endpoint."""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from shared.database import get_db
from webui.auth import get_current_user
from webui.main import app


@pytest_asyncio.fixture
async def non_superuser_client(db_session):
    """Provide AsyncClient with non-superuser authentication."""
    mock_regular_user = {
        "id": 2,
        "username": "user",
        "email": "user@example.com",
        "full_name": "Regular User",
        "is_superuser": False,
    }

    async def override_get_current_user() -> dict[str, Any]:
        return mock_regular_user

    async def override_get_db() -> AsyncGenerator[Any, None]:
        yield db_session

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def superuser_client(db_session):
    """Provide AsyncClient with superuser authentication."""
    mock_admin_user = {
        "id": 1,
        "username": "admin",
        "email": "admin@example.com",
        "full_name": "Admin User",
        "is_superuser": True,
    }

    async def override_get_current_user() -> dict[str, Any]:
        return mock_admin_user

    async def override_get_db() -> AsyncGenerator[Any, None]:
        yield db_session

    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


class TestGetTaskProgress:
    """Tests for GET /api/v2/models/tasks/{task_id}."""

    @pytest.mark.asyncio()
    async def test_returns_task_progress(self, superuser_client) -> None:
        """Test successful retrieval of task progress."""
        mock_progress = {
            "task_id": "task-123",
            "model_id": "test/model",
            "operation": "download",
            "status": "running",
            "bytes_downloaded": 1024,
            "bytes_total": 2048,
            "error": None,
            "updated_at": 1704067200.0,
        }

        with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
            mock_redis_manager = AsyncMock()
            mock_redis_client = AsyncMock()
            mock_redis_manager.async_client.return_value = mock_redis_client
            mock_get_redis.return_value = mock_redis_manager

            with patch("webui.api.v2.model_manager.task_state.get_task_progress") as mock_get_progress:
                mock_get_progress.return_value = mock_progress

                response = await superuser_client.get("/api/v2/models/tasks/task-123")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-123"
        assert data["model_id"] == "test/model"
        assert data["operation"] == "download"
        assert data["status"] == "running"
        assert data["bytes_downloaded"] == 1024
        assert data["bytes_total"] == 2048

    @pytest.mark.asyncio()
    async def test_returns_404_for_unknown_task(self, superuser_client) -> None:
        """Test 404 response for non-existent task."""
        with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
            mock_redis_manager = AsyncMock()
            mock_redis_client = AsyncMock()
            mock_redis_manager.async_client.return_value = mock_redis_client
            mock_get_redis.return_value = mock_redis_manager

            with patch("webui.api.v2.model_manager.task_state.get_task_progress") as mock_get_progress:
                mock_get_progress.return_value = None

                response = await superuser_client.get("/api/v2/models/tasks/nonexistent")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio()
    async def test_requires_superuser(self, non_superuser_client) -> None:
        """Test 403 response for non-superuser."""
        response = await non_superuser_client.get("/api/v2/models/tasks/task-123")

        assert response.status_code == 403
        data = response.json()
        assert "superuser" in data["detail"].lower()

    @pytest.mark.asyncio()
    async def test_handles_completed_status(self, superuser_client) -> None:
        """Test completed task progress."""
        mock_progress = {
            "task_id": "task-456",
            "model_id": "test/model",
            "operation": "download",
            "status": "completed",
            "bytes_downloaded": 2048,
            "bytes_total": 2048,
            "error": None,
            "updated_at": 1704067300.0,
        }

        with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
            mock_redis_manager = AsyncMock()
            mock_redis_client = AsyncMock()
            mock_redis_manager.async_client.return_value = mock_redis_client
            mock_get_redis.return_value = mock_redis_manager

            with patch("webui.api.v2.model_manager.task_state.get_task_progress") as mock_get_progress:
                mock_get_progress.return_value = mock_progress

                response = await superuser_client.get("/api/v2/models/tasks/task-456")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["bytes_downloaded"] == data["bytes_total"]

    @pytest.mark.asyncio()
    async def test_handles_failed_status_with_error(self, superuser_client) -> None:
        """Test failed task with error message."""
        mock_progress = {
            "task_id": "task-789",
            "model_id": "test/model",
            "operation": "download",
            "status": "failed",
            "bytes_downloaded": 512,
            "bytes_total": 2048,
            "error": "Network connection failed",
            "updated_at": 1704067400.0,
        }

        with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
            mock_redis_manager = AsyncMock()
            mock_redis_client = AsyncMock()
            mock_redis_manager.async_client.return_value = mock_redis_client
            mock_get_redis.return_value = mock_redis_manager

            with patch("webui.api.v2.model_manager.task_state.get_task_progress") as mock_get_progress:
                mock_get_progress.return_value = mock_progress

                response = await superuser_client.get("/api/v2/models/tasks/task-789")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Network connection failed"


class TestTaskProgressResponseSchema:
    """Tests for TaskProgressResponse schema validation."""

    def test_valid_response_schema(self):
        """Test that valid progress data passes schema validation."""
        from webui.api.v2.model_manager_schemas import TaskProgressResponse, TaskStatus

        response = TaskProgressResponse(
            task_id="task-123",
            model_id="test/model",
            operation="download",
            status=TaskStatus.RUNNING,
            bytes_downloaded=1024,
            bytes_total=2048,
            error=None,
            updated_at=1704067200.0,
        )

        assert response.task_id == "task-123"
        assert response.status == TaskStatus.RUNNING

    def test_schema_with_error(self):
        """Test schema with error field populated."""
        from webui.api.v2.model_manager_schemas import TaskProgressResponse, TaskStatus

        response = TaskProgressResponse(
            task_id="task-456",
            model_id="test/model",
            operation="delete",
            status=TaskStatus.FAILED,
            error="Permission denied",
            updated_at=1704067200.0,
        )

        assert response.error == "Permission denied"
        assert response.status == TaskStatus.FAILED

    def test_default_bytes_values(self):
        """Test default values for bytes fields."""
        from webui.api.v2.model_manager_schemas import TaskProgressResponse, TaskStatus

        response = TaskProgressResponse(
            task_id="task-789",
            model_id="test/model",
            operation="delete",
            status=TaskStatus.PENDING,
            updated_at=1704067200.0,
        )

        assert response.bytes_downloaded == 0
        assert response.bytes_total == 0
