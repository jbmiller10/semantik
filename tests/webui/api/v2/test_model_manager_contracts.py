"""Contract tests for model manager API schemas and access control."""

from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from shared.database import get_db
from webui.api.v2.model_manager_schemas import (
    CacheSizeInfo,
    ConflictType,
    ModelManagerConflictResponse,
    TaskResponse,
    TaskStatus,
)
from webui.auth import get_current_user
from webui.main import app


class TestModelManagerConflictResponse:
    """Tests for 409 Conflict response schema."""

    def test_cross_op_exclusion_response(self) -> None:
        """Test 409 response for concurrent operation conflict."""
        response = ModelManagerConflictResponse(
            conflict_type=ConflictType.CROSS_OP_EXCLUSION,
            detail="A download is already in progress for this model",
            model_id="Qwen/Qwen3-Embedding-0.6B",
            active_operation="download",
            active_task_id="task-abc-123",
        )
        assert response.conflict_type == ConflictType.CROSS_OP_EXCLUSION
        assert response.active_task_id == "task-abc-123"
        assert response.blocked_by_collections == []
        assert response.requires_confirmation is False

    def test_in_use_block_response(self) -> None:
        """Test 409 response for model in use by collections."""
        response = ModelManagerConflictResponse(
            conflict_type=ConflictType.IN_USE_BLOCK,
            detail="Model is used by 2 collections and cannot be deleted",
            model_id="BAAI/bge-small-en-v1.5",
            blocked_by_collections=["Collection A", "Collection B"],
        )
        assert response.conflict_type == ConflictType.IN_USE_BLOCK
        assert len(response.blocked_by_collections) == 2
        assert response.active_operation is None

    def test_requires_confirmation_response(self) -> None:
        """Test 409 response when warnings require confirmation."""
        response = ModelManagerConflictResponse(
            conflict_type=ConflictType.REQUIRES_CONFIRMATION,
            detail="Deletion requires confirmation due to warnings",
            model_id="Qwen/Qwen3-Embedding-0.6B",
            requires_confirmation=True,
            warnings=[
                "Model is the system default embedding model",
                "3 users have this as their default preference",
            ],
        )
        assert response.conflict_type == ConflictType.REQUIRES_CONFIRMATION
        assert response.requires_confirmation is True
        assert len(response.warnings) == 2

    def test_required_fields(self) -> None:
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            ModelManagerConflictResponse(
                conflict_type=ConflictType.CROSS_OP_EXCLUSION,
                # Missing required: detail, model_id
            )


class TestTaskResponse:
    """Tests for task response schema."""

    def test_idempotent_already_installed(self) -> None:
        """Test response when model is already installed (download no-op)."""
        response = TaskResponse(
            task_id=None,
            model_id="BAAI/bge-small-en-v1.5",
            operation="download",
            status=TaskStatus.ALREADY_INSTALLED,
        )
        assert response.task_id is None
        assert response.status == TaskStatus.ALREADY_INSTALLED
        assert response.warnings == []

    def test_idempotent_not_installed(self) -> None:
        """Test response when model is not installed (delete no-op)."""
        response = TaskResponse(
            task_id=None,
            model_id="BAAI/bge-small-en-v1.5",
            operation="delete",
            status=TaskStatus.NOT_INSTALLED,
        )
        assert response.task_id is None
        assert response.status == TaskStatus.NOT_INSTALLED

    def test_download_started(self) -> None:
        """Test response when download task is started."""
        response = TaskResponse(
            task_id="task-xyz-456",
            model_id="Qwen/Qwen3-Embedding-0.6B",
            operation="download",
            status=TaskStatus.PENDING,
        )
        assert response.task_id == "task-xyz-456"
        assert response.status == TaskStatus.PENDING

    def test_delete_with_warnings(self) -> None:
        """Test delete response includes warnings when confirm=true."""
        response = TaskResponse(
            task_id="task-del-789",
            model_id="Qwen/Qwen3-Embedding-0.6B",
            operation="delete",
            status=TaskStatus.PENDING,
            warnings=["Model was the system default"],
        )
        assert response.task_id == "task-del-789"
        assert len(response.warnings) == 1


class TestCacheSizeInfo:
    """Tests for cache size info schema."""

    def test_cache_size_fields(self) -> None:
        """Test cache size breakdown fields."""
        info = CacheSizeInfo(
            total_cache_size_mb=5000,
            managed_cache_size_mb=3500,
            unmanaged_cache_size_mb=1500,
            unmanaged_repo_count=5,
        )
        assert info.total_cache_size_mb == 5000
        assert info.managed_cache_size_mb == 3500
        assert info.unmanaged_cache_size_mb == 1500
        assert info.unmanaged_repo_count == 5

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            CacheSizeInfo(
                total_cache_size_mb=5000,
                managed_cache_size_mb=3500,
                unmanaged_cache_size_mb=1500,
                unmanaged_repo_count=5,
                extra_field="should fail",  # type: ignore[call-arg]
            )


# =============================================================================
# API-level superuser gating tests
# =============================================================================


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


class TestModelManagerSuperuserGating:
    """Tests for superuser-only access control."""

    @pytest.mark.asyncio()
    async def test_list_models_requires_superuser(self, non_superuser_client) -> None:
        """Test GET /api/v2/models returns 403 for non-superuser."""
        response = await non_superuser_client.get("/api/v2/models")
        assert response.status_code == 403
        data = response.json()
        assert "superuser" in data["detail"].lower()

    @pytest.mark.asyncio()
    async def test_list_models_allowed_for_superuser(self, superuser_client) -> None:
        """Test GET /api/v2/models returns 200 for superuser."""
        response = await superuser_client.get("/api/v2/models")
        assert response.status_code == 200
