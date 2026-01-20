"""Unit tests for model manager write API endpoints.

Tests for:
- POST /api/v2/models/download
- GET /api/v2/models/usage
- DELETE /api/v2/models/cache
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from shared.database import get_db
from webui.api.v2.model_manager_schemas import (
    ModelDownloadRequest,
    ModelUsageResponse,
)
from webui.auth import get_current_user
from webui.main import app
from webui.model_manager.task_state import CrossOpConflictError

# =============================================================================
# Fixtures
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


# =============================================================================
# Schema Tests
# =============================================================================


class TestModelDownloadRequestSchema:
    """Tests for ModelDownloadRequest schema."""

    def test_valid_request(self) -> None:
        """Test valid download request."""
        request = ModelDownloadRequest(model_id="Qwen/Qwen3-Embedding-0.6B")
        assert request.model_id == "Qwen/Qwen3-Embedding-0.6B"

    def test_missing_model_id_raises(self) -> None:
        """Test that missing model_id raises validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ModelDownloadRequest()


class TestModelUsageResponseSchema:
    """Tests for ModelUsageResponse schema."""

    def test_installed_model_response(self) -> None:
        """Test response for installed model with usage."""
        response = ModelUsageResponse(
            model_id="test/model",
            is_installed=True,
            size_on_disk_mb=500,
            estimated_freed_size_mb=500,
            blocked_by_collections=["Collection A"],
            user_preferences_count=3,
            llm_config_count=2,
            is_default_embedding_model=True,
            loaded_in_vecpipe=True,
            loaded_vecpipe_model_types=["embedding"],
            warnings=["Model is used by 1 collection(s)"],
            can_delete=False,
            requires_confirmation=False,
        )
        assert response.is_installed is True
        assert response.can_delete is False
        assert len(response.blocked_by_collections) == 1

    def test_not_installed_model_response(self) -> None:
        """Test response for model that isn't installed."""
        response = ModelUsageResponse(
            model_id="test/model",
            is_installed=False,
            can_delete=True,
            requires_confirmation=False,
        )
        assert response.is_installed is False
        assert response.size_on_disk_mb is None

    def test_deletable_with_confirmation(self) -> None:
        """Test response for model deletable with confirmation."""
        response = ModelUsageResponse(
            model_id="test/model",
            is_installed=True,
            size_on_disk_mb=500,
            warnings=["3 users have this as default"],
            can_delete=True,
            requires_confirmation=True,
        )
        assert response.can_delete is True
        assert response.requires_confirmation is True


# =============================================================================
# POST /api/v2/models/download Tests
# =============================================================================


class TestDownloadModelEndpoint:
    """Tests for POST /api/v2/models/download."""

    @pytest.mark.asyncio()
    async def test_requires_superuser(self, non_superuser_client) -> None:
        """Test 403 response for non-superuser."""
        response = await non_superuser_client.post(
            "/api/v2/models/download",
            json={"model_id": "test/model"},
        )
        assert response.status_code == 403
        assert "superuser" in response.json()["detail"].lower()

    @pytest.mark.asyncio()
    async def test_non_curated_model_rejected(self, superuser_client) -> None:
        """Test 400 response for model not in curated list."""
        with patch("webui.api.v2.model_manager.get_curated_model_ids") as mock_curated:
            mock_curated.return_value = {"Qwen/Qwen3-Embedding-0.6B"}

            response = await superuser_client.post(
                "/api/v2/models/download",
                json={"model_id": "unknown/model"},
            )

        assert response.status_code == 400
        assert "not in the curated model list" in response.json()["detail"]

    @pytest.mark.asyncio()
    async def test_already_installed_returns_idempotent(self, superuser_client) -> None:
        """Test idempotent response when model is already installed."""
        with patch("webui.api.v2.model_manager.get_curated_model_ids") as mock_curated:
            mock_curated.return_value = {"test/model"}

            with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
                mock_installed.return_value = True

                response = await superuser_client.post(
                    "/api/v2/models/download",
                    json={"model_id": "test/model"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "already_installed"
        assert data["task_id"] is None

    @pytest.mark.asyncio()
    async def test_download_success(self, superuser_client) -> None:
        """Test successful download initiation."""
        with patch("webui.api.v2.model_manager.get_curated_model_ids") as mock_curated:
            mock_curated.return_value = {"test/model"}

            with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
                mock_installed.return_value = False

                with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
                    mock_redis_manager = AsyncMock()
                    mock_redis_client = AsyncMock()
                    mock_redis_manager.async_client.return_value = mock_redis_client
                    mock_get_redis.return_value = mock_redis_manager

                    with patch("webui.api.v2.model_manager.task_state.claim_model_operation") as mock_claim:
                        mock_claim.return_value = (True, None)

                        with patch("webui.api.v2.model_manager.task_state.init_task_progress") as mock_init:
                            mock_init.return_value = None

                            with patch("webui.api.v2.model_manager.celery_app.send_task") as mock_send:
                                mock_send.return_value = MagicMock()

                                response = await superuser_client.post(
                                    "/api/v2/models/download",
                                    json={"model_id": "test/model"},
                                )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["task_id"] is not None
        assert data["operation"] == "download"

    @pytest.mark.asyncio()
    async def test_download_dedupe(self, superuser_client) -> None:
        """Test de-duplication when same download is already running."""
        with patch("webui.api.v2.model_manager.get_curated_model_ids") as mock_curated:
            mock_curated.return_value = {"test/model"}

            with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
                mock_installed.return_value = False

                with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
                    mock_redis_manager = AsyncMock()
                    mock_redis_client = AsyncMock()
                    mock_redis_manager.async_client.return_value = mock_redis_client
                    mock_get_redis.return_value = mock_redis_manager

                    with patch("webui.api.v2.model_manager.task_state.claim_model_operation") as mock_claim:
                        # Not claimed, existing task returned
                        mock_claim.return_value = (False, "existing-task-123")

                        response = await superuser_client.post(
                            "/api/v2/models/download",
                            json={"model_id": "test/model"},
                        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "existing-task-123"
        assert data["status"] == "running"

    @pytest.mark.asyncio()
    async def test_cross_op_conflict(self, superuser_client) -> None:
        """Test 409 response when delete is active for same model."""
        with patch("webui.api.v2.model_manager.get_curated_model_ids") as mock_curated:
            mock_curated.return_value = {"test/model"}

            with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
                mock_installed.return_value = False

                with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
                    mock_redis_manager = AsyncMock()
                    mock_redis_client = AsyncMock()
                    mock_redis_manager.async_client.return_value = mock_redis_client
                    mock_get_redis.return_value = mock_redis_manager

                    with patch("webui.api.v2.model_manager.task_state.claim_model_operation") as mock_claim:
                        mock_claim.side_effect = CrossOpConflictError(
                            model_id="test/model",
                            active_operation="delete",
                            active_task_id="delete-task-456",
                        )

                        response = await superuser_client.post(
                            "/api/v2/models/download",
                            json={"model_id": "test/model"},
                        )

        assert response.status_code == 409
        data = response.json()["detail"]
        assert data["conflict_type"] == "cross_op_exclusion"
        assert data["active_operation"] == "delete"


# =============================================================================
# GET /api/v2/models/usage Tests
# =============================================================================


class TestModelUsageEndpoint:
    """Tests for GET /api/v2/models/usage."""

    @pytest.mark.asyncio()
    async def test_requires_superuser(self, non_superuser_client) -> None:
        """Test 403 response for non-superuser."""
        response = await non_superuser_client.get(
            "/api/v2/models/usage",
            params={"model_id": "test/model"},
        )
        assert response.status_code == 403

    @pytest.mark.asyncio()
    async def test_not_installed_model(self, superuser_client) -> None:
        """Test response for model that is not installed."""
        with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
            mock_installed.return_value = False

            with patch("webui.api.v2.model_manager._get_collections_using_model") as mock_cols:
                mock_cols.return_value = []

                with patch("webui.api.v2.model_manager._count_user_preferences_using_model") as mock_prefs:
                    mock_prefs.return_value = 0

                    with patch("webui.api.v2.model_manager._count_llm_configs_using_model") as mock_llm:
                        mock_llm.return_value = 0

                        with patch("webui.api.v2.model_manager.settings") as mock_settings:
                            mock_settings.DEFAULT_EMBEDDING_MODEL = "other/model"

                            with patch("webui.api.v2.model_manager._get_vecpipe_loaded_models") as mock_vec:
                                mock_vec.return_value = (False, [])

                                response = await superuser_client.get(
                                    "/api/v2/models/usage",
                                    params={"model_id": "test/model"},
                                )

        assert response.status_code == 200
        data = response.json()
        assert data["is_installed"] is False
        assert data["can_delete"] is True
        assert data["requires_confirmation"] is False

    @pytest.mark.asyncio()
    async def test_installed_model_with_collections(self, superuser_client) -> None:
        """Test response for installed model used by collections."""
        with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
            mock_installed.return_value = True

            with patch("webui.api.v2.model_manager.get_model_size_on_disk") as mock_size:
                mock_size.return_value = 1024

                with patch("webui.api.v2.model_manager._get_collections_using_model") as mock_cols:
                    mock_cols.return_value = ["Collection A", "Collection B"]

                    with patch("webui.api.v2.model_manager._count_user_preferences_using_model") as mock_prefs:
                        mock_prefs.return_value = 0

                        with patch("webui.api.v2.model_manager._count_llm_configs_using_model") as mock_llm:
                            mock_llm.return_value = 0

                            with patch("webui.api.v2.model_manager.settings") as mock_settings:
                                mock_settings.DEFAULT_EMBEDDING_MODEL = "other/model"

                                with patch("webui.api.v2.model_manager._get_vecpipe_loaded_models") as mock_vec:
                                    mock_vec.return_value = (False, [])

                                    response = await superuser_client.get(
                                        "/api/v2/models/usage",
                                        params={"model_id": "test/model"},
                                    )

        assert response.status_code == 200
        data = response.json()
        assert data["is_installed"] is True
        assert data["size_on_disk_mb"] == 1024
        assert len(data["blocked_by_collections"]) == 2
        assert data["can_delete"] is False

    @pytest.mark.asyncio()
    async def test_installed_model_with_warnings(self, superuser_client) -> None:
        """Test response for installed model with warning conditions."""
        with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
            mock_installed.return_value = True

            with patch("webui.api.v2.model_manager.get_model_size_on_disk") as mock_size:
                mock_size.return_value = 512

                with patch("webui.api.v2.model_manager._get_collections_using_model") as mock_cols:
                    mock_cols.return_value = []

                    with patch("webui.api.v2.model_manager._count_user_preferences_using_model") as mock_prefs:
                        mock_prefs.return_value = 3

                        with patch("webui.api.v2.model_manager._count_llm_configs_using_model") as mock_llm:
                            mock_llm.return_value = 2

                            with patch("webui.api.v2.model_manager.settings") as mock_settings:
                                mock_settings.DEFAULT_EMBEDDING_MODEL = "test/model"

                                with patch("webui.api.v2.model_manager._get_vecpipe_loaded_models") as mock_vec:
                                    mock_vec.return_value = (True, ["embedding"])

                                    response = await superuser_client.get(
                                        "/api/v2/models/usage",
                                        params={"model_id": "test/model"},
                                    )

        assert response.status_code == 200
        data = response.json()
        assert data["can_delete"] is True
        assert data["requires_confirmation"] is True
        assert data["user_preferences_count"] == 3
        assert data["llm_config_count"] == 2
        assert data["is_default_embedding_model"] is True
        assert data["loaded_in_vecpipe"] is True
        assert len(data["warnings"]) >= 4

    @pytest.mark.asyncio()
    async def test_vecpipe_query_failure_handled(self, superuser_client) -> None:
        """Test that VecPipe query failures don't break the endpoint."""
        with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
            mock_installed.return_value = True

            with patch("webui.api.v2.model_manager.get_model_size_on_disk") as mock_size:
                mock_size.return_value = 256

                with patch("webui.api.v2.model_manager._get_collections_using_model") as mock_cols:
                    mock_cols.return_value = []

                    with patch("webui.api.v2.model_manager._count_user_preferences_using_model") as mock_prefs:
                        mock_prefs.return_value = 0

                        with patch("webui.api.v2.model_manager._count_llm_configs_using_model") as mock_llm:
                            mock_llm.return_value = 0

                            with patch("webui.api.v2.model_manager.settings") as mock_settings:
                                mock_settings.DEFAULT_EMBEDDING_MODEL = "other/model"

                                with patch("webui.api.v2.model_manager._get_vecpipe_loaded_models") as mock_vec:
                                    # Simulate VecPipe failure - should return defaults
                                    mock_vec.return_value = (False, [])

                                    response = await superuser_client.get(
                                        "/api/v2/models/usage",
                                        params={"model_id": "test/model"},
                                    )

        assert response.status_code == 200
        data = response.json()
        assert data["loaded_in_vecpipe"] is False
        assert data["loaded_vecpipe_model_types"] == []


# =============================================================================
# DELETE /api/v2/models/cache Tests
# =============================================================================


class TestDeleteModelCacheEndpoint:
    """Tests for DELETE /api/v2/models/cache."""

    @pytest.mark.asyncio()
    async def test_requires_superuser(self, non_superuser_client) -> None:
        """Test 403 response for non-superuser."""
        response = await non_superuser_client.delete(
            "/api/v2/models/cache",
            params={"model_id": "test/model"},
        )
        assert response.status_code == 403

    @pytest.mark.asyncio()
    async def test_not_installed_returns_idempotent(self, superuser_client) -> None:
        """Test idempotent response when model is not installed."""
        with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
            mock_installed.return_value = False

            response = await superuser_client.delete(
                "/api/v2/models/cache",
                params={"model_id": "test/model"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_installed"
        assert data["task_id"] is None

    @pytest.mark.asyncio()
    async def test_blocked_by_collections(self, superuser_client) -> None:
        """Test 409 response when collections are using the model."""
        with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
            mock_installed.return_value = True

            with patch("webui.api.v2.model_manager._get_collections_using_model") as mock_cols:
                mock_cols.return_value = ["Collection A", "Collection B"]

                response = await superuser_client.delete(
                    "/api/v2/models/cache",
                    params={"model_id": "test/model"},
                )

        assert response.status_code == 409
        data = response.json()["detail"]
        assert data["conflict_type"] == "in_use_block"
        assert len(data["blocked_by_collections"]) == 2

    @pytest.mark.asyncio()
    async def test_requires_confirmation(self, superuser_client) -> None:
        """Test 409 response when warnings exist and confirm=false."""
        with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
            mock_installed.return_value = True

            with patch("webui.api.v2.model_manager._get_collections_using_model") as mock_cols:
                mock_cols.return_value = []

                with patch("webui.api.v2.model_manager._count_user_preferences_using_model") as mock_prefs:
                    mock_prefs.return_value = 3

                    with patch("webui.api.v2.model_manager._count_llm_configs_using_model") as mock_llm:
                        mock_llm.return_value = 0

                        with patch("webui.api.v2.model_manager.settings") as mock_settings:
                            mock_settings.DEFAULT_EMBEDDING_MODEL = "other/model"

                            with patch("webui.api.v2.model_manager._get_vecpipe_loaded_models") as mock_vec:
                                mock_vec.return_value = (False, [])

                                response = await superuser_client.delete(
                                    "/api/v2/models/cache",
                                    params={"model_id": "test/model"},
                                )

        assert response.status_code == 409
        data = response.json()["detail"]
        assert data["conflict_type"] == "requires_confirmation"
        assert data["requires_confirmation"] is True
        assert len(data["warnings"]) >= 1

    @pytest.mark.asyncio()
    async def test_delete_with_confirm(self, superuser_client) -> None:
        """Test successful delete when confirm=true bypasses warnings."""
        with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
            mock_installed.return_value = True

            with patch("webui.api.v2.model_manager._get_collections_using_model") as mock_cols:
                mock_cols.return_value = []

                with patch("webui.api.v2.model_manager._count_user_preferences_using_model") as mock_prefs:
                    mock_prefs.return_value = 3

                    with patch("webui.api.v2.model_manager._count_llm_configs_using_model") as mock_llm:
                        mock_llm.return_value = 0

                        with patch("webui.api.v2.model_manager.settings") as mock_settings:
                            mock_settings.DEFAULT_EMBEDDING_MODEL = "other/model"

                            with patch("webui.api.v2.model_manager._get_vecpipe_loaded_models") as mock_vec:
                                mock_vec.return_value = (False, [])

                                with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
                                    mock_redis_manager = AsyncMock()
                                    mock_redis_client = AsyncMock()
                                    mock_redis_manager.async_client.return_value = mock_redis_client
                                    mock_get_redis.return_value = mock_redis_manager

                                    with patch(
                                        "webui.api.v2.model_manager.task_state.claim_model_operation"
                                    ) as mock_claim:
                                        mock_claim.return_value = (True, None)

                                        with patch(
                                            "webui.api.v2.model_manager.task_state.init_task_progress"
                                        ) as mock_init:
                                            mock_init.return_value = None

                                            with patch("webui.api.v2.model_manager.celery_app.send_task") as mock_send:
                                                mock_send.return_value = MagicMock()

                                                response = await superuser_client.delete(
                                                    "/api/v2/models/cache",
                                                    params={"model_id": "test/model", "confirm": "true"},
                                                )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["task_id"] is not None
        assert data["operation"] == "delete"
        assert len(data["warnings"]) >= 1

    @pytest.mark.asyncio()
    async def test_delete_no_warnings(self, superuser_client) -> None:
        """Test successful delete without warnings doesn't require confirm."""
        with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
            mock_installed.return_value = True

            with patch("webui.api.v2.model_manager._get_collections_using_model") as mock_cols:
                mock_cols.return_value = []

                with patch("webui.api.v2.model_manager._count_user_preferences_using_model") as mock_prefs:
                    mock_prefs.return_value = 0

                    with patch("webui.api.v2.model_manager._count_llm_configs_using_model") as mock_llm:
                        mock_llm.return_value = 0

                        with patch("webui.api.v2.model_manager.settings") as mock_settings:
                            mock_settings.DEFAULT_EMBEDDING_MODEL = "other/model"

                            with patch("webui.api.v2.model_manager._get_vecpipe_loaded_models") as mock_vec:
                                mock_vec.return_value = (False, [])

                                with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
                                    mock_redis_manager = AsyncMock()
                                    mock_redis_client = AsyncMock()
                                    mock_redis_manager.async_client.return_value = mock_redis_client
                                    mock_get_redis.return_value = mock_redis_manager

                                    with patch(
                                        "webui.api.v2.model_manager.task_state.claim_model_operation"
                                    ) as mock_claim:
                                        mock_claim.return_value = (True, None)

                                        with patch(
                                            "webui.api.v2.model_manager.task_state.init_task_progress"
                                        ) as mock_init:
                                            mock_init.return_value = None

                                            with patch("webui.api.v2.model_manager.celery_app.send_task") as mock_send:
                                                mock_send.return_value = MagicMock()

                                                response = await superuser_client.delete(
                                                    "/api/v2/models/cache",
                                                    params={"model_id": "test/model"},
                                                )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"

    @pytest.mark.asyncio()
    async def test_delete_dedupe(self, superuser_client) -> None:
        """Test de-duplication when same delete is already running."""
        with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
            mock_installed.return_value = True

            with patch("webui.api.v2.model_manager._get_collections_using_model") as mock_cols:
                mock_cols.return_value = []

                with patch("webui.api.v2.model_manager._count_user_preferences_using_model") as mock_prefs:
                    mock_prefs.return_value = 0

                    with patch("webui.api.v2.model_manager._count_llm_configs_using_model") as mock_llm:
                        mock_llm.return_value = 0

                        with patch("webui.api.v2.model_manager.settings") as mock_settings:
                            mock_settings.DEFAULT_EMBEDDING_MODEL = "other/model"

                            with patch("webui.api.v2.model_manager._get_vecpipe_loaded_models") as mock_vec:
                                mock_vec.return_value = (False, [])

                                with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
                                    mock_redis_manager = AsyncMock()
                                    mock_redis_client = AsyncMock()
                                    mock_redis_manager.async_client.return_value = mock_redis_client
                                    mock_get_redis.return_value = mock_redis_manager

                                    with patch(
                                        "webui.api.v2.model_manager.task_state.claim_model_operation"
                                    ) as mock_claim:
                                        mock_claim.return_value = (False, "existing-delete-task")

                                        response = await superuser_client.delete(
                                            "/api/v2/models/cache",
                                            params={"model_id": "test/model"},
                                        )

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "existing-delete-task"
        assert data["status"] == "running"

    @pytest.mark.asyncio()
    async def test_cross_op_conflict(self, superuser_client) -> None:
        """Test 409 response when download is active for same model."""
        with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
            mock_installed.return_value = True

            with patch("webui.api.v2.model_manager._get_collections_using_model") as mock_cols:
                mock_cols.return_value = []

                with patch("webui.api.v2.model_manager._count_user_preferences_using_model") as mock_prefs:
                    mock_prefs.return_value = 0

                    with patch("webui.api.v2.model_manager._count_llm_configs_using_model") as mock_llm:
                        mock_llm.return_value = 0

                        with patch("webui.api.v2.model_manager.settings") as mock_settings:
                            mock_settings.DEFAULT_EMBEDDING_MODEL = "other/model"

                            with patch("webui.api.v2.model_manager._get_vecpipe_loaded_models") as mock_vec:
                                mock_vec.return_value = (False, [])

                                with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
                                    mock_redis_manager = AsyncMock()
                                    mock_redis_client = AsyncMock()
                                    mock_redis_manager.async_client.return_value = mock_redis_client
                                    mock_get_redis.return_value = mock_redis_manager

                                    with patch(
                                        "webui.api.v2.model_manager.task_state.claim_model_operation"
                                    ) as mock_claim:
                                        mock_claim.side_effect = CrossOpConflictError(
                                            model_id="test/model",
                                            active_operation="download",
                                            active_task_id="download-task-789",
                                        )

                                        response = await superuser_client.delete(
                                            "/api/v2/models/cache",
                                            params={"model_id": "test/model"},
                                        )

        assert response.status_code == 409
        data = response.json()["detail"]
        assert data["conflict_type"] == "cross_op_exclusion"
        assert data["active_operation"] == "download"
