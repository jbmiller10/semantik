"""Integration tests for model manager write API endpoints.

These tests verify the end-to-end flow of model download and delete operations,
including task creation, progress polling, and cross-operation exclusion.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from shared.database import get_db
from webui.auth import get_current_user
from webui.main import app
from webui.model_manager.task_state import CrossOpConflictError

# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def superuser_client(db_session):
    """Provide AsyncClient with superuser authentication."""

    async def override_get_db() -> AsyncGenerator[Any, None]:
        yield db_session

    async def override_get_current_user() -> dict[str, Any]:
        return {
            "id": 1,
            "username": "admin",
            "email": "admin@test.com",
            "full_name": "Admin User",
            "is_superuser": True,
        }

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    app.dependency_overrides.clear()


def _mock_redis_manager():
    """Create a mock Redis manager with async client."""
    mock_redis_client = AsyncMock()
    mock_redis_client.get = AsyncMock(return_value=None)
    mock_redis_client.set = AsyncMock(return_value=True)
    mock_redis_client.delete = AsyncMock(return_value=1)

    mock_manager = AsyncMock()
    mock_manager.async_client.return_value = mock_redis_client
    mock_manager.get_client = MagicMock(return_value=mock_redis_client)

    return mock_manager, mock_redis_client


# =============================================================================
# Test Classes
# =============================================================================


class TestDownloadProgressFlow:
    """Test download endpoint returns task_id for progress polling."""

    @pytest.mark.asyncio()
    async def test_download_returns_task_id_for_polling(self, superuser_client):
        """Verify download returns task_id that can be used for polling."""
        with patch("webui.api.v2.model_manager.get_curated_model_ids") as mock_curated:
            mock_curated.return_value = {"BAAI/bge-small-en-v1.5"}

            with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
                mock_installed.return_value = False

                with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
                    mock_redis_manager, _ = _mock_redis_manager()
                    mock_get_redis.return_value = mock_redis_manager

                    with patch(
                        "webui.api.v2.model_manager.task_state.claim_model_operation",
                        new_callable=AsyncMock,
                    ) as mock_claim:
                        mock_claim.return_value = (True, None)

                        with patch(
                            "webui.api.v2.model_manager.task_state.init_task_progress",
                            new_callable=AsyncMock,
                        ) as mock_init:
                            mock_init.return_value = None

                            with patch("webui.api.v2.model_manager.celery_app.send_task") as mock_send:
                                mock_send.return_value = MagicMock()

                                response = await superuser_client.post(
                                    "/api/v2/models/download",
                                    json={"model_id": "BAAI/bge-small-en-v1.5"},
                                )

                                assert response.status_code == 200
                                data = response.json()
                                assert data["task_id"] is not None
                                assert data["operation"] == "download"
                                assert data["status"] == "pending"

    @pytest.mark.asyncio()
    async def test_concurrent_downloads_deduplicated(self, superuser_client):
        """Verify concurrent download requests return same task_id."""
        existing_task_id = "existing-task-123"

        with patch("webui.api.v2.model_manager.get_curated_model_ids") as mock_curated:
            mock_curated.return_value = {"BAAI/bge-small-en-v1.5"}

            with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
                mock_installed.return_value = False

                with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
                    mock_redis_manager, _ = _mock_redis_manager()
                    mock_get_redis.return_value = mock_redis_manager

                    with patch(
                        "webui.api.v2.model_manager.task_state.claim_model_operation",
                        new_callable=AsyncMock,
                    ) as mock_claim:
                        # Not claimed - another task already exists
                        mock_claim.return_value = (False, existing_task_id)

                        response = await superuser_client.post(
                            "/api/v2/models/download",
                            json={"model_id": "BAAI/bge-small-en-v1.5"},
                        )

                        assert response.status_code == 200
                        data = response.json()
                        assert data["task_id"] == existing_task_id
                        assert data["status"] == "running"


class TestUsagePreflightDeleteFlow:
    """Test usage preflight check followed by delete with confirmation."""

    @pytest.mark.asyncio()
    async def test_usage_then_delete_flow(self, superuser_client):
        """Verify usage preflight provides info needed for confirmed delete."""
        # Step 1: Check usage
        with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
            mock_installed.return_value = True

            with patch("webui.api.v2.model_manager.get_model_size_on_disk") as mock_size:
                mock_size.return_value = 500

                with patch("webui.api.v2.model_manager._get_collections_using_model") as mock_colls:
                    mock_colls.return_value = []

                    with patch(
                        "webui.api.v2.model_manager._count_user_preferences_using_model",
                        new_callable=AsyncMock,
                    ) as mock_prefs:
                        mock_prefs.return_value = 1

                        with patch(
                            "webui.api.v2.model_manager._count_llm_configs_using_model",
                            new_callable=AsyncMock,
                        ) as mock_llm:
                            mock_llm.return_value = 0

                            with patch(
                                "webui.api.v2.model_manager._get_vecpipe_loaded_models",
                                new_callable=AsyncMock,
                            ) as mock_vp:
                                mock_vp.return_value = (False, [])

                                with patch("webui.api.v2.model_manager.settings") as mock_settings:
                                    mock_settings.DEFAULT_EMBEDDING_MODEL = "other-model"

                                    response = await superuser_client.get(
                                        "/api/v2/models/usage",
                                        params={"model_id": "BAAI/bge-small-en-v1.5"},
                                    )

                                    assert response.status_code == 200
                                    usage_data = response.json()
                                    assert usage_data["can_delete"] is True
                                    assert usage_data["requires_confirmation"] is True
                                    assert usage_data["user_preferences_count"] == 1

        # Step 2: Delete with confirmation
        with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
            mock_installed.return_value = True

            with patch("webui.api.v2.model_manager._get_collections_using_model") as mock_colls:
                mock_colls.return_value = []

                with patch(
                    "webui.api.v2.model_manager._count_user_preferences_using_model",
                    new_callable=AsyncMock,
                ) as mock_prefs:
                    mock_prefs.return_value = 1

                    with patch(
                        "webui.api.v2.model_manager._count_llm_configs_using_model",
                        new_callable=AsyncMock,
                    ) as mock_llm:
                        mock_llm.return_value = 0

                        with patch(
                            "webui.api.v2.model_manager._get_vecpipe_loaded_models",
                            new_callable=AsyncMock,
                        ) as mock_vp:
                            mock_vp.return_value = (False, [])

                            with patch("webui.api.v2.model_manager.settings") as mock_settings:
                                mock_settings.DEFAULT_EMBEDDING_MODEL = "other-model"

                                with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
                                    mock_redis_manager, _ = _mock_redis_manager()
                                    mock_get_redis.return_value = mock_redis_manager

                                    with patch(
                                        "webui.api.v2.model_manager.task_state.claim_model_operation",
                                        new_callable=AsyncMock,
                                    ) as mock_claim:
                                        mock_claim.return_value = (True, None)

                                        with patch(
                                            "webui.api.v2.model_manager.task_state.init_task_progress",
                                            new_callable=AsyncMock,
                                        ) as mock_init:
                                            mock_init.return_value = None

                                            with patch("webui.api.v2.model_manager.celery_app.send_task") as mock_send:
                                                mock_send.return_value = MagicMock()

                                                response = await superuser_client.delete(
                                                    "/api/v2/models/cache",
                                                    params={
                                                        "model_id": "BAAI/bge-small-en-v1.5",
                                                        "confirm": "true",
                                                    },
                                                )

                                                assert response.status_code == 200
                                                delete_data = response.json()
                                                assert delete_data["task_id"] is not None
                                                assert delete_data["operation"] == "delete"


class TestCrossOpExclusion:
    """Test cross-operation exclusion between download and delete."""

    @pytest.mark.asyncio()
    async def test_download_blocks_delete(self, superuser_client):
        """Verify active download blocks delete attempt."""
        with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
            mock_installed.return_value = True

            with patch("webui.api.v2.model_manager._get_collections_using_model") as mock_colls:
                mock_colls.return_value = []

                with patch(
                    "webui.api.v2.model_manager._count_user_preferences_using_model",
                    new_callable=AsyncMock,
                ) as mock_prefs:
                    mock_prefs.return_value = 0

                    with patch(
                        "webui.api.v2.model_manager._count_llm_configs_using_model",
                        new_callable=AsyncMock,
                    ) as mock_llm:
                        mock_llm.return_value = 0

                        with patch(
                            "webui.api.v2.model_manager._get_vecpipe_loaded_models",
                            new_callable=AsyncMock,
                        ) as mock_vp:
                            mock_vp.return_value = (False, [])

                            with patch("webui.api.v2.model_manager.settings") as mock_settings:
                                mock_settings.DEFAULT_EMBEDDING_MODEL = "other-model"

                                with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
                                    mock_redis_manager, _ = _mock_redis_manager()
                                    mock_get_redis.return_value = mock_redis_manager

                                    with patch(
                                        "webui.api.v2.model_manager.task_state.claim_model_operation",
                                        new_callable=AsyncMock,
                                    ) as mock_claim:
                                        # Simulate conflict - download is active
                                        mock_claim.side_effect = CrossOpConflictError(
                                            "BAAI/bge-small-en-v1.5", "download", "download-task-123"
                                        )

                                        response = await superuser_client.delete(
                                            "/api/v2/models/cache",
                                            params={"model_id": "BAAI/bge-small-en-v1.5"},
                                        )

                                        assert response.status_code == 409
                                        data = response.json()
                                        # Response is nested under "detail"
                                        assert data["detail"]["conflict_type"] == "cross_op_exclusion"
                                        assert data["detail"]["active_operation"] == "download"

    @pytest.mark.asyncio()
    async def test_delete_blocks_download(self, superuser_client):
        """Verify active delete blocks download attempt."""
        with patch("webui.api.v2.model_manager.get_curated_model_ids") as mock_curated:
            mock_curated.return_value = {"BAAI/bge-small-en-v1.5"}

            with patch("webui.api.v2.model_manager.is_model_installed") as mock_installed:
                mock_installed.return_value = False

                with patch("webui.api.v2.model_manager.get_redis_manager") as mock_get_redis:
                    mock_redis_manager, _ = _mock_redis_manager()
                    mock_get_redis.return_value = mock_redis_manager

                    with patch(
                        "webui.api.v2.model_manager.task_state.claim_model_operation",
                        new_callable=AsyncMock,
                    ) as mock_claim:
                        # Simulate conflict - delete is active
                        mock_claim.side_effect = CrossOpConflictError(
                            "BAAI/bge-small-en-v1.5", "delete", "delete-task-456"
                        )

                        response = await superuser_client.post(
                            "/api/v2/models/download",
                            json={"model_id": "BAAI/bge-small-en-v1.5"},
                        )

                        assert response.status_code == 409
                        data = response.json()
                        # Response is nested under "detail"
                        assert data["detail"]["conflict_type"] == "cross_op_exclusion"
                        assert data["detail"]["active_operation"] == "delete"
