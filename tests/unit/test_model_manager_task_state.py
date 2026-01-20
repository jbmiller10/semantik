"""Unit tests for model manager task state Redis operations."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from webui.model_manager import task_state
from webui.model_manager.task_state import CrossOpConflictError


class TestClaimModelOperation:
    """Tests for claim_model_operation async function."""

    @pytest.mark.asyncio()
    async def test_claim_succeeds_when_no_active_operation(self):
        """Test successful claim when no operation is active."""
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True  # SET NX succeeds

        claimed, existing_task_id = await task_state.claim_model_operation(
            mock_redis, "test/model", "download", "task-123"
        )

        assert claimed is True
        assert existing_task_id is None
        mock_redis.set.assert_called_once_with(
            "model-manager:active:test/model",
            "download:task-123",
            nx=True,
            ex=task_state.ACTIVE_KEY_TTL,
        )

    @pytest.mark.asyncio()
    async def test_claim_returns_existing_task_for_same_operation(self):
        """Test de-duplication when same operation already active."""
        mock_redis = AsyncMock()
        mock_redis.set.return_value = False  # SET NX fails
        mock_redis.get.return_value = "download:existing-task"

        claimed, existing_task_id = await task_state.claim_model_operation(
            mock_redis, "test/model", "download", "task-123"
        )

        assert claimed is False
        assert existing_task_id == "existing-task"

    @pytest.mark.asyncio()
    async def test_claim_raises_conflict_for_different_operation(self):
        """Test CrossOpConflictError when different operation is active."""
        mock_redis = AsyncMock()
        mock_redis.set.return_value = False  # SET NX fails
        mock_redis.get.return_value = "download:existing-task"

        with pytest.raises(CrossOpConflictError) as exc_info:
            await task_state.claim_model_operation(
                mock_redis, "test/model", "delete", "task-123"
            )

        assert exc_info.value.model_id == "test/model"
        assert exc_info.value.active_operation == "download"
        assert exc_info.value.active_task_id == "existing-task"

    @pytest.mark.asyncio()
    async def test_claim_handles_race_condition(self):
        """Test retry when key deleted between SET and GET."""
        mock_redis = AsyncMock()
        # First SET fails, GET returns None (key deleted), second SET succeeds
        mock_redis.set.side_effect = [False, True]
        mock_redis.get.return_value = None

        claimed, existing_task_id = await task_state.claim_model_operation(
            mock_redis, "test/model", "download", "task-123"
        )

        assert claimed is True
        assert existing_task_id is None
        assert mock_redis.set.call_count == 2


class TestReleaseModelOperation:
    """Tests for release_model_operation async function."""

    @pytest.mark.asyncio()
    async def test_release_deletes_key(self):
        """Test that release deletes the active key."""
        mock_redis = AsyncMock()

        await task_state.release_model_operation(mock_redis, "test/model")

        mock_redis.delete.assert_called_once_with("model-manager:active:test/model")


class TestGetActiveOperation:
    """Tests for get_active_operation async function."""

    @pytest.mark.asyncio()
    async def test_returns_operation_and_task_id(self):
        """Test parsing of active operation value."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "download:task-abc"

        result = await task_state.get_active_operation(mock_redis, "test/model")

        assert result == ("download", "task-abc")

    @pytest.mark.asyncio()
    async def test_returns_none_when_no_operation(self):
        """Test None return when no active operation."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None

        result = await task_state.get_active_operation(mock_redis, "test/model")

        assert result is None

    @pytest.mark.asyncio()
    async def test_returns_none_for_invalid_format(self):
        """Test None return for malformed value."""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "invalid-format"

        result = await task_state.get_active_operation(mock_redis, "test/model")

        assert result is None


class TestInitTaskProgress:
    """Tests for init_task_progress async function."""

    @pytest.mark.asyncio()
    async def test_creates_progress_hash(self):
        """Test that progress hash is created with correct fields."""
        mock_redis = AsyncMock()

        await task_state.init_task_progress(
            mock_redis, "task-123", "test/model", "download"
        )

        # Verify hset was called
        mock_redis.hset.assert_called_once()
        call_kwargs = mock_redis.hset.call_args
        mapping = call_kwargs.kwargs["mapping"]

        assert mapping["task_id"] == "task-123"
        assert mapping["model_id"] == "test/model"
        assert mapping["operation"] == "download"
        assert mapping["status"] == "pending"
        assert mapping["bytes_downloaded"] == "0"
        assert mapping["bytes_total"] == "0"
        assert "updated_at" in mapping

        # Verify TTL was set
        mock_redis.expire.assert_called_once_with(
            "model-manager:task:task-123",
            task_state.PROGRESS_KEY_TTL,
        )


class TestUpdateTaskProgress:
    """Tests for update_task_progress async function."""

    @pytest.mark.asyncio()
    async def test_updates_progress_fields(self):
        """Test progress update with bytes and status."""
        mock_redis = AsyncMock()
        mock_redis.hgetall.return_value = {"model_id": "test/model"}

        await task_state.update_task_progress(
            mock_redis,
            "task-123",
            status="running",
            bytes_downloaded=1024,
            bytes_total=2048,
        )

        call_kwargs = mock_redis.hset.call_args
        mapping = call_kwargs.kwargs["mapping"]

        assert mapping["status"] == "running"
        assert mapping["bytes_downloaded"] == "1024"
        assert mapping["bytes_total"] == "2048"

    @pytest.mark.asyncio()
    async def test_updates_error_field_when_provided(self):
        """Test error field is updated when provided."""
        mock_redis = AsyncMock()
        mock_redis.hgetall.return_value = {}

        await task_state.update_task_progress(
            mock_redis,
            "task-123",
            status="failed",
            error="Something went wrong",
        )

        call_kwargs = mock_redis.hset.call_args
        mapping = call_kwargs.kwargs["mapping"]

        assert mapping["error"] == "Something went wrong"

    @pytest.mark.asyncio()
    async def test_refreshes_active_key_ttl_when_running(self):
        """Test that active key TTL is refreshed when status is running."""
        mock_redis = AsyncMock()
        mock_redis.hgetall.return_value = {"model_id": "test/model"}

        await task_state.update_task_progress(
            mock_redis,
            "task-123",
            status="running",
        )

        # Should have two expire calls: progress key and active key
        assert mock_redis.expire.call_count == 2


class TestGetTaskProgress:
    """Tests for get_task_progress async function."""

    @pytest.mark.asyncio()
    async def test_returns_progress_dict(self):
        """Test parsing of progress hash to dict."""
        mock_redis = AsyncMock()
        mock_redis.hgetall.return_value = {
            "task_id": "task-123",
            "model_id": "test/model",
            "operation": "download",
            "status": "running",
            "bytes_downloaded": "1024",
            "bytes_total": "2048",
            "error": "",
            "updated_at": "1704067200.0",
        }

        result = await task_state.get_task_progress(mock_redis, "task-123")

        assert result is not None
        assert result["task_id"] == "task-123"
        assert result["model_id"] == "test/model"
        assert result["operation"] == "download"
        assert result["status"] == "running"
        assert result["bytes_downloaded"] == 1024
        assert result["bytes_total"] == 2048
        assert result["error"] is None  # Empty string converted to None
        assert result["updated_at"] == 1704067200.0

    @pytest.mark.asyncio()
    async def test_returns_none_when_not_found(self):
        """Test None return when task doesn't exist."""
        mock_redis = AsyncMock()
        mock_redis.hgetall.return_value = {}

        result = await task_state.get_task_progress(mock_redis, "nonexistent")

        assert result is None


class TestSyncFunctions:
    """Tests for synchronous versions of task state functions."""

    def test_claim_model_operation_sync_succeeds(self):
        """Test sync claim when no operation is active."""
        mock_redis = MagicMock()
        mock_redis.set.return_value = True

        claimed, existing_task_id = task_state.claim_model_operation_sync(
            mock_redis, "test/model", "download", "task-123"
        )

        assert claimed is True
        assert existing_task_id is None

    def test_claim_model_operation_sync_returns_existing(self):
        """Test sync de-duplication for same operation."""
        mock_redis = MagicMock()
        mock_redis.set.return_value = False
        mock_redis.get.return_value = "download:existing-task"

        claimed, existing_task_id = task_state.claim_model_operation_sync(
            mock_redis, "test/model", "download", "task-123"
        )

        assert claimed is False
        assert existing_task_id == "existing-task"

    def test_claim_model_operation_sync_raises_conflict(self):
        """Test sync CrossOpConflictError for different operation."""
        mock_redis = MagicMock()
        mock_redis.set.return_value = False
        mock_redis.get.return_value = "download:existing-task"

        with pytest.raises(CrossOpConflictError):
            task_state.claim_model_operation_sync(
                mock_redis, "test/model", "delete", "task-123"
            )

    def test_release_model_operation_sync(self):
        """Test sync release deletes key."""
        mock_redis = MagicMock()

        task_state.release_model_operation_sync(mock_redis, "test/model")

        mock_redis.delete.assert_called_once_with("model-manager:active:test/model")

    def test_update_task_progress_sync(self):
        """Test sync progress update."""
        mock_redis = MagicMock()
        mock_redis.hgetall.return_value = {}

        task_state.update_task_progress_sync(
            mock_redis,
            "task-123",
            status="completed",
            bytes_downloaded=2048,
            bytes_total=2048,
        )

        mock_redis.hset.assert_called_once()

    def test_init_task_progress_sync(self):
        """Test sync progress initialization."""
        mock_redis = MagicMock()

        task_state.init_task_progress_sync(
            mock_redis, "task-123", "test/model", "download"
        )

        mock_redis.hset.assert_called_once()
        mock_redis.expire.assert_called_once()


class TestCrossOpConflictError:
    """Tests for CrossOpConflictError exception."""

    def test_error_attributes(self):
        """Test exception stores correct attributes."""
        error = CrossOpConflictError("test/model", "download", "task-abc")

        assert error.model_id == "test/model"
        assert error.active_operation == "download"
        assert error.active_task_id == "task-abc"
        assert "download" in str(error)
        assert "test/model" in str(error)
        assert "task-abc" in str(error)
