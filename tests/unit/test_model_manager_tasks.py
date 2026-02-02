"""Unit tests for model manager Celery tasks."""

from unittest.mock import MagicMock, patch

import pytest

from webui.model_manager.task_state import CrossOpConflictError


class TestDownloadModelTask:
    """Tests for download_model Celery task."""

    @patch("webui.tasks.model_manager._get_sync_redis_client")
    @patch("webui.tasks.model_manager.task_state")
    def test_download_succeeds(self, mock_task_state, mock_get_redis):
        """Test successful download flow."""
        from webui.tasks.model_manager import download_model

        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        # Active lock is already owned by this task (normal path via API).
        mock_task_state.get_active_operation_sync.return_value = ("download", "task-123")
        mock_task_state.task_progress_exists_sync.return_value = True

        # Mock huggingface_hub.snapshot_download
        with patch("huggingface_hub.snapshot_download") as mock_download:
            mock_download.return_value = "/path/to/model"

            # Call the task function directly via .run() method
            result = download_model.run("test/model", "task-123")

        assert result["status"] == "completed"
        assert result["task_id"] == "task-123"
        assert result["model_id"] == "test/model"
        assert result["local_dir"] == "/path/to/model"

        # Verify update calls (running + completed)
        assert mock_task_state.update_task_progress_sync.call_count >= 2  # running + completed

        # Verify release was called
        mock_task_state.release_model_operation_if_owner_sync.assert_called_once_with(
            mock_redis, "test/model", "download", "task-123"
        )

    @patch("webui.tasks.model_manager._get_sync_redis_client")
    @patch("webui.tasks.model_manager.task_state")
    def test_download_deduplicated(self, mock_task_state, mock_get_redis):
        """Test de-duplication when download already in progress."""
        from webui.tasks.model_manager import download_model

        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        # Active lock exists for a different task_id.
        mock_task_state.get_active_operation_sync.return_value = ("download", "existing-task")

        result = download_model.run("test/model", "task-123")

        assert result["status"] == "deduplicated"
        assert result["existing_task_id"] == "existing-task"

        # Should not have called init or update
        mock_task_state.init_task_progress_sync.assert_not_called()

    @patch("webui.tasks.model_manager._get_sync_redis_client")
    @patch("webui.tasks.model_manager.task_state")
    def test_download_cross_op_conflict(self, mock_task_state, mock_get_redis):
        """Test conflict when delete is in progress."""
        from webui.tasks.model_manager import download_model

        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        # Active lock exists for the other operation.
        mock_task_state.get_active_operation_sync.return_value = ("delete", "delete-task")
        mock_task_state.CrossOpConflictError = CrossOpConflictError

        result = download_model.run("test/model", "task-123")

        assert result["status"] == "conflict"
        assert "delete" in result["error"]

    @patch("webui.tasks.model_manager._get_sync_redis_client")
    @patch("webui.tasks.model_manager.task_state")
    def test_download_handles_failure(self, mock_task_state, mock_get_redis):
        """Test error handling on download failure."""
        from webui.tasks.model_manager import download_model

        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        mock_task_state.get_active_operation_sync.return_value = ("download", "task-123")
        mock_task_state.task_progress_exists_sync.return_value = True
        mock_task_state.CrossOpConflictError = CrossOpConflictError

        with patch("huggingface_hub.snapshot_download") as mock_download:
            mock_download.side_effect = OSError(13, "Permission denied")

            result = download_model.run("test/model", "task-123")

        assert result["status"] == "failed"
        assert "Permission denied" in result["error"]

        # Verify release was still called
        mock_task_state.release_model_operation_if_owner_sync.assert_called_once()

    @patch("webui.tasks.model_manager._get_sync_redis_client")
    @patch("webui.tasks.model_manager.task_state")
    def test_retry_does_not_release_operation_lock(self, mock_task_state, mock_get_redis):
        """Retryable errors should preserve the operation lock for the retry."""
        from webui.tasks.model_manager import download_model

        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        mock_task_state.get_active_operation_sync.return_value = ("download", "task-123")
        mock_task_state.task_progress_exists_sync.return_value = True
        mock_task_state.CrossOpConflictError = CrossOpConflictError

        with patch("huggingface_hub.snapshot_download", side_effect=ConnectionError("Temporary network error")):
            with patch.object(download_model, "retry", side_effect=RuntimeError("retry")):
                with pytest.raises(RuntimeError, match="retry"):
                    download_model.run("test/model", "task-123")

        # The operation lock must not be released when retrying.
        mock_task_state.release_model_operation_if_owner_sync.assert_not_called()


class TestDeleteModelTask:
    """Tests for delete_model Celery task."""

    @patch("webui.tasks.model_manager._get_sync_redis_client")
    @patch("webui.tasks.model_manager.task_state")
    def test_delete_succeeds(self, mock_task_state, mock_get_redis):
        """Test successful delete flow."""
        from webui.tasks.model_manager import delete_model

        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        mock_task_state.get_active_operation_sync.return_value = ("delete", "task-123")
        mock_task_state.task_progress_exists_sync.return_value = True

        # Mock scan_cache_dir
        with patch("huggingface_hub.scan_cache_dir") as mock_scan:
            mock_repo = MagicMock()
            mock_repo.repo_id = "test/model"
            mock_repo.repo_type = "model"
            mock_revision = MagicMock()
            mock_revision.commit_hash = "abc123"
            mock_repo.revisions = [mock_revision]

            mock_cache_info = MagicMock()
            mock_cache_info.repos = [mock_repo]

            mock_delete_strategy = MagicMock()
            mock_delete_strategy.expected_freed_size = 1024 * 1024 * 100
            mock_cache_info.delete_revisions.return_value = mock_delete_strategy

            mock_scan.return_value = mock_cache_info

            result = delete_model.run("test/model", "task-123")

        assert result["status"] == "completed"
        assert result["task_id"] == "task-123"
        assert result["model_id"] == "test/model"
        assert result["revisions_deleted"] == 1
        assert result["freed_bytes"] == 1024 * 1024 * 100

        mock_task_state.release_model_operation_if_owner_sync.assert_called_once()

    @patch("webui.tasks.model_manager._get_sync_redis_client")
    @patch("webui.tasks.model_manager.task_state")
    def test_delete_model_not_found(self, mock_task_state, mock_get_redis):
        """Test delete when model is not in cache - returns not_installed (no-op)."""
        from webui.tasks.model_manager import delete_model

        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        mock_task_state.get_active_operation_sync.return_value = ("delete", "task-123")
        mock_task_state.task_progress_exists_sync.return_value = True

        with patch("huggingface_hub.scan_cache_dir") as mock_scan:
            mock_cache_info = MagicMock()
            mock_cache_info.repos = []  # No repos
            mock_scan.return_value = mock_cache_info

            result = delete_model.run("test/model", "task-123")

        # Model not found is treated as successful no-op (not an error)
        assert result["status"] == "not_installed"

    @patch("webui.tasks.model_manager._get_sync_redis_client")
    @patch("webui.tasks.model_manager.task_state")
    def test_delete_deduplicated(self, mock_task_state, mock_get_redis):
        """Test de-duplication when delete already in progress."""
        from webui.tasks.model_manager import delete_model

        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        mock_task_state.get_active_operation_sync.return_value = ("delete", "existing-task")

        result = delete_model.run("test/model", "task-123")

        assert result["status"] == "deduplicated"
        assert result["existing_task_id"] == "existing-task"

    @patch("webui.tasks.model_manager._get_sync_redis_client")
    @patch("webui.tasks.model_manager.task_state")
    def test_delete_cross_op_conflict(self, mock_task_state, mock_get_redis):
        """Test conflict when download is in progress."""
        from webui.tasks.model_manager import delete_model

        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        mock_task_state.get_active_operation_sync.return_value = ("download", "download-task")
        mock_task_state.CrossOpConflictError = CrossOpConflictError

        result = delete_model.run("test/model", "task-123")

        assert result["status"] == "conflict"
        assert "download" in result["error"]


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_is_retryable_error_connection_error(self):
        """Test that ConnectionError is retryable."""
        from webui.tasks.model_manager import _is_retryable_error

        assert _is_retryable_error(ConnectionError("Network error")) is True

    def test_is_retryable_error_timeout(self):
        """Test that TimeoutError is retryable."""
        from webui.tasks.model_manager import _is_retryable_error

        assert _is_retryable_error(TimeoutError("Timeout")) is True

    def test_is_retryable_error_other(self):
        """Test that other errors are not retryable."""
        from webui.tasks.model_manager import _is_retryable_error

        assert _is_retryable_error(ValueError("Bad value")) is False

    def test_is_retryable_error_http_5xx(self):
        """HTTP 5xx errors should be retryable."""
        import httpx

        from webui.tasks.model_manager import _is_retryable_error

        request = httpx.Request("GET", "https://example.invalid")
        response = httpx.Response(503, request=request)
        err = httpx.HTTPStatusError("Server error", request=request, response=response)
        assert _is_retryable_error(err) is True

    def test_is_retryable_error_http_429(self):
        """HTTP 429 errors should be retryable."""
        import httpx

        from webui.tasks.model_manager import _is_retryable_error

        request = httpx.Request("GET", "https://example.invalid")
        response = httpx.Response(429, request=request)
        err = httpx.HTTPStatusError("Rate limited", request=request, response=response)
        assert _is_retryable_error(err) is True

    def test_is_fatal_error_permission_denied(self):
        """Test that permission denied is fatal."""
        from webui.tasks.model_manager import _is_fatal_error

        error = OSError(13, "Permission denied")
        assert _is_fatal_error(error) is True

    def test_is_fatal_error_disk_full(self):
        """Test that disk full is fatal."""
        from webui.tasks.model_manager import _is_fatal_error

        error = OSError(28, "No space left on device")
        assert _is_fatal_error(error) is True

    def test_is_fatal_error_other(self):
        """Test that other errors are not fatal."""
        from webui.tasks.model_manager import _is_fatal_error

        assert _is_fatal_error(ValueError("Bad value")) is False

    def test_is_fatal_error_http_401_403_404(self):
        """HTTP 401/403/404 errors should be fatal."""
        import httpx

        from webui.tasks.model_manager import _is_fatal_error

        for code in (401, 403, 404):
            request = httpx.Request("GET", "https://example.invalid")
            response = httpx.Response(code, request=request)
            err = httpx.HTTPStatusError("HTTP error", request=request, response=response)
            assert _is_fatal_error(err) is True


class TestDownloadProgressAggregator:
    """Unit tests for download progress aggregation."""

    @patch("webui.tasks.model_manager.task_state.update_task_progress_sync")
    def test_updates_progress_with_aggregated_bytes(self, mock_update_progress):
        from webui.tasks.model_manager import _DownloadProgressAggregator

        redis_client = MagicMock()
        aggregator = _DownloadProgressAggregator(redis_client, "task-123", min_update_interval_seconds=0)

        aggregator.set_bar(1, downloaded=10, total=100)
        aggregator.set_bar(2, downloaded=5, total=50)

        assert aggregator.latest() == (15, 150)
        assert mock_update_progress.call_count >= 2
        mock_update_progress.assert_called_with(
            redis_client,
            "task-123",
            status="running",
            bytes_downloaded=15,
            bytes_total=150,
        )

    @patch("webui.tasks.model_manager.task_state.update_task_progress_sync")
    def test_redis_failure_does_not_crash_progress_updates(self, mock_update_progress):
        from webui.tasks.model_manager import _DownloadProgressAggregator

        mock_update_progress.side_effect = ConnectionError("Redis is down")
        redis_client = MagicMock()
        aggregator = _DownloadProgressAggregator(redis_client, "task-123", min_update_interval_seconds=0)

        # Should not raise
        aggregator.set_bar(1, downloaded=10, total=100)
        aggregator.set_bar(1, downloaded=20, total=100)
        aggregator.set_bar(1, downloaded=30, total=100)

    @patch("webui.tasks.model_manager.task_state.update_task_progress_sync")
    def test_disable_stops_tracking_new_bars(self, mock_update_progress):
        from webui.tasks.model_manager import _DownloadProgressAggregator

        redis_client = MagicMock()
        aggregator = _DownloadProgressAggregator(redis_client, "task-123", min_update_interval_seconds=0)

        aggregator.set_bar(1, downloaded=10, total=100)
        aggregator.disable()
        aggregator.set_bar(2, downloaded=5, total=50)

        assert mock_update_progress.call_count == 1
