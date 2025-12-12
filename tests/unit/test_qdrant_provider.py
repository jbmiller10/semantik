"""Unit tests for the unified Qdrant provider (webui.qdrant)."""

import threading
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from webui.qdrant import UnifiedQdrantManager, get_qdrant_manager, set_qdrant_manager_for_tests
from webui.utils.retry import exponential_backoff_retry


class TestExponentialBackoffRetry:
    """Test suite for the exponential_backoff_retry decorator"""

    def test_sync_function_success_first_try(self) -> None:
        mock_func = Mock(return_value="success")
        decorated = exponential_backoff_retry(max_retries=3)(mock_func)
        assert decorated() == "success"
        assert mock_func.call_count == 1

    def test_sync_function_retry_then_success(self) -> None:
        mock_func = Mock(side_effect=[ConnectionError("Network error"), TimeoutError("Timeout"), "success"])
        decorated = exponential_backoff_retry(max_retries=3, initial_delay=0.1, max_delay=1.0)(mock_func)

        with patch("time.sleep") as mock_sleep:
            assert decorated() == "success"

        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2

    def test_sync_function_max_retries_exceeded(self) -> None:
        mock_func = Mock(side_effect=ConnectionError("Persistent error"))
        decorated = exponential_backoff_retry(max_retries=2, initial_delay=0.1)(mock_func)

        with patch("time.sleep"), pytest.raises(ConnectionError, match="Persistent error"):
            decorated()

        assert mock_func.call_count == 3

    @pytest.mark.asyncio()
    async def test_async_function_retry_then_success(self) -> None:
        mock_func = AsyncMock(side_effect=[ConnectionError("Network error"), TimeoutError("Timeout"), "async success"])
        decorated = exponential_backoff_retry(max_retries=3, initial_delay=0.1, max_delay=1.0)(mock_func)

        with patch("asyncio.sleep") as mock_sleep:
            assert await decorated() == "async success"

        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2


class TestUnifiedQdrantProvider:
    """Validate singleton behavior of get_qdrant_manager."""

    def setup_method(self) -> None:
        # Ensure clean singleton before each test
        set_qdrant_manager_for_tests(None)

    def teardown_method(self) -> None:
        set_qdrant_manager_for_tests(None)

    @patch("webui.qdrant._build_client")
    def test_get_qdrant_manager_singleton(self, mock_build_client: Mock) -> None:
        mock_client = MagicMock()
        mock_build_client.return_value = mock_client

        manager1 = get_qdrant_manager()
        manager2 = get_qdrant_manager()

        assert manager1 is manager2
        assert isinstance(manager1, UnifiedQdrantManager)
        assert manager1.get_client() is mock_client
        mock_build_client.assert_called_once()

    @patch("webui.qdrant._build_client")
    def test_thread_safe_singleton(self, mock_build_client: Mock) -> None:
        mock_client = MagicMock()
        mock_build_client.return_value = mock_client

        managers: list[UnifiedQdrantManager] = []

        def create_manager() -> None:
            managers.append(get_qdrant_manager())

        threads = [threading.Thread(target=create_manager) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len({id(m) for m in managers}) == 1
        mock_build_client.assert_called_once()

    @patch("webui.qdrant._build_client")
    def test_set_qdrant_manager_for_tests_overrides(self, mock_build_client: Mock) -> None:
        mock_client = MagicMock()
        mock_build_client.return_value = mock_client
        original = get_qdrant_manager()

        replacement = UnifiedQdrantManager(MagicMock())
        set_qdrant_manager_for_tests(replacement)

        assert get_qdrant_manager() is replacement

        # Reset to original for cleanliness
        set_qdrant_manager_for_tests(original)
