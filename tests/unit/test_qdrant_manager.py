"""Unit tests for packages/webui/utils/qdrant_manager.py"""

import threading
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from webui.utils.qdrant_manager import QdrantConnectionManager
from webui.utils.retry import exponential_backoff_retry

# from qdrant_client.exceptions import UnexpectedResponse  # Not available in all versions


class TestExponentialBackoffRetry:
    """Test suite for the exponential_backoff_retry decorator"""

    def test_sync_function_success_first_try(self):
        """Test that sync function succeeds on first try"""
        mock_func = Mock(return_value="success")
        decorated = exponential_backoff_retry(max_retries=3)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_sync_function_retry_then_success(self):
        """Test sync function that fails twice then succeeds"""
        mock_func = Mock(side_effect=[ConnectionError("Network error"), TimeoutError("Timeout"), "success"])

        decorated = exponential_backoff_retry(max_retries=3, initial_delay=0.1, max_delay=1.0)(mock_func)

        with patch("time.sleep") as mock_sleep:
            result = decorated()

        assert result == "success"
        assert mock_func.call_count == 3

        # Verify exponential backoff delays
        assert mock_sleep.call_count == 2
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert delays[0] == 0.1  # initial_delay
        assert delays[1] == 0.2  # initial_delay * 2

    def test_sync_function_max_retries_exceeded(self):
        """Test sync function that exceeds max retries"""
        mock_func = Mock(side_effect=ConnectionError("Persistent error"))

        decorated = exponential_backoff_retry(max_retries=2, initial_delay=0.1)(mock_func)

        with patch("time.sleep"), pytest.raises(ConnectionError, match="Persistent error"):
            decorated()

        assert mock_func.call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio()
    async def test_async_function_success_first_try(self):
        """Test that async function succeeds on first try"""
        mock_func = AsyncMock(return_value="async success")
        decorated = exponential_backoff_retry(max_retries=3)(mock_func)

        result = await decorated()

        assert result == "async success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio()
    async def test_async_function_retry_then_success(self):
        """Test async function that fails twice then succeeds"""
        mock_func = AsyncMock(side_effect=[ConnectionError("Network error"), TimeoutError("Timeout"), "async success"])

        decorated = exponential_backoff_retry(max_retries=3, initial_delay=0.1, max_delay=1.0)(mock_func)

        with patch("asyncio.sleep") as mock_sleep:
            result = await decorated()

        assert result == "async success"
        assert mock_func.call_count == 3

        # Verify exponential backoff delays
        assert mock_sleep.call_count == 2

    def test_decorator_with_different_configurations(self):
        """Test decorator with various configuration parameters"""
        mock_func = Mock(side_effect=[Exception("Error 1"), Exception("Error 2"), Exception("Error 3"), "success"])

        decorated = exponential_backoff_retry(max_retries=4, initial_delay=0.5, max_delay=2.0, exponential_base=3)(
            mock_func
        )

        with patch("time.sleep") as mock_sleep:
            result = decorated()

        assert result == "success"
        assert mock_func.call_count == 4

        # Verify delays follow exponential pattern with base 3
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert delays[0] == 0.5  # initial_delay
        assert delays[1] == 1.5  # 0.5 * 3
        assert delays[2] == 2.0  # Would be 4.5 but capped at max_delay

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves the original function's metadata"""

        def test_function(x: int, y: str) -> str:
            """Test function docstring"""
            return f"{x}-{y}"

        decorated = exponential_backoff_retry()(test_function)

        assert decorated.__name__ == "test_function"
        assert decorated.__doc__ == "Test function docstring"


class TestQdrantConnectionManager:
    """Test suite for QdrantConnectionManager class"""

    def test_singleton_pattern(self):
        """Test that QdrantConnectionManager follows singleton pattern"""
        manager1 = QdrantConnectionManager()
        manager2 = QdrantConnectionManager()

        assert manager1 is manager2

    def test_thread_safe_singleton(self):
        """Test that singleton is thread-safe"""
        managers = []

        def create_manager():
            managers.append(QdrantConnectionManager())

        threads = [threading.Thread(target=create_manager) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All instances should be the same
        assert all(m is managers[0] for m in managers)

    @patch("webui.utils.qdrant_manager.QdrantClient")
    def test_get_client_creates_new_client(self, mock_qdrant_client_class):
        """Test get_client creates a new client when none exists"""
        # Reset singleton
        QdrantConnectionManager._instance = None

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()  # Simulate successful connection
        mock_qdrant_client_class.return_value = mock_client

        manager = QdrantConnectionManager()

        client = manager.get_client()

        assert client is mock_client
        mock_qdrant_client_class.assert_called_once_with(
            url="http://localhost:6333"  # No timeout parameter in actual implementation
        )
        mock_client.get_collections.assert_called_once()  # Verify connection check

    @patch("webui.utils.qdrant_manager.QdrantClient")
    def test_get_client_reuses_existing_client(self, mock_qdrant_client_class):
        """Test get_client reuses existing valid client"""
        # Reset singleton
        QdrantConnectionManager._instance = None

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_qdrant_client_class.return_value = mock_client

        manager = QdrantConnectionManager()

        client1 = manager.get_client()
        client2 = manager.get_client()

        assert client1 is client2
        assert mock_qdrant_client_class.call_count == 1  # Only created once

    @patch("webui.utils.qdrant_manager.QdrantClient")
    def test_get_client_recreates_on_connection_failure(self, mock_qdrant_client_class):
        """Test get_client creates new client when existing one fails verification"""
        # Reset singleton
        QdrantConnectionManager._instance = None

        # First client that will fail verification when checked
        mock_client1 = MagicMock()
        mock_client1.get_collections.side_effect = Exception("Connection lost")

        # Second client that works
        mock_client2 = MagicMock()
        mock_client2.get_collections.return_value = MagicMock()

        # Only return mock_client2 when creating new client
        mock_qdrant_client_class.return_value = mock_client2

        manager = QdrantConnectionManager()

        # Manually set the cached client to the failing one
        manager._client = mock_client1

        # This should detect the failure and create client2
        client = manager.get_client()

        assert client is mock_client2
        assert mock_qdrant_client_class.call_count == 1  # Only one new client creation

    @patch("webui.utils.qdrant_manager.QdrantClient")
    def test_get_client_retry_on_creation_failure(self, mock_qdrant_client_class):
        """Test get_client retries when client creation fails"""
        # Reset singleton
        QdrantConnectionManager._instance = None

        # Fail twice, then succeed
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()

        mock_qdrant_client_class.side_effect = [ConnectionError("Network error"), TimeoutError("Timeout"), mock_client]

        manager = QdrantConnectionManager()

        with patch("time.sleep"):  # Speed up test
            client = manager.get_client()

        assert client is mock_client
        assert mock_qdrant_client_class.call_count == 3

    @patch("webui.utils.qdrant_manager.QdrantClient")
    def test_create_collection_success(self, mock_qdrant_client_class):
        """Test successful collection creation"""
        # Reset singleton
        QdrantConnectionManager._instance = None

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_client.create_collection.return_value = None
        mock_qdrant_client_class.return_value = mock_client

        manager = QdrantConnectionManager()

        manager.create_collection(collection_name="test_collection", vector_size=768)

        # create_collection returns None
        mock_client.create_collection.assert_called_once()

        # Verify collection configuration
        call_args = mock_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "test_collection"
        assert call_args.kwargs["vectors_config"].size == 768

    @patch("webui.utils.qdrant_manager.QdrantClient")
    def test_create_collection_with_retry(self, mock_qdrant_client_class):
        """Test collection creation with retry on failure"""
        # Reset singleton
        QdrantConnectionManager._instance = None

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()

        # Fail twice, then succeed
        mock_client.create_collection.side_effect = [
            Exception("Server error"),
            ConnectionError("Network error"),
            None,  # Success returns None
        ]

        mock_qdrant_client_class.return_value = mock_client

        manager = QdrantConnectionManager()

        with patch("time.sleep"):  # Speed up test
            manager.create_collection(collection_name="test_collection", vector_size=768)

        # create_collection returns None
        assert mock_client.create_collection.call_count == 3

    @patch("webui.utils.qdrant_manager.QdrantClient")
    def test_verify_collection_exists(self, mock_qdrant_client_class):
        """Test verify_collection when collection exists"""
        # Reset singleton
        QdrantConnectionManager._instance = None

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_client.get_collection.return_value = {"name": "test_collection", "config": {}}
        mock_qdrant_client_class.return_value = mock_client

        manager = QdrantConnectionManager()

        result = manager.verify_collection("test_collection")

        assert isinstance(result, dict)
        assert result["name"] == "test_collection"
        mock_client.get_collection.assert_called_once_with("test_collection")

    @patch("webui.utils.qdrant_manager.QdrantClient")
    def test_verify_collection_not_exists(self, mock_qdrant_client_class):
        """Test verify_collection when collection doesn't exist"""
        # Reset singleton
        QdrantConnectionManager._instance = None

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_qdrant_client_class.return_value = mock_client

        manager = QdrantConnectionManager()

        with pytest.raises(Exception, match="Collection not found"):
            manager.verify_collection("test_collection")

    def test_close_connection(self):
        """Test closing the connection"""
        # Reset singleton
        QdrantConnectionManager._instance = None

        mock_client = MagicMock()
        manager = QdrantConnectionManager()
        manager._client = mock_client

        manager.close()

        mock_client.close.assert_called_once()
        assert manager._client is None


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""

    @patch("webui.utils.qdrant_manager.QdrantClient")
    def test_connection_recovery_scenario(self, mock_qdrant_client_class):
        """Test that manager recovers from connection failures"""
        # Reset singleton
        QdrantConnectionManager._instance = None

        # Create clients with different behaviors
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()

        # Client 1 works initially, then fails
        mock_client1.get_collections.side_effect = [
            MagicMock(),  # Works for initial get_client
            Exception("Connection lost"),  # Fails on second check
        ]

        # Client 2 always works
        mock_client2.get_collections.return_value = MagicMock()

        mock_qdrant_client_class.side_effect = [mock_client1, mock_client2]

        manager = QdrantConnectionManager()

        # First call succeeds with client1
        client1 = manager.get_client()
        assert client1 is mock_client1

        # Second call detects failure and creates client2
        client2 = manager.get_client()
        assert client2 is mock_client2
