"""Test suite for Qdrant collection cleanup tasks."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCleanupOldCollections:
    """Test suite for cleanup_old_collections task."""

    @pytest.fixture()
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        client = MagicMock()
        client.get_collections = MagicMock()
        client.delete_collection = MagicMock()
        return client

    @pytest.fixture()
    def mock_connection_manager(self, mock_qdrant_client):
        """Create a mock connection manager."""
        manager = MagicMock()
        manager.get_client = MagicMock(return_value=mock_qdrant_client)
        return manager

    def test_cleanup_old_collections_empty_list(self):
        """Test cleanup with empty collection list."""
        from webui.tasks import cleanup_old_collections
        
        result = cleanup_old_collections([], "collection-123")

        assert result["collections_deleted"] == 0
        assert result["collections_failed"] == 0
        assert result["errors"] == []
        assert result["collection_id"] == "collection-123"

    @patch("webui.tasks.QdrantManager")
    def test_cleanup_old_collections_success(self, mock_qdrant_manager_class, mock_qdrant_client):
        """Test successful cleanup of collections."""
        from webui.tasks import cleanup_old_collections
        
        # Setup mocks
        mock_qdrant_manager = MagicMock()
        mock_qdrant_manager.client = mock_qdrant_client
        mock_qdrant_manager_class.return_value = mock_qdrant_manager

        # Mock collections exist
        mock_collections = MagicMock()
        mock_collections.collections = [
            MagicMock(name="col_old_1"),
            MagicMock(name="col_old_2"),
        ]
        mock_qdrant_client.get_collections.return_value = mock_collections

        # Run cleanup
        result = cleanup_old_collections(["col_old_1", "col_old_2"], "collection-123")

        # Verify results
        assert result["collections_deleted"] == 2
        assert result["collections_failed"] == 0
        assert result["errors"] == []

        # Verify delete was called
        assert mock_qdrant_client.delete_collection.call_count == 2
        mock_qdrant_client.delete_collection.assert_any_call("col_old_1")
        mock_qdrant_client.delete_collection.assert_any_call("col_old_2")

    @patch("webui.tasks.QdrantManager")
    def test_cleanup_old_collections_not_found(self, mock_qdrant_manager_class, mock_qdrant_client):
        """Test cleanup when collection doesn't exist."""
        from webui.tasks import cleanup_old_collections
        
        # Setup mocks
        mock_qdrant_manager = MagicMock()
        mock_qdrant_manager.client = mock_qdrant_client
        mock_qdrant_manager_class.return_value = mock_qdrant_manager

        # Mock empty collections
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_qdrant_client.get_collections.return_value = mock_collections

        # Run cleanup
        result = cleanup_old_collections(["col_missing"], "collection-123")

        # Verify results
        assert result["collections_deleted"] == 0
        assert result["collections_failed"] == 0
        assert mock_qdrant_client.delete_collection.call_count == 0

    @patch("webui.tasks.QdrantManager")
    def test_cleanup_old_collections_partial_failure(self, mock_qdrant_manager_class, mock_qdrant_client):
        """Test cleanup with partial failures."""
        from webui.tasks import cleanup_old_collections
        
        # Setup mocks
        mock_qdrant_manager = MagicMock()
        mock_qdrant_manager.client = mock_qdrant_client
        mock_qdrant_manager_class.return_value = mock_qdrant_manager

        # Mock collections exist
        mock_collections = MagicMock()
        mock_collections.collections = [
            MagicMock(name="col_success"),
            MagicMock(name="col_fail"),
        ]
        mock_qdrant_client.get_collections.return_value = mock_collections

        # Mock delete failure for second collection
        mock_qdrant_client.delete_collection.side_effect = [None, Exception("Delete failed")]

        # Run cleanup
        result = cleanup_old_collections(["col_success", "col_fail"], "collection-123")

        # Verify results
        assert result["collections_deleted"] == 1
        assert result["collections_failed"] == 1
        assert len(result["errors"]) == 1
        assert "col_fail" in result["errors"][0]


class TestCleanupQdrantCollections:
    """Test suite for enhanced cleanup_qdrant_collections task."""

    @pytest.fixture()
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        client = MagicMock()
        client.get_collections = MagicMock()
        client.delete_collection = MagicMock()
        return client

    @pytest.fixture()
    def mock_connection_manager(self, mock_qdrant_client):
        """Create a mock connection manager."""
        manager = MagicMock()
        manager.get_client = MagicMock(return_value=mock_qdrant_client)
        return manager

    @pytest.fixture()
    def mock_qdrant_manager(self, mock_qdrant_client):
        """Create a mock QdrantManager instance."""
        manager = MagicMock()
        manager.client = mock_qdrant_client
        manager.collection_exists = MagicMock()
        manager.get_collection_info = MagicMock()
        manager._is_staging_collection_old = MagicMock()
        return manager

    @pytest.fixture()
    def mock_collection_repo(self):
        """Create a mock collection repository."""
        repo = AsyncMock()
        repo.list_all = AsyncMock()
        return repo

    def test_cleanup_qdrant_collections_empty_list(self):
        """Test cleanup with empty collection list."""
        from webui.tasks import cleanup_qdrant_collections
        
        result = cleanup_qdrant_collections([])

        assert result["collections_deleted"] == 0
        assert result["collections_skipped"] == 0
        assert result["collections_failed"] == 0
        assert result["errors"] == []
        assert "timestamp" in result

    @patch("webui.tasks._audit_collection_deletion")
    @patch("webui.tasks._get_active_collections")
    @patch("webui.tasks.connection_manager")
    @patch("webui.tasks.QdrantManager")
    def test_cleanup_qdrant_collections_skip_system(
        self,
        mock_qdrant_manager_class,
        mock_conn_manager,
        mock_get_active,
        mock_audit,
        mock_qdrant_manager,
        mock_qdrant_client,
    ):
        """Test that system collections are skipped."""
        # Setup mocks
        mock_qdrant_manager_class.return_value = mock_qdrant_manager
        mock_conn_manager.get_client.return_value = mock_qdrant_client
        mock_get_active.return_value = set()

        # Run cleanup with system collection
        from webui.tasks import cleanup_qdrant_collections
        
        result = cleanup_qdrant_collections(["_system_collection"])

        # Verify results
        assert result["collections_deleted"] == 0
        assert result["collections_skipped"] == 1
        assert result["safety_checks"]["_system_collection"] == "system_collection"

        # Verify no deletion attempted
        mock_qdrant_manager.collection_exists.assert_not_called()
        mock_qdrant_client.delete_collection.assert_not_called()

    @patch("webui.tasks._audit_collection_deletion")
    @patch("webui.tasks._get_active_collections")
    @patch("webui.tasks.connection_manager")
    @patch("webui.tasks.QdrantManager")
    def test_cleanup_qdrant_collections_skip_active(
        self,
        mock_qdrant_manager_class,
        mock_conn_manager,
        mock_get_active,
        mock_audit,
        mock_qdrant_manager,
        mock_qdrant_client,
    ):
        """Test that active collections are skipped."""
        # Setup mocks
        mock_qdrant_manager_class.return_value = mock_qdrant_manager
        mock_conn_manager.get_client.return_value = mock_qdrant_client
        mock_get_active.return_value = {"col_active", "col_in_use"}

        # Run cleanup
        from webui.tasks import cleanup_qdrant_collections
        
        result = cleanup_qdrant_collections(["col_active", "col_inactive"])

        # Verify active collection skipped
        assert result["collections_skipped"] >= 1
        assert result["safety_checks"]["col_active"] == "active_collection"

    @patch("webui.tasks._audit_collection_deletion")
    @patch("webui.tasks._get_active_collections")
    @patch("webui.tasks.connection_manager")
    @patch("webui.tasks.QdrantManager")
    def test_cleanup_qdrant_collections_skip_recent_staging(
        self,
        mock_qdrant_manager_class,
        mock_conn_manager,
        mock_get_active,
        mock_audit,
        mock_qdrant_manager,
        mock_qdrant_client,
    ):
        """Test that recent staging collections are skipped."""
        # Setup mocks
        mock_qdrant_manager_class.return_value = mock_qdrant_manager
        mock_conn_manager.get_client.return_value = mock_qdrant_client
        mock_get_active.return_value = set()

        # Mock collection exists but is recent staging
        mock_qdrant_manager.collection_exists.return_value = True
        mock_qdrant_manager._is_staging_collection_old.return_value = False

        # Run cleanup
        from webui.tasks import cleanup_qdrant_collections
        
        result = cleanup_qdrant_collections(["staging_col_123_20240115_120000"])

        # Verify results
        assert result["collections_deleted"] == 0
        assert result["collections_skipped"] == 1
        assert result["safety_checks"]["staging_col_123_20240115_120000"] == "staging_too_recent"

    @patch("webui.tasks._audit_collection_deletion")
    @patch("webui.tasks._get_active_collections")
    @patch("webui.tasks.connection_manager")
    @patch("webui.tasks.QdrantManager")
    def test_cleanup_qdrant_collections_successful_deletion(
        self,
        mock_qdrant_manager_class,
        mock_conn_manager,
        mock_get_active,
        mock_audit,
        mock_qdrant_manager,
        mock_qdrant_client,
    ):
        """Test successful collection deletion with all safety checks passed."""
        # Setup mocks
        mock_qdrant_manager_class.return_value = mock_qdrant_manager
        mock_conn_manager.get_client.return_value = mock_qdrant_client
        mock_get_active.return_value = set()

        # Mock collection exists and is old
        mock_qdrant_manager.collection_exists.return_value = True
        mock_collection_info = MagicMock()
        mock_collection_info.vectors_count = 1000
        mock_qdrant_manager.get_collection_info.return_value = mock_collection_info
        mock_qdrant_manager._is_staging_collection_old.return_value = True

        # Run cleanup
        from webui.tasks import cleanup_qdrant_collections
        
        result = cleanup_qdrant_collections(["staging_col_old_20240101_120000"])

        # Verify results
        assert result["collections_deleted"] == 1
        assert result["collections_skipped"] == 0
        assert result["collections_failed"] == 0
        assert result["safety_checks"]["staging_col_old_20240101_120000"] == "deleted"

        # Verify deletion was called
        mock_qdrant_client.delete_collection.assert_called_once_with("staging_col_old_20240101_120000")

        # Verify audit was called
        mock_audit.assert_called_once_with("staging_col_old_20240101_120000", 1000)

    @patch("webui.tasks._get_active_collections")
    @patch("webui.tasks.connection_manager")
    @patch("webui.tasks.QdrantManager")
    def test_cleanup_qdrant_collections_error_handling(
        self, mock_qdrant_manager_class, mock_conn_manager, mock_get_active, mock_qdrant_manager, mock_qdrant_client
    ):
        """Test error handling during cleanup."""
        # Setup mocks
        mock_qdrant_manager_class.return_value = mock_qdrant_manager
        mock_conn_manager.get_client.return_value = mock_qdrant_client
        mock_get_active.return_value = set()

        # Mock collection exists but deletion fails
        mock_qdrant_manager.collection_exists.return_value = True
        mock_qdrant_manager.get_collection_info.return_value = MagicMock(vectors_count=100)
        mock_qdrant_client.delete_collection.side_effect = Exception("Qdrant error")

        # Run cleanup
        from webui.tasks import cleanup_qdrant_collections
        
        result = cleanup_qdrant_collections(["col_error"])

        # Verify results
        assert result["collections_deleted"] == 0
        assert result["collections_failed"] == 1
        assert len(result["errors"]) == 1
        assert "col_error" in result["errors"][0]
        assert "error: Qdrant error" in result["safety_checks"]["col_error"]


@pytest.mark.asyncio()
class TestGetActiveCollections:
    """Test suite for _get_active_collections helper function."""

    @pytest.fixture()
    def mock_collections(self):
        """Create mock collection data."""
        return [
            {
                "id": "col1",
                "vector_store_name": "qdrant_col_1",
                "qdrant_collections": ["col_1_active"],
                "qdrant_staging": None,
            },
            {
                "id": "col2",
                "vector_store_name": "qdrant_col_2",
                "qdrant_collections": ["col_2_active", "col_2_backup"],
                "qdrant_staging": {"collection_name": "staging_col_2_20240115_120000"},
            },
            {
                "id": "col3",
                "vector_store_name": None,
                "qdrant_collections": [],
                "qdrant_staging": None,
            },
        ]

    @patch("webui.tasks.AsyncSessionLocal")
    async def test_get_active_collections(self, mock_session_local, mock_collections):
        """Test getting active collections from database."""
        from webui.tasks import _get_active_collections

        # Setup mocks
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session

        mock_repo = AsyncMock()
        mock_repo.list_all.return_value = mock_collections

        with patch("webui.tasks.CollectionRepository", return_value=mock_repo):
            # Run function
            active_collections = await _get_active_collections()

            # Verify results
            assert isinstance(active_collections, set)
            assert "qdrant_col_1" in active_collections
            assert "qdrant_col_2" in active_collections
            assert "col_1_active" in active_collections
            assert "col_2_active" in active_collections
            assert "col_2_backup" in active_collections
            assert "staging_col_2_20240115_120000" in active_collections

            # Verify col3 values not included (None/empty)
            assert None not in active_collections
            assert len(active_collections) == 6


@pytest.mark.asyncio()
class TestAuditCollectionDeletion:
    """Test suite for _audit_collection_deletion helper function."""

    @patch("webui.tasks.AsyncSessionLocal")
    async def test_audit_collection_deletion_success(self, mock_session_local):
        """Test successful audit log creation."""
        from webui.tasks import _audit_collection_deletion

        # Setup mocks
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session

        # Run function
        await _audit_collection_deletion("test_collection", 5000)

        # Verify audit log was added
        assert mock_session.add.call_count == 1
        audit_log = mock_session.add.call_args[0][0]

        # Verify audit log fields
        assert audit_log.collection_id is None
        assert audit_log.operation_id is None
        assert audit_log.user_id is None
        assert audit_log.action == "qdrant_collection_deleted"
        assert audit_log.details["collection_name"] == "test_collection"
        assert audit_log.details["vector_count"] == 5000
        assert "deleted_at" in audit_log.details

        # Verify commit was called
        mock_session.commit.assert_called_once()

    @patch("webui.tasks.AsyncSessionLocal")
    async def test_audit_collection_deletion_error(self, mock_session_local):
        """Test audit log creation handles errors gracefully."""
        from webui.tasks import _audit_collection_deletion

        # Setup mocks with error
        mock_session = AsyncMock()
        mock_session.commit.side_effect = Exception("Database error")
        mock_session_local.return_value.__aenter__.return_value = mock_session

        # Run function - should not raise
        await _audit_collection_deletion("test_collection", 100)

        # Verify attempt was made
        assert mock_session.add.call_count == 1
        mock_session.commit.assert_called_once()
