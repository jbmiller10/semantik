"""Test suite for Qdrant collection cleanup tasks."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCleanupOldCollections:
    """Test suite for cleanup_old_collections task."""

    def test_cleanup_old_collections_empty_list(self):
        """Test cleanup with empty collection list."""
        from webui.tasks import cleanup_old_collections

        result = cleanup_old_collections([], "collection-123")

        assert result["collections_deleted"] == 0
        assert result["collections_failed"] == 0
        assert result["errors"] == []
        assert result["collection_id"] == "collection-123"

    @patch("shared.metrics.collection_metrics.QdrantOperationTimer")
    @patch("webui.utils.qdrant_manager.qdrant_manager")
    @patch("shared.managers.qdrant_manager.QdrantManager")
    def test_cleanup_old_collections_success(self, mock_qdrant_manager_class, mock_conn_manager, mock_timer):
        """Test successful cleanup of collections."""
        from webui.tasks import cleanup_old_collections

        # Setup mocks
        mock_qdrant_manager = MagicMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_manager.client = mock_qdrant_client
        mock_qdrant_manager_class.return_value = mock_qdrant_manager
        mock_conn_manager.get_client.return_value = mock_qdrant_client

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


class TestCleanupQdrantCollections:
    """Test suite for enhanced cleanup_qdrant_collections task."""

    def test_cleanup_qdrant_collections_empty_list(self):
        """Test cleanup with empty collection list."""
        from webui.tasks import cleanup_qdrant_collections

        result = cleanup_qdrant_collections([])

        assert result["collections_deleted"] == 0
        assert result["collections_skipped"] == 0
        assert result["collections_failed"] == 0
        assert result["errors"] == []
        assert "timestamp" in result

    @patch("shared.metrics.collection_metrics.QdrantOperationTimer")
    @patch("webui.tasks._audit_collection_deletions_batch")
    @patch("webui.tasks._get_active_collections")
    @patch("webui.utils.qdrant_manager.qdrant_manager")
    @patch("shared.managers.qdrant_manager.QdrantManager")
    def test_cleanup_qdrant_collections_skip_system(
        self,
        mock_qdrant_manager_class,
        mock_conn_manager,
        mock_get_active,
        mock_audit,
        mock_timer,
    ):
        """Test that system collections are skipped."""
        # Setup mocks
        mock_qdrant_manager = MagicMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_manager.client = mock_qdrant_client
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

    @patch("shared.metrics.collection_metrics.QdrantOperationTimer")
    @patch("webui.tasks._audit_collection_deletions_batch")
    @patch("webui.tasks._get_active_collections")
    @patch("webui.utils.qdrant_manager.qdrant_manager")
    @patch("shared.managers.qdrant_manager.QdrantManager")
    def test_cleanup_qdrant_collections_skip_active(
        self,
        mock_qdrant_manager_class,
        mock_conn_manager,
        mock_get_active,
        mock_audit,
        mock_timer,
    ):
        """Test that active collections are skipped."""
        # Setup mocks
        mock_qdrant_manager = MagicMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_manager.client = mock_qdrant_client
        mock_qdrant_manager_class.return_value = mock_qdrant_manager
        mock_conn_manager.get_client.return_value = mock_qdrant_client
        mock_get_active.return_value = {"col_active", "col_in_use"}

        # Run cleanup
        from webui.tasks import cleanup_qdrant_collections

        result = cleanup_qdrant_collections(["col_active", "col_inactive"])

        # Verify active collection skipped
        assert result["collections_skipped"] >= 1
        assert result["safety_checks"]["col_active"] == "active_collection"

    @patch("shared.metrics.collection_metrics.QdrantOperationTimer")
    @patch("webui.tasks._audit_collection_deletions_batch")
    @patch("webui.tasks._get_active_collections")
    @patch("webui.utils.qdrant_manager.qdrant_manager")
    @patch("shared.managers.qdrant_manager.QdrantManager")
    def test_cleanup_qdrant_collections_skip_recent_staging(
        self,
        mock_qdrant_manager_class,
        mock_conn_manager,
        mock_get_active,
        mock_audit,
        mock_timer,
    ):
        """Test that recent staging collections are skipped."""
        # Setup mocks
        mock_qdrant_manager = MagicMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_manager.client = mock_qdrant_client
        mock_qdrant_manager_class.return_value = mock_qdrant_manager
        mock_conn_manager.get_client.return_value = mock_qdrant_client
        mock_get_active.return_value = set()

        # Mock collection exists but is recent staging
        mock_qdrant_manager.collection_exists.return_value = True
        mock_qdrant_manager._is_staging_collection_old.return_value = False

        # Run cleanup
        from webui.tasks import cleanup_qdrant_collections

        result = cleanup_qdrant_collections(["staging_col_123_20240115_120000"], staging_age_hours=1)

        # Verify results
        assert result["collections_deleted"] == 0
        assert result["collections_skipped"] == 1
        assert result["safety_checks"]["staging_col_123_20240115_120000"] == "staging_too_recent"

    @patch("shared.metrics.collection_metrics.QdrantOperationTimer")
    @patch("webui.tasks._audit_collection_deletions_batch")
    @patch("webui.tasks._get_active_collections")
    @patch("webui.utils.qdrant_manager.qdrant_manager")
    @patch("shared.managers.qdrant_manager.QdrantManager")
    def test_cleanup_qdrant_collections_successful_deletion(
        self,
        mock_qdrant_manager_class,
        mock_conn_manager,
        mock_get_active,
        mock_audit_batch,
        mock_timer,
    ):
        """Test successful collection deletion with all safety checks passed."""
        # Setup mocks
        mock_qdrant_manager = MagicMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_manager.client = mock_qdrant_client
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

        # Verify batch audit was called with correct data
        mock_audit_batch.assert_called_once_with([("staging_col_old_20240101_120000", 1000)])


@pytest.mark.asyncio()
class TestGetActiveCollections:
    """Test suite for _get_active_collections helper function."""

    @patch("shared.database.database.AsyncSessionLocal")
    @patch("shared.database.repositories.collection_repository.CollectionRepository")
    async def test_get_active_collections(self, mock_repo_class, mock_session_local):
        """Test getting active collections from database."""
        from webui.tasks import _get_active_collections

        # Setup mocks
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session

        mock_repo = AsyncMock()
        mock_collections = [
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
        ]
        mock_repo.list_all.return_value = mock_collections
        mock_repo_class.return_value = mock_repo

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


@pytest.mark.asyncio()
class TestAuditCollectionDeletion:
    """Test suite for _audit_collection_deletions_batch helper function."""

    @patch("shared.database.database.AsyncSessionLocal")
    @patch("shared.database.models.CollectionAuditLog")
    async def test_audit_collection_deletions_batch_success(self, mock_audit_log_class, mock_session_local):
        """Test successful batch audit log creation."""
        from webui.tasks import _audit_collection_deletions_batch

        # Setup mocks
        mock_session = AsyncMock()
        mock_session_local.return_value.__aenter__.return_value = mock_session

        # Run function
        deletions = [("test_collection_1", 1000), ("test_collection_2", 2000)]
        await _audit_collection_deletions_batch(deletions)

        # Verify audit logs were created
        assert mock_session.add.call_count == 2
        assert mock_session.commit.call_count == 1

    @patch("shared.database.database.AsyncSessionLocal")
    async def test_audit_collection_deletions_batch_empty(self, mock_session_local):
        """Test batch audit with empty list."""
        from webui.tasks import _audit_collection_deletions_batch

        # Run function with empty list
        await _audit_collection_deletions_batch([])

        # Verify no session created
        mock_session_local.assert_not_called()
