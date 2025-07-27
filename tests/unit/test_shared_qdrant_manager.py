"""
Unit tests for packages/shared/managers/qdrant_manager.py

Tests the QdrantManager service for managing Qdrant collections with
blue-green deployment support.
"""

# mypy: ignore-errors

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import CollectionInfo, Distance

from packages.shared.managers.qdrant_manager import QdrantManager


class TestQdrantManager:
    """Test suite for QdrantManager class"""

    @pytest.fixture()
    def mock_qdrant_client(self) -> None:
        """Create a mock Qdrant client"""
        return MagicMock()

    @pytest.fixture()
    def qdrant_manager(self, mock_qdrant_client) -> None:
        """Create a QdrantManager instance with mocked client"""
        return QdrantManager(mock_qdrant_client)

    def test_initialization(self, mock_qdrant_client) -> None:
        """Test QdrantManager initialization"""
        manager = QdrantManager(mock_qdrant_client)

        assert manager.client is mock_qdrant_client
        assert manager._staging_prefix == "staging_"
        assert manager._collection_prefix == "collection_"

    def test_create_staging_collection_success(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test successful creation of staging collection"""
        # Mock successful collection creation
        mock_collection_info = Mock(spec=CollectionInfo)
        mock_collection_info.vectors_count = 0
        mock_qdrant_client.get_collection.return_value = mock_collection_info

        with patch("packages.shared.managers.qdrant_manager.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15, 10, 30, 45, tzinfo=UTC)

            result = qdrant_manager.create_staging_collection(
                base_name="collection_abc123", vector_size=768, distance=Distance.COSINE
            )

        expected_name = "staging_collection_abc123_20240115_103045"
        assert result == expected_name

        # Verify collection was created with correct config
        mock_qdrant_client.create_collection.assert_called_once()
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == expected_name
        assert call_args.kwargs["vectors_config"].size == 768
        assert call_args.kwargs["vectors_config"].distance == Distance.COSINE
        assert call_args.kwargs["optimizers_config"]["indexing_threshold"] == 20000

    def test_create_staging_collection_with_custom_config(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test staging collection creation with custom optimizer config"""
        mock_collection_info = Mock(spec=CollectionInfo)
        mock_collection_info.vectors_count = 0
        mock_qdrant_client.get_collection.return_value = mock_collection_info

        custom_config = {"indexing_threshold": 50000, "memmap_threshold": 100000}

        qdrant_manager.create_staging_collection(
            base_name="test_collection", vector_size=512, distance=Distance.DOT, optimizers_config=custom_config
        )

        # Verify custom config was used
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args.kwargs["optimizers_config"] == custom_config
        assert call_args.kwargs["vectors_config"].distance == Distance.DOT

    def test_create_staging_collection_failure_with_cleanup(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test staging collection creation failure triggers cleanup"""
        # Mock creation success but verification failure
        mock_qdrant_client.get_collection.side_effect = Exception("Verification failed")

        with pytest.raises(Exception, match="Verification failed"):
            qdrant_manager.create_staging_collection(base_name="failing_collection", vector_size=768)

        # Verify cleanup was attempted
        mock_qdrant_client.delete_collection.assert_called_once()

    def test_cleanup_orphaned_collections_basic(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test basic orphaned collection cleanup"""
        # Mock existing collections
        mock_collections = MagicMock()

        # Create mock collection objects with name attribute
        collection_mocks = []
        for name in [
            "collection_active1",
            "collection_active2",
            "collection_orphaned1",
            "staging_old_collection_20240101_000000",
            "_collection_metadata",
        ]:
            col_mock = Mock()
            col_mock.name = name
            collection_mocks.append(col_mock)

        mock_collections.collections = collection_mocks
        mock_qdrant_client.get_collections.return_value = mock_collections

        # Mock collection info for orphaned collections
        mock_info = Mock(spec=CollectionInfo)
        mock_info.vectors_count = 100
        mock_qdrant_client.get_collection.return_value = mock_info

        # Define active collections
        active_collections = ["collection_active1", "collection_active2"]

        # Patch time.sleep to speed up test
        with patch("packages.shared.managers.qdrant_manager.time.sleep"):
            # Run cleanup
            deleted = qdrant_manager.cleanup_orphaned_collections(active_collections)

        # Should delete orphaned collections but not system collections
        assert len(deleted) == 2
        assert "collection_orphaned1" in deleted
        assert "staging_old_collection_20240101_000000" in deleted

        # Verify delete was called for each orphaned collection
        assert mock_qdrant_client.delete_collection.call_count == 2

    def test_cleanup_orphaned_collections_dry_run(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test cleanup in dry run mode doesn't delete anything"""
        # Mock existing collections
        mock_collections = MagicMock()

        # Create mock collection objects with name attribute
        col1 = Mock()
        col1.name = "collection_active"
        col2 = Mock()
        col2.name = "collection_orphaned"

        mock_collections.collections = [col1, col2]
        mock_qdrant_client.get_collections.return_value = mock_collections

        # Mock collection info
        mock_info = Mock(spec=CollectionInfo)
        mock_info.vectors_count = 50
        mock_qdrant_client.get_collection.return_value = mock_info

        # Patch time.sleep to speed up test
        with patch("packages.shared.managers.qdrant_manager.time.sleep"):
            # Run cleanup in dry run mode
            deleted = qdrant_manager.cleanup_orphaned_collections(
                active_collections=["collection_active"], dry_run=True
            )

        # Should report what would be deleted
        assert len(deleted) == 1
        assert "collection_orphaned" in deleted

        # But should not actually delete
        mock_qdrant_client.delete_collection.assert_not_called()

    def test_cleanup_orphaned_collections_recent_staging(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test that recent staging collections are not deleted"""
        # Create recent and old staging collection names
        now = datetime.now(UTC)
        recent_time = (now - timedelta(hours=12)).strftime("%Y%m%d_%H%M%S")
        old_time = (now - timedelta(hours=48)).strftime("%Y%m%d_%H%M%S")

        mock_collections = MagicMock()

        # Create mock collection objects with name attribute
        col1 = Mock()
        col1.name = f"staging_collection_test_{recent_time}"
        col2 = Mock()
        col2.name = f"staging_collection_test_{old_time}"

        mock_collections.collections = [col1, col2]
        mock_qdrant_client.get_collections.return_value = mock_collections

        # Mock collection info
        mock_info = Mock(spec=CollectionInfo)
        mock_info.vectors_count = 0
        mock_qdrant_client.get_collection.return_value = mock_info

        # Patch time.sleep to speed up test
        with patch("packages.shared.managers.qdrant_manager.time.sleep"):
            # Run cleanup
            deleted = qdrant_manager.cleanup_orphaned_collections(active_collections=[])

        # Should only delete the old staging collection
        assert len(deleted) == 1
        assert f"staging_collection_test_{old_time}" in deleted
        assert f"staging_collection_test_{recent_time}" not in deleted

    def test_cleanup_orphaned_collections_error_handling(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test cleanup continues after individual delete failures"""
        mock_collections = MagicMock()

        # Create mock collection objects with name attribute
        collection_mocks = []
        for name in ["collection_orphaned1", "collection_orphaned2", "collection_orphaned3"]:
            col_mock = Mock()
            col_mock.name = name
            collection_mocks.append(col_mock)

        mock_collections.collections = collection_mocks
        mock_qdrant_client.get_collections.return_value = mock_collections

        # Mock collection info
        mock_info = Mock(spec=CollectionInfo)
        mock_info.vectors_count = 10
        mock_qdrant_client.get_collection.return_value = mock_info

        # Make second delete fail
        mock_qdrant_client.delete_collection.side_effect = [
            None,  # First succeeds
            Exception("Delete failed"),  # Second fails
            None,  # Third succeeds
        ]

        # Patch time.sleep to speed up test
        with patch("packages.shared.managers.qdrant_manager.time.sleep"):
            # Run cleanup
            deleted = qdrant_manager.cleanup_orphaned_collections(active_collections=[])

        # Should successfully delete 2 collections despite one failure
        assert len(deleted) == 2
        assert "collection_orphaned1" in deleted
        assert "collection_orphaned3" in deleted
        assert "collection_orphaned2" not in deleted

    def test_list_collections(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test listing all collections"""
        mock_collections = MagicMock()

        # Create mock collection objects with name attribute
        col1 = Mock()
        col1.name = "collection1"
        col2 = Mock()
        col2.name = "collection2"
        col3 = Mock()
        col3.name = "collection3"

        mock_collections.collections = [col1, col2, col3]
        mock_qdrant_client.get_collections.return_value = mock_collections

        result = qdrant_manager.list_collections()

        assert result == ["collection1", "collection2", "collection3"]

    def test_get_collection_info_success(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test getting collection info successfully"""
        mock_info = Mock(spec=CollectionInfo)
        mock_info.vectors_count = 1000
        mock_info.status = "green"
        mock_qdrant_client.get_collection.return_value = mock_info

        result = qdrant_manager.get_collection_info("test_collection")

        assert result == mock_info
        assert result.vectors_count == 1000

    def test_get_collection_info_not_found(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test getting info for non-existent collection"""
        mock_qdrant_client.get_collection.side_effect = UnexpectedResponse(status_code=404, reason_phrase="Not Found", content=b"", headers={})  # type: ignore

        with pytest.raises(ValueError, match="Collection test_collection not found"):
            qdrant_manager.get_collection_info("test_collection")

    def test_collection_exists_true(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test checking if collection exists - positive case"""
        mock_info = Mock(spec=CollectionInfo)
        mock_qdrant_client.get_collection.return_value = mock_info

        result = qdrant_manager.collection_exists("existing_collection")

        assert result is True

    def test_collection_exists_false(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test checking if collection exists - negative case"""
        mock_qdrant_client.get_collection.side_effect = UnexpectedResponse(status_code=404, reason_phrase="Not Found", content=b"", headers={})  # type: ignore

        result = qdrant_manager.collection_exists("non_existent_collection")

        assert result is False

    def test_validate_collection_health_healthy(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test validating health of a healthy collection"""
        mock_info = Mock(spec=CollectionInfo)
        mock_info.vectors_count = 5000
        mock_info.indexed_vectors_count = 5000
        mock_info.status = "green"
        mock_info.optimizer_status = Mock()
        mock_info.optimizer_status.error = None

        mock_qdrant_client.get_collection.return_value = mock_info

        result = qdrant_manager.validate_collection_health("healthy_collection")

        assert result["healthy"] is True
        assert result["exists"] is True
        assert result["vectors_count"] == 5000
        assert result["status"] == "green"

    def test_validate_collection_health_degraded(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test validating health of a degraded collection"""
        mock_info = Mock(spec=CollectionInfo)
        mock_info.vectors_count = 1000
        mock_info.indexed_vectors_count = 800
        mock_info.status = "yellow"
        mock_info.optimizer_status = Mock()
        mock_info.optimizer_status.error = "Indexing error"

        mock_qdrant_client.get_collection.return_value = mock_info

        result = qdrant_manager.validate_collection_health("degraded_collection")

        assert result["healthy"] is False
        assert result["exists"] is True
        assert result["status"] == "yellow"
        assert "warning" in result
        assert "optimizer_error" in result

    def test_validate_collection_health_not_found(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test validating health of non-existent collection"""
        mock_qdrant_client.get_collection.side_effect = UnexpectedResponse(status_code=404, reason_phrase="Not Found", content=b"", headers={})  # type: ignore

        result = qdrant_manager.validate_collection_health("missing_collection")

        assert result["healthy"] is False
        assert result["exists"] is False
        assert "error" in result

    def test_is_staging_collection_old_valid_format(self, qdrant_manager) -> None:
        """Test checking age of staging collections with valid format"""
        # Create collection names with specific timestamps
        old_name = "staging_collection_test_20240101_120000"
        recent_name = f"staging_collection_test_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        # Test old collection (should be True)
        assert qdrant_manager._is_staging_collection_old(old_name, hours=24) is True

        # Test recent collection (should be False)
        assert qdrant_manager._is_staging_collection_old(recent_name, hours=24) is False

    def test_is_staging_collection_old_invalid_format(self, qdrant_manager) -> None:
        """Test checking age with invalid collection name format"""
        invalid_names = [
            "staging_collection",  # No timestamp
            "staging_collection_test",  # Missing timestamp
            "staging_collection_test_invalid",  # Invalid timestamp format
            "staging_test_20240101",  # Incomplete timestamp
        ]

        for name in invalid_names:
            # Should return True (consider old) for invalid formats
            assert qdrant_manager._is_staging_collection_old(name) is True

    def test_rename_collection_warning(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test rename collection creates new collection but warns about data migration"""
        # Create proper mock structure for CollectionInfo
        mock_info = Mock(spec=CollectionInfo)
        mock_config = Mock()
        mock_params = Mock()
        mock_vectors = Mock(size=768, distance=Distance.COSINE)

        # Set up the nested structure
        mock_params.vectors = mock_vectors
        mock_config.params = mock_params
        mock_config.optimizer_config = {"indexing_threshold": 20000}
        mock_info.config = mock_config

        mock_qdrant_client.get_collection.return_value = mock_info

        # Test rename operation
        qdrant_manager.rename_collection("old_collection", "new_collection")

        # Should create new collection with same config
        mock_qdrant_client.create_collection.assert_called_once()
        call_args = mock_qdrant_client.create_collection.call_args
        assert call_args.kwargs["collection_name"] == "new_collection"

        # Should not delete old collection (data migration not implemented)
        mock_qdrant_client.delete_collection.assert_not_called()

    def test_sleep_calls_in_cleanup(self, qdrant_manager, mock_qdrant_client) -> None:
        """Test that cleanup includes delays between deletions"""
        mock_collections = MagicMock()

        # Create mock collection objects with name attribute
        col1 = Mock()
        col1.name = "orphaned1"
        col2 = Mock()
        col2.name = "orphaned2"

        mock_collections.collections = [col1, col2]
        mock_qdrant_client.get_collections.return_value = mock_collections

        mock_info = Mock(spec=CollectionInfo)
        mock_info.vectors_count = 10
        mock_qdrant_client.get_collection.return_value = mock_info

        with patch("packages.shared.managers.qdrant_manager.time.sleep") as mock_sleep:
            qdrant_manager.cleanup_orphaned_collections(active_collections=[])

            # Should have small delays between deletions
            assert mock_sleep.call_count == 2
            for call in mock_sleep.call_args_list:
                assert call[0][0] == 0.1  # 100ms delay
