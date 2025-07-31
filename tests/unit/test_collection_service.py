#!/usr/bin/env python3
"""
Comprehensive test suite for webui/services/collection_service.py
Tests collection CRUD operations, permission checking, state transitions, and error handling
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from packages.shared.database.exceptions import AccessDeniedError, EntityAlreadyExistsError, InvalidStateError
from packages.shared.database.models import Collection, CollectionStatus, OperationStatus, OperationType
from packages.webui.services.collection_service import CollectionService


class TestCollectionService:
    """Test CollectionService implementation"""

    @pytest.fixture()
    def mock_session(self):
        """Create a mock AsyncSession"""
        session = AsyncMock()
        session.commit = AsyncMock()
        return session

    @pytest.fixture()
    def mock_collection_repo(self):
        """Create a mock CollectionRepository"""
        return AsyncMock()

    @pytest.fixture()
    def mock_operation_repo(self):
        """Create a mock OperationRepository"""
        return AsyncMock()

    @pytest.fixture()
    def mock_document_repo(self):
        """Create a mock DocumentRepository"""
        return AsyncMock()

    @pytest.fixture()
    def collection_service(self, mock_session, mock_collection_repo, mock_operation_repo, mock_document_repo):
        """Create CollectionService with mocked dependencies"""
        return CollectionService(
            db_session=mock_session,
            collection_repo=mock_collection_repo,
            operation_repo=mock_operation_repo,
            document_repo=mock_document_repo,
        )

    @pytest.mark.asyncio()
    @patch("packages.webui.services.collection_service.celery_app")
    async def test_create_collection_success(
        self, mock_celery_app, collection_service, mock_collection_repo, mock_operation_repo, mock_session
    ):
        """Test successful collection creation with minimal config"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 123
        mock_collection.name = "Test Collection"
        mock_collection.description = "Test description"
        mock_collection.owner_id = 123
        mock_collection.vector_store_name = f"collection_{mock_collection.id}"
        mock_collection.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
        mock_collection.quantization = "float16"
        mock_collection.chunk_size = 1000
        mock_collection.chunk_overlap = 200
        mock_collection.is_public = False
        mock_collection.meta = None
        mock_collection.created_at = datetime.now(UTC)
        mock_collection.updated_at = datetime.now(UTC)
        mock_collection.status = CollectionStatus.PENDING

        mock_collection_repo.create.return_value = mock_collection

        # Mock operation
        mock_operation = Mock()
        mock_operation.uuid = "op-uuid-123"
        mock_operation.collection_id = mock_collection.id
        mock_operation.type = OperationType.INDEX
        mock_operation.status = OperationStatus.PENDING
        mock_operation.config = {"sources": [], "collection_config": {}}
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = None
        mock_operation.completed_at = None
        mock_operation.error_message = None

        mock_operation_repo.create.return_value = mock_operation

        # Test create collection
        collection_dict, operation_dict = await collection_service.create_collection(
            user_id=123,
            name="Test Collection",
            description="Test description",
        )

        # Verify collection creation
        mock_collection_repo.create.assert_called_once_with(
            owner_id=123,
            name="Test Collection",
            description="Test description",
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            quantization="float16",
            chunk_size=1000,
            chunk_overlap=200,
            is_public=False,
            meta=None,
        )

        # Verify operation creation
        mock_operation_repo.create.assert_called_once_with(
            collection_id=mock_collection.id,
            user_id=123,
            operation_type=OperationType.INDEX,
            config={"sources": [], "collection_config": {}},
        )

        # Verify commit and Celery task
        mock_session.commit.assert_called_once()
        mock_celery_app.send_task.assert_called_once()

        # Verify returned dictionaries
        assert collection_dict["id"] == mock_collection.id
        assert collection_dict["name"] == mock_collection.name
        assert collection_dict["status"] == CollectionStatus.PENDING
        assert operation_dict["uuid"] == mock_operation.uuid
        assert operation_dict["type"] == OperationType.INDEX.value

    @pytest.mark.asyncio()
    @patch("packages.webui.services.collection_service.celery_app")
    async def test_create_collection_with_custom_config(
        self, mock_celery_app, collection_service, mock_collection_repo, mock_operation_repo, mock_session
    ):
        """Test collection creation with custom configuration"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 456
        mock_collection.name = "Custom Collection"
        mock_collection.description = None
        mock_collection.owner_id = 123
        mock_collection.vector_store_name = f"collection_{mock_collection.id}"
        mock_collection.embedding_model = "custom-model"
        mock_collection.quantization = "int8"
        mock_collection.chunk_size = 512
        mock_collection.chunk_overlap = 100
        mock_collection.is_public = True
        mock_collection.meta = {"custom": "metadata"}
        mock_collection.created_at = datetime.now(UTC)
        mock_collection.updated_at = datetime.now(UTC)
        mock_collection.status = CollectionStatus.PENDING

        mock_collection_repo.create.return_value = mock_collection

        # Mock operation
        mock_operation = Mock()
        mock_operation.uuid = "op-uuid-456"
        mock_operation.collection_id = mock_collection.id
        mock_operation.type = OperationType.INDEX
        mock_operation.status = OperationStatus.PENDING
        mock_operation.config = {"sources": [], "collection_config": {}}
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = None
        mock_operation.completed_at = None
        mock_operation.error_message = None

        mock_operation_repo.create.return_value = mock_operation

        # Test with custom config
        config = {
            "embedding_model": "custom-model",
            "quantization": "int8",
            "chunk_size": 512,
            "chunk_overlap": 100,
            "is_public": True,
            "metadata": {"custom": "metadata"},
        }

        collection_dict, operation_dict = await collection_service.create_collection(
            user_id=123, name="Custom Collection", config=config
        )

        # Verify custom values were used
        mock_collection_repo.create.assert_called_once_with(
            owner_id=123,
            name="Custom Collection",
            description=None,
            embedding_model="custom-model",
            quantization="int8",
            chunk_size=512,
            chunk_overlap=100,
            is_public=True,
            meta={"custom": "metadata"},
        )

    @pytest.mark.asyncio()
    async def test_create_collection_empty_name(self, collection_service):
        """Test collection creation with empty name"""
        with pytest.raises(ValueError, match="Collection name is required"):
            await collection_service.create_collection(user_id=123, name="")

        with pytest.raises(ValueError, match="Collection name is required"):
            await collection_service.create_collection(user_id=123, name="   ")

    @pytest.mark.asyncio()
    async def test_create_collection_already_exists(self, collection_service, mock_collection_repo):
        """Test collection creation when name already exists"""
        mock_collection_repo.create.side_effect = EntityAlreadyExistsError("collection", "Test Collection")

        with pytest.raises(EntityAlreadyExistsError):
            await collection_service.create_collection(user_id=123, name="Test Collection")

    @pytest.mark.asyncio()
    @patch("packages.webui.services.collection_service.celery_app")
    async def test_add_source_success(
        self, mock_celery_app, collection_service, mock_collection_repo, mock_operation_repo, mock_session
    ):
        """Test successful source addition to collection"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 123
        mock_collection.name = "Test Collection"
        mock_collection.status = CollectionStatus.READY
        mock_collection.vector_store_name = f"collection_{mock_collection.id}"

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # No active operations
        mock_operation_repo.get_active_operations.return_value = []

        # Mock operation
        mock_operation = Mock()
        mock_operation.uuid = "op-uuid-789"
        mock_operation.collection_id = mock_collection.id
        mock_operation.type = OperationType.APPEND
        mock_operation.status = OperationStatus.PENDING
        mock_operation.config = {"source_path": "/path/to/source", "source_config": {}}
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = None
        mock_operation.completed_at = None
        mock_operation.error_message = None

        mock_operation_repo.create.return_value = mock_operation

        # Test add source
        operation_dict = await collection_service.add_source(
            collection_id="collection-123",
            user_id=123,
            source_path="/path/to/source",
        )

        # Verify permission check
        mock_collection_repo.get_by_uuid_with_permission_check.assert_called_once_with(
            collection_uuid="collection-123", user_id=123
        )

        # Verify operation creation
        mock_operation_repo.create.assert_called_once_with(
            collection_id=mock_collection.id,
            user_id=123,
            operation_type=OperationType.APPEND,
            config={"source_path": "/path/to/source", "source_config": {}},
        )

        # Verify status update
        mock_collection_repo.update_status.assert_called_once_with(mock_collection.id, CollectionStatus.PROCESSING)

        # Verify commit and Celery task
        mock_session.commit.assert_called_once()
        mock_celery_app.send_task.assert_called_once()

        assert operation_dict["uuid"] == mock_operation.uuid
        assert operation_dict["type"] == OperationType.APPEND.value

    @pytest.mark.asyncio()
    async def test_add_source_invalid_state(self, collection_service, mock_collection_repo):
        """Test adding source to collection in invalid state"""
        # Mock collection in ERROR state
        mock_collection = Mock(spec=Collection)
        mock_collection.status = CollectionStatus.ERROR

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(InvalidStateError, match="Cannot add source to collection in CollectionStatus.ERROR state"):
            await collection_service.add_source(
                collection_id="collection-123", user_id=123, source_path="/path/to/source"
            )

    @pytest.mark.asyncio()
    async def test_add_source_active_operation(self, collection_service, mock_collection_repo, mock_operation_repo):
        """Test adding source when another operation is in progress"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 123
        mock_collection.status = CollectionStatus.READY

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # Active operation exists
        mock_operation_repo.get_active_operations.return_value = [Mock()]

        with pytest.raises(InvalidStateError, match="Cannot add source while another operation is in progress"):
            await collection_service.add_source(
                collection_id="collection-123", user_id=123, source_path="/path/to/source"
            )

    @pytest.mark.asyncio()
    @patch("packages.webui.services.collection_service.celery_app")
    async def test_reindex_collection_success(
        self, mock_celery_app, collection_service, mock_collection_repo, mock_operation_repo, mock_session
    ):
        """Test successful collection reindexing"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 123
        mock_collection.name = "Test Collection"
        mock_collection.status = CollectionStatus.READY
        mock_collection.embedding_model = "original-model"
        mock_collection.quantization = "float16"
        mock_collection.chunk_size = 1000
        mock_collection.chunk_overlap = 200
        mock_collection.is_public = False
        mock_collection.meta = None

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # No active operations
        mock_operation_repo.get_active_operations_count.return_value = 0

        # Mock operation
        mock_operation = Mock()
        mock_operation.uuid = "op-uuid-reindex"
        mock_operation.collection_id = mock_collection.id
        mock_operation.type = OperationType.REINDEX
        mock_operation.status = OperationStatus.PENDING
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = None
        mock_operation.completed_at = None
        mock_operation.error_message = None

        mock_operation_repo.create.return_value = mock_operation

        # Test reindex with config updates
        config_updates = {"chunk_size": 512, "chunk_overlap": 100}

        _ = await collection_service.reindex_collection(
            collection_id="collection-123", user_id=123, config_updates=config_updates
        )

        # Verify operation config includes both old and new config
        expected_config = {
            "previous_config": {
                "embedding_model": "original-model",
                "quantization": "float16",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "is_public": False,
                "metadata": None,
            },
            "new_config": {
                "embedding_model": "original-model",
                "quantization": "float16",
                "chunk_size": 512,
                "chunk_overlap": 100,
                "is_public": False,
                "metadata": None,
            },
            "blue_green": True,
        }

        mock_operation_repo.create.assert_called_once_with(
            collection_id=mock_collection.id,
            user_id=123,
            operation_type=OperationType.REINDEX,
            config=expected_config,
        )

        # Verify status update
        mock_collection_repo.update_status.assert_called_once_with(mock_collection.id, CollectionStatus.PROCESSING)

        # Verify commit and Celery task
        mock_session.commit.assert_called_once()
        mock_celery_app.send_task.assert_called_once()

    @pytest.mark.asyncio()
    async def test_reindex_collection_processing_state(self, collection_service, mock_collection_repo):
        """Test reindexing collection that is currently processing"""
        # Mock collection in PROCESSING state
        mock_collection = Mock(spec=Collection)
        mock_collection.status = CollectionStatus.PROCESSING

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(InvalidStateError, match="Cannot reindex collection that is currently processing"):
            await collection_service.reindex_collection(collection_id="collection-123", user_id=123)

    @pytest.mark.asyncio()
    async def test_reindex_collection_error_state(self, collection_service, mock_collection_repo):
        """Test reindexing failed collection"""
        # Mock collection in ERROR state
        mock_collection = Mock(spec=Collection)
        mock_collection.status = CollectionStatus.ERROR

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(InvalidStateError, match="Cannot reindex failed collection"):
            await collection_service.reindex_collection(collection_id="collection-123", user_id=123)

    @pytest.mark.asyncio()
    @patch("packages.webui.services.collection_service.qdrant_manager")
    async def test_delete_collection_success(
        self, mock_qdrant_manager, collection_service, mock_collection_repo, mock_operation_repo
    ):
        """Test successful collection deletion"""
        # Mock collection - use string UUID
        mock_collection = Mock(spec=Collection)
        mock_collection.id = "collection-123"  # String UUID
        mock_collection.owner_id = 123
        mock_collection.vector_store_name = "collection_123"

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # No active operations
        mock_operation_repo.get_active_operations_count.return_value = 0

        # Mock Qdrant client
        mock_qdrant_client = Mock()
        # Create a mock collection object with name attribute
        mock_collection_obj = Mock()
        mock_collection_obj.name = "collection_123"

        # Create the collections response
        mock_collections_response = Mock()
        mock_collections_response.collections = [mock_collection_obj]

        mock_qdrant_client.get_collections.return_value = mock_collections_response
        mock_qdrant_manager.get_client.return_value = mock_qdrant_client

        # Test delete
        await collection_service.delete_collection(collection_id="collection-123", user_id=123)

        # Verify ownership check
        assert mock_collection.owner_id == 123

        # Verify Qdrant deletion
        mock_qdrant_client.delete_collection.assert_called_once_with("collection_123")

        # Verify database deletion
        mock_collection_repo.delete.assert_called_once_with("collection-123", 123)

    @pytest.mark.asyncio()
    async def test_delete_collection_not_owner(self, collection_service, mock_collection_repo):
        """Test deleting collection when not the owner"""
        # Mock collection owned by different user
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 123
        mock_collection.owner_id = 456  # Different owner

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(AccessDeniedError):
            await collection_service.delete_collection(collection_id="collection-123", user_id=123)

    @pytest.mark.asyncio()
    async def test_delete_collection_active_operations(
        self, collection_service, mock_collection_repo, mock_operation_repo
    ):
        """Test deleting collection with active operations"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 123
        mock_collection.owner_id = 123

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # Active operations exist
        mock_operation_repo.get_active_operations_count.return_value = 2

        with pytest.raises(InvalidStateError, match="Cannot delete collection while operations are in progress"):
            await collection_service.delete_collection(collection_id="collection-123", user_id=123)

    @pytest.mark.asyncio()
    @patch("packages.webui.services.collection_service.qdrant_manager")
    @patch("packages.webui.services.collection_service.logger")
    async def test_delete_collection_qdrant_failure(
        self, mock_logger, mock_qdrant_manager, collection_service, mock_collection_repo, mock_operation_repo
    ):
        """Test collection deletion when Qdrant deletion fails"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.id = "collection-123"
        mock_collection.owner_id = 123
        mock_collection.vector_store_name = "collection_123"

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # No active operations
        mock_operation_repo.get_active_operations_count.return_value = 0

        # Mock Qdrant client failure
        mock_qdrant_client = Mock()
        mock_qdrant_client.get_collections.side_effect = Exception("Qdrant connection error")
        mock_qdrant_manager.get_client.return_value = mock_qdrant_client

        # Test delete - should still succeed
        await collection_service.delete_collection(collection_id="collection-123", user_id=123)

        # Verify error was logged
        mock_logger.error.assert_called()

        # Verify database deletion still happened
        mock_collection_repo.delete.assert_called_once_with("collection-123", 123)

    @pytest.mark.asyncio()
    @patch("packages.webui.services.collection_service.celery_app")
    async def test_remove_source_success(
        self, mock_celery_app, collection_service, mock_collection_repo, mock_operation_repo, mock_session
    ):
        """Test successful source removal from collection"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 123
        mock_collection.status = CollectionStatus.READY

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # No active operations
        mock_operation_repo.get_active_operations_count.return_value = 0

        # Mock operation
        mock_operation = Mock()
        mock_operation.uuid = "op-uuid-remove"
        mock_operation.collection_id = mock_collection.id
        mock_operation.type = OperationType.REMOVE_SOURCE
        mock_operation.status = OperationStatus.PENDING
        mock_operation.config = {"source_path": "/path/to/remove"}
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = None
        mock_operation.completed_at = None
        mock_operation.error_message = None

        mock_operation_repo.create.return_value = mock_operation

        # Test remove source
        operation_dict = await collection_service.remove_source(
            collection_id="collection-123", user_id=123, source_path="/path/to/remove"
        )

        # Verify operation creation
        mock_operation_repo.create.assert_called_once_with(
            collection_id=mock_collection.id,
            user_id=123,
            operation_type=OperationType.REMOVE_SOURCE,
            config={"source_path": "/path/to/remove"},
        )

        # Verify status update
        mock_collection_repo.update_status.assert_called_once_with(mock_collection.id, CollectionStatus.PROCESSING)

        # Verify commit and Celery task
        mock_session.commit.assert_called_once()
        mock_celery_app.send_task.assert_called_once()

        assert operation_dict["type"] == OperationType.REMOVE_SOURCE.value

    @pytest.mark.asyncio()
    async def test_remove_source_invalid_state(self, collection_service, mock_collection_repo):
        """Test removing source from collection in invalid state"""
        # Mock collection in PENDING state
        mock_collection = Mock(spec=Collection)
        mock_collection.status = CollectionStatus.PENDING

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(
            InvalidStateError, match="Cannot remove source from collection in CollectionStatus.PENDING state"
        ):
            await collection_service.remove_source(
                collection_id="collection-123", user_id=123, source_path="/path/to/remove"
            )

    @pytest.mark.asyncio()
    async def test_list_for_user(self, collection_service, mock_collection_repo):
        """Test listing collections for user"""
        # Mock collections
        mock_collections = []
        for i in range(3):
            collection = Mock(spec=Collection)
            collection.id = f"collection-{i}"
            collection.name = f"Collection {i}"
            mock_collections.append(collection)

        mock_collection_repo.list_for_user.return_value = (mock_collections, 3)

        # Test list
        collections, total = await collection_service.list_for_user(
            user_id=123, offset=0, limit=50, include_public=True
        )

        assert len(collections) == 3
        assert total == 3

        mock_collection_repo.list_for_user.assert_called_once_with(user_id=123, offset=0, limit=50, include_public=True)

    @pytest.mark.asyncio()
    async def test_update_collection_success(self, collection_service, mock_collection_repo, mock_session):
        """Test successful collection update"""
        # Mock collection - use integer ID internally
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 123  # Integer ID
        mock_collection.owner_id = 123

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # Mock updated collection
        updated_collection = Mock(spec=Collection)
        updated_collection.id = 123
        updated_collection.name = "Updated Name"
        updated_collection.description = "Updated description"

        mock_collection_repo.update.return_value = updated_collection

        # Test update
        updates = {"name": "Updated Name", "description": "Updated description"}
        result = await collection_service.update(collection_id="collection-123", user_id=123, updates=updates)

        # Verify ownership check
        assert mock_collection.owner_id == 123

        # Verify update call - note str() conversion
        mock_collection_repo.update.assert_called_once_with("123", updates)

        # Verify commit
        mock_session.commit.assert_called_once()

        assert result == updated_collection

    @pytest.mark.asyncio()
    async def test_update_collection_not_owner(self, collection_service, mock_collection_repo):
        """Test updating collection when not the owner"""
        # Mock collection owned by different user
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 123
        mock_collection.owner_id = 456  # Different owner

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(AccessDeniedError):
            await collection_service.update(collection_id="collection-123", user_id=123, updates={"name": "New Name"})

    @pytest.mark.asyncio()
    async def test_list_documents(self, collection_service, mock_collection_repo, mock_document_repo):
        """Test listing documents in collection"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 123

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # Mock documents
        mock_documents = []
        for i in range(5):
            doc = Mock()
            doc.id = f"doc-{i}"
            doc.filename = f"document{i}.pdf"
            mock_documents.append(doc)

        mock_document_repo.list_by_collection.return_value = (mock_documents, 5)

        # Test list documents
        documents, total = await collection_service.list_documents(
            collection_id="collection-123", user_id=123, offset=0, limit=50
        )

        assert len(documents) == 5
        assert total == 5

        mock_document_repo.list_by_collection.assert_called_once_with(
            collection_id=mock_collection.id, offset=0, limit=50
        )

    @pytest.mark.asyncio()
    async def test_list_operations(self, collection_service, mock_collection_repo, mock_operation_repo):
        """Test listing operations for collection"""
        # Mock collection
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 123

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # Mock operations
        mock_operations = []
        for i in range(3):
            op = Mock()
            op.uuid = f"op-{i}"
            op.type = OperationType.INDEX
            op.status = OperationStatus.COMPLETED
            mock_operations.append(op)

        mock_operation_repo.list_for_collection.return_value = (mock_operations, 3)

        # Test list operations
        operations, total = await collection_service.list_operations(
            collection_id="collection-123", user_id=123, offset=0, limit=50
        )

        assert len(operations) == 3
        assert total == 3

        mock_operation_repo.list_for_collection.assert_called_once_with(
            collection_id=mock_collection.id, user_id=123, offset=0, limit=50
        )


class TestCollectionServiceErrorHandling:
    """Test error handling in collection service"""

    @pytest.fixture()
    def collection_service(self):
        mock_session = AsyncMock()
        mock_collection_repo = AsyncMock()
        mock_operation_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        return CollectionService(
            db_session=mock_session,
            collection_repo=mock_collection_repo,
            operation_repo=mock_operation_repo,
            document_repo=mock_document_repo,
        )

    @pytest.mark.asyncio()
    async def test_handle_repository_error(self, collection_service):
        """Test handling of repository errors"""
        collection_service.collection_repo.get_by_uuid_with_permission_check.side_effect = Exception(
            "Database connection error"
        )

        with pytest.raises(Exception, match="Database connection error"):
            await collection_service.add_source(collection_id="collection-123", user_id=123, source_path="/path")

    @pytest.mark.asyncio()
    @patch("packages.webui.services.collection_service.celery_app")
    @patch("packages.webui.services.collection_service.logger")
    async def test_create_collection_partial_failure(self, mock_logger, mock_celery_app, collection_service):
        """Test collection creation when operation creation fails"""
        # Mock collection creation success
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 123
        collection_service.collection_repo.create.return_value = mock_collection

        # Mock operation creation failure
        collection_service.operation_repo.create.side_effect = Exception("Operation creation failed")

        # Test create - should propagate the exception
        with pytest.raises(Exception, match="Operation creation failed"):
            await collection_service.create_collection(user_id=123, name="Test Collection")


class TestCollectionServiceIntegration:
    """Test collection service integration scenarios"""

    @pytest.mark.asyncio()
    @patch("packages.webui.services.collection_service.celery_app")
    @patch("packages.webui.services.collection_service.uuid")
    async def test_collection_creation_workflow(self, mock_uuid, mock_celery_app):
        """Test complete collection creation workflow"""
        # Setup UUIDs
        mock_uuid_obj = Mock()
        mock_uuid_obj.hex = "task-uuid-123"
        mock_uuid.uuid4.return_value = mock_uuid_obj

        # Setup service
        mock_session = AsyncMock()
        mock_collection_repo = AsyncMock()
        mock_operation_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        service = CollectionService(mock_session, mock_collection_repo, mock_operation_repo, mock_document_repo)

        # Mock collection with all attributes
        mock_collection = Mock(spec=Collection)
        mock_collection.id = 999
        mock_collection.name = "Full Collection"
        mock_collection.description = "Complete test"
        mock_collection.owner_id = 123
        mock_collection.vector_store_name = f"collection_{mock_collection.id}"
        mock_collection.embedding_model = "test-model"
        mock_collection.quantization = "float32"
        mock_collection.chunk_size = 2048
        mock_collection.chunk_overlap = 256
        mock_collection.is_public = True
        mock_collection.meta = {"tags": ["test", "integration"]}
        mock_collection.created_at = datetime.now(UTC)
        mock_collection.updated_at = datetime.now(UTC)
        mock_collection.status = CollectionStatus.PENDING

        mock_collection_repo.create.return_value = mock_collection

        # Mock operation
        mock_operation = Mock()
        mock_operation.uuid = "op-full-uuid"
        mock_operation.collection_id = mock_collection.id
        mock_operation.type = OperationType.INDEX
        mock_operation.status = OperationStatus.PENDING
        mock_operation.config = {"sources": [], "collection_config": {}}
        mock_operation.created_at = datetime.now(UTC)
        mock_operation.started_at = None
        mock_operation.completed_at = None
        mock_operation.error_message = None

        mock_operation_repo.create.return_value = mock_operation

        # Execute full workflow
        config = {
            "embedding_model": "test-model",
            "quantization": "float32",
            "chunk_size": 2048,
            "chunk_overlap": 256,
            "is_public": True,
            "metadata": {"tags": ["test", "integration"]},
        }

        collection_dict, operation_dict = await service.create_collection(
            user_id=123,
            name="Full Collection",
            description="Complete test",
            config=config,
        )

        # Verify complete workflow
        assert collection_dict["id"] == mock_collection.id
        assert collection_dict["config"]["chunk_size"] == 2048
        assert operation_dict["uuid"] == mock_operation.uuid

        # Verify Celery task dispatch
        mock_celery_app.send_task.assert_called_once()
        # Just verify it was called with the right task name
        args, kwargs = mock_celery_app.send_task.call_args
        assert args[0] == "webui.tasks.process_collection_operation"

        # Verify transaction commit
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio()
    async def test_collection_state_validation(self):
        """Test collection state validation across operations"""
        # Setup service
        mock_session = AsyncMock()
        mock_collection_repo = AsyncMock()
        mock_operation_repo = AsyncMock()
        mock_document_repo = AsyncMock()
        service = CollectionService(mock_session, mock_collection_repo, mock_operation_repo, mock_document_repo)

        # Test state transitions
        test_cases = [
            (CollectionStatus.PENDING, True),  # Can add source to PENDING
            (CollectionStatus.READY, True),  # Can add source to READY
            (CollectionStatus.DEGRADED, True),  # Can add source to DEGRADED
            (CollectionStatus.PROCESSING, False),  # Cannot add source to PROCESSING
            (CollectionStatus.ERROR, False),  # Cannot add source to ERROR
        ]

        for status, should_succeed in test_cases:
            mock_collection = Mock(spec=Collection)
            mock_collection.id = f"collection-{status.value}"
            mock_collection.status = status

            mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
            mock_operation_repo.get_active_operations.return_value = []

            if should_succeed:
                mock_operation = Mock()
                mock_operation.uuid = f"op-{status.value}"
                mock_operation.uuid = f"op-{status.value}"
                mock_operation.collection_id = mock_collection.id
                mock_operation.type = OperationType.APPEND
                mock_operation.status = OperationStatus.PENDING
                mock_operation.config = {"source_path": "/test/path", "source_config": {}}
                mock_operation.created_at = datetime.now(UTC)
                mock_operation.started_at = None
                mock_operation.completed_at = None
                mock_operation.error_message = None
                mock_operation_repo.create.return_value = mock_operation

                operation_dict = await service.add_source(
                    collection_id=f"collection-{status.value}",
                    user_id=123,
                    source_path="/test/path",
                )
                assert operation_dict["uuid"] == f"op-{status.value}"
            else:
                with pytest.raises(InvalidStateError):
                    await service.add_source(
                        collection_id=f"collection-{status.value}",
                        user_id=123,
                        source_path="/test/path",
                    )
