"""
Comprehensive tests for CollectionService covering all methods and edge cases.
"""

import uuid
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.exceptions import (
    AccessDeniedError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidStateError,
)
from packages.shared.database.models import Collection, CollectionStatus, Operation, OperationType
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.shared.database.repositories.operation_repository import OperationRepository
from packages.webui.services.collection_service import CollectionService

# Fixtures are now imported from conftest.py


@pytest.fixture()
def collection_service(
    mock_db_session: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_document_repo: AsyncMock,
) -> CollectionService:
    """Create a CollectionService instance with mocked dependencies."""
    return CollectionService(
        db_session=mock_db_session,
        collection_repo=mock_collection_repo,
        operation_repo=mock_operation_repo,
        document_repo=mock_document_repo,
    )


# Collection and operation fixtures are now imported from conftest.py


class TestCollectionServiceInit:
    """Test CollectionService initialization."""

    def test_init(
        self,
        mock_db_session: AsyncMock,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_document_repo: AsyncMock,
    ) -> None:
        """Test service initialization."""
        service = CollectionService(
            db_session=mock_db_session,
            collection_repo=mock_collection_repo,
            operation_repo=mock_operation_repo,
            document_repo=mock_document_repo,
        )

        assert service.db_session == mock_db_session
        assert service.collection_repo == mock_collection_repo
        assert service.operation_repo == mock_operation_repo
        assert service.document_repo == mock_document_repo


class TestCreateCollection:
    """Test create_collection method."""

    @pytest.mark.asyncio()
    async def test_create_collection_success(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test successful collection creation."""
        mock_collection_repo.create.return_value = mock_collection
        mock_operation_repo.create.return_value = mock_operation

        with patch("packages.webui.celery_app.celery_app.send_task") as mock_send_task:
            collection_dict, operation_dict = await collection_service.create_collection(
                user_id=1,
                name="Test Collection",
                description="Test description",
                config={
                    "embedding_model": "custom-model",
                    "quantization": "int8",
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "is_public": True,
                    "metadata": {"custom": "data"},
                },
            )

        # Verify repository calls
        mock_collection_repo.create.assert_called_once_with(
            owner_id=1,
            name="Test Collection",
            description="Test description",
            embedding_model="custom-model",
            quantization="int8",
            chunk_size=500,
            chunk_overlap=100,
            is_public=True,
            meta={"custom": "data"},
        )

        mock_operation_repo.create.assert_called_once_with(
            collection_id=mock_collection.id,
            user_id=1,
            operation_type=OperationType.INDEX,
            config={
                "sources": [],
                "collection_config": {
                    "embedding_model": "custom-model",
                    "quantization": "int8",
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "is_public": True,
                    "metadata": {"custom": "data"},
                },
            },
        )

        # Verify commit before task dispatch
        mock_db_session.commit.assert_called_once()

        # Verify Celery task dispatch
        mock_send_task.assert_called_once_with(
            "webui.tasks.process_collection_operation",
            args=[mock_operation.uuid],
            task_id=ANY,
        )

        # Verify return values
        assert collection_dict["id"] == mock_collection.id
        assert collection_dict["name"] == mock_collection.name
        assert collection_dict["description"] == mock_collection.description
        assert collection_dict["owner_id"] == mock_collection.owner_id
        assert collection_dict["embedding_model"] == mock_collection.embedding_model
        assert collection_dict["document_count"] == 0
        assert collection_dict["vector_count"] == 0

        assert operation_dict["uuid"] == mock_operation.uuid
        assert operation_dict["collection_id"] == mock_operation.collection_id
        assert operation_dict["type"] == mock_operation.type.value

    @pytest.mark.asyncio()
    async def test_create_collection_with_defaults(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test collection creation with default values."""
        mock_collection_repo.create.return_value = mock_collection
        mock_operation_repo.create.return_value = mock_operation

        with patch("packages.webui.celery_app.celery_app.send_task"):
            await collection_service.create_collection(
                user_id=1,
                name="Test Collection",
            )

        # Verify default values were used
        mock_collection_repo.create.assert_called_once_with(
            owner_id=1,
            name="Test Collection",
            description=None,
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            quantization="float16",
            chunk_size=1000,
            chunk_overlap=200,
            is_public=False,
            meta=None,
        )

    @pytest.mark.asyncio()
    async def test_create_collection_empty_name(
        self,
        collection_service: CollectionService,
    ) -> None:
        """Test collection creation with empty name."""
        with pytest.raises(ValueError) as exc_info:
            await collection_service.create_collection(
                user_id=1,
                name="",
            )

        assert "Collection name is required" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_create_collection_whitespace_name(
        self,
        collection_service: CollectionService,
    ) -> None:
        """Test collection creation with whitespace-only name."""
        with pytest.raises(ValueError) as exc_info:
            await collection_service.create_collection(
                user_id=1,
                name="   ",
            )

        assert "Collection name is required" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_create_collection_already_exists(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
    ) -> None:
        """Test collection creation when name already exists."""
        mock_collection_repo.create.side_effect = EntityAlreadyExistsError("Collection", "Existing Collection")

        with pytest.raises(EntityAlreadyExistsError):
            await collection_service.create_collection(
                user_id=1,
                name="Existing Collection",
            )

    @pytest.mark.asyncio()
    async def test_create_collection_database_error(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
    ) -> None:
        """Test collection creation with database error."""
        mock_collection_repo.create.side_effect = Exception("Database error")

        with pytest.raises(Exception) as exc_info:
            await collection_service.create_collection(
                user_id=1,
                name="Test Collection",
            )

        assert "Database error" in str(exc_info.value)


class TestAddSource:
    """Test add_source method."""

    @pytest.mark.asyncio()
    async def test_add_source_success(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test successful source addition."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.return_value = []
        mock_operation_repo.create.return_value = mock_operation

        with patch("packages.webui.celery_app.celery_app.send_task") as mock_send_task:
            result = await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_path="/path/to/source",
                source_config={"recursive": True},
            )

        # Verify permission check
        mock_collection_repo.get_by_uuid_with_permission_check.assert_called_once_with(
            collection_uuid=str(mock_collection.uuid),
            user_id=1,
        )

        # Verify active operations check
        mock_operation_repo.get_active_operations.assert_called_once_with(mock_collection.id)

        # Verify operation creation
        mock_operation_repo.create.assert_called_once_with(
            collection_id=mock_collection.id,
            user_id=1,
            operation_type=OperationType.APPEND,
            config={
                "source_path": "/path/to/source",
                "source_config": {"recursive": True},
            },
        )

        # Verify status update
        mock_collection_repo.update_status.assert_called_once_with(
            mock_collection.id, CollectionStatus.PROCESSING
        )

        # Verify commit and task dispatch
        mock_db_session.commit.assert_called_once()
        mock_send_task.assert_called_once()

        # Verify return value
        assert result["uuid"] == mock_operation.uuid
        assert result["type"] == mock_operation.type.value

    @pytest.mark.asyncio()
    async def test_add_source_invalid_status(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test adding source to collection in invalid status."""
        mock_collection.status = CollectionStatus.ERROR
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(InvalidStateError) as exc_info:
            await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_path="/path/to/source",
            )

        assert "Cannot add source to collection in" in str(exc_info.value)
        assert "ERROR" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_add_source_active_operation_exists(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test adding source when active operation exists."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.return_value = [MagicMock()]  # Active operation exists

        with pytest.raises(InvalidStateError) as exc_info:
            await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_path="/path/to/source",
            )

        assert "Cannot add source while another operation is in progress" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_add_source_collection_not_found(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
    ) -> None:
        """Test adding source to non-existent collection."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = EntityNotFoundError(
            "Collection", "nonexistent-uuid"
        )

        with pytest.raises(EntityNotFoundError):
            await collection_service.add_source(
                collection_id="nonexistent-uuid",
                user_id=1,
                source_path="/path/to/source",
            )

    @pytest.mark.asyncio()
    async def test_add_source_access_denied(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
    ) -> None:
        """Test adding source without permission."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = AccessDeniedError(
            "2", "Collection", "some-uuid"
        )

        with pytest.raises(AccessDeniedError):
            await collection_service.add_source(
                collection_id="some-uuid",
                user_id=2,
                source_path="/path/to/source",
            )

    @pytest.mark.asyncio()
    async def test_add_source_with_pending_status(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test adding source to pending collection."""
        mock_collection.status = CollectionStatus.PENDING
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.return_value = []
        mock_operation_repo.create.return_value = mock_operation

        with patch("packages.webui.celery_app.celery_app.send_task"):
            result = await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_path="/path/to/source",
            )

        assert result["uuid"] == mock_operation.uuid

    @pytest.mark.asyncio()
    async def test_add_source_with_degraded_status(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test adding source to degraded collection."""
        mock_collection.status = CollectionStatus.DEGRADED
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.return_value = []
        mock_operation_repo.create.return_value = mock_operation

        with patch("packages.webui.celery_app.celery_app.send_task"):
            result = await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_path="/path/to/source",
            )

        assert result["uuid"] == mock_operation.uuid


class TestReindexCollection:
    """Test reindex_collection method."""

    @pytest.mark.asyncio()
    async def test_reindex_collection_success(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test successful collection reindexing."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations_count.return_value = 0
        mock_operation_repo.create.return_value = mock_operation

        with patch("packages.webui.celery_app.celery_app.send_task") as mock_send_task:
            result = await collection_service.reindex_collection(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                config_updates={
                    "embedding_model": "new-model",
                    "chunk_size": 1500,
                },
            )

        # Verify operation creation with config merging
        expected_config = {
            "previous_config": {
                "embedding_model": mock_collection.embedding_model,
                "quantization": mock_collection.quantization,
                "chunk_size": mock_collection.chunk_size,
                "chunk_overlap": mock_collection.chunk_overlap,
                "is_public": mock_collection.is_public,
                "metadata": mock_collection.meta,
            },
            "new_config": {
                "embedding_model": "new-model",
                "quantization": mock_collection.quantization,
                "chunk_size": 1500,
                "chunk_overlap": mock_collection.chunk_overlap,
                "is_public": mock_collection.is_public,
                "metadata": mock_collection.meta,
            },
            "blue_green": True,
        }

        mock_operation_repo.create.assert_called_once_with(
            collection_id=mock_collection.id,
            user_id=1,
            operation_type=OperationType.REINDEX,
            config=expected_config,
        )

        # Verify status update
        mock_collection_repo.update_status.assert_called_once_with(
            mock_collection.id, CollectionStatus.PROCESSING
        )

        # Verify commit and task dispatch
        mock_db_session.commit.assert_called_once()
        mock_send_task.assert_called_once()

        assert result["uuid"] == mock_operation.uuid

    @pytest.mark.asyncio()
    async def test_reindex_collection_no_config_updates(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test reindexing without configuration updates."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations_count.return_value = 0
        mock_operation_repo.create.return_value = mock_operation

        with patch("packages.webui.celery_app.celery_app.send_task"):
            result = await collection_service.reindex_collection(
                collection_id=str(mock_collection.uuid),
                user_id=1,
            )

        # Verify new_config is same as previous_config
        call_args = mock_operation_repo.create.call_args[1]
        assert call_args["config"]["previous_config"] == call_args["config"]["new_config"]

    @pytest.mark.asyncio()
    async def test_reindex_collection_processing_status(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test reindexing collection in processing status."""
        mock_collection.status = CollectionStatus.PROCESSING
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(InvalidStateError) as exc_info:
            await collection_service.reindex_collection(
                collection_id=str(mock_collection.uuid),
                user_id=1,
            )

        assert "Cannot reindex collection that is currently processing" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_reindex_collection_error_status(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test reindexing collection in error status."""
        mock_collection.status = CollectionStatus.ERROR
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(InvalidStateError) as exc_info:
            await collection_service.reindex_collection(
                collection_id=str(mock_collection.uuid),
                user_id=1,
            )

        assert "Cannot reindex failed collection" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_reindex_collection_active_operation_exists(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test reindexing when active operation exists."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations_count.return_value = 1

        with pytest.raises(InvalidStateError) as exc_info:
            await collection_service.reindex_collection(
                collection_id=str(mock_collection.uuid),
                user_id=1,
            )

        assert "Cannot reindex while another operation is in progress" in str(exc_info.value)


class TestDeleteCollection:
    """Test delete_collection method."""

    @pytest.mark.asyncio()
    async def test_delete_collection_success(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test successful collection deletion."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations_count.return_value = 0

        with patch("packages.webui.utils.qdrant_manager.qdrant_manager.get_client") as mock_get_client:
            mock_qdrant_client = MagicMock()
            mock_get_client.return_value = mock_qdrant_client
            mock_collections = MagicMock()
            mock_collections.collections = [
                MagicMock(name=mock_collection.vector_store_name),
                MagicMock(name="other_collection"),
            ]
            mock_qdrant_client.get_collections.return_value = mock_collections

            await collection_service.delete_collection(
                collection_id=str(mock_collection.uuid),
                user_id=mock_collection.owner_id,
            )

        # Verify Qdrant deletion
        mock_qdrant_client.delete_collection.assert_called_once_with(mock_collection.vector_store_name)

        # Verify database deletion
        mock_collection_repo.delete.assert_called_once_with(mock_collection.id, mock_collection.owner_id)

    @pytest.mark.asyncio()
    async def test_delete_collection_not_owner(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test deleting collection by non-owner."""
        mock_collection.owner_id = 1
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(AccessDeniedError) as exc_info:
            await collection_service.delete_collection(
                collection_id=str(mock_collection.uuid),
                user_id=2,  # Different user
            )

        assert "does not have access to Collection" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_delete_collection_active_operations(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test deleting collection with active operations."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations_count.return_value = 1

        with pytest.raises(InvalidStateError) as exc_info:
            await collection_service.delete_collection(
                collection_id=str(mock_collection.uuid),
                user_id=mock_collection.owner_id,
            )

        assert "Cannot delete collection while operations are in progress" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_delete_collection_qdrant_not_found(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test deleting collection when not found in Qdrant."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations_count.return_value = 0

        with patch("packages.webui.utils.qdrant_manager.qdrant_manager.get_client") as mock_get_client:
            mock_qdrant_client = MagicMock()
            mock_get_client.return_value = mock_qdrant_client
            mock_collections = MagicMock()
            mock_collections.collections = []  # No collections in Qdrant
            mock_qdrant_client.get_collections.return_value = mock_collections

            await collection_service.delete_collection(
                collection_id=str(mock_collection.uuid),
                user_id=mock_collection.owner_id,
            )

        # Verify Qdrant deletion was not called
        mock_qdrant_client.delete_collection.assert_not_called()

        # Verify database deletion was still called
        mock_collection_repo.delete.assert_called_once()

    @pytest.mark.asyncio()
    async def test_delete_collection_qdrant_error(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test deleting collection with Qdrant error (continues with DB deletion)."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations_count.return_value = 0

        with patch("packages.webui.utils.qdrant_manager.qdrant_manager.get_client") as mock_get_client:
            mock_qdrant_client = MagicMock()
            mock_get_client.return_value = mock_qdrant_client
            mock_qdrant_client.get_collections.side_effect = Exception("Qdrant error")

            await collection_service.delete_collection(
                collection_id=str(mock_collection.uuid),
                user_id=mock_collection.owner_id,
            )

        # Verify database deletion was still called despite Qdrant error
        mock_collection_repo.delete.assert_called_once()

    @pytest.mark.asyncio()
    async def test_delete_collection_no_vector_store_name(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test deleting collection without vector store name."""
        mock_collection.vector_store_name = None
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations_count.return_value = 0

        with patch("packages.webui.utils.qdrant_manager.qdrant_manager.get_client") as mock_get_client:
            await collection_service.delete_collection(
                collection_id=str(mock_collection.uuid),
                user_id=mock_collection.owner_id,
            )

        # Verify Qdrant client was not even created
        mock_get_client.assert_not_called()

        # Verify database deletion was called
        mock_collection_repo.delete.assert_called_once()


class TestRemoveSource:
    """Test remove_source method."""

    @pytest.mark.asyncio()
    async def test_remove_source_success(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test successful source removal."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations_count.return_value = 0
        mock_operation_repo.create.return_value = mock_operation

        with patch("packages.webui.celery_app.celery_app.send_task") as mock_send_task:
            result = await collection_service.remove_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_path="/path/to/remove",
            )

        # Verify operation creation
        mock_operation_repo.create.assert_called_once_with(
            collection_id=mock_collection.id,
            user_id=1,
            operation_type=OperationType.REMOVE_SOURCE,
            config={
                "source_path": "/path/to/remove",
            },
        )

        # Verify status update
        mock_collection_repo.update_status.assert_called_once_with(
            mock_collection.id, CollectionStatus.PROCESSING
        )

        # Verify commit and task dispatch
        mock_db_session.commit.assert_called_once()
        mock_send_task.assert_called_once()

        assert result["uuid"] == mock_operation.uuid

    @pytest.mark.asyncio()
    async def test_remove_source_invalid_status(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test removing source from collection in invalid status."""
        mock_collection.status = CollectionStatus.PENDING
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(InvalidStateError) as exc_info:
            await collection_service.remove_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_path="/path/to/remove",
            )

        assert "Cannot remove source from collection in" in str(exc_info.value)
        assert "PENDING" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_remove_source_degraded_status(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test removing source from degraded collection."""
        mock_collection.status = CollectionStatus.DEGRADED
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations_count.return_value = 0
        mock_operation_repo.create.return_value = mock_operation

        with patch("packages.webui.celery_app.celery_app.send_task"):
            result = await collection_service.remove_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_path="/path/to/remove",
            )

        assert result["uuid"] == mock_operation.uuid

    @pytest.mark.asyncio()
    async def test_remove_source_active_operations(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test removing source with active operations."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations_count.return_value = 1

        with pytest.raises(InvalidStateError) as exc_info:
            await collection_service.remove_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_path="/path/to/remove",
            )

        assert "Cannot remove source while another operation is in progress" in str(exc_info.value)


class TestListForUser:
    """Test list_for_user method."""

    @pytest.mark.asyncio()
    async def test_list_for_user_success(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
    ) -> None:
        """Test listing collections for user."""
        mock_collections = [MagicMock(), MagicMock()]
        mock_collection_repo.list_for_user.return_value = (mock_collections, 2)

        collections, total = await collection_service.list_for_user(
            user_id=1,
            offset=0,
            limit=50,
            include_public=True,
        )

        mock_collection_repo.list_for_user.assert_called_once_with(
            user_id=1,
            offset=0,
            limit=50,
            include_public=True,
        )

        assert collections == mock_collections
        assert total == 2

    @pytest.mark.asyncio()
    async def test_list_for_user_with_pagination(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
    ) -> None:
        """Test listing collections with pagination."""
        mock_collections = [MagicMock()]
        mock_collection_repo.list_for_user.return_value = (mock_collections, 100)

        collections, total = await collection_service.list_for_user(
            user_id=1,
            offset=50,
            limit=10,
            include_public=False,
        )

        mock_collection_repo.list_for_user.assert_called_once_with(
            user_id=1,
            offset=50,
            limit=10,
            include_public=False,
        )

        assert len(collections) == 1
        assert total == 100


class TestUpdate:
    """Test update method."""

    @pytest.mark.asyncio()
    async def test_update_success(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test successful collection update."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        updated_collection = MagicMock()
        mock_collection_repo.update.return_value = updated_collection

        result = await collection_service.update(
            collection_id=str(mock_collection.uuid),
            user_id=mock_collection.owner_id,
            updates={
                "name": "Updated Name",
                "description": "Updated description",
                "is_public": True,
            },
        )

        # Verify permission check
        mock_collection_repo.get_by_uuid_with_permission_check.assert_called_once_with(
            collection_uuid=str(mock_collection.uuid),
            user_id=mock_collection.owner_id,
        )

        # Verify update call
        mock_collection_repo.update.assert_called_once_with(
            str(mock_collection.id),
            {
                "name": "Updated Name",
                "description": "Updated description",
                "is_public": True,
            },
        )

        # Verify commit
        mock_db_session.commit.assert_called_once()

        assert result == updated_collection

    @pytest.mark.asyncio()
    async def test_update_not_owner(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test updating collection by non-owner."""
        mock_collection.owner_id = 1
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(AccessDeniedError) as exc_info:
            await collection_service.update(
                collection_id=str(mock_collection.uuid),
                user_id=2,  # Different user
                updates={"name": "New Name"},
            )

        assert "does not have access to Collection" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_update_collection_not_found(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
    ) -> None:
        """Test updating non-existent collection."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = EntityNotFoundError(
            "Collection", "nonexistent-uuid"
        )

        with pytest.raises(EntityNotFoundError):
            await collection_service.update(
                collection_id="nonexistent-uuid",
                user_id=1,
                updates={"name": "New Name"},
            )

    @pytest.mark.asyncio()
    async def test_update_already_exists(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test updating collection with name that already exists."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_collection_repo.update.side_effect = EntityAlreadyExistsError("Collection", "Existing Name")

        with pytest.raises(EntityAlreadyExistsError):
            await collection_service.update(
                collection_id=str(mock_collection.uuid),
                user_id=mock_collection.owner_id,
                updates={"name": "Existing Name"},
            )


class TestListDocuments:
    """Test list_documents method."""

    @pytest.mark.asyncio()
    async def test_list_documents_success(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_document_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test listing documents in collection."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_documents = [MagicMock(), MagicMock()]
        mock_document_repo.list_by_collection.return_value = (mock_documents, 2)

        documents, total = await collection_service.list_documents(
            collection_id=str(mock_collection.uuid),
            user_id=1,
            offset=0,
            limit=50,
        )

        # Verify permission check
        mock_collection_repo.get_by_uuid_with_permission_check.assert_called_once_with(
            collection_uuid=str(mock_collection.uuid),
            user_id=1,
        )

        # Verify document listing
        mock_document_repo.list_by_collection.assert_called_once_with(
            collection_id=mock_collection.id,
            offset=0,
            limit=50,
        )

        assert documents == mock_documents
        assert total == 2

    @pytest.mark.asyncio()
    async def test_list_documents_with_pagination(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_document_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test listing documents with pagination."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_documents = [MagicMock()]
        mock_document_repo.list_by_collection.return_value = (mock_documents, 100)

        documents, total = await collection_service.list_documents(
            collection_id=str(mock_collection.uuid),
            user_id=1,
            offset=20,
            limit=10,
        )

        mock_document_repo.list_by_collection.assert_called_once_with(
            collection_id=mock_collection.id,
            offset=20,
            limit=10,
        )

        assert len(documents) == 1
        assert total == 100

    @pytest.mark.asyncio()
    async def test_list_documents_access_denied(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
    ) -> None:
        """Test listing documents without permission."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = AccessDeniedError(
            "2", "Collection", "some-uuid"
        )

        with pytest.raises(AccessDeniedError):
            await collection_service.list_documents(
                collection_id="some-uuid",
                user_id=2,
            )


class TestListOperations:
    """Test list_operations method."""

    @pytest.mark.asyncio()
    async def test_list_operations_success(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test listing operations for collection."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operations = [MagicMock(), MagicMock()]
        mock_operation_repo.list_for_collection.return_value = (mock_operations, 2)

        operations, total = await collection_service.list_operations(
            collection_id=str(mock_collection.uuid),
            user_id=1,
            offset=0,
            limit=50,
        )

        # Verify permission check
        mock_collection_repo.get_by_uuid_with_permission_check.assert_called_once_with(
            collection_uuid=str(mock_collection.uuid),
            user_id=1,
        )

        # Verify operation listing
        mock_operation_repo.list_for_collection.assert_called_once_with(
            collection_id=mock_collection.id,
            user_id=1,
            offset=0,
            limit=50,
        )

        assert operations == mock_operations
        assert total == 2

    @pytest.mark.asyncio()
    async def test_list_operations_with_pagination(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test listing operations with pagination."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operations = [MagicMock()]
        mock_operation_repo.list_for_collection.return_value = (mock_operations, 100)

        operations, total = await collection_service.list_operations(
            collection_id=str(mock_collection.uuid),
            user_id=1,
            offset=10,
            limit=5,
        )

        mock_operation_repo.list_for_collection.assert_called_once_with(
            collection_id=mock_collection.id,
            user_id=1,
            offset=10,
            limit=5,
        )

        assert len(operations) == 1
        assert total == 100

    @pytest.mark.asyncio()
    async def test_list_operations_collection_not_found(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
    ) -> None:
        """Test listing operations for non-existent collection."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = EntityNotFoundError(
            "Collection", "nonexistent-uuid"
        )

        with pytest.raises(EntityNotFoundError):
            await collection_service.list_operations(
                collection_id="nonexistent-uuid",
                user_id=1,
            )


class TestCollectionServiceEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio()
    async def test_create_collection_with_none_config(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test creating collection with None config."""
        mock_collection_repo.create.return_value = mock_collection
        mock_operation_repo.create.return_value = mock_operation

        with patch("packages.webui.celery_app.celery_app.send_task"):
            await collection_service.create_collection(
                user_id=1,
                name="Test Collection",
                config=None,
            )

        # Verify defaults were used
        call_args = mock_collection_repo.create.call_args[1]
        assert call_args["embedding_model"] == "Qwen/Qwen3-Embedding-0.6B"
        assert call_args["quantization"] == "float16"
        assert call_args["chunk_size"] == 1000
        assert call_args["chunk_overlap"] == 200
        assert call_args["is_public"] is False
        assert call_args["meta"] is None

    @pytest.mark.asyncio()
    async def test_add_source_with_none_config(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test adding source with None config."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.return_value = []
        mock_operation_repo.create.return_value = mock_operation

        with patch("packages.webui.celery_app.celery_app.send_task"):
            await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_path="/path/to/source",
                source_config=None,
            )

        # Verify empty dict was used for source_config
        call_args = mock_operation_repo.create.call_args[1]
        assert call_args["config"]["source_config"] == {}

    @pytest.mark.asyncio()
    async def test_multiple_operations_coordination(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test that operations properly check for active operations."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # First operation succeeds
        mock_operation_repo.get_active_operations.return_value = []
        mock_operation_repo.get_active_operations_count.return_value = 0
        mock_operation_repo.create.return_value = mock_operation

        with patch("packages.webui.celery_app.celery_app.send_task"):
            await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_path="/path1",
            )

        # Second operation should fail if active operation exists
        mock_operation_repo.get_active_operations.return_value = [mock_operation]
        mock_operation_repo.get_active_operations_count.return_value = 1

        with pytest.raises(InvalidStateError):
            await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_path="/path2",
            )

        with pytest.raises(InvalidStateError):
            await collection_service.reindex_collection(
                collection_id=str(mock_collection.uuid),
                user_id=1,
            )

        with pytest.raises(InvalidStateError):
            await collection_service.remove_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_path="/path1",
            )

    @pytest.mark.asyncio()
    async def test_uuid_generation_for_celery_tasks(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test that unique UUIDs are generated for Celery tasks."""
        mock_collection_repo.create.return_value = mock_collection
        mock_operation_repo.create.return_value = mock_operation

        task_ids = []
        with patch("packages.webui.celery_app.celery_app.send_task") as mock_send_task:
            with patch("uuid.uuid4", side_effect=[
                uuid.UUID("11111111-1111-1111-1111-111111111111"),
                uuid.UUID("22222222-2222-2222-2222-222222222222"),
            ]):
                await collection_service.create_collection(
                    user_id=1,
                    name="Collection 1",
                )
                task_ids.append(mock_send_task.call_args[1]["task_id"])

                await collection_service.create_collection(
                    user_id=1,
                    name="Collection 2",
                )
                task_ids.append(mock_send_task.call_args[1]["task_id"])

        # Verify unique task IDs
        assert len(task_ids) == 2
        assert task_ids[0] != task_ids[1]
        assert task_ids[0] == "11111111-1111-1111-1111-111111111111"
        assert task_ids[1] == "22222222-2222-2222-2222-222222222222"