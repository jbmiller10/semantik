"""
Comprehensive tests for CollectionService covering all methods and edge cases.
"""

import uuid
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

import pytest

from shared.database.exceptions import (
    AccessDeniedError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    InvalidStateError,
)
from shared.database.models import CollectionStatus, OperationType
from shared.managers import QdrantManager
from webui.services.collection_service import CollectionService

# Fixtures are now imported from conftest.py


@pytest.fixture()
def mock_qdrant_manager() -> MagicMock:
    manager = MagicMock(spec=QdrantManager)
    manager.rename_collection = AsyncMock()
    manager.list_collections.return_value = []
    manager.client = MagicMock()
    return manager


@pytest.fixture()
def collection_service(
    mock_db_session: AsyncMock,
    mock_collection_repo: AsyncMock,
    mock_operation_repo: AsyncMock,
    mock_document_repo: AsyncMock,
    mock_collection_source_repo: AsyncMock,
    mock_qdrant_manager: AsyncMock,
) -> CollectionService:
    """Create a CollectionService instance with mocked dependencies."""
    return CollectionService(
        db_session=mock_db_session,
        collection_repo=mock_collection_repo,
        operation_repo=mock_operation_repo,
        document_repo=mock_document_repo,
        collection_source_repo=mock_collection_source_repo,
        qdrant_manager=mock_qdrant_manager,
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
        mock_collection_source_repo: AsyncMock,
    ) -> None:
        """Test service initialization."""
        service = CollectionService(
            db_session=mock_db_session,
            collection_repo=mock_collection_repo,
            operation_repo=mock_operation_repo,
            document_repo=mock_document_repo,
            collection_source_repo=mock_collection_source_repo,
            qdrant_manager=AsyncMock(),
        )

        assert service.db_session == mock_db_session
        assert service.collection_repo == mock_collection_repo
        assert service.operation_repo == mock_operation_repo
        assert service.document_repo == mock_document_repo
        assert service.collection_source_repo == mock_collection_source_repo


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

        with patch("webui.celery_app.celery_app.send_task") as mock_send_task:
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
            chunking_strategy=None,
            chunking_config=None,
            is_public=True,
            meta={"custom": "data"},
            sync_mode="one_time",
            sync_interval_minutes=None,
            sync_next_run_at=None,
            pipeline_config=ANY,
            persist_originals=False,
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
            "webui.tasks.process_collection_operation", args=[mock_operation.uuid], task_id=ANY
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

        with patch("webui.celery_app.celery_app.send_task"):
            await collection_service.create_collection(user_id=1, name="Test Collection")

        # Verify default values were used
        mock_collection_repo.create.assert_called_once_with(
            owner_id=1,
            name="Test Collection",
            description=None,
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            quantization="float16",
            chunk_size=1000,
            chunk_overlap=200,
            chunking_strategy=None,
            chunking_config=None,
            is_public=False,
            meta=None,
            sync_mode="one_time",
            sync_interval_minutes=None,
            sync_next_run_at=None,
            pipeline_config=ANY,
            persist_originals=False,
        )

    @pytest.mark.asyncio()
    async def test_create_collection_empty_name(self, collection_service: CollectionService) -> None:
        """Test collection creation with empty name."""
        with pytest.raises(ValueError, match="Collection name is required"):
            await collection_service.create_collection(user_id=1, name="")

    @pytest.mark.asyncio()
    async def test_create_collection_whitespace_name(self, collection_service: CollectionService) -> None:
        """Test collection creation with whitespace-only name."""
        with pytest.raises(ValueError, match="Collection name is required"):
            await collection_service.create_collection(user_id=1, name="   ")

    @pytest.mark.asyncio()
    async def test_create_collection_already_exists(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock
    ) -> None:
        """Test collection creation when name already exists."""
        mock_collection_repo.create.side_effect = EntityAlreadyExistsError("Collection", "Existing Collection")

        with pytest.raises(EntityAlreadyExistsError):
            await collection_service.create_collection(user_id=1, name="Existing Collection")

    @pytest.mark.asyncio()
    async def test_create_collection_database_error(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock
    ) -> None:
        """Test collection creation with database error."""
        mock_collection_repo.create.side_effect = Exception("Database error")

        with pytest.raises(Exception, match="Database error") as exc_info:
            await collection_service.create_collection(user_id=1, name="Test Collection")

        assert "Database error" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_create_collection_with_none_chunk_values_defaults_applied(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Explicit None for chunk fields should use default numeric values."""
        mock_collection_repo.create.return_value = mock_collection
        mock_operation_repo.create.return_value = mock_operation

        with patch("webui.celery_app.celery_app.send_task"):
            await collection_service.create_collection(
                user_id=1,
                name="Test Collection",
                config={
                    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                    "quantization": "float16",
                    "chunk_size": None,
                    "chunk_overlap": None,
                    "is_public": False,
                    "metadata": None,
                },
            )

        # Verify defaults applied in repo call
        mock_collection_repo.create.assert_called_once_with(
            owner_id=1,
            name="Test Collection",
            description=None,
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            quantization="float16",
            chunk_size=1000,
            chunk_overlap=200,
            chunking_strategy=None,
            chunking_config=None,
            is_public=False,
            meta=None,
            sync_mode="one_time",
            sync_interval_minutes=None,
            sync_next_run_at=None,
            pipeline_config=ANY,
            persist_originals=False,
        )


class TestAddSource:
    """Test add_source method."""

    @pytest.mark.asyncio()
    async def test_add_source_success(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection_source_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_collection_source: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test successful source addition."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.return_value = []
        mock_operation_repo.create.return_value = mock_operation
        # Mock get_or_create to return a new source
        mock_collection_source_repo.get_or_create.return_value = (mock_collection_source, True)

        with patch("webui.celery_app.celery_app.send_task") as mock_send_task:
            result = await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_type="directory",
                source_config={"path": "/path/to/source", "recursive": True},
                additional_config={"chunk_size": 500},
            )

        # Verify permission check
        mock_collection_repo.get_by_uuid_with_permission_check.assert_called_once_with(
            collection_uuid=str(mock_collection.uuid), user_id=1
        )

        # Verify active operations check
        mock_operation_repo.get_active_operations.assert_called_once_with(mock_collection.id)

        # Verify CollectionSource get_or_create was called
        mock_collection_source_repo.get_or_create.assert_called_once_with(
            collection_id=mock_collection.id,
            source_type="directory",
            source_path="/path/to/source",
            source_config={"path": "/path/to/source", "recursive": True},
        )

        # Verify operation creation with source_id included
        mock_operation_repo.create.assert_called_once_with(
            collection_id=mock_collection.id,
            user_id=1,
            operation_type=OperationType.APPEND,
            config={
                "source_id": mock_collection_source.id,  # Now includes source_id
                "source_type": "directory",
                "source_config": {"path": "/path/to/source", "recursive": True},
                "source_path": "/path/to/source",  # Extracted from source_config for audit
                "additional_config": {"chunk_size": 500},
            },
        )

        # Verify status update
        mock_collection_repo.update_status.assert_called_once_with(mock_collection.id, CollectionStatus.PROCESSING)

        # Verify commit and task dispatch
        mock_db_session.commit.assert_called_once()
        mock_send_task.assert_called_once()

        # Verify return value
        assert result["uuid"] == mock_operation.uuid
        assert result["type"] == mock_operation.type.value

    @pytest.mark.asyncio()
    async def test_add_source_merges_legacy_source_path_into_directory_config(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection_source_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_collection_source: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test that legacy_source_path fills in directory source_config.path when missing."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.return_value = []
        mock_operation_repo.create.return_value = mock_operation
        mock_collection_source_repo.get_or_create.return_value = (mock_collection_source, True)

        with patch("webui.celery_app.celery_app.send_task"):
            await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_type="directory",
                source_config={"recursive": True},
                legacy_source_path="/path/to/source",
                additional_config=None,
            )

        mock_collection_source_repo.get_or_create.assert_called_once_with(
            collection_id=mock_collection.id,
            source_type="directory",
            source_path="/path/to/source",
            source_config={"recursive": True, "path": "/path/to/source"},
        )

    @pytest.mark.asyncio()
    async def test_add_source_reuses_existing_collection_source(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection_source_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_collection_source: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test that add_source reuses existing CollectionSource for same path."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.return_value = []
        mock_operation_repo.create.return_value = mock_operation
        # Mock get_or_create to return existing source (is_new=False)
        mock_collection_source_repo.get_or_create.return_value = (mock_collection_source, False)

        with patch("webui.celery_app.celery_app.send_task"):
            await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                source_type="directory",
                source_config={"path": "/path/to/source"},
            )

        # Verify get_or_create was called (should reuse existing)
        mock_collection_source_repo.get_or_create.assert_called_once()

        # Verify operation still gets the source_id
        call_args = mock_operation_repo.create.call_args[1]
        assert call_args["config"]["source_id"] == mock_collection_source.id

    @pytest.mark.asyncio()
    async def test_add_source_invalid_status(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock, mock_collection: MagicMock
    ) -> None:
        """Test adding source to collection in invalid status."""
        mock_collection.status = CollectionStatus.ERROR
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(InvalidStateError) as exc_info:
            await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                legacy_source_path="/path/to/source",
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
                legacy_source_path="/path/to/source",
            )

        assert "Cannot add source while another operation is in progress" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_add_source_waits_for_active_operations(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection_source_repo: AsyncMock,
        mock_collection: MagicMock,
        mock_collection_source: MagicMock,
        mock_operation: MagicMock,
        mock_db_session: AsyncMock,
    ) -> None:
        """Service should tolerate briefly active operations before enqueuing APPEND."""

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.side_effect = [
            [MagicMock()],
            [MagicMock()],
            [],
            [],
        ]
        mock_operation_repo.create.return_value = mock_operation
        mock_collection_source_repo.get_or_create.return_value = (mock_collection_source, True)

        with (
            patch("webui.services.collection_service.asyncio.sleep", new=AsyncMock()) as mock_sleep,
            patch("webui.celery_app.celery_app.send_task") as mock_send_task,
        ):
            result = await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                legacy_source_path="/path/to/source",
            )

        # Should poll until operations clear
        assert mock_operation_repo.get_active_operations.await_count >= 3
        mock_sleep.assert_awaited()
        mock_operation_repo.create.assert_awaited_once()
        mock_collection_repo.update_status.assert_awaited_once()
        mock_db_session.commit.assert_awaited_once()
        mock_send_task.assert_called_once()
        assert result["uuid"] == mock_operation.uuid

    @pytest.mark.asyncio()
    async def test_add_source_collection_not_found(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock
    ) -> None:
        """Test adding source to non-existent collection."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = EntityNotFoundError(
            "Collection", "nonexistent-uuid"
        )

        with pytest.raises(EntityNotFoundError):
            await collection_service.add_source(
                collection_id="nonexistent-uuid",
                user_id=1,
                legacy_source_path="/path/to/source",
            )

    @pytest.mark.asyncio()
    async def test_add_source_access_denied(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock
    ) -> None:
        """Test adding source without permission."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = AccessDeniedError(
            "2", "Collection", "some-uuid"
        )

        with pytest.raises(AccessDeniedError):
            await collection_service.add_source(
                collection_id="some-uuid",
                user_id=2,
                legacy_source_path="/path/to/source",
            )

    @pytest.mark.asyncio()
    async def test_add_source_with_pending_status(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection_source_repo: AsyncMock,
        mock_collection: MagicMock,
        mock_collection_source: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test adding source to pending collection."""
        mock_collection.status = CollectionStatus.PENDING
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.return_value = []
        mock_operation_repo.create.return_value = mock_operation
        mock_collection_source_repo.get_or_create.return_value = (mock_collection_source, True)

        with patch("webui.celery_app.celery_app.send_task"):
            result = await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                legacy_source_path="/path/to/source",
            )

        assert result["uuid"] == mock_operation.uuid

    @pytest.mark.asyncio()
    async def test_add_source_with_ready_status(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection_source_repo: AsyncMock,
        mock_collection: MagicMock,
        mock_collection_source: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test adding source to ready collection."""
        mock_collection.status = CollectionStatus.READY
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.return_value = []
        mock_operation_repo.create.return_value = mock_operation
        mock_collection_source_repo.get_or_create.return_value = (mock_collection_source, True)

        with patch("webui.celery_app.celery_app.send_task"):
            result = await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                legacy_source_path="/path/to/source",
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

        with patch("webui.celery_app.celery_app.send_task") as mock_send_task:
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
            collection_id=mock_collection.id, user_id=1, operation_type=OperationType.REINDEX, config=expected_config
        )

        # Verify status update
        mock_collection_repo.update_status.assert_called_once_with(mock_collection.id, CollectionStatus.PROCESSING)

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

        with patch("webui.celery_app.celery_app.send_task"):
            await collection_service.reindex_collection(collection_id=str(mock_collection.uuid), user_id=1)

        # Verify new_config is same as previous_config
        call_args = mock_operation_repo.create.call_args[1]
        assert call_args["config"]["previous_config"] == call_args["config"]["new_config"]

    @pytest.mark.asyncio()
    async def test_reindex_collection_processing_status(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock, mock_collection: MagicMock
    ) -> None:
        """Test reindexing collection in processing status."""
        mock_collection.status = CollectionStatus.PROCESSING
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(InvalidStateError) as exc_info:
            await collection_service.reindex_collection(collection_id=str(mock_collection.uuid), user_id=1)

        assert "Cannot reindex collection that is currently processing" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_reindex_collection_error_status(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock, mock_collection: MagicMock
    ) -> None:
        """Test reindexing collection in error status."""
        mock_collection.status = CollectionStatus.ERROR
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(InvalidStateError) as exc_info:
            await collection_service.reindex_collection(collection_id=str(mock_collection.uuid), user_id=1)

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
            await collection_service.reindex_collection(collection_id=str(mock_collection.uuid), user_id=1)

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

        collection_service.qdrant_manager.list_collections.return_value = [
            mock_collection.vector_store_name,
            "other_collection",
        ]

        await collection_service.delete_collection(
            collection_id=str(mock_collection.uuid), user_id=mock_collection.owner_id
        )

        # Verify Qdrant deletion
        collection_service.qdrant_manager.client.delete_collection.assert_called_once_with(
            mock_collection.vector_store_name
        )

        # Verify database deletion
        mock_collection_repo.delete.assert_called_once_with(mock_collection.id, mock_collection.owner_id)

    @pytest.mark.asyncio()
    async def test_delete_collection_not_owner(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock, mock_collection: MagicMock
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
                collection_id=str(mock_collection.uuid), user_id=mock_collection.owner_id
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

        collection_service.qdrant_manager.list_collections.return_value = []

        await collection_service.delete_collection(
            collection_id=str(mock_collection.uuid), user_id=mock_collection.owner_id
        )

        # Verify Qdrant deletion was not called
        collection_service.qdrant_manager.client.delete_collection.assert_not_called()

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

        collection_service.qdrant_manager.list_collections.side_effect = Exception("Qdrant error")

        await collection_service.delete_collection(
            collection_id=str(mock_collection.uuid), user_id=mock_collection.owner_id
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

        await collection_service.delete_collection(
            collection_id=str(mock_collection.uuid), user_id=mock_collection.owner_id
        )

        # Verify Qdrant helper was not touched
        collection_service.qdrant_manager.list_collections.assert_not_called()
        collection_service.qdrant_manager.client.delete_collection.assert_not_called()

        # Verify database deletion was called
        mock_collection_repo.delete.assert_called_once()


class TestRemoveSource:
    """Test remove_source method."""

    @pytest.mark.asyncio()
    async def test_remove_source_success(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection_source_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_collection_source: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test successful source removal."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_collection_source_repo.get_by_collection_and_path.return_value = mock_collection_source
        mock_operation_repo.get_active_operations_count.return_value = 0
        mock_operation_repo.create.return_value = mock_operation

        with patch("webui.celery_app.celery_app.send_task") as mock_send_task:
            result = await collection_service.remove_source(
                collection_id=str(mock_collection.uuid), user_id=1, source_path="/path/to/remove"
            )

        # Verify source lookup
        mock_collection_source_repo.get_by_collection_and_path.assert_called_once_with(
            collection_id=mock_collection.id, source_path="/path/to/remove"
        )

        # Verify operation creation
        mock_operation_repo.create.assert_called_once_with(
            collection_id=mock_collection.id,
            user_id=1,
            operation_type=OperationType.REMOVE_SOURCE,
            config={
                "source_id": mock_collection_source.id,
                "source_path": "/path/to/remove",
            },
        )

        # Verify status update
        mock_collection_repo.update_status.assert_called_once_with(mock_collection.id, CollectionStatus.PROCESSING)

        # Verify commit and task dispatch
        mock_db_session.commit.assert_called_once()
        mock_send_task.assert_called_once()

        assert result["uuid"] == mock_operation.uuid

    @pytest.mark.asyncio()
    async def test_remove_source_invalid_status(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock, mock_collection: MagicMock
    ) -> None:
        """Test removing source from collection in invalid status."""
        mock_collection.status = CollectionStatus.PENDING
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(InvalidStateError) as exc_info:
            await collection_service.remove_source(
                collection_id=str(mock_collection.uuid), user_id=1, source_path="/path/to/remove"
            )

        assert "Cannot remove source from collection in" in str(exc_info.value)
        assert "PENDING" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_remove_source_ready_status(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection_source_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
        mock_collection_source: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test removing source from ready collection."""
        mock_collection.status = CollectionStatus.READY
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_collection_source_repo.get_by_collection_and_path.return_value = mock_collection_source
        mock_operation_repo.get_active_operations_count.return_value = 0
        mock_operation_repo.create.return_value = mock_operation

        with patch("webui.celery_app.celery_app.send_task"):
            result = await collection_service.remove_source(
                collection_id=str(mock_collection.uuid), user_id=1, source_path="/path/to/remove"
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
                collection_id=str(mock_collection.uuid), user_id=1, source_path="/path/to/remove"
            )

        assert "Cannot remove source while another operation is in progress" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_remove_source_not_found(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection_source_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test removing source that doesn't exist."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations_count.return_value = 0
        mock_collection_source_repo.get_by_collection_and_path.return_value = None

        with pytest.raises(EntityNotFoundError) as exc_info:
            await collection_service.remove_source(
                collection_id=str(mock_collection.uuid), user_id=1, source_path="/nonexistent/path"
            )

        assert "collection_source" in str(exc_info.value)


class TestListForUser:
    """Test list_for_user method."""

    @pytest.mark.asyncio()
    async def test_list_for_user_success(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock
    ) -> None:
        """Test listing collections for user."""
        mock_collections = [MagicMock(), MagicMock()]
        mock_collection_repo.list_for_user.return_value = (mock_collections, 2)

        collections, total = await collection_service.list_for_user(user_id=1, offset=0, limit=50, include_public=True)

        mock_collection_repo.list_for_user.assert_called_once_with(user_id=1, offset=0, limit=50, include_public=True)

        assert collections == mock_collections
        assert total == 2

    @pytest.mark.asyncio()
    async def test_list_for_user_with_pagination(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock
    ) -> None:
        """Test listing collections with pagination."""
        mock_collections = [MagicMock()]
        mock_collection_repo.list_for_user.return_value = (mock_collections, 100)

        collections, total = await collection_service.list_for_user(
            user_id=1, offset=50, limit=10, include_public=False
        )

        mock_collection_repo.list_for_user.assert_called_once_with(user_id=1, offset=50, limit=10, include_public=False)

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

        new_name = "Updated Name"
        expected_vector_store = CollectionService._build_vector_store_name(str(mock_collection.id), new_name)
        result = await collection_service.update(
            collection_id=str(mock_collection.uuid),
            user_id=mock_collection.owner_id,
            updates={
                "name": new_name,
                "description": "Updated description",
                "is_public": True,
            },
        )

        # Verify permission check
        mock_collection_repo.get_by_uuid_with_permission_check.assert_called_once_with(
            collection_uuid=str(mock_collection.uuid), user_id=mock_collection.owner_id
        )

        # Verify update call
        mock_collection_repo.update.assert_called_once_with(
            str(mock_collection.id),
            {
                "name": new_name,
                "description": "Updated description",
                "is_public": True,
                "vector_store_name": expected_vector_store,
            },
        )

        # Verify commit
        mock_db_session.commit.assert_called_once()
        collection_service.qdrant_manager.rename_collection.assert_awaited_once_with(
            old_name=mock_collection.vector_store_name,
            new_name=expected_vector_store,
        )

        assert result == updated_collection

    @pytest.mark.asyncio()
    async def test_update_not_owner(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock, mock_collection: MagicMock
    ) -> None:
        """Test updating collection by non-owner."""
        mock_collection.owner_id = 1
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(AccessDeniedError) as exc_info:
            await collection_service.update(
                collection_id=str(mock_collection.uuid),
                user_id=2,
                updates={"name": "New Name"},  # Different user
            )

        assert "does not have access to Collection" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_update_collection_not_found(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock
    ) -> None:
        """Test updating non-existent collection."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = EntityNotFoundError(
            "Collection", "nonexistent-uuid"
        )

        with pytest.raises(EntityNotFoundError):
            await collection_service.update(collection_id="nonexistent-uuid", user_id=1, updates={"name": "New Name"})

    @pytest.mark.asyncio()
    async def test_update_already_exists(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock, mock_collection: MagicMock
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
            collection_id=str(mock_collection.uuid), user_id=1, offset=0, limit=50
        )

        # Verify permission check
        mock_collection_repo.get_by_uuid_with_permission_check.assert_called_once_with(
            collection_uuid=str(mock_collection.uuid), user_id=1
        )

        # Verify document listing
        mock_document_repo.list_by_collection.assert_called_once_with(
            collection_id=mock_collection.id, offset=0, limit=50
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
            collection_id=str(mock_collection.uuid), user_id=1, offset=20, limit=10
        )

        mock_document_repo.list_by_collection.assert_called_once_with(
            collection_id=mock_collection.id, offset=20, limit=10
        )

        assert len(documents) == 1
        assert total == 100

    @pytest.mark.asyncio()
    async def test_list_documents_access_denied(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock
    ) -> None:
        """Test listing documents without permission."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = AccessDeniedError(
            "2", "Collection", "some-uuid"
        )

        with pytest.raises(AccessDeniedError):
            await collection_service.list_documents(collection_id="some-uuid", user_id=2)


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
            collection_id=str(mock_collection.uuid), user_id=1, offset=0, limit=50
        )

        # Verify permission check
        mock_collection_repo.get_by_uuid_with_permission_check.assert_called_once_with(
            collection_uuid=str(mock_collection.uuid), user_id=1
        )

        # Verify operation listing
        mock_operation_repo.list_for_collection.assert_called_once_with(
            collection_id=mock_collection.id, user_id=1, offset=0, limit=50
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
            collection_id=str(mock_collection.uuid), user_id=1, offset=10, limit=5
        )

        mock_operation_repo.list_for_collection.assert_called_once_with(
            collection_id=mock_collection.id, user_id=1, offset=10, limit=5
        )

        assert len(operations) == 1
        assert total == 100

    @pytest.mark.asyncio()
    async def test_list_operations_collection_not_found(
        self, collection_service: CollectionService, mock_collection_repo: AsyncMock
    ) -> None:
        """Test listing operations for non-existent collection."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = EntityNotFoundError(
            "Collection", "nonexistent-uuid"
        )

        with pytest.raises(EntityNotFoundError):
            await collection_service.list_operations(collection_id="nonexistent-uuid", user_id=1)


class TestRenameCollectionWithQdrantSync:
    """Tests ensuring collection rename stays in sync with Qdrant."""

    @pytest.mark.asyncio()
    async def test_rename_collection_updates_qdrant_and_database(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Renaming should update Qdrant and commit DB changes."""

        mock_collection.vector_store_name = "col_123e4567_e89b_12d3_a456_426614174000"
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        updated_collection = MagicMock()
        new_name = "Renamed"
        expected_vector_store = CollectionService._build_vector_store_name(str(mock_collection.id), new_name)
        updated_collection.name = new_name
        updated_collection.vector_store_name = expected_vector_store
        mock_collection_repo.update.return_value = updated_collection

        result = await collection_service.update(
            collection_id=str(mock_collection.uuid),
            user_id=mock_collection.owner_id,
            updates={"name": new_name},
        )

        mock_collection_repo.update.assert_called_once_with(
            str(mock_collection.id),
            {"name": new_name, "vector_store_name": expected_vector_store},
        )
        mock_db_session.commit.assert_called_once()
        mock_db_session.rollback.assert_not_called()
        collection_service.qdrant_manager.rename_collection.assert_awaited_once_with(
            old_name=mock_collection.vector_store_name,
            new_name=expected_vector_store,
        )
        assert result == updated_collection

    @pytest.mark.asyncio()
    async def test_rename_collection_qdrant_failure_triggers_rollback(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """If Qdrant rename fails, the DB state should be reverted."""

        mock_collection.vector_store_name = "col_123e4567_e89b_12d3_a456_426614174000"
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        updated_collection = MagicMock()
        updated_collection.name = "Renamed"
        updated_collection.vector_store_name = CollectionService._build_vector_store_name(
            str(mock_collection.id), "Renamed"
        )

        reverted_collection = MagicMock()
        reverted_collection.name = mock_collection.name
        reverted_collection.vector_store_name = mock_collection.vector_store_name

        mock_collection_repo.update.side_effect = [updated_collection, reverted_collection]

        expected_vector_store = CollectionService._build_vector_store_name(str(mock_collection.id), "Renamed")
        collection_service.qdrant_manager.rename_collection.side_effect = RuntimeError("Qdrant rename failed")

        with pytest.raises(RuntimeError, match="Qdrant rename failed"):
            await collection_service.update(
                collection_id=str(mock_collection.uuid),
                user_id=mock_collection.owner_id,
                updates={"name": "Renamed"},
            )

        assert mock_collection_repo.update.await_args_list == [
            call(str(mock_collection.id), {"name": "Renamed", "vector_store_name": expected_vector_store}),
            call(
                str(mock_collection.id),
                {"name": mock_collection.name, "vector_store_name": mock_collection.vector_store_name},
            ),
        ]
        collection_service.qdrant_manager.rename_collection.assert_awaited_once_with(
            old_name=mock_collection.vector_store_name,
            new_name=expected_vector_store,
        )
        assert mock_db_session.commit.await_count == 2
        mock_db_session.rollback.assert_not_called()

    @pytest.mark.asyncio()
    async def test_rename_collection_commit_failure_reverts_qdrant(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """If the DB commit fails, skip Qdrant rename and roll back the DB change."""

        mock_collection.vector_store_name = "col_123e4567_e89b_12d3_a456_426614174000"
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        updated_collection = MagicMock()
        new_name = "Renamed"
        expected_vector_store = CollectionService._build_vector_store_name(str(mock_collection.id), new_name)
        updated_collection.name = new_name
        updated_collection.vector_store_name = expected_vector_store
        mock_collection_repo.update.return_value = updated_collection

        mock_db_session.commit.side_effect = RuntimeError("DB commit failed")

        with pytest.raises(RuntimeError, match="DB commit failed"):
            await collection_service.update(
                collection_id=str(mock_collection.uuid),
                user_id=mock_collection.owner_id,
                updates={"name": new_name},
            )

        mock_collection_repo.update.assert_called_once_with(
            str(mock_collection.id),
            {"name": new_name, "vector_store_name": expected_vector_store},
        )
        collection_service.qdrant_manager.rename_collection.assert_not_called()
        mock_db_session.rollback.assert_called_once()
        mock_db_session.commit.assert_called_once()


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

        with patch("webui.celery_app.celery_app.send_task"):
            await collection_service.create_collection(user_id=1, name="Test Collection", config=None)

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
        mock_collection_source_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_collection_source: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test adding source with None config."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.return_value = []
        mock_operation_repo.create.return_value = mock_operation
        mock_collection_source_repo.get_or_create.return_value = (mock_collection_source, True)

        with patch("webui.celery_app.celery_app.send_task"):
            await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                legacy_source_path="/path/to/source",
                # source_config and additional_config default to None
            )

        # Verify empty dict was used for source_config and additional_config
        call_args = mock_operation_repo.create.call_args[1]
        assert call_args["config"]["source_config"] == {"path": "/path/to/source"}
        assert call_args["config"]["additional_config"] == {}
        assert call_args["config"]["source_id"] == mock_collection_source.id

    @pytest.mark.asyncio()
    async def test_multiple_operations_coordination(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection_source_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_collection_source: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test that operations properly check for active operations."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_collection_source_repo.get_or_create.return_value = (mock_collection_source, True)

        # First operation succeeds
        mock_operation_repo.get_active_operations.return_value = []
        mock_operation_repo.get_active_operations_count.return_value = 0
        mock_operation_repo.create.return_value = mock_operation

        with patch("webui.celery_app.celery_app.send_task"):
            await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                legacy_source_path="/path1",
            )

        # Second operation should fail if active operation exists
        mock_operation_repo.get_active_operations.return_value = [mock_operation]
        mock_operation_repo.get_active_operations_count.return_value = 1

        with pytest.raises(InvalidStateError):
            await collection_service.add_source(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                legacy_source_path="/path2",
            )

        with pytest.raises(InvalidStateError):
            await collection_service.reindex_collection(collection_id=str(mock_collection.uuid), user_id=1)

        with pytest.raises(InvalidStateError):
            await collection_service.remove_source(
                collection_id=str(mock_collection.uuid), user_id=1, source_path="/path1"
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
        with (
            patch("webui.celery_app.celery_app.send_task") as mock_send_task,
            patch(
                "uuid.uuid4",
                side_effect=[
                    uuid.UUID("11111111-1111-1111-1111-111111111111"),
                    uuid.UUID("22222222-2222-2222-2222-222222222222"),
                ],
            ),
        ):
            await collection_service.create_collection(user_id=1, name="Collection 1")
            task_ids.append(mock_send_task.call_args[1]["task_id"])

            await collection_service.create_collection(user_id=1, name="Collection 2")
            task_ids.append(mock_send_task.call_args[1]["task_id"])

        # Verify unique task IDs
        assert len(task_ids) == 2
        assert task_ids[0] != task_ids[1]
        assert task_ids[0] == "11111111-1111-1111-1111-111111111111"
        assert task_ids[1] == "22222222-2222-2222-2222-222222222222"


class TestDeriveSourcePath:
    """Test _derive_source_path helper method."""

    def test_derive_source_path_directory(self, collection_service: CollectionService) -> None:
        """Test deriving path for directory source type."""
        result = collection_service._derive_source_path("directory", {"path": "/data/documents"})
        assert result == "/data/documents"

    def test_derive_source_path_web(self, collection_service: CollectionService) -> None:
        """Test deriving path for web source type."""
        result = collection_service._derive_source_path("web", {"url": "https://example.com"})
        assert result == "https://example.com"

    def test_derive_source_path_empty_config(self, collection_service: CollectionService) -> None:
        """Test deriving path with empty config."""
        result = collection_service._derive_source_path("directory", None)
        assert result == ""

    def test_derive_source_path_fallback_keys(self, collection_service: CollectionService) -> None:
        """Test deriving path using fallback keys."""
        result = collection_service._derive_source_path("slack", {"channel": "#general"})
        assert result == "#general"

    def test_derive_source_path_identifier_key(self, collection_service: CollectionService) -> None:
        """Test deriving path using identifier key."""
        result = collection_service._derive_source_path("custom", {"identifier": "my-source-id"})
        assert result == "my-source-id"

    def test_derive_source_path_fallback_first_string(self, collection_service: CollectionService) -> None:
        """Test deriving path using first string value as fallback."""
        result = collection_service._derive_source_path("unknown", {"some_key": "some_value", "number": 123})
        assert result == "some_value"

    def test_derive_source_path_no_string_values(self, collection_service: CollectionService) -> None:
        """Test deriving path with no string values returns empty string."""
        result = collection_service._derive_source_path("custom", {"count": 123, "enabled": True})
        assert result == ""

    def test_derive_source_path_none_path_value(self, collection_service: CollectionService) -> None:
        """Test deriving path when path is None."""
        result = collection_service._derive_source_path("directory", {"path": None})
        assert result == ""


class TestBuildVectorStoreName:
    """Test _build_vector_store_name static method."""

    def test_build_vector_store_name_basic(self) -> None:
        """Test basic vector store name generation."""
        result = CollectionService._build_vector_store_name("123e4567-e89b-12d3-a456-426614174000", "My Collection")
        assert result.startswith("col_")
        assert "my_collection" in result

    def test_build_vector_store_name_special_characters(self) -> None:
        """Test vector store name with special characters removed."""
        result = CollectionService._build_vector_store_name("123e4567-e89b-12d3-a456-426614174000", "My!@#$%Collection")
        assert "!" not in result
        assert "@" not in result
        assert "#" not in result

    def test_build_vector_store_name_empty_slug(self) -> None:
        """Test vector store name when slug becomes empty."""
        result = CollectionService._build_vector_store_name("123e4567-e89b-12d3-a456-426614174000", "!@#$%")
        assert result.startswith("col_")

    def test_build_vector_store_name_truncation(self) -> None:
        """Test vector store name truncation for long names."""
        long_name = "a" * 200
        result = CollectionService._build_vector_store_name("123e4567-e89b-12d3-a456-426614174000", long_name)
        assert len(result) <= 120


class TestListOperationsFiltered:
    """Test list_operations_filtered method."""

    @pytest.mark.asyncio()
    async def test_list_operations_filtered_by_status(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test listing operations filtered by status."""
        from shared.database.models import OperationStatus

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        # Create operations with different statuses
        pending_op = MagicMock()
        pending_op.status = OperationStatus.PENDING
        pending_op.type = OperationType.INDEX

        completed_op = MagicMock()
        completed_op.status = OperationStatus.COMPLETED
        completed_op.type = OperationType.INDEX

        mock_operation_repo.list_for_collection.return_value = ([pending_op, completed_op], 2)

        operations, total = await collection_service.list_operations_filtered(
            collection_id=str(mock_collection.uuid),
            user_id=1,
            status="pending",
        )

        assert len(operations) == 1
        assert operations[0].status == OperationStatus.PENDING

    @pytest.mark.asyncio()
    async def test_list_operations_filtered_by_type(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test listing operations filtered by type."""
        from shared.database.models import OperationStatus

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        index_op = MagicMock()
        index_op.status = OperationStatus.PENDING
        index_op.type = OperationType.INDEX

        reindex_op = MagicMock()
        reindex_op.status = OperationStatus.PENDING
        reindex_op.type = OperationType.REINDEX

        mock_operation_repo.list_for_collection.return_value = ([index_op, reindex_op], 2)

        operations, total = await collection_service.list_operations_filtered(
            collection_id=str(mock_collection.uuid),
            user_id=1,
            operation_type="reindex",
        )

        assert len(operations) == 1
        assert operations[0].type == OperationType.REINDEX

    @pytest.mark.asyncio()
    async def test_list_operations_filtered_invalid_status(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test listing operations with invalid status raises ValueError."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(ValueError, match="Invalid status"):
            await collection_service.list_operations_filtered(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                status="invalid_status",
            )

    @pytest.mark.asyncio()
    async def test_list_operations_filtered_invalid_type(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test listing operations with invalid type raises ValueError."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(ValueError, match="Invalid operation type"):
            await collection_service.list_operations_filtered(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                operation_type="invalid_type",
            )

    @pytest.mark.asyncio()
    async def test_list_operations_filtered_no_filters(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test listing operations without filters returns all."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        ops = [MagicMock(), MagicMock()]
        mock_operation_repo.list_for_collection.return_value = (ops, 2)

        operations, total = await collection_service.list_operations_filtered(
            collection_id=str(mock_collection.uuid),
            user_id=1,
        )

        assert len(operations) == 2
        assert total == 2


class TestListDocumentsFiltered:
    """Test list_documents_filtered method."""

    @pytest.mark.asyncio()
    async def test_list_documents_filtered_by_status(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_document_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test listing documents filtered by status."""
        from shared.database.models import DocumentStatus

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        completed_doc = MagicMock()
        completed_doc.status = DocumentStatus.COMPLETED

        pending_doc = MagicMock()
        pending_doc.status = DocumentStatus.PENDING

        mock_document_repo.list_by_collection.return_value = ([completed_doc, pending_doc], 2)

        documents, total = await collection_service.list_documents_filtered(
            collection_id=str(mock_collection.uuid),
            user_id=1,
            status="completed",
        )

        assert len(documents) == 1
        assert documents[0].status == DocumentStatus.COMPLETED

    @pytest.mark.asyncio()
    async def test_list_documents_filtered_invalid_status(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test listing documents with invalid status raises ValueError."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(ValueError, match="Invalid status"):
            await collection_service.list_documents_filtered(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                status="invalid_status",
            )

    @pytest.mark.asyncio()
    async def test_list_documents_filtered_no_filter(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_document_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test listing documents without filter returns all."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        docs = [MagicMock(), MagicMock()]
        mock_document_repo.list_by_collection.return_value = (docs, 2)

        documents, total = await collection_service.list_documents_filtered(
            collection_id=str(mock_collection.uuid),
            user_id=1,
        )

        assert len(documents) == 2
        assert total == 2


class TestCreateOperation:
    """Test create_operation method."""

    @pytest.mark.asyncio()
    async def test_create_operation_index(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test creating an index operation."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation.meta = {"operation_type": "index"}
        mock_operation_repo.create.return_value = mock_operation

        result = await collection_service.create_operation(
            collection_id=str(mock_collection.uuid),
            operation_type="index",
            config={"source_path": "/data"},
            user_id=1,
        )

        mock_operation_repo.create.assert_called_once()
        mock_db_session.commit.assert_called_once()

        assert result["uuid"] == mock_operation.uuid
        assert result["type"] == mock_operation.type.value

    @pytest.mark.asyncio()
    async def test_create_operation_chunking_maps_to_index(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test that 'chunking' operation type maps to INDEX."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation.meta = {"operation_type": "chunking"}
        mock_operation_repo.create.return_value = mock_operation

        await collection_service.create_operation(
            collection_id=str(mock_collection.uuid),
            operation_type="chunking",
            config={},
            user_id=1,
        )

        call_args = mock_operation_repo.create.call_args[1]
        assert call_args["operation_type"] == OperationType.INDEX
        assert call_args["meta"] == {"operation_type": "chunking"}

    @pytest.mark.asyncio()
    async def test_create_operation_rechunking_maps_to_reindex(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test that 'rechunking' operation type maps to REINDEX."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation.meta = {"operation_type": "rechunking"}
        mock_operation_repo.create.return_value = mock_operation

        await collection_service.create_operation(
            collection_id=str(mock_collection.uuid),
            operation_type="rechunking",
            config={},
            user_id=1,
        )

        call_args = mock_operation_repo.create.call_args[1]
        assert call_args["operation_type"] == OperationType.REINDEX

    @pytest.mark.asyncio()
    async def test_create_operation_collection_not_found(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
    ) -> None:
        """Test creating operation for non-existent collection."""
        mock_collection_repo.get_by_uuid_with_permission_check.side_effect = EntityNotFoundError(
            "Collection", "nonexistent"
        )

        with pytest.raises(EntityNotFoundError):
            await collection_service.create_operation(
                collection_id="nonexistent",
                operation_type="index",
                config={},
                user_id=1,
            )


class TestUpdateCollection:
    """Test update_collection method (alias for update that returns dict)."""

    @pytest.mark.asyncio()
    async def test_update_collection_returns_dict(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test update_collection returns a dictionary."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        updated_collection = MagicMock()
        updated_collection.id = mock_collection.id
        updated_collection.name = "Updated Name"
        updated_collection.description = "Updated description"
        updated_collection.owner_id = mock_collection.owner_id
        updated_collection.vector_store_name = mock_collection.vector_store_name
        updated_collection.embedding_model = mock_collection.embedding_model
        updated_collection.quantization = mock_collection.quantization
        updated_collection.chunk_size = mock_collection.chunk_size
        updated_collection.chunk_overlap = mock_collection.chunk_overlap
        updated_collection.chunking_strategy = None
        updated_collection.chunking_config = None
        updated_collection.is_public = False
        updated_collection.meta = None
        updated_collection.created_at = mock_collection.created_at
        updated_collection.updated_at = mock_collection.updated_at
        updated_collection.document_count = 0
        updated_collection.vector_count = 0
        updated_collection.status = CollectionStatus.READY
        updated_collection.status_message = None
        mock_collection_repo.update.return_value = updated_collection

        result = await collection_service.update_collection(
            collection_id=str(mock_collection.uuid),
            updates={"description": "Updated description"},
            user_id=mock_collection.owner_id,
        )

        assert isinstance(result, dict)
        assert "id" in result
        assert "name" in result
        assert "config" in result


class TestCollectionSyncMethods:
    """Test collection sync methods."""

    @pytest.fixture()
    def mock_sync_run_repo(self) -> AsyncMock:
        """Create a mock sync run repository."""
        from shared.database.repositories.collection_sync_run_repository import CollectionSyncRunRepository

        mock = AsyncMock(spec=CollectionSyncRunRepository)
        mock.create = AsyncMock()
        mock.list_for_collection = AsyncMock()
        return mock

    @pytest.fixture()
    def collection_service_with_sync(
        self,
        mock_db_session: AsyncMock,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_document_repo: AsyncMock,
        mock_collection_source_repo: AsyncMock,
        mock_qdrant_manager: AsyncMock,
        mock_sync_run_repo: AsyncMock,
    ) -> CollectionService:
        """Create a CollectionService with sync run repository."""
        return CollectionService(
            db_session=mock_db_session,
            collection_repo=mock_collection_repo,
            operation_repo=mock_operation_repo,
            document_repo=mock_document_repo,
            collection_source_repo=mock_collection_source_repo,
            qdrant_manager=mock_qdrant_manager,
            sync_run_repo=mock_sync_run_repo,
        )

    @pytest.mark.asyncio()
    async def test_run_collection_sync_success(
        self,
        collection_service_with_sync: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection_source_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_sync_run_repo: AsyncMock,
        mock_collection: MagicMock,
        mock_collection_source: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test successful sync run creation."""
        mock_collection.sync_mode = None
        mock_collection.sync_interval_minutes = None
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.return_value = []
        mock_collection_source_repo.list_by_collection.return_value = ([mock_collection_source], 1)

        sync_run = MagicMock()
        sync_run.id = 1
        mock_sync_run_repo.create.return_value = sync_run

        mock_operation_repo.create.return_value = mock_operation

        with patch("webui.celery_app.celery_app.send_task"):
            result = await collection_service_with_sync.run_collection_sync(
                collection_id=str(mock_collection.uuid),
                user_id=1,
                triggered_by="manual",
            )

        assert result == sync_run
        mock_sync_run_repo.create.assert_called_once()
        mock_operation_repo.create.assert_called_once()
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio()
    async def test_run_collection_sync_invalid_status(
        self,
        collection_service_with_sync: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test sync run fails for collection in PROCESSING state."""
        mock_collection.status = CollectionStatus.PROCESSING
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(InvalidStateError, match="Cannot sync collection"):
            await collection_service_with_sync.run_collection_sync(
                collection_id=str(mock_collection.uuid),
                user_id=1,
            )

    @pytest.mark.asyncio()
    async def test_run_collection_sync_active_operations(
        self,
        collection_service_with_sync: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test sync run fails when active operations exist."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.return_value = [MagicMock()]

        with pytest.raises(InvalidStateError, match="Cannot start sync while another operation"):
            await collection_service_with_sync.run_collection_sync(
                collection_id=str(mock_collection.uuid),
                user_id=1,
            )

    @pytest.mark.asyncio()
    async def test_run_collection_sync_no_sources(
        self,
        collection_service_with_sync: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_collection_source_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test sync run fails when collection has no sources."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection
        mock_operation_repo.get_active_operations.return_value = []
        mock_collection_source_repo.list_by_collection.return_value = ([], 0)

        with pytest.raises(InvalidStateError, match="Cannot sync collection with no sources"):
            await collection_service_with_sync.run_collection_sync(
                collection_id=str(mock_collection.uuid),
                user_id=1,
            )

    @pytest.mark.asyncio()
    async def test_pause_collection_sync_success(
        self,
        collection_service_with_sync: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test successful pause of collection sync."""
        mock_collection.sync_mode = "continuous"
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        paused_collection = MagicMock()
        mock_collection_repo.pause_sync.return_value = paused_collection

        result = await collection_service_with_sync.pause_collection_sync(
            collection_id=str(mock_collection.uuid),
            user_id=1,
        )

        assert result == paused_collection
        mock_collection_repo.pause_sync.assert_called_once_with(mock_collection.id)
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio()
    async def test_pause_collection_sync_not_continuous(
        self,
        collection_service_with_sync: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test pause fails when not in continuous mode."""
        from shared.database.exceptions import ValidationError

        mock_collection.sync_mode = "manual"
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(ValidationError, match="not in continuous sync mode"):
            await collection_service_with_sync.pause_collection_sync(
                collection_id=str(mock_collection.uuid),
                user_id=1,
            )

    @pytest.mark.asyncio()
    async def test_resume_collection_sync_success(
        self,
        collection_service_with_sync: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test successful resume of collection sync."""
        from datetime import UTC, datetime

        mock_collection.sync_mode = "continuous"
        mock_collection.sync_paused_at = datetime.now(UTC)
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        resumed_collection = MagicMock()
        mock_collection_repo.resume_sync.return_value = resumed_collection

        result = await collection_service_with_sync.resume_collection_sync(
            collection_id=str(mock_collection.uuid),
            user_id=1,
        )

        assert result == resumed_collection
        mock_collection_repo.resume_sync.assert_called_once_with(mock_collection.id)
        mock_db_session.commit.assert_called_once()

    @pytest.mark.asyncio()
    async def test_resume_collection_sync_not_paused(
        self,
        collection_service_with_sync: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test resume fails when not paused."""
        from shared.database.exceptions import ValidationError

        mock_collection.sync_mode = "continuous"
        mock_collection.sync_paused_at = None
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(ValidationError, match="not paused"):
            await collection_service_with_sync.resume_collection_sync(
                collection_id=str(mock_collection.uuid),
                user_id=1,
            )

    @pytest.mark.asyncio()
    async def test_list_collection_sync_runs_success(
        self,
        collection_service_with_sync: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_sync_run_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test listing sync runs for a collection."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        sync_runs = [MagicMock(), MagicMock()]
        mock_sync_run_repo.list_for_collection.return_value = (sync_runs, 2)

        result, total = await collection_service_with_sync.list_collection_sync_runs(
            collection_id=str(mock_collection.uuid),
            user_id=1,
            offset=0,
            limit=50,
        )

        assert result == sync_runs
        assert total == 2
        mock_sync_run_repo.list_for_collection.assert_called_once_with(
            collection_id=mock_collection.id,
            offset=0,
            limit=50,
        )


class TestChunkingStrategyValidation:
    """Test chunking strategy validation in create_collection."""

    @pytest.mark.asyncio()
    async def test_create_collection_with_valid_chunking_strategy(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_operation_repo: AsyncMock,
        mock_db_session: AsyncMock,
        mock_collection: MagicMock,
        mock_operation: MagicMock,
    ) -> None:
        """Test collection creation with valid chunking strategy."""
        mock_collection_repo.create.return_value = mock_collection
        mock_operation_repo.create.return_value = mock_operation

        with patch("webui.celery_app.celery_app.send_task"):
            await collection_service.create_collection(
                user_id=1,
                name="Test Collection",
                config={
                    "chunking_strategy": "semantic",
                    "chunking_config": {"max_chunk_size": 1000},
                },
            )

        call_args = mock_collection_repo.create.call_args[1]
        assert call_args["chunking_strategy"] == "semantic"
        assert call_args["chunking_config"] is not None

    @pytest.mark.asyncio()
    async def test_create_collection_with_invalid_chunking_strategy(
        self,
        collection_service: CollectionService,
    ) -> None:
        """Test collection creation with invalid chunking strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid chunking_strategy"):
            await collection_service.create_collection(
                user_id=1,
                name="Test Collection",
                config={
                    "chunking_strategy": "nonexistent_strategy",
                },
            )

    @pytest.mark.asyncio()
    async def test_create_collection_with_config_but_no_strategy(
        self,
        collection_service: CollectionService,
    ) -> None:
        """Test collection creation with config but no strategy raises ValueError."""
        with pytest.raises(ValueError, match="chunking_config requires chunking_strategy"):
            await collection_service.create_collection(
                user_id=1,
                name="Test Collection",
                config={
                    "chunking_config": {"max_chunk_size": 1000},
                },
            )


class TestUpdateChunkingValidation:
    """Test chunking validation in update method."""

    @pytest.mark.asyncio()
    async def test_update_with_invalid_chunking_strategy(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test update with invalid chunking strategy raises ValueError."""
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(ValueError, match="Invalid chunking strategy"):
            await collection_service.update(
                collection_id=str(mock_collection.uuid),
                user_id=mock_collection.owner_id,
                updates={"chunking_strategy": "nonexistent_strategy"},
            )

    @pytest.mark.asyncio()
    async def test_update_chunking_config_without_strategy(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """Test update with chunking config but no strategy raises ValueError."""
        mock_collection.chunking_strategy = None  # No existing strategy
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        with pytest.raises(ValueError, match="chunking_config requires chunking_strategy"):
            await collection_service.update(
                collection_id=str(mock_collection.uuid),
                user_id=mock_collection.owner_id,
                updates={"chunking_config": {"max_chunk_size": 1000}},
            )


class TestEnableSparseIndex:
    @pytest.mark.asyncio()
    async def test_enable_sparse_index_loads_sparse_indexer_plugins_before_validation(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        """enable_sparse_index should load sparse indexer plugins before validating plugin_id."""

        class DummySparseIndexer:
            SPARSE_TYPE = "bm25"

        plugin_record = MagicMock()
        plugin_record.plugin_type = "sparse_indexer"
        plugin_record.plugin_class = DummySparseIndexer

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        mock_async_qdrant = AsyncMock()
        mock_async_qdrant.close = AsyncMock()

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_async_qdrant) as mock_qdrant_client,
            patch(
                "shared.config.settings",
                new=MagicMock(QDRANT_HOST="localhost", QDRANT_PORT=6333, QDRANT_API_KEY="test-api-key"),
            ),
            patch("shared.database.collection_metadata.get_sparse_index_config", new=AsyncMock(return_value=None)),
            patch("shared.database.collection_metadata.store_sparse_index_config", new=AsyncMock()),
            patch("vecpipe.sparse.ensure_sparse_collection", new=AsyncMock()),
            patch("vecpipe.sparse.generate_sparse_collection_name", return_value="sparse_test_collection"),
            patch("shared.plugins.load_plugins") as mock_load_plugins,
            patch("shared.plugins.plugin_registry.find_by_id", return_value=plugin_record) as mock_find_by_id,
        ):
            result = await collection_service.enable_sparse_index(
                collection_id=str(mock_collection.id),
                user_id=mock_collection.owner_id,
                plugin_id="bm25-local",
            )

        mock_qdrant_client.assert_called_once_with(url="http://localhost:6333", api_key="test-api-key")
        mock_load_plugins.assert_called_once_with(plugin_types={"sparse_indexer"})
        mock_find_by_id.assert_called_once_with("bm25-local")
        assert result["enabled"] is True
        assert result["plugin_id"] == "bm25-local"
        assert result["sparse_collection_name"] == "sparse_test_collection"


class TestGetSparseIndexConfig:
    @pytest.mark.asyncio()
    async def test_get_sparse_index_config_passes_api_key(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        mock_async_qdrant = AsyncMock()
        mock_async_qdrant.close = AsyncMock()
        expected_config = {"enabled": True, "plugin_id": "bm25-local"}

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_async_qdrant) as mock_qdrant_client,
            patch(
                "shared.config.settings",
                new=MagicMock(QDRANT_HOST="localhost", QDRANT_PORT=6333, QDRANT_API_KEY="test-api-key"),
            ),
            patch(
                "shared.database.collection_metadata.get_sparse_index_config",
                new=AsyncMock(return_value=expected_config),
            ),
        ):
            result = await collection_service.get_sparse_index_config(
                collection_id=str(mock_collection.id),
                user_id=mock_collection.owner_id,
            )

        mock_qdrant_client.assert_called_once_with(url="http://localhost:6333", api_key="test-api-key")
        assert result == expected_config


class TestDisableSparseIndex:
    @pytest.mark.asyncio()
    async def test_disable_sparse_index_deletes_collection_and_config(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        mock_async_qdrant = AsyncMock()
        mock_async_qdrant.close = AsyncMock()

        sparse_cfg = {"enabled": True, "sparse_collection_name": "dense_sparse_bm25"}

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_async_qdrant) as mock_qdrant_client,
            patch(
                "shared.config.settings",
                new=MagicMock(QDRANT_HOST="localhost", QDRANT_PORT=6333, QDRANT_API_KEY="test-api-key"),
            ),
            patch(
                "shared.database.collection_metadata.get_sparse_index_config", new=AsyncMock(return_value=sparse_cfg)
            ),
            patch("shared.database.collection_metadata.delete_sparse_index_config", new=AsyncMock()) as mock_delete_cfg,
            patch("vecpipe.sparse.delete_sparse_collection", new=AsyncMock()) as mock_delete_collection,
        ):
            await collection_service.disable_sparse_index(
                collection_id=str(mock_collection.id),
                user_id=mock_collection.owner_id,
            )

        mock_qdrant_client.assert_called_once_with(url="http://localhost:6333", api_key="test-api-key")
        mock_delete_collection.assert_awaited_once_with("dense_sparse_bm25", mock_async_qdrant)
        mock_delete_cfg.assert_awaited_once_with(mock_async_qdrant, mock_collection.vector_store_name)
        mock_async_qdrant.close.assert_awaited_once()


class TestTriggerSparseReindex:
    @pytest.mark.asyncio()
    async def test_trigger_sparse_reindex_dispatches_celery_task(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        mock_async_qdrant = AsyncMock()
        mock_async_qdrant.close = AsyncMock()

        sparse_cfg = {"enabled": True, "plugin_id": "bm25-local", "model_config": {"k1": 1.2}}

        job = MagicMock()
        job.id = "job-123"

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_async_qdrant),
            patch(
                "shared.config.settings",
                new=MagicMock(QDRANT_HOST="localhost", QDRANT_PORT=6333, QDRANT_API_KEY="test-api-key"),
            ),
            patch(
                "shared.database.collection_metadata.get_sparse_index_config", new=AsyncMock(return_value=sparse_cfg)
            ),
            patch("webui.services.collection_service.celery_app.send_task", return_value=job) as mock_send_task,
        ):
            result = await collection_service.trigger_sparse_reindex(
                collection_id=str(mock_collection.id),
                user_id=mock_collection.owner_id,
            )

        mock_send_task.assert_called_once_with(
            "sparse.reindex_collection",
            args=[str(mock_collection.id), "bm25-local", {"k1": 1.2}],
        )
        assert result["job_id"] == "job-123"
        assert result["status"] == "queued"
        assert result["plugin_id"] == "bm25-local"

    @pytest.mark.asyncio()
    async def test_trigger_sparse_reindex_raises_when_not_enabled(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        from shared.database.exceptions import EntityNotFoundError

        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        mock_async_qdrant = AsyncMock()
        mock_async_qdrant.close = AsyncMock()

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_async_qdrant),
            patch(
                "shared.config.settings",
                new=MagicMock(QDRANT_HOST="localhost", QDRANT_PORT=6333, QDRANT_API_KEY="test-api-key"),
            ),
            patch("shared.database.collection_metadata.get_sparse_index_config", new=AsyncMock(return_value=None)),
        ):
            with pytest.raises(EntityNotFoundError):
                await collection_service.trigger_sparse_reindex(
                    collection_id=str(mock_collection.id),
                    user_id=mock_collection.owner_id,
                )


class TestGetSparseReindexProgress:
    @pytest.mark.asyncio()
    async def test_get_sparse_reindex_progress_progress_state(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        result_obj = MagicMock()
        result_obj.state = "PROGRESS"
        result_obj.info = {"progress": 12.5, "documents_processed": 3, "total_documents": 10}

        with patch("celery.result.AsyncResult", return_value=result_obj):
            progress = await collection_service.get_sparse_reindex_progress(
                collection_id=str(mock_collection.id),
                user_id=mock_collection.owner_id,
                job_id="job-1",
            )

        assert progress["status"] == "PROGRESS"
        assert progress["progress"] == 12.5
        assert progress["documents_processed"] == 3
        assert progress["total_documents"] == 10

    @pytest.mark.asyncio()
    async def test_get_sparse_reindex_progress_failure_state(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        result_obj = MagicMock()
        result_obj.state = "FAILURE"
        result_obj.result = RuntimeError("boom")

        with patch("celery.result.AsyncResult", return_value=result_obj):
            progress = await collection_service.get_sparse_reindex_progress(
                collection_id=str(mock_collection.id),
                user_id=mock_collection.owner_id,
                job_id="job-1",
            )

        assert progress["status"] == "FAILURE"
        assert "boom" in progress["error"]

    @pytest.mark.asyncio()
    async def test_get_sparse_reindex_progress_success_state(
        self,
        collection_service: CollectionService,
        mock_collection_repo: AsyncMock,
        mock_collection: MagicMock,
    ) -> None:
        mock_collection_repo.get_by_uuid_with_permission_check.return_value = mock_collection

        result_obj = MagicMock()
        result_obj.state = "SUCCESS"
        result_obj.result = {"documents_processed": 7, "total_documents": 7}

        with patch("celery.result.AsyncResult", return_value=result_obj):
            progress = await collection_service.get_sparse_reindex_progress(
                collection_id=str(mock_collection.id),
                user_id=mock_collection.owner_id,
                job_id="job-1",
            )

        assert progress["status"] == "SUCCESS"
        assert progress["progress"] == 100.0
        assert progress["documents_processed"] == 7
        assert progress["total_documents"] == 7
