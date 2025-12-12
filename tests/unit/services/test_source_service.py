"""Unit tests for SourceService."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from shared.database.exceptions import AccessDeniedError, EntityNotFoundError, InvalidStateError
from shared.database.models import Collection, CollectionSource, Operation, OperationStatus, OperationType
from webui.services.source_service import SourceService


class TestSourceService:
    """Test cases for SourceService."""

    @pytest.fixture()
    def mock_db_session(self) -> AsyncMock:
        """Create a mock async session."""
        session = AsyncMock()
        session.commit = AsyncMock()
        return session

    @pytest.fixture()
    def mock_collection_repo(self) -> AsyncMock:
        """Create a mock collection repository."""
        return AsyncMock()

    @pytest.fixture()
    def mock_source_repo(self) -> AsyncMock:
        """Create a mock source repository."""
        return AsyncMock()

    @pytest.fixture()
    def mock_operation_repo(self) -> AsyncMock:
        """Create a mock operation repository."""
        return AsyncMock()

    @pytest.fixture()
    def service(
        self, mock_db_session, mock_collection_repo, mock_source_repo, mock_operation_repo
    ) -> SourceService:
        """Create a SourceService instance with mocked dependencies."""
        return SourceService(
            db_session=mock_db_session,
            collection_repo=mock_collection_repo,
            source_repo=mock_source_repo,
            operation_repo=mock_operation_repo,
        )

    @pytest.fixture()
    def sample_collection(self) -> Collection:
        """Create a sample collection for testing."""
        collection = Collection(
            id=str(uuid4()),
            name="test-collection",
            is_public=False,
        )
        # Set owner_id attribute directly (used for access control checks)
        collection.owner_id = 1
        return collection

    @pytest.fixture()
    def sample_source(self, sample_collection) -> CollectionSource:
        """Create a sample source for testing."""
        return CollectionSource(
            id=1,
            collection_id=sample_collection.id,
            source_type="directory",
            source_path="/data/test",
            source_config={"path": "/data/test"},
            sync_mode="one_time",
        )

    # --- create_source tests ---

    @pytest.mark.asyncio()
    async def test_create_source_success(
        self, service, mock_collection_repo, mock_source_repo, sample_collection
    ) -> None:
        """Test successful source creation."""
        mock_collection_repo.get_by_id.return_value = sample_collection

        expected_source = CollectionSource(
            id=1,
            collection_id=sample_collection.id,
            source_type="directory",
            source_path="/data/new",
            source_config={"path": "/data/new"},
            sync_mode="one_time",
        )
        mock_source_repo.create.return_value = expected_source

        result = await service.create_source(
            user_id=sample_collection.owner_id,
            collection_id=sample_collection.id,
            source_type="directory",
            source_path="/data/new",
            source_config={"path": "/data/new"},
        )

        assert result == expected_source
        mock_collection_repo.get_by_id.assert_called_once_with(sample_collection.id)
        mock_source_repo.create.assert_called_once()

    @pytest.mark.asyncio()
    async def test_create_source_collection_not_found(
        self, service, mock_collection_repo
    ) -> None:
        """Test source creation with non-existent collection."""
        mock_collection_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFoundError) as exc_info:
            await service.create_source(
                user_id=1,
                collection_id=str(uuid4()),
                source_type="directory",
                source_path="/data/new",
                source_config={},
            )
        assert "collection" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_create_source_access_denied(
        self, service, mock_collection_repo, sample_collection
    ) -> None:
        """Test source creation with wrong user."""
        mock_collection_repo.get_by_id.return_value = sample_collection

        with pytest.raises(AccessDeniedError):
            await service.create_source(
                user_id=999,  # Different user
                collection_id=sample_collection.id,
                source_type="directory",
                source_path="/data/new",
                source_config={},
            )

    # --- update_source tests ---

    @pytest.mark.asyncio()
    async def test_update_source_success(
        self, service, mock_collection_repo, mock_source_repo, sample_collection, sample_source
    ) -> None:
        """Test successful source update."""
        mock_source_repo.get_by_id.return_value = sample_source
        mock_collection_repo.get_by_id.return_value = sample_collection

        updated_source = CollectionSource(
            id=sample_source.id,
            collection_id=sample_collection.id,
            source_type="directory",
            source_path=sample_source.source_path,
            source_config={"path": "/data/updated"},
            sync_mode="continuous",
            interval_minutes=30,
        )
        mock_source_repo.update.return_value = updated_source

        result = await service.update_source(
            user_id=sample_collection.owner_id,
            source_id=sample_source.id,
            source_config={"path": "/data/updated"},
            sync_mode="continuous",
            interval_minutes=30,
        )

        assert result == updated_source
        mock_source_repo.update.assert_called_once()

    @pytest.mark.asyncio()
    async def test_update_source_not_found(self, service, mock_source_repo) -> None:
        """Test updating non-existent source."""
        mock_source_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFoundError) as exc_info:
            await service.update_source(user_id=1, source_id=999)
        assert "collection_source" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_update_source_access_denied(
        self, service, mock_collection_repo, mock_source_repo, sample_collection, sample_source
    ) -> None:
        """Test updating source without access."""
        mock_source_repo.get_by_id.return_value = sample_source
        mock_collection_repo.get_by_id.return_value = sample_collection

        with pytest.raises(AccessDeniedError):
            await service.update_source(user_id=999, source_id=sample_source.id)

    # --- delete_source tests ---

    @pytest.mark.asyncio()
    @patch("webui.services.source_service.celery_app")
    async def test_delete_source_success(
        self,
        mock_celery,
        service,
        mock_db_session,
        mock_collection_repo,
        mock_source_repo,
        mock_operation_repo,
        sample_collection,
        sample_source,
    ) -> None:
        """Test successful source deletion."""
        mock_source_repo.get_by_id.return_value = sample_source
        mock_collection_repo.get_by_id.return_value = sample_collection
        mock_operation_repo.get_active_operations.return_value = []

        operation = Operation(
            id=1,
            uuid=str(uuid4()),
            collection_id=sample_collection.id,
            user_id=sample_collection.owner_id,
            type=OperationType.REMOVE_SOURCE,
            status=OperationStatus.PENDING,
            config={"source_id": sample_source.id},
        )
        mock_operation_repo.create.return_value = operation

        result = await service.delete_source(
            user_id=sample_collection.owner_id,
            source_id=sample_source.id,
        )

        assert result["uuid"] == operation.uuid
        assert result["type"] == "remove_source"
        mock_db_session.commit.assert_called_once()
        mock_celery.send_task.assert_called_once()

    @pytest.mark.asyncio()
    async def test_delete_source_active_operation(
        self,
        service,
        mock_collection_repo,
        mock_source_repo,
        mock_operation_repo,
        sample_collection,
        sample_source,
    ) -> None:
        """Test delete source blocked by active operation."""
        mock_source_repo.get_by_id.return_value = sample_source
        mock_collection_repo.get_by_id.return_value = sample_collection
        mock_operation_repo.get_active_operations.return_value = [MagicMock()]

        with pytest.raises(InvalidStateError) as exc_info:
            await service.delete_source(
                user_id=sample_collection.owner_id,
                source_id=sample_source.id,
            )
        assert "active operation" in str(exc_info.value).lower()

    # --- get_source tests ---

    @pytest.mark.asyncio()
    async def test_get_source_success(
        self, service, mock_collection_repo, mock_source_repo, sample_collection, sample_source
    ) -> None:
        """Test successful source retrieval."""
        mock_source_repo.get_by_id.return_value = sample_source
        mock_collection_repo.get_by_id.return_value = sample_collection

        result = await service.get_source(
            user_id=sample_collection.owner_id,
            source_id=sample_source.id,
        )

        assert result == sample_source

    @pytest.mark.asyncio()
    async def test_get_source_not_found(self, service, mock_source_repo) -> None:
        """Test getting non-existent source."""
        mock_source_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFoundError):
            await service.get_source(user_id=1, source_id=999)

    # --- list_sources tests ---

    @pytest.mark.asyncio()
    async def test_list_sources_success(
        self, service, mock_collection_repo, mock_source_repo, sample_collection, sample_source
    ) -> None:
        """Test successful source listing."""
        mock_collection_repo.get_by_id.return_value = sample_collection
        mock_source_repo.list_by_collection.return_value = ([sample_source], 1)

        sources, total = await service.list_sources(
            user_id=sample_collection.owner_id,
            collection_id=sample_collection.id,
        )

        assert len(sources) == 1
        assert total == 1
        mock_source_repo.list_by_collection.assert_called_once()

    @pytest.mark.asyncio()
    async def test_list_sources_collection_not_found(
        self, service, mock_collection_repo
    ) -> None:
        """Test listing sources for non-existent collection."""
        mock_collection_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFoundError):
            await service.list_sources(user_id=1, collection_id=str(uuid4()))

    # --- run_now tests ---

    @pytest.mark.asyncio()
    @patch("webui.services.source_service.celery_app")
    async def test_run_now_success(
        self,
        mock_celery,
        service,
        mock_db_session,
        mock_collection_repo,
        mock_source_repo,
        mock_operation_repo,
        sample_collection,
        sample_source,
    ) -> None:
        """Test successful manual sync trigger."""
        mock_source_repo.get_by_id.return_value = sample_source
        mock_collection_repo.get_by_id.return_value = sample_collection
        mock_operation_repo.get_active_operations.return_value = []

        operation = Operation(
            id=1,
            uuid=str(uuid4()),
            collection_id=sample_collection.id,
            user_id=sample_collection.owner_id,
            type=OperationType.APPEND,
            status=OperationStatus.PENDING,
            config={},
        )
        mock_operation_repo.create.return_value = operation

        result = await service.run_now(
            user_id=sample_collection.owner_id,
            source_id=sample_source.id,
        )

        assert result["uuid"] == operation.uuid
        assert result["type"] == "append"
        mock_source_repo.update_sync_status.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_celery.send_task.assert_called_once()

    @pytest.mark.asyncio()
    @patch("webui.services.source_service.celery_app")
    async def test_run_now_continuous_updates_next_run(
        self,
        mock_celery,
        service,
        mock_db_session,
        mock_collection_repo,
        mock_source_repo,
        mock_operation_repo,
        sample_collection,
    ) -> None:
        """Test run_now updates next_run_at for continuous sync sources."""
        continuous_source = CollectionSource(
            id=1,
            collection_id=sample_collection.id,
            source_type="directory",
            source_path="/data/test",
            source_config={},
            sync_mode="continuous",
            interval_minutes=30,
        )
        mock_source_repo.get_by_id.return_value = continuous_source
        mock_collection_repo.get_by_id.return_value = sample_collection
        mock_operation_repo.get_active_operations.return_value = []

        operation = Operation(
            id=1,
            uuid=str(uuid4()),
            collection_id=sample_collection.id,
            user_id=sample_collection.owner_id,
            type=OperationType.APPEND,
            status=OperationStatus.PENDING,
            config={},
        )
        mock_operation_repo.create.return_value = operation

        await service.run_now(
            user_id=sample_collection.owner_id,
            source_id=continuous_source.id,
        )

        mock_source_repo.set_next_run.assert_called_once_with(continuous_source.id)

    @pytest.mark.asyncio()
    async def test_run_now_active_operation(
        self,
        service,
        mock_collection_repo,
        mock_source_repo,
        mock_operation_repo,
        sample_collection,
        sample_source,
    ) -> None:
        """Test run_now blocked by active operation."""
        mock_source_repo.get_by_id.return_value = sample_source
        mock_collection_repo.get_by_id.return_value = sample_collection
        mock_operation_repo.get_active_operations.return_value = [MagicMock()]

        with pytest.raises(InvalidStateError) as exc_info:
            await service.run_now(
                user_id=sample_collection.owner_id,
                source_id=sample_source.id,
            )
        assert "active operation" in str(exc_info.value).lower()

    # --- pause tests ---

    @pytest.mark.asyncio()
    async def test_pause_success(
        self, service, mock_collection_repo, mock_source_repo, sample_collection
    ) -> None:
        """Test successful source pause."""
        continuous_source = CollectionSource(
            id=1,
            collection_id=sample_collection.id,
            source_type="directory",
            source_path="/data/test",
            source_config={},
            sync_mode="continuous",
            interval_minutes=30,
        )
        mock_source_repo.get_by_id.return_value = continuous_source
        mock_collection_repo.get_by_id.return_value = sample_collection

        paused_source = CollectionSource(
            id=continuous_source.id,
            collection_id=sample_collection.id,
            source_type="directory",
            source_path="/data/test",
            source_config={},
            sync_mode="continuous",
            interval_minutes=30,
            paused_at=datetime.now(UTC),
        )
        mock_source_repo.pause.return_value = paused_source

        result = await service.pause(
            user_id=sample_collection.owner_id,
            source_id=continuous_source.id,
        )

        assert result.paused_at is not None
        mock_source_repo.pause.assert_called_once_with(continuous_source.id)

    @pytest.mark.asyncio()
    async def test_pause_source_not_found(self, service, mock_source_repo) -> None:
        """Test pausing non-existent source."""
        mock_source_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFoundError):
            await service.pause(user_id=1, source_id=999)

    # --- resume tests ---

    @pytest.mark.asyncio()
    async def test_resume_success(
        self, service, mock_collection_repo, mock_source_repo, sample_collection
    ) -> None:
        """Test successful source resume."""
        paused_source = CollectionSource(
            id=1,
            collection_id=sample_collection.id,
            source_type="directory",
            source_path="/data/test",
            source_config={},
            sync_mode="continuous",
            interval_minutes=30,
            paused_at=datetime.now(UTC),
        )
        mock_source_repo.get_by_id.return_value = paused_source
        mock_collection_repo.get_by_id.return_value = sample_collection

        resumed_source = CollectionSource(
            id=paused_source.id,
            collection_id=sample_collection.id,
            source_type="directory",
            source_path="/data/test",
            source_config={},
            sync_mode="continuous",
            interval_minutes=30,
            paused_at=None,
            next_run_at=datetime.now(UTC),
        )
        mock_source_repo.resume.return_value = resumed_source

        result = await service.resume(
            user_id=sample_collection.owner_id,
            source_id=paused_source.id,
        )

        assert result.paused_at is None
        assert result.next_run_at is not None
        mock_source_repo.resume.assert_called_once_with(paused_source.id)

    @pytest.mark.asyncio()
    async def test_resume_source_not_found(self, service, mock_source_repo) -> None:
        """Test resuming non-existent source."""
        mock_source_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFoundError):
            await service.resume(user_id=1, source_id=999)

    @pytest.mark.asyncio()
    async def test_resume_access_denied(
        self, service, mock_collection_repo, mock_source_repo, sample_collection
    ) -> None:
        """Test resuming source without access."""
        paused_source = CollectionSource(
            id=1,
            collection_id=sample_collection.id,
            source_type="directory",
            source_path="/data/test",
            source_config={},
            sync_mode="continuous",
            interval_minutes=30,
            paused_at=datetime.now(UTC),
        )
        mock_source_repo.get_by_id.return_value = paused_source
        mock_collection_repo.get_by_id.return_value = sample_collection

        with pytest.raises(AccessDeniedError):
            await service.resume(user_id=999, source_id=paused_source.id)
