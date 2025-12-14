"""Unit tests for SourceService.

Note: Sync policy (mode, interval, pause/resume) is now managed at collection level.
Sources only track per-source telemetry (last_run_* fields).
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from shared.database.exceptions import AccessDeniedError, EntityNotFoundError, InvalidStateError
from shared.database.models import Collection, CollectionSource, Operation, OperationStatus, OperationType
from shared.utils.encryption import EncryptionNotConfiguredError
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
    def service(self, mock_db_session, mock_collection_repo, mock_source_repo, mock_operation_repo) -> SourceService:
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
    def sample_source(self, sample_collection) -> MagicMock:
        """Create a sample source for testing.

        Note: sync_mode, interval_minutes, paused_at, next_run_at are no longer
        on CollectionSource model - they're at collection level. However, the service
        code still references them, so we use MagicMock to provide these attributes.
        """
        source = MagicMock(spec=CollectionSource)
        source.id = 1
        source.collection_id = sample_collection.id
        source.source_type = "directory"
        source.source_path = "/data/test"
        source.source_config = {"path": "/data/test"}
        # Legacy attributes still referenced in service code
        source.sync_mode = "one_time"
        source.interval_minutes = None
        source.paused_at = None
        return source

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
        )
        mock_source_repo.update.return_value = updated_source

        result = await service.update_source(
            user_id=sample_collection.owner_id,
            source_id=sample_source.id,
            source_config={"path": "/data/updated"},
        )

        # update_source returns (source, secret_types) tuple
        source, secret_types = result
        assert source == updated_source
        assert secret_types == []
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

    @pytest.mark.asyncio()
    async def test_update_source_rejects_secrets_without_encryption(
        self, service, mock_collection_repo, mock_source_repo, sample_collection, sample_source
    ) -> None:
        """Reject secrets updates when connector secrets encryption is disabled."""
        mock_source_repo.get_by_id.return_value = sample_source
        mock_collection_repo.get_by_id.return_value = sample_collection

        with pytest.raises(EncryptionNotConfiguredError):
            await service.update_source(
                user_id=sample_collection.owner_id,
                source_id=sample_source.id,
                secrets={"password": "super-secret"},
            )

        mock_source_repo.update.assert_not_called()

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

        # get_source returns just source when include_secret_types=False (default)
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

        # list_sources returns list of (source, secret_types) tuples when include_secret_types=True
        result, total = await service.list_sources(
            user_id=sample_collection.owner_id,
            collection_id=sample_collection.id,
            include_secret_types=True,
        )

        assert len(result) == 1
        assert total == 1
        mock_source_repo.list_by_collection.assert_called_once()

    @pytest.mark.asyncio()
    async def test_list_sources_collection_not_found(self, service, mock_collection_repo) -> None:
        """Test listing sources for non-existent collection."""
        mock_collection_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFoundError):
            await service.list_sources(user_id=1, collection_id=str(uuid4()))

    # --- Additional fixtures for secret repository ---

    @pytest.fixture()
    def mock_secret_repo(self) -> AsyncMock:
        """Create a mock connector secret repository."""
        mock = AsyncMock()
        mock.set_secret = AsyncMock()
        mock.get_secret_types_for_source = AsyncMock(return_value=[])
        mock.delete_secret = AsyncMock()
        return mock

    @pytest.fixture()
    def service_with_secrets(
        self, mock_db_session, mock_collection_repo, mock_source_repo, mock_operation_repo, mock_secret_repo
    ) -> SourceService:
        """Create a SourceService instance with secret repository enabled."""
        return SourceService(
            db_session=mock_db_session,
            collection_repo=mock_collection_repo,
            source_repo=mock_source_repo,
            operation_repo=mock_operation_repo,
            secret_repo=mock_secret_repo,
        )

    @pytest.fixture()
    def sample_continuous_source(self, sample_collection) -> MagicMock:
        """Create a mock source with continuous sync mode.

        Note: sync_mode and interval_minutes are now at collection level,
        but the service code still references them on source. We use MagicMock
        to provide these attributes for testing.
        """
        source = MagicMock(spec=CollectionSource)
        source.id = 2
        source.collection_id = sample_collection.id
        source.source_type = "git"
        source.source_path = "https://github.com/example/repo"
        source.source_config = {"url": "https://github.com/example/repo"}
        # Legacy attributes still referenced in service code
        source.sync_mode = "continuous"
        source.interval_minutes = 30
        source.paused_at = None
        return source

    # --- create_source tests ---

    @pytest.mark.asyncio()
    async def test_create_source_success_without_secrets(
        self, service, mock_collection_repo, mock_source_repo, sample_collection
    ) -> None:
        """Test successful source creation without secrets."""
        mock_collection_repo.get_by_id.return_value = sample_collection

        new_source = CollectionSource(
            id=1,
            collection_id=sample_collection.id,
            source_type="directory",
            source_path="/data/test",
            source_config={"path": "/data/test"},
        )
        mock_source_repo.create.return_value = new_source

        result = await service.create_source(
            user_id=sample_collection.owner_id,
            collection_id=sample_collection.id,
            source_type="directory",
            source_path="/data/test",
            source_config={"path": "/data/test"},
        )

        source, secret_types = result
        assert source == new_source
        assert secret_types == []
        mock_source_repo.create.assert_called_once()

    @pytest.mark.asyncio()
    async def test_create_source_success_with_secrets(
        self, service_with_secrets, mock_collection_repo, mock_source_repo, mock_secret_repo, sample_collection
    ) -> None:
        """Test successful source creation with encrypted secrets."""
        mock_collection_repo.get_by_id.return_value = sample_collection

        new_source = CollectionSource(
            id=1,
            collection_id=sample_collection.id,
            source_type="git",
            source_path="https://github.com/example/repo",
            source_config={"url": "https://github.com/example/repo"},
        )
        mock_source_repo.create.return_value = new_source

        result = await service_with_secrets.create_source(
            user_id=sample_collection.owner_id,
            collection_id=sample_collection.id,
            source_type="git",
            source_path="https://github.com/example/repo",
            source_config={"url": "https://github.com/example/repo"},
            secrets={"password": "super-secret", "token": "api-token"},
        )

        source, secret_types = result
        assert source == new_source
        assert "password" in secret_types
        assert "token" in secret_types
        assert mock_secret_repo.set_secret.call_count == 2

    @pytest.mark.asyncio()
    async def test_create_source_collection_not_found(self, service, mock_collection_repo) -> None:
        """Test creating source for non-existent collection."""
        mock_collection_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFoundError) as exc_info:
            await service.create_source(
                user_id=1,
                collection_id=str(uuid4()),
                source_type="directory",
                source_path="/data/test",
                source_config={"path": "/data/test"},
            )
        assert "collection" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_create_source_access_denied(self, service, mock_collection_repo, sample_collection) -> None:
        """Test creating source without access to collection."""
        mock_collection_repo.get_by_id.return_value = sample_collection

        with pytest.raises(AccessDeniedError):
            await service.create_source(
                user_id=999,  # Different user
                collection_id=sample_collection.id,
                source_type="directory",
                source_path="/data/test",
                source_config={"path": "/data/test"},
            )

    @pytest.mark.asyncio()
    async def test_create_source_secrets_without_encryption(
        self, service, mock_collection_repo, sample_collection
    ) -> None:
        """Test creating source with secrets when encryption not configured."""
        mock_collection_repo.get_by_id.return_value = sample_collection

        with pytest.raises(EncryptionNotConfiguredError):
            await service.create_source(
                user_id=sample_collection.owner_id,
                collection_id=sample_collection.id,
                source_type="git",
                source_path="https://github.com/example/repo",
                source_config={"url": "https://github.com/example/repo"},
                secrets={"password": "super-secret"},
            )

    @pytest.mark.asyncio()
    async def test_create_source_returns_stored_secret_types(
        self, service_with_secrets, mock_collection_repo, mock_source_repo, mock_secret_repo, sample_collection
    ) -> None:
        """Test that create_source returns the list of stored secret types."""
        mock_collection_repo.get_by_id.return_value = sample_collection

        new_source = MagicMock(spec=CollectionSource)
        new_source.id = 1
        new_source.collection_id = sample_collection.id
        new_source.source_type = "imap"
        new_source.source_path = "imap.example.com"
        new_source.source_config = {"host": "imap.example.com"}
        mock_source_repo.create.return_value = new_source

        result = await service_with_secrets.create_source(
            user_id=sample_collection.owner_id,
            collection_id=sample_collection.id,
            source_type="imap",
            source_path="imap.example.com",
            source_config={"host": "imap.example.com"},
            secrets={"password": "secret", "empty_key": ""},  # Empty VALUE should be filtered
        )

        source, secret_types = result
        assert source == new_source
        assert secret_types == ["password"]  # Empty value secret should be filtered out
        mock_secret_repo.set_secret.assert_called_once_with(1, "password", "secret")

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
            config={"source_id": sample_source.id},
        )
        mock_operation_repo.create.return_value = operation

        result = await service.run_now(
            user_id=sample_collection.owner_id,
            source_id=sample_source.id,
        )

        assert result["uuid"] == operation.uuid
        assert result["type"] == "append"
        mock_db_session.commit.assert_called_once()
        mock_celery.send_task.assert_called_once()

    @pytest.mark.asyncio()
    async def test_run_now_source_not_found(self, service, mock_source_repo) -> None:
        """Test run_now for non-existent source."""
        mock_source_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFoundError) as exc_info:
            await service.run_now(user_id=1, source_id=999)
        assert "collection_source" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_run_now_access_denied(
        self, service, mock_collection_repo, mock_source_repo, sample_collection, sample_source
    ) -> None:
        """Test run_now without access to collection."""
        mock_source_repo.get_by_id.return_value = sample_source
        mock_collection_repo.get_by_id.return_value = sample_collection

        with pytest.raises(AccessDeniedError):
            await service.run_now(user_id=999, source_id=sample_source.id)

    @pytest.mark.asyncio()
    async def test_run_now_active_operation_blocks(
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

    @pytest.mark.asyncio()
    @patch("webui.services.source_service.celery_app")
    async def test_run_now_dispatches_celery_task(
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
        """Test that run_now dispatches Celery task with correct args."""
        mock_source_repo.get_by_id.return_value = sample_source
        mock_collection_repo.get_by_id.return_value = sample_collection
        mock_operation_repo.get_active_operations.return_value = []

        operation_uuid = str(uuid4())
        operation = Operation(
            id=1,
            uuid=operation_uuid,
            collection_id=sample_collection.id,
            user_id=sample_collection.owner_id,
            type=OperationType.APPEND,
            status=OperationStatus.PENDING,
            config={"source_id": sample_source.id},
        )
        mock_operation_repo.create.return_value = operation

        await service.run_now(
            user_id=sample_collection.owner_id,
            source_id=sample_source.id,
        )

        mock_celery.send_task.assert_called_once_with(
            "webui.tasks.process_collection_operation",
            args=[operation_uuid],
            queue="default",
        )

    @pytest.mark.asyncio()
    @patch("webui.services.source_service.celery_app")
    async def test_run_now_updates_sync_status(
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
        """Test that run_now updates sync status to partial."""
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
            config={"source_id": sample_source.id},
        )
        mock_operation_repo.create.return_value = operation

        await service.run_now(
            user_id=sample_collection.owner_id,
            source_id=sample_source.id,
        )

        mock_source_repo.update_sync_status.assert_called_once()
        call_kwargs = mock_source_repo.update_sync_status.call_args[1]
        assert call_kwargs["source_id"] == sample_source.id
        assert call_kwargs["status"] == "partial"

    # --- pause tests ---

    @pytest.mark.asyncio()
    async def test_pause_success(
        self, service, mock_collection_repo, mock_source_repo, sample_collection, sample_continuous_source
    ) -> None:
        """Test successful pause of continuous sync."""
        mock_source_repo.get_by_id.return_value = sample_continuous_source
        mock_collection_repo.get_by_id.return_value = sample_collection

        # Create a mock paused source with paused_at set
        paused_source = MagicMock(spec=CollectionSource)
        paused_source.id = sample_continuous_source.id
        paused_source.collection_id = sample_collection.id
        paused_source.source_type = "git"
        paused_source.source_path = sample_continuous_source.source_path
        paused_source.source_config = sample_continuous_source.source_config
        paused_source.sync_mode = "continuous"
        paused_source.paused_at = datetime.now(UTC)
        mock_source_repo.pause.return_value = paused_source

        result = await service.pause(
            user_id=sample_collection.owner_id,
            source_id=sample_continuous_source.id,
        )

        assert result.paused_at is not None
        mock_source_repo.pause.assert_called_once_with(sample_continuous_source.id)

    @pytest.mark.asyncio()
    async def test_pause_source_not_found(self, service, mock_source_repo) -> None:
        """Test pausing non-existent source."""
        mock_source_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFoundError) as exc_info:
            await service.pause(user_id=1, source_id=999)
        assert "collection_source" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_pause_access_denied(
        self, service, mock_collection_repo, mock_source_repo, sample_collection, sample_continuous_source
    ) -> None:
        """Test pausing without access to collection."""
        mock_source_repo.get_by_id.return_value = sample_continuous_source
        mock_collection_repo.get_by_id.return_value = sample_collection

        with pytest.raises(AccessDeniedError):
            await service.pause(user_id=999, source_id=sample_continuous_source.id)

    # --- resume tests ---

    @pytest.mark.asyncio()
    async def test_resume_success(
        self, service, mock_collection_repo, mock_source_repo, sample_collection, sample_continuous_source
    ) -> None:
        """Test successful resume of paused sync."""
        sample_continuous_source.paused_at = datetime.now(UTC)
        mock_source_repo.get_by_id.return_value = sample_continuous_source
        mock_collection_repo.get_by_id.return_value = sample_collection

        # Create a mock resumed source with paused_at cleared
        resumed_source = MagicMock(spec=CollectionSource)
        resumed_source.id = sample_continuous_source.id
        resumed_source.collection_id = sample_collection.id
        resumed_source.source_type = "git"
        resumed_source.source_path = sample_continuous_source.source_path
        resumed_source.source_config = sample_continuous_source.source_config
        resumed_source.sync_mode = "continuous"
        resumed_source.paused_at = None
        mock_source_repo.resume.return_value = resumed_source

        result = await service.resume(
            user_id=sample_collection.owner_id,
            source_id=sample_continuous_source.id,
        )

        assert result.paused_at is None
        mock_source_repo.resume.assert_called_once_with(sample_continuous_source.id)

    @pytest.mark.asyncio()
    async def test_resume_source_not_found(self, service, mock_source_repo) -> None:
        """Test resuming non-existent source."""
        mock_source_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFoundError) as exc_info:
            await service.resume(user_id=1, source_id=999)
        assert "collection_source" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_resume_access_denied(
        self, service, mock_collection_repo, mock_source_repo, sample_collection, sample_continuous_source
    ) -> None:
        """Test resuming without access to collection."""
        sample_continuous_source.paused_at = datetime.now(UTC)
        mock_source_repo.get_by_id.return_value = sample_continuous_source
        mock_collection_repo.get_by_id.return_value = sample_collection

        with pytest.raises(AccessDeniedError):
            await service.resume(user_id=999, source_id=sample_continuous_source.id)

    # --- Edge case tests for existing methods ---

    @pytest.mark.asyncio()
    async def test_get_source_with_secret_types(
        self,
        service_with_secrets,
        mock_collection_repo,
        mock_source_repo,
        mock_secret_repo,
        sample_collection,
        sample_source,
    ) -> None:
        """Test get_source with include_secret_types=True."""
        mock_source_repo.get_by_id.return_value = sample_source
        mock_collection_repo.get_by_id.return_value = sample_collection
        mock_secret_repo.get_secret_types_for_source.return_value = ["password", "token"]

        result = await service_with_secrets.get_source(
            user_id=sample_collection.owner_id,
            source_id=sample_source.id,
            include_secret_types=True,
        )

        source, secret_types = result
        assert source == sample_source
        assert secret_types == ["password", "token"]
        mock_secret_repo.get_secret_types_for_source.assert_called_once_with(sample_source.id)

    @pytest.mark.asyncio()
    async def test_get_source_access_denied(
        self, service, mock_collection_repo, mock_source_repo, sample_collection, sample_source
    ) -> None:
        """Test get_source without access to collection."""
        mock_source_repo.get_by_id.return_value = sample_source
        mock_collection_repo.get_by_id.return_value = sample_collection

        with pytest.raises(AccessDeniedError):
            await service.get_source(user_id=999, source_id=sample_source.id)

    @pytest.mark.asyncio()
    async def test_list_sources_access_denied(self, service, mock_collection_repo, sample_collection) -> None:
        """Test list_sources without access to collection."""
        mock_collection_repo.get_by_id.return_value = sample_collection

        with pytest.raises(AccessDeniedError):
            await service.list_sources(user_id=999, collection_id=sample_collection.id)

    @pytest.mark.asyncio()
    async def test_update_source_deletes_secret_with_empty_string(
        self,
        service_with_secrets,
        mock_collection_repo,
        mock_source_repo,
        mock_secret_repo,
        sample_collection,
        sample_source,
    ) -> None:
        """Test that passing empty string for a secret deletes it."""
        mock_source_repo.get_by_id.return_value = sample_source
        mock_collection_repo.get_by_id.return_value = sample_collection
        mock_source_repo.update.return_value = sample_source
        mock_secret_repo.get_secret_types_for_source.return_value = []

        await service_with_secrets.update_source(
            user_id=sample_collection.owner_id,
            source_id=sample_source.id,
            secrets={"password": ""},  # Empty string should trigger delete
        )

        mock_secret_repo.delete_secret.assert_called_once_with(sample_source.id, "password")
        mock_secret_repo.set_secret.assert_not_called()

    @pytest.mark.asyncio()
    async def test_delete_source_not_found(self, service, mock_source_repo) -> None:
        """Test deleting non-existent source."""
        mock_source_repo.get_by_id.return_value = None

        with pytest.raises(EntityNotFoundError) as exc_info:
            await service.delete_source(user_id=1, source_id=999)
        assert "collection_source" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_delete_source_access_denied(
        self, service, mock_collection_repo, mock_source_repo, sample_collection, sample_source
    ) -> None:
        """Test deleting source without access to collection."""
        mock_source_repo.get_by_id.return_value = sample_source
        mock_collection_repo.get_by_id.return_value = sample_collection

        with pytest.raises(AccessDeniedError):
            await service.delete_source(user_id=999, source_id=sample_source.id)
