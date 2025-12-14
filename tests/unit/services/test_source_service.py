"""Unit tests for SourceService.

Note: Sync policy (mode, interval, pause/resume) is now managed at collection level.
Sources only track per-source telemetry (last_run_* fields).
"""

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
    def sample_source(self, sample_collection) -> CollectionSource:
        """Create a sample source for testing.

        Note: sync_mode, interval_minutes, paused_at, next_run_at are no longer
        on CollectionSource - they're at collection level now.
        """
        return CollectionSource(
            id=1,
            collection_id=sample_collection.id,
            source_type="directory",
            source_path="/data/test",
            source_config={"path": "/data/test"},
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
