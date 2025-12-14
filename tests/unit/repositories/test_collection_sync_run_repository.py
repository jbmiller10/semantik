"""Unit tests for CollectionSyncRunRepository."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.models import CollectionSyncRun
from shared.database.repositories.collection_sync_run_repository import CollectionSyncRunRepository


class TestCollectionSyncRunRepository:
    """Tests for CollectionSyncRunRepository."""

    @pytest.fixture()
    def mock_session(self) -> MagicMock:
        """Create a mock async session."""
        session = MagicMock(spec=AsyncSession)
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.execute = AsyncMock()
        session.scalar = AsyncMock()
        return session

    @pytest.fixture()
    def repository(self, mock_session: MagicMock) -> CollectionSyncRunRepository:
        """Create repository instance with mock session."""
        return CollectionSyncRunRepository(mock_session)

    @pytest.fixture()
    def sample_sync_run(self) -> CollectionSyncRun:
        """Create a sample sync run for testing."""
        return CollectionSyncRun(
            id=1,
            collection_id="coll-uuid-1",
            triggered_by="scheduler",
            started_at=datetime.now(UTC),
            completed_at=None,
            status="running",
            expected_sources=3,
            completed_sources=0,
            failed_sources=0,
            partial_sources=0,
            error_summary=None,
            meta=None,
        )

    @pytest.mark.asyncio()
    async def test_create_sync_run(
        self,
        repository: CollectionSyncRunRepository,
        mock_session: MagicMock,
    ) -> None:
        """Test creating a new sync run."""
        result = await repository.create(
            collection_id="coll-uuid-1",
            triggered_by="scheduler",
            expected_sources=5,
        )

        assert result is not None
        assert result.collection_id == "coll-uuid-1"
        assert result.triggered_by == "scheduler"
        assert result.expected_sources == 5
        assert result.status == "running"
        assert result.completed_sources == 0
        assert result.failed_sources == 0
        assert result.partial_sources == 0

        mock_session.add.assert_called_once()
        mock_session.flush.assert_called_once()

    @pytest.mark.asyncio()
    async def test_get_by_id_found(
        self,
        repository: CollectionSyncRunRepository,
        mock_session: MagicMock,
        sample_sync_run: CollectionSyncRun,
    ) -> None:
        """Test getting a sync run by ID when it exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_sync_run
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(1)

        assert result == sample_sync_run

    @pytest.mark.asyncio()
    async def test_get_by_id_not_found(
        self,
        repository: CollectionSyncRunRepository,
        mock_session: MagicMock,
    ) -> None:
        """Test getting a sync run by ID when it doesn't exist."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(999)

        assert result is None

    @pytest.mark.asyncio()
    async def test_get_active_run_found(
        self,
        repository: CollectionSyncRunRepository,
        mock_session: MagicMock,
        sample_sync_run: CollectionSyncRun,
    ) -> None:
        """Test getting an active run for a collection."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_sync_run
        mock_session.execute.return_value = mock_result

        result = await repository.get_active_run("coll-uuid-1")

        assert result == sample_sync_run

    @pytest.mark.asyncio()
    async def test_get_active_run_not_found(
        self,
        repository: CollectionSyncRunRepository,
        mock_session: MagicMock,
    ) -> None:
        """Test getting an active run when none exists."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_active_run("coll-uuid-1")

        assert result is None

    @pytest.mark.asyncio()
    async def test_update_source_completion_success(
        self,
        repository: CollectionSyncRunRepository,
        mock_session: MagicMock,
        sample_sync_run: CollectionSyncRun,
    ) -> None:
        """Test incrementing completed_sources on success."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_sync_run
        mock_session.execute.return_value = mock_result

        result = await repository.update_source_completion(1, "success")

        assert result.completed_sources == 1
        assert result.failed_sources == 0
        assert result.partial_sources == 0

    @pytest.mark.asyncio()
    async def test_update_source_completion_failed(
        self,
        repository: CollectionSyncRunRepository,
        mock_session: MagicMock,
        sample_sync_run: CollectionSyncRun,
    ) -> None:
        """Test incrementing failed_sources on failure."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_sync_run
        mock_session.execute.return_value = mock_result

        result = await repository.update_source_completion(1, "failed")

        assert result.completed_sources == 0
        assert result.failed_sources == 1
        assert result.partial_sources == 0

    @pytest.mark.asyncio()
    async def test_update_source_completion_partial(
        self,
        repository: CollectionSyncRunRepository,
        mock_session: MagicMock,
        sample_sync_run: CollectionSyncRun,
    ) -> None:
        """Test incrementing partial_sources on partial."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sample_sync_run
        mock_session.execute.return_value = mock_result

        result = await repository.update_source_completion(1, "partial")

        assert result.completed_sources == 0
        assert result.failed_sources == 0
        assert result.partial_sources == 1

    @pytest.mark.asyncio()
    async def test_complete_run_all_success(
        self,
        repository: CollectionSyncRunRepository,
        mock_session: MagicMock,
    ) -> None:
        """Test completing a run when all sources succeeded."""
        sync_run = CollectionSyncRun(
            id=1,
            collection_id="coll-uuid-1",
            triggered_by="scheduler",
            started_at=datetime.now(UTC) - timedelta(minutes=5),
            status="running",
            expected_sources=3,
            completed_sources=3,
            failed_sources=0,
            partial_sources=0,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sync_run
        mock_session.execute.return_value = mock_result

        result = await repository.complete_run(1)

        assert result.status == "success"
        assert result.completed_at is not None

    @pytest.mark.asyncio()
    async def test_complete_run_all_failed(
        self,
        repository: CollectionSyncRunRepository,
        mock_session: MagicMock,
    ) -> None:
        """Test completing a run when all sources failed."""
        sync_run = CollectionSyncRun(
            id=1,
            collection_id="coll-uuid-1",
            triggered_by="scheduler",
            started_at=datetime.now(UTC) - timedelta(minutes=5),
            status="running",
            expected_sources=3,
            completed_sources=0,
            failed_sources=3,
            partial_sources=0,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sync_run
        mock_session.execute.return_value = mock_result

        result = await repository.complete_run(1)

        assert result.status == "failed"
        assert result.completed_at is not None

    @pytest.mark.asyncio()
    async def test_complete_run_partial_failures(
        self,
        repository: CollectionSyncRunRepository,
        mock_session: MagicMock,
    ) -> None:
        """Test completing a run when some sources failed."""
        sync_run = CollectionSyncRun(
            id=1,
            collection_id="coll-uuid-1",
            triggered_by="scheduler",
            started_at=datetime.now(UTC) - timedelta(minutes=5),
            status="running",
            expected_sources=3,
            completed_sources=2,
            failed_sources=1,
            partial_sources=0,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sync_run
        mock_session.execute.return_value = mock_result

        result = await repository.complete_run(1)

        assert result.status == "partial"
        assert result.completed_at is not None

    @pytest.mark.asyncio()
    async def test_complete_run_with_partial_sources(
        self,
        repository: CollectionSyncRunRepository,
        mock_session: MagicMock,
    ) -> None:
        """Test completing a run when some sources have partial results."""
        sync_run = CollectionSyncRun(
            id=1,
            collection_id="coll-uuid-1",
            triggered_by="scheduler",
            started_at=datetime.now(UTC) - timedelta(minutes=5),
            status="running",
            expected_sources=3,
            completed_sources=2,
            failed_sources=0,
            partial_sources=1,
        )

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = sync_run
        mock_session.execute.return_value = mock_result

        result = await repository.complete_run(1)

        assert result.status == "partial"
        assert result.completed_at is not None

    @pytest.mark.asyncio()
    async def test_list_for_collection(
        self,
        repository: CollectionSyncRunRepository,
        mock_session: MagicMock,
    ) -> None:
        """Test listing sync runs for a collection."""
        sync_runs = [
            CollectionSyncRun(
                id=1,
                collection_id="coll-uuid-1",
                triggered_by="scheduler",
                started_at=datetime.now(UTC) - timedelta(hours=1),
                status="success",
                expected_sources=2,
                completed_sources=2,
                failed_sources=0,
                partial_sources=0,
            ),
            CollectionSyncRun(
                id=2,
                collection_id="coll-uuid-1",
                triggered_by="manual",
                started_at=datetime.now(UTC),
                status="running",
                expected_sources=2,
                completed_sources=0,
                failed_sources=0,
                partial_sources=0,
            ),
        ]

        # Mock the count query result
        count_result = MagicMock()
        count_result.scalar.return_value = 2

        # Mock the list query result
        list_scalars = MagicMock()
        list_scalars.all.return_value = sync_runs
        list_result = MagicMock()
        list_result.scalars.return_value = list_scalars

        # execute is called twice: first for count, then for list
        mock_session.execute = AsyncMock(side_effect=[count_result, list_result])

        result, total = await repository.list_for_collection(
            collection_id="coll-uuid-1",
            offset=0,
            limit=10,
        )

        assert len(result) == 2
        assert total == 2
        assert result[0].id == 1
        assert result[1].id == 2
