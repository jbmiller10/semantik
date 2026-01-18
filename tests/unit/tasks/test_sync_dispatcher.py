"""Unit tests for sync_dispatcher task (collection-level sync)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the module directly to avoid webui.tasks proxy issues
import webui.tasks.sync_dispatcher as sync_dispatcher_module
from shared.database.models import Collection, CollectionSource, CollectionStatus, CollectionSyncRun, OperationType


class TestDispatchDueSyncsAsync:
    """Tests for the _dispatch_due_syncs_async function (collection-level dispatch)."""

    @pytest.fixture()
    def mock_session(self) -> MagicMock:
        """Create mock database session."""
        session = MagicMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture()
    def mock_collection_repo(self) -> MagicMock:
        """Create mock collection repository."""
        repo = MagicMock()
        repo.get_due_for_sync = AsyncMock(return_value=[])
        repo.update_sync_status = AsyncMock()
        repo.set_next_sync_run = AsyncMock()
        repo.update_status = AsyncMock()
        return repo

    @pytest.fixture()
    def mock_source_repo(self) -> MagicMock:
        """Create mock source repository."""
        repo = MagicMock()
        repo.list_by_collection = AsyncMock(return_value=([], 0))
        return repo

    @pytest.fixture()
    def mock_sync_run_repo(self) -> MagicMock:
        """Create mock sync run repository."""
        repo = MagicMock()
        repo.create = AsyncMock()
        return repo

    @pytest.fixture()
    def mock_operation_repo(self) -> MagicMock:
        """Create mock operation repository."""
        repo = MagicMock()
        repo.get_active_operations = AsyncMock(return_value=[])
        repo.create = AsyncMock()
        return repo

    @pytest.fixture()
    def sample_collection(self) -> Collection:
        """Create a sample collection for testing."""
        collection = MagicMock(spec=Collection)
        collection.id = "coll-uuid-1"
        collection.name = "test-collection"
        collection.status = CollectionStatus.READY
        collection.is_public = False
        collection.owner_id = 1
        collection.sync_mode = "continuous"
        collection.sync_interval_minutes = 30
        collection.sync_paused_at = None
        collection.sync_next_run_at = datetime.now(UTC) - timedelta(minutes=5)
        return collection

    @pytest.fixture()
    def sample_source(self) -> CollectionSource:
        """Create a sample source for testing."""
        source = MagicMock(spec=CollectionSource)
        source.id = 1
        source.collection_id = "coll-uuid-1"
        source.source_type = "directory"
        source.source_path = "/data/test"
        source.source_config = {"path": "/data/test"}
        return source

    @pytest.fixture()
    def sample_sync_run(self) -> CollectionSyncRun:
        """Create a sample sync run for testing."""
        sync_run = MagicMock(spec=CollectionSyncRun)
        sync_run.id = 1
        sync_run.collection_id = "coll-uuid-1"
        sync_run.triggered_by = "scheduler"
        sync_run.expected_sources = 1
        sync_run.completed_sources = 0
        sync_run.failed_sources = 0
        sync_run.partial_sources = 0
        return sync_run

    @pytest.mark.asyncio()
    async def test_no_collections_due(
        self,
        mock_session: MagicMock,
        mock_collection_repo: MagicMock,
        mock_source_repo: MagicMock,
        mock_sync_run_repo: MagicMock,
        mock_operation_repo: MagicMock,
    ) -> None:
        """Test when no collections are due for sync."""
        mock_collection_repo.get_due_for_sync.return_value = []

        with (
            patch.object(sync_dispatcher_module, "pg_connection_manager") as mock_pg,
            patch.object(sync_dispatcher_module, "CollectionRepository", return_value=mock_collection_repo),
            patch.object(sync_dispatcher_module, "CollectionSourceRepository", return_value=mock_source_repo),
            patch.object(sync_dispatcher_module, "CollectionSyncRunRepository", return_value=mock_sync_run_repo),
            patch.object(sync_dispatcher_module, "OperationRepository", return_value=mock_operation_repo),
        ):
            mock_pg._engine = MagicMock()  # Simulate initialized engine
            mock_pg.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_pg.get_session.return_value.__aexit__ = AsyncMock()

            result = await sync_dispatcher_module._dispatch_due_syncs_async()

        assert result["collections_checked"] == 0
        assert result["collections_dispatched"] == 0
        assert result["sources_dispatched"] == 0
        assert result["collections_skipped"] == 0
        assert result["errors"] == []

    @pytest.mark.asyncio()
    async def test_collection_has_active_operations(
        self,
        mock_session: MagicMock,
        mock_collection_repo: MagicMock,
        mock_source_repo: MagicMock,
        mock_sync_run_repo: MagicMock,
        mock_operation_repo: MagicMock,
        sample_collection: Collection,
    ) -> None:
        """Test that collection with active operations is skipped (collection-level gating)."""
        mock_collection_repo.get_due_for_sync.return_value = [sample_collection]
        mock_operation_repo.get_active_operations.return_value = [MagicMock()]  # Active op

        with (
            patch.object(sync_dispatcher_module, "pg_connection_manager") as mock_pg,
            patch.object(sync_dispatcher_module, "CollectionRepository", return_value=mock_collection_repo),
            patch.object(sync_dispatcher_module, "CollectionSourceRepository", return_value=mock_source_repo),
            patch.object(sync_dispatcher_module, "CollectionSyncRunRepository", return_value=mock_sync_run_repo),
            patch.object(sync_dispatcher_module, "OperationRepository", return_value=mock_operation_repo),
        ):
            mock_pg._engine = MagicMock()
            mock_pg.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_pg.get_session.return_value.__aexit__ = AsyncMock()

            result = await sync_dispatcher_module._dispatch_due_syncs_async()

        assert result["collections_checked"] == 1
        assert result["collections_dispatched"] == 0
        assert result["collections_skipped"] == 1
        assert result["errors"] == []

    @pytest.mark.asyncio()
    async def test_collection_with_no_sources(
        self,
        mock_session: MagicMock,
        mock_collection_repo: MagicMock,
        mock_source_repo: MagicMock,
        mock_sync_run_repo: MagicMock,
        mock_operation_repo: MagicMock,
        sample_collection: Collection,
    ) -> None:
        """Test that collection with no sources is skipped but next_run is updated."""
        mock_collection_repo.get_due_for_sync.return_value = [sample_collection]
        mock_operation_repo.get_active_operations.return_value = []
        mock_source_repo.list_by_collection.return_value = ([], 0)

        with (
            patch.object(sync_dispatcher_module, "pg_connection_manager") as mock_pg,
            patch.object(sync_dispatcher_module, "CollectionRepository", return_value=mock_collection_repo),
            patch.object(sync_dispatcher_module, "CollectionSourceRepository", return_value=mock_source_repo),
            patch.object(sync_dispatcher_module, "CollectionSyncRunRepository", return_value=mock_sync_run_repo),
            patch.object(sync_dispatcher_module, "OperationRepository", return_value=mock_operation_repo),
        ):
            mock_pg._engine = MagicMock()
            mock_pg.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_pg.get_session.return_value.__aexit__ = AsyncMock()

            result = await sync_dispatcher_module._dispatch_due_syncs_async()

        assert result["collections_checked"] == 1
        assert result["collections_dispatched"] == 0
        assert result["collections_skipped"] == 1
        # Verify next_run was still updated
        mock_collection_repo.set_next_sync_run.assert_called_once()

    @pytest.mark.asyncio()
    async def test_successful_dispatch_with_sources(
        self,
        mock_session: MagicMock,
        mock_collection_repo: MagicMock,
        mock_source_repo: MagicMock,
        mock_sync_run_repo: MagicMock,
        mock_operation_repo: MagicMock,
        sample_collection: Collection,
        sample_source: CollectionSource,
        sample_sync_run: CollectionSyncRun,
    ) -> None:
        """Test successful dispatch creates sync run and fans out to sources."""
        mock_collection_repo.get_due_for_sync.return_value = [sample_collection]
        mock_operation_repo.get_active_operations.return_value = []
        mock_source_repo.list_by_collection.return_value = ([sample_source], 1)
        mock_sync_run_repo.create.return_value = sample_sync_run

        mock_operation = MagicMock()
        mock_operation.uuid = "op-uuid-1"
        mock_operation_repo.create.return_value = mock_operation

        with (
            patch.object(sync_dispatcher_module, "pg_connection_manager") as mock_pg,
            patch.object(sync_dispatcher_module, "CollectionRepository", return_value=mock_collection_repo),
            patch.object(sync_dispatcher_module, "CollectionSourceRepository", return_value=mock_source_repo),
            patch.object(sync_dispatcher_module, "CollectionSyncRunRepository", return_value=mock_sync_run_repo),
            patch.object(sync_dispatcher_module, "OperationRepository", return_value=mock_operation_repo),
            patch.object(sync_dispatcher_module, "celery_app") as mock_celery,
        ):
            mock_pg._engine = MagicMock()
            mock_pg.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_pg.get_session.return_value.__aexit__ = AsyncMock()

            result = await sync_dispatcher_module._dispatch_due_syncs_async()

        assert result["collections_checked"] == 1
        assert result["collections_dispatched"] == 1
        assert result["sources_dispatched"] == 1
        assert result["errors"] == []

        # Verify sync run was created
        mock_sync_run_repo.create.assert_called_once()
        create_kwargs = mock_sync_run_repo.create.call_args.kwargs
        assert create_kwargs["collection_id"] == sample_collection.id
        assert create_kwargs["triggered_by"] == "scheduler"
        assert create_kwargs["expected_sources"] == 1

        # Verify operation was created with sync_run_id
        mock_operation_repo.create.assert_called_once()
        op_kwargs = mock_operation_repo.create.call_args.kwargs
        assert op_kwargs["collection_id"] == sample_collection.id
        assert op_kwargs["operation_type"] == OperationType.APPEND
        assert op_kwargs["config"]["source_id"] == sample_source.id
        assert op_kwargs["config"]["sync_run_id"] == sample_sync_run.id

        # Verify collection status was updated
        mock_collection_repo.update_sync_status.assert_called_once()
        mock_collection_repo.update_status.assert_called_once()

        # Verify next run was scheduled
        mock_collection_repo.set_next_sync_run.assert_called_once()

        # Verify task was dispatched
        mock_celery.send_task.assert_called_once()

    @pytest.mark.asyncio()
    async def test_multiple_sources_fan_out(
        self,
        mock_session: MagicMock,
        mock_collection_repo: MagicMock,
        mock_source_repo: MagicMock,
        mock_sync_run_repo: MagicMock,
        mock_operation_repo: MagicMock,
        sample_collection: Collection,
        sample_sync_run: CollectionSyncRun,
    ) -> None:
        """Test that multiple sources result in multiple operations."""
        mock_collection_repo.get_due_for_sync.return_value = [sample_collection]
        mock_operation_repo.get_active_operations.return_value = []

        # Create 3 sources
        sources = []
        for i in range(3):
            source = MagicMock(spec=CollectionSource)
            source.id = i + 1
            source.collection_id = sample_collection.id
            source.source_type = "directory"
            source.source_path = f"/data/test{i}"
            source.source_config = {"path": f"/data/test{i}"}
            sources.append(source)

        mock_source_repo.list_by_collection.return_value = (sources, 3)
        sample_sync_run.expected_sources = 3
        mock_sync_run_repo.create.return_value = sample_sync_run

        mock_operation = MagicMock()
        mock_operation.uuid = "op-uuid"
        mock_operation_repo.create.return_value = mock_operation

        with (
            patch.object(sync_dispatcher_module, "pg_connection_manager") as mock_pg,
            patch.object(sync_dispatcher_module, "CollectionRepository", return_value=mock_collection_repo),
            patch.object(sync_dispatcher_module, "CollectionSourceRepository", return_value=mock_source_repo),
            patch.object(sync_dispatcher_module, "CollectionSyncRunRepository", return_value=mock_sync_run_repo),
            patch.object(sync_dispatcher_module, "OperationRepository", return_value=mock_operation_repo),
            patch.object(sync_dispatcher_module, "celery_app") as mock_celery,
        ):
            mock_pg._engine = MagicMock()
            mock_pg.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_pg.get_session.return_value.__aexit__ = AsyncMock()

            result = await sync_dispatcher_module._dispatch_due_syncs_async()

        assert result["collections_dispatched"] == 1
        assert result["sources_dispatched"] == 3

        # Verify 3 operations were created
        assert mock_operation_repo.create.call_count == 3

        # Verify 3 tasks were dispatched
        assert mock_celery.send_task.call_count == 3

        # Verify sync run was created with expected_sources=3
        create_kwargs = mock_sync_run_repo.create.call_args.kwargs
        assert create_kwargs["expected_sources"] == 3

    @pytest.mark.asyncio()
    async def test_error_handling_continues_to_next_collection(
        self,
        mock_session: MagicMock,
        mock_collection_repo: MagicMock,
        mock_source_repo: MagicMock,
        mock_sync_run_repo: MagicMock,
        mock_operation_repo: MagicMock,
        sample_source: CollectionSource,
        sample_sync_run: CollectionSyncRun,
    ) -> None:
        """Test that errors for one collection don't stop processing others."""
        # Create two collections
        collection1 = MagicMock(spec=Collection)
        collection1.id = "coll-uuid-1"
        collection1.owner_id = 1
        collection1.sync_interval_minutes = 30

        collection2 = MagicMock(spec=Collection)
        collection2.id = "coll-uuid-2"
        collection2.owner_id = 1
        collection2.sync_interval_minutes = 60

        mock_collection_repo.get_due_for_sync.return_value = [collection1, collection2]

        # First collection raises error, second succeeds
        mock_operation_repo.get_active_operations.side_effect = [
            Exception("DB error"),  # First collection fails
            [],  # Second collection succeeds
        ]
        mock_source_repo.list_by_collection.return_value = ([sample_source], 1)
        mock_sync_run_repo.create.return_value = sample_sync_run

        mock_operation = MagicMock()
        mock_operation.uuid = "op-uuid"
        mock_operation_repo.create.return_value = mock_operation

        with (
            patch.object(sync_dispatcher_module, "pg_connection_manager") as mock_pg,
            patch.object(sync_dispatcher_module, "CollectionRepository", return_value=mock_collection_repo),
            patch.object(sync_dispatcher_module, "CollectionSourceRepository", return_value=mock_source_repo),
            patch.object(sync_dispatcher_module, "CollectionSyncRunRepository", return_value=mock_sync_run_repo),
            patch.object(sync_dispatcher_module, "OperationRepository", return_value=mock_operation_repo),
            patch.object(sync_dispatcher_module, "celery_app"),
        ):
            mock_pg._engine = MagicMock()
            mock_pg.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_pg.get_session.return_value.__aexit__ = AsyncMock()

            result = await sync_dispatcher_module._dispatch_due_syncs_async()

        assert result["collections_checked"] == 2
        assert result["collections_dispatched"] == 1  # Second succeeded
        assert len(result["errors"]) == 1  # First failed
        assert "DB error" in result["errors"][0]["error"]

        # Verify rollback was called after error
        mock_session.rollback.assert_called_once()

    @pytest.mark.asyncio()
    async def test_initializes_db_when_engine_is_none(
        self,
        mock_session: MagicMock,
        mock_collection_repo: MagicMock,
        mock_source_repo: MagicMock,
        mock_sync_run_repo: MagicMock,
        mock_operation_repo: MagicMock,
    ) -> None:
        """Test that database is initialized when engine is None."""
        mock_collection_repo.get_due_for_sync.return_value = []

        with (
            patch.object(sync_dispatcher_module, "pg_connection_manager") as mock_pg,
            patch.object(sync_dispatcher_module, "CollectionRepository", return_value=mock_collection_repo),
            patch.object(sync_dispatcher_module, "CollectionSourceRepository", return_value=mock_source_repo),
            patch.object(sync_dispatcher_module, "CollectionSyncRunRepository", return_value=mock_sync_run_repo),
            patch.object(sync_dispatcher_module, "OperationRepository", return_value=mock_operation_repo),
        ):
            mock_pg._engine = None  # Engine not initialized
            mock_pg.initialize = AsyncMock()
            mock_pg.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_pg.get_session.return_value.__aexit__ = AsyncMock()

            await sync_dispatcher_module._dispatch_due_syncs_async()

        # Verify initialize was called
        mock_pg.initialize.assert_called_once()

    @pytest.mark.asyncio()
    async def test_next_run_calculated_from_collection_interval(
        self,
        mock_session: MagicMock,
        mock_collection_repo: MagicMock,
        mock_source_repo: MagicMock,
        mock_sync_run_repo: MagicMock,
        mock_operation_repo: MagicMock,
        sample_source: CollectionSource,
        sample_sync_run: CollectionSyncRun,
    ) -> None:
        """Test that next_run_at is calculated based on collection's sync_interval_minutes."""
        collection = MagicMock(spec=Collection)
        collection.id = "coll-uuid-1"
        collection.owner_id = 1
        collection.sync_interval_minutes = 45  # Custom interval

        mock_collection_repo.get_due_for_sync.return_value = [collection]
        mock_operation_repo.get_active_operations.return_value = []
        mock_source_repo.list_by_collection.return_value = ([sample_source], 1)
        mock_sync_run_repo.create.return_value = sample_sync_run

        mock_operation = MagicMock()
        mock_operation.uuid = "op-uuid"
        mock_operation_repo.create.return_value = mock_operation

        with (
            patch.object(sync_dispatcher_module, "pg_connection_manager") as mock_pg,
            patch.object(sync_dispatcher_module, "CollectionRepository", return_value=mock_collection_repo),
            patch.object(sync_dispatcher_module, "CollectionSourceRepository", return_value=mock_source_repo),
            patch.object(sync_dispatcher_module, "CollectionSyncRunRepository", return_value=mock_sync_run_repo),
            patch.object(sync_dispatcher_module, "OperationRepository", return_value=mock_operation_repo),
            patch.object(sync_dispatcher_module, "celery_app"),
        ):
            mock_pg._engine = MagicMock()
            mock_pg.get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_pg.get_session.return_value.__aexit__ = AsyncMock()

            before_call = datetime.now(UTC)
            await sync_dispatcher_module._dispatch_due_syncs_async()
            after_call = datetime.now(UTC)

        # Verify set_next_sync_run was called with approximately now + 45 minutes
        mock_collection_repo.set_next_sync_run.assert_called_once()
        call_args = mock_collection_repo.set_next_sync_run.call_args
        collection_id, next_run = call_args[0]
        assert collection_id == collection.id
        # next_run should be within the expected range
        expected_min = before_call + timedelta(minutes=45)
        expected_max = after_call + timedelta(minutes=45)
        assert expected_min <= next_run <= expected_max


class TestDispatchDueSyncsTask:
    """Tests for the dispatch_due_syncs Celery task."""

    def test_task_returns_result(self) -> None:
        """Test that the Celery task returns the async function result."""
        expected_result = {
            "checked_at": datetime.now(UTC).isoformat(),
            "collections_checked": 2,
            "collections_dispatched": 1,
            "sources_dispatched": 3,
            "collections_skipped": 1,
            "errors": [],
        }

        # Use AsyncMock to properly mock the async function
        mock_async = AsyncMock(return_value=expected_result)

        with patch.object(sync_dispatcher_module, "_dispatch_due_syncs_async", mock_async):
            # For bound Celery tasks, use .run() method
            result = sync_dispatcher_module.dispatch_due_syncs.run()

        assert result == expected_result

    def test_task_raises_on_error(self) -> None:
        """Test that the Celery task raises exceptions from async function."""
        # Use AsyncMock with side_effect to raise an exception
        mock_async = AsyncMock(side_effect=RuntimeError("Test error"))

        with (
            patch.object(sync_dispatcher_module, "_dispatch_due_syncs_async", mock_async),
            pytest.raises(RuntimeError, match="Test error"),
        ):
            sync_dispatcher_module.dispatch_due_syncs.run()
