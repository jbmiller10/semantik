"""Integration tests for collection-level continuous sync functionality.

This test suite verifies the full sync flow from trigger to completion,
including pause/resume and edge cases.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.exceptions import ValidationError
from shared.database.models import CollectionStatus, CollectionSyncRun
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.collection_sync_run_repository import CollectionSyncRunRepository


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestCollectionSyncFlow:
    """Test the full sync flow: trigger -> dispatch -> complete."""

    async def test_create_collection_with_continuous_sync(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ) -> None:
        """Test creating a collection with continuous sync mode."""
        collection = await collection_factory(
            owner_id=test_user_db.id,
            sync_mode="continuous",
            sync_interval_minutes=30,
        )

        assert collection.sync_mode == "continuous"
        assert collection.sync_interval_minutes == 30
        assert collection.sync_paused_at is None

    async def test_get_due_for_sync_returns_due_collections(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ) -> None:
        """Test that get_due_for_sync returns collections ready to sync."""
        repo = CollectionRepository(db_session)

        # Create a continuous sync collection with next_run_at in the past
        collection = await collection_factory(
            owner_id=test_user_db.id,
            sync_mode="continuous",
            sync_interval_minutes=15,
            sync_next_run_at=datetime.now(UTC) - timedelta(minutes=5),
            status=CollectionStatus.READY,
        )
        await db_session.commit()

        due_collections = await repo.get_due_for_sync(limit=10)

        collection_ids = [c.id for c in due_collections]
        assert collection.id in collection_ids

    async def test_get_due_for_sync_excludes_paused_collections(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ) -> None:
        """Test that paused collections are not returned by get_due_for_sync."""
        repo = CollectionRepository(db_session)

        # Create a paused collection
        collection = await collection_factory(
            owner_id=test_user_db.id,
            sync_mode="continuous",
            sync_interval_minutes=15,
            sync_next_run_at=datetime.now(UTC) - timedelta(minutes=5),
            sync_paused_at=datetime.now(UTC),
            status=CollectionStatus.READY,
        )
        await db_session.commit()

        due_collections = await repo.get_due_for_sync(limit=10)

        collection_ids = [c.id for c in due_collections]
        assert collection.id not in collection_ids

    async def test_get_due_for_sync_excludes_one_time_collections(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ) -> None:
        """Test that one_time collections are not returned by get_due_for_sync."""
        repo = CollectionRepository(db_session)

        # Create a one-time collection
        collection = await collection_factory(
            owner_id=test_user_db.id,
            sync_mode="one_time",
            status=CollectionStatus.READY,
        )
        await db_session.commit()

        due_collections = await repo.get_due_for_sync(limit=10)

        collection_ids = [c.id for c in due_collections]
        assert collection.id not in collection_ids

    async def test_sync_run_creation_and_completion(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ) -> None:
        """Test creating and completing a sync run."""
        collection = await collection_factory(
            owner_id=test_user_db.id,
            sync_mode="continuous",
            sync_interval_minutes=30,
        )

        sync_run_repo = CollectionSyncRunRepository(db_session)

        # Create sync run
        sync_run = await sync_run_repo.create(
            collection_id=collection.id,
            triggered_by="scheduler",
            expected_sources=3,
        )
        await db_session.flush()

        assert sync_run.id is not None
        assert sync_run.collection_id == collection.id
        assert sync_run.triggered_by == "scheduler"
        assert sync_run.expected_sources == 3
        assert sync_run.status == "running"
        assert sync_run.completed_sources == 0
        assert sync_run.failed_sources == 0

        # Simulate source completions
        await sync_run_repo.update_source_completion(sync_run.id, "success")
        await sync_run_repo.update_source_completion(sync_run.id, "success")
        await sync_run_repo.update_source_completion(sync_run.id, "failed")

        # Refresh sync run
        sync_run = await sync_run_repo.get_by_id(sync_run.id)
        assert sync_run.completed_sources == 2
        assert sync_run.failed_sources == 1

        # Complete the run
        completed_run = await sync_run_repo.complete_run(sync_run.id)
        assert completed_run.status == "partial"  # Mixed results
        assert completed_run.completed_at is not None

    async def test_sync_run_all_success(self, db_session: AsyncSession, test_user_db, collection_factory) -> None:
        """Test sync run with all successful sources."""
        collection = await collection_factory(owner_id=test_user_db.id)
        sync_run_repo = CollectionSyncRunRepository(db_session)

        sync_run = await sync_run_repo.create(
            collection_id=collection.id,
            triggered_by="manual",
            expected_sources=2,
        )
        await db_session.flush()

        await sync_run_repo.update_source_completion(sync_run.id, "success")
        await sync_run_repo.update_source_completion(sync_run.id, "success")

        completed_run = await sync_run_repo.complete_run(sync_run.id)
        assert completed_run.status == "success"

    async def test_sync_run_all_failed(self, db_session: AsyncSession, test_user_db, collection_factory) -> None:
        """Test sync run with all failed sources."""
        collection = await collection_factory(owner_id=test_user_db.id)
        sync_run_repo = CollectionSyncRunRepository(db_session)

        sync_run = await sync_run_repo.create(
            collection_id=collection.id,
            triggered_by="scheduler",
            expected_sources=2,
        )
        await db_session.flush()

        await sync_run_repo.update_source_completion(sync_run.id, "failed")
        await sync_run_repo.update_source_completion(sync_run.id, "failed")

        completed_run = await sync_run_repo.complete_run(sync_run.id)
        assert completed_run.status == "failed"


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestCollectionSyncPauseResume:
    """Test pause and resume functionality for collection sync."""

    async def test_pause_continuous_collection(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ) -> None:
        """Test pausing a continuous sync collection."""
        repo = CollectionRepository(db_session)

        collection = await collection_factory(
            owner_id=test_user_db.id,
            sync_mode="continuous",
            sync_interval_minutes=30,
            sync_next_run_at=datetime.now(UTC) + timedelta(minutes=15),
        )
        await db_session.commit()

        paused = await repo.pause_sync(collection.id)

        assert paused.sync_paused_at is not None
        # next_run_at is preserved but won't be used while paused

    async def test_pause_one_time_collection_raises_error(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ) -> None:
        """Test that pausing a one_time collection raises an error."""
        repo = CollectionRepository(db_session)

        collection = await collection_factory(
            owner_id=test_user_db.id,
            sync_mode="one_time",
        )
        await db_session.commit()

        with pytest.raises(ValidationError) as exc_info:
            await repo.pause_sync(collection.id)
        assert "continuous" in str(exc_info.value).lower()

    async def test_resume_paused_collection(self, db_session: AsyncSession, test_user_db, collection_factory) -> None:
        """Test resuming a paused collection."""
        repo = CollectionRepository(db_session)

        collection = await collection_factory(
            owner_id=test_user_db.id,
            sync_mode="continuous",
            sync_interval_minutes=30,
            sync_paused_at=datetime.now(UTC),
        )
        await db_session.commit()

        resumed = await repo.resume_sync(collection.id)

        assert resumed.sync_paused_at is None
        assert resumed.sync_next_run_at is not None

    async def test_resume_not_paused_is_noop(self, db_session: AsyncSession, test_user_db, collection_factory) -> None:
        """Test that resuming an unpaused collection is a no-op."""
        repo = CollectionRepository(db_session)

        original_next_run = datetime.now(UTC) + timedelta(minutes=15)
        collection = await collection_factory(
            owner_id=test_user_db.id,
            sync_mode="continuous",
            sync_interval_minutes=30,
            sync_next_run_at=original_next_run,
        )
        await db_session.commit()

        resumed = await repo.resume_sync(collection.id)

        assert resumed.sync_paused_at is None
        # next_run_at should remain close to original (within a second)
        assert abs((resumed.sync_next_run_at - original_next_run).total_seconds()) < 1

    async def test_paused_collection_excluded_from_due_list(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ) -> None:
        """Test that a paused collection doesn't appear in due list after pause."""
        repo = CollectionRepository(db_session)

        collection = await collection_factory(
            owner_id=test_user_db.id,
            sync_mode="continuous",
            sync_interval_minutes=15,
            sync_next_run_at=datetime.now(UTC) - timedelta(minutes=5),
            status=CollectionStatus.READY,
        )
        await db_session.commit()

        # Verify it's in the due list
        due_before = await repo.get_due_for_sync(limit=10)
        assert collection.id in [c.id for c in due_before]

        # Pause it
        await repo.pause_sync(collection.id)
        await db_session.commit()

        # Verify it's no longer in the due list
        due_after = await repo.get_due_for_sync(limit=10)
        assert collection.id not in [c.id for c in due_after]


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestCollectionSyncEdgeCases:
    """Test edge cases for collection sync."""

    async def test_sync_run_with_partial_sources(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ) -> None:
        """Test sync run with partial source completions."""
        collection = await collection_factory(owner_id=test_user_db.id)
        sync_run_repo = CollectionSyncRunRepository(db_session)

        sync_run = await sync_run_repo.create(
            collection_id=collection.id,
            triggered_by="scheduler",
            expected_sources=3,
        )
        await db_session.flush()

        # Mix of success and partial
        await sync_run_repo.update_source_completion(sync_run.id, "success")
        await sync_run_repo.update_source_completion(sync_run.id, "partial")
        await sync_run_repo.update_source_completion(sync_run.id, "success")

        completed_run = await sync_run_repo.complete_run(sync_run.id)
        assert completed_run.status == "partial"
        assert completed_run.partial_sources == 1
        assert completed_run.completed_sources == 2

    async def test_get_active_run_returns_running_run(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ) -> None:
        """Test that get_active_run returns a running sync run."""
        collection = await collection_factory(owner_id=test_user_db.id)
        sync_run_repo = CollectionSyncRunRepository(db_session)

        # Create a running sync run
        sync_run = await sync_run_repo.create(
            collection_id=collection.id,
            triggered_by="manual",
            expected_sources=1,
        )
        await db_session.flush()

        active = await sync_run_repo.get_active_run(collection.id)
        assert active is not None
        assert active.id == sync_run.id
        assert active.status == "running"

    async def test_get_active_run_returns_none_after_completion(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ) -> None:
        """Test that get_active_run returns None after run completion."""
        collection = await collection_factory(owner_id=test_user_db.id)
        sync_run_repo = CollectionSyncRunRepository(db_session)

        sync_run = await sync_run_repo.create(
            collection_id=collection.id,
            triggered_by="manual",
            expected_sources=1,
        )
        await db_session.flush()

        # Complete the run
        await sync_run_repo.update_source_completion(sync_run.id, "success")
        await sync_run_repo.complete_run(sync_run.id)

        active = await sync_run_repo.get_active_run(collection.id)
        assert active is None

    async def test_list_sync_runs_for_collection(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ) -> None:
        """Test listing sync runs for a collection."""
        collection = await collection_factory(owner_id=test_user_db.id)
        sync_run_repo = CollectionSyncRunRepository(db_session)

        # Create multiple sync runs
        run1 = await sync_run_repo.create(
            collection_id=collection.id,
            triggered_by="scheduler",
            expected_sources=1,
        )
        await sync_run_repo.update_source_completion(run1.id, "success")
        await sync_run_repo.complete_run(run1.id)

        _run2 = await sync_run_repo.create(
            collection_id=collection.id,
            triggered_by="manual",
            expected_sources=1,
        )
        await db_session.flush()

        runs, total = await sync_run_repo.list_for_collection(
            collection_id=collection.id,
            offset=0,
            limit=10,
        )

        assert total == 2
        assert len(runs) == 2

    async def test_update_sync_status_on_collection(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ) -> None:
        """Test updating sync status on a collection."""
        repo = CollectionRepository(db_session)

        collection = await collection_factory(
            owner_id=test_user_db.id,
            sync_mode="continuous",
            sync_interval_minutes=30,
        )
        await db_session.commit()

        now = datetime.now(UTC)
        await repo.update_sync_status(
            collection.id,
            status="running",
            started_at=now,
        )
        await db_session.commit()

        # Refresh and check
        updated = await repo.get_by_uuid(collection.id)
        assert updated.sync_last_run_status == "running"
        assert updated.sync_last_run_started_at is not None

    async def test_set_next_sync_run(self, db_session: AsyncSession, test_user_db, collection_factory) -> None:
        """Test setting the next sync run time."""
        repo = CollectionRepository(db_session)

        collection = await collection_factory(
            owner_id=test_user_db.id,
            sync_mode="continuous",
            sync_interval_minutes=30,
        )
        await db_session.commit()

        future_time = datetime.now(UTC) + timedelta(hours=1)
        await repo.set_next_sync_run(collection.id, future_time)
        await db_session.commit()

        updated = await repo.get_by_uuid(collection.id)
        assert updated.sync_next_run_at is not None
        # Should be close to the specified time
        assert abs((updated.sync_next_run_at - future_time).total_seconds()) < 1

    async def test_sync_run_cascades_on_collection_delete(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ) -> None:
        """Test that sync runs are deleted when collection is deleted."""
        collection = await collection_factory(owner_id=test_user_db.id)
        collection_id = collection.id

        sync_run_repo = CollectionSyncRunRepository(db_session)
        collection_repo = CollectionRepository(db_session)

        # Create a sync run
        sync_run = await sync_run_repo.create(
            collection_id=collection_id,
            triggered_by="manual",
            expected_sources=1,
        )
        sync_run_id = sync_run.id
        await db_session.flush()

        # Delete the collection
        await collection_repo.delete(collection_id, test_user_db.id)
        await db_session.commit()

        # Sync run should be gone (cascade delete)
        result = await db_session.execute(select(CollectionSyncRun).where(CollectionSyncRun.id == sync_run_id))
        assert result.scalar_one_or_none() is None
