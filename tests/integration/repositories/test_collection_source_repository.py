"""Integration tests for CollectionSourceRepository using the real database session."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from shared.database.exceptions import (
    EntityAlreadyExistsError,
    EntityNotFoundError,
    ValidationError,
)
from shared.database.repositories.collection_source_repository import CollectionSourceRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestCollectionSourceRepositoryIntegration:
    """Integration coverage for the collection source repository."""

    @pytest.fixture()
    def repository(self, db_session: AsyncSession) -> CollectionSourceRepository:
        """Construct the repository with a real async session."""
        return CollectionSourceRepository(db_session)

    async def test_create_source_persists_expected_fields(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Creating a source should persist it with expected fields."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source_path = f"/data/test-{uuid4().hex[:8]}"

        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=source_path,
            source_config={"path": source_path, "recursive": True},
            meta={"created_by": "test"},
        )

        assert source.id is not None
        assert source.collection_id == collection.id
        assert source.source_type == "directory"
        assert source.source_path == source_path
        assert source.source_config == {"path": source_path, "recursive": True}
        assert source.meta == {"created_by": "test"}
        assert source.document_count == 0
        assert source.size_bytes == 0
        assert source.created_at is not None
        assert source.updated_at is not None

    async def test_create_source_duplicate_path_raises_conflict(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Duplicate source paths in same collection should raise EntityAlreadyExistsError."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source_path = f"/data/duplicate-{uuid4().hex[:8]}"

        await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=source_path,
        )

        with pytest.raises(EntityAlreadyExistsError):
            await repository.create(
                collection_id=collection.id,
                source_type="directory",
                source_path=source_path,
            )

    async def test_create_source_same_path_different_collection_allowed(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Same source path in different collections should be allowed."""
        collection1 = await collection_factory(owner_id=test_user_db.id)
        collection2 = await collection_factory(owner_id=test_user_db.id)
        source_path = f"/data/shared-{uuid4().hex[:8]}"

        source1 = await repository.create(
            collection_id=collection1.id,
            source_type="directory",
            source_path=source_path,
        )

        source2 = await repository.create(
            collection_id=collection2.id,
            source_type="directory",
            source_path=source_path,
        )

        assert source1.id != source2.id
        assert source1.collection_id == collection1.id
        assert source2.collection_id == collection2.id

    async def test_create_source_validation_errors(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Invalid inputs should raise ValidationError."""
        collection = await collection_factory(owner_id=test_user_db.id)

        with pytest.raises(ValidationError) as exc_info:
            await repository.create(
                collection_id="",
                source_type="directory",
                source_path="/data/test",
            )
        assert "collection_id" in str(exc_info.value).lower()

        with pytest.raises(ValidationError) as exc_info:
            await repository.create(
                collection_id=collection.id,
                source_type="",
                source_path="/data/test",
            )
        assert "source_type" in str(exc_info.value).lower()

        with pytest.raises(ValidationError) as exc_info:
            await repository.create(
                collection_id=collection.id,
                source_type="directory",
                source_path="",
            )
        assert "source_path" in str(exc_info.value).lower()

    async def test_get_by_id_returns_source(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Repository should fetch source by ID."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/getbyid-{uuid4().hex[:8]}",
        )

        fetched = await repository.get_by_id(source.id)

        assert fetched is not None
        assert fetched.id == source.id
        assert fetched.source_path == source.source_path

    async def test_get_by_id_returns_none_for_missing(self, repository: CollectionSourceRepository) -> None:
        """Missing sources should return None."""
        fetched = await repository.get_by_id(999999)
        assert fetched is None

    async def test_get_by_collection_and_path_returns_source(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Repository should fetch source by collection and path."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source_path = f"/data/bypath-{uuid4().hex[:8]}"
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=source_path,
        )

        fetched = await repository.get_by_collection_and_path(collection.id, source_path)

        assert fetched is not None
        assert fetched.id == source.id

    async def test_get_by_collection_and_path_returns_none_for_wrong_collection(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Should return None when path exists but in different collection."""
        collection1 = await collection_factory(owner_id=test_user_db.id)
        collection2 = await collection_factory(owner_id=test_user_db.id)
        source_path = f"/data/wrongcoll-{uuid4().hex[:8]}"

        await repository.create(
            collection_id=collection1.id,
            source_type="directory",
            source_path=source_path,
        )

        fetched = await repository.get_by_collection_and_path(collection2.id, source_path)
        assert fetched is None

    async def test_get_or_create_creates_new_source(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """get_or_create should create source when it doesn't exist."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source_path = f"/data/new-{uuid4().hex[:8]}"

        source, is_new = await repository.get_or_create(
            collection_id=collection.id,
            source_type="directory",
            source_path=source_path,
            source_config={"path": source_path},
        )

        assert is_new is True
        assert source.id is not None
        assert source.source_path == source_path

    async def test_get_or_create_returns_existing_source(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """get_or_create should return existing source when it exists."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source_path = f"/data/existing-{uuid4().hex[:8]}"

        # First call creates
        source1, is_new1 = await repository.get_or_create(
            collection_id=collection.id,
            source_type="directory",
            source_path=source_path,
        )

        # Second call returns existing
        source2, is_new2 = await repository.get_or_create(
            collection_id=collection.id,
            source_type="directory",
            source_path=source_path,
        )

        assert is_new1 is True
        assert is_new2 is False
        assert source1.id == source2.id

    async def test_get_or_create_updates_config_on_existing(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """get_or_create should update source_config when returning existing."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source_path = f"/data/updatecfg-{uuid4().hex[:8]}"

        # Create with initial config
        source1, _ = await repository.get_or_create(
            collection_id=collection.id,
            source_type="directory",
            source_path=source_path,
            source_config={"path": source_path, "recursive": False},
        )

        # Get again with updated config
        source2, is_new = await repository.get_or_create(
            collection_id=collection.id,
            source_type="directory",
            source_path=source_path,
            source_config={"path": source_path, "recursive": True},
        )

        assert is_new is False
        assert source2.id == source1.id
        assert source2.source_config["recursive"] is True

    async def test_list_by_collection_returns_sources(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """list_by_collection should return sources for the collection."""
        collection = await collection_factory(owner_id=test_user_db.id)

        # Create multiple sources
        for i in range(3):
            await repository.create(
                collection_id=collection.id,
                source_type="directory",
                source_path=f"/data/list-{uuid4().hex[:8]}-{i}",
            )

        sources, total = await repository.list_by_collection(collection.id)

        assert total == 3
        assert len(sources) == 3

    async def test_list_by_collection_pagination(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """list_by_collection should respect pagination parameters."""
        collection = await collection_factory(owner_id=test_user_db.id)

        # Create 5 sources
        for i in range(5):
            await repository.create(
                collection_id=collection.id,
                source_type="directory",
                source_path=f"/data/page-{uuid4().hex[:8]}-{i}",
            )

        sources_page1, total = await repository.list_by_collection(collection.id, offset=0, limit=2)
        sources_page2, _ = await repository.list_by_collection(collection.id, offset=2, limit=2)

        assert total == 5
        assert len(sources_page1) == 2
        assert len(sources_page2) == 2

    async def test_list_by_collection_excludes_other_collections(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """list_by_collection should only return sources for the specified collection."""
        collection1 = await collection_factory(owner_id=test_user_db.id)
        collection2 = await collection_factory(owner_id=test_user_db.id)

        await repository.create(
            collection_id=collection1.id,
            source_type="directory",
            source_path=f"/data/coll1-{uuid4().hex[:8]}",
        )
        await repository.create(
            collection_id=collection2.id,
            source_type="directory",
            source_path=f"/data/coll2-{uuid4().hex[:8]}",
        )

        sources1, total1 = await repository.list_by_collection(collection1.id)
        sources2, total2 = await repository.list_by_collection(collection2.id)

        assert total1 == 1
        assert total2 == 1
        assert sources1[0].collection_id == collection1.id
        assert sources2[0].collection_id == collection2.id

    async def test_update_stats_updates_counters(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """update_stats should update document_count, size_bytes, and last_indexed_at."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/stats-{uuid4().hex[:8]}",
        )

        now = datetime.now(UTC)
        updated = await repository.update_stats(
            source_id=source.id,
            document_count=10,
            size_bytes=50000,
            last_indexed_at=now,
        )

        assert updated.document_count == 10
        assert updated.size_bytes == 50000
        assert updated.last_indexed_at is not None

    async def test_update_stats_raises_for_missing_source(self, repository: CollectionSourceRepository) -> None:
        """update_stats should raise EntityNotFoundError for missing source."""
        with pytest.raises(EntityNotFoundError):
            await repository.update_stats(source_id=999999, document_count=5)

    async def test_update_stats_rejects_negative_counts(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """update_stats should reject negative counts."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/negcnt-{uuid4().hex[:8]}",
        )

        with pytest.raises(ValidationError) as exc_info:
            await repository.update_stats(source_id=source.id, document_count=-1)
        assert "document_count" in str(exc_info.value).lower()

        with pytest.raises(ValidationError) as exc_info:
            await repository.update_stats(source_id=source.id, size_bytes=-1)
        assert "size_bytes" in str(exc_info.value).lower()

    async def test_delete_removes_source(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """delete should remove the source."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/delete-{uuid4().hex[:8]}",
        )

        await repository.delete(source.id)

        fetched = await repository.get_by_id(source.id)
        assert fetched is None

    async def test_delete_raises_for_missing_source(self, repository: CollectionSourceRepository) -> None:
        """delete should raise EntityNotFoundError for missing source."""
        with pytest.raises(EntityNotFoundError):
            await repository.delete(999999)

    # -------------------------------------------------------------------------
    # Sync-related method tests
    # -------------------------------------------------------------------------

    async def test_create_source_with_sync_mode_one_time(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Creating a source with one_time sync mode should work without interval."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source_path = f"/data/onetime-{uuid4().hex[:8]}"

        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=source_path,
            sync_mode="one_time",
        )

        assert source.sync_mode == "one_time"
        assert source.interval_minutes is None
        assert source.next_run_at is None
        assert source.paused_at is None

    async def test_create_source_one_time_ignores_interval_minutes_even_if_too_short(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """one_time sync mode should ignore interval_minutes to match DB constraint."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source_path = f"/data/onetime-int-{uuid4().hex[:8]}"

        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=source_path,
            sync_mode="one_time",
            interval_minutes=5,
        )

        assert source.sync_mode == "one_time"
        assert source.interval_minutes is None

    async def test_create_source_with_sync_mode_continuous(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Creating a source with continuous sync mode should set next_run_at."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source_path = f"/data/continuous-{uuid4().hex[:8]}"

        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=source_path,
            sync_mode="continuous",
            interval_minutes=30,
        )

        assert source.sync_mode == "continuous"
        assert source.interval_minutes == 30
        assert source.next_run_at is not None  # Should be scheduled immediately
        assert source.paused_at is None

    async def test_create_source_continuous_requires_interval(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Continuous sync mode requires interval_minutes."""
        collection = await collection_factory(owner_id=test_user_db.id)

        with pytest.raises(ValidationError) as exc_info:
            await repository.create(
                collection_id=collection.id,
                source_type="directory",
                source_path=f"/data/noint-{uuid4().hex[:8]}",
                sync_mode="continuous",
            )
        assert "interval_minutes" in str(exc_info.value).lower()

    async def test_create_source_continuous_min_interval(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Continuous sync mode requires minimum 15 minute interval."""
        collection = await collection_factory(owner_id=test_user_db.id)

        with pytest.raises(ValidationError) as exc_info:
            await repository.create(
                collection_id=collection.id,
                source_type="directory",
                source_path=f"/data/shortint-{uuid4().hex[:8]}",
                sync_mode="continuous",
                interval_minutes=5,  # Too short
            )
        assert "15" in str(exc_info.value)

    async def test_create_source_invalid_sync_mode(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Invalid sync mode should raise ValidationError."""
        collection = await collection_factory(owner_id=test_user_db.id)

        with pytest.raises(ValidationError) as exc_info:
            await repository.create(
                collection_id=collection.id,
                source_type="directory",
                source_path=f"/data/invalid-{uuid4().hex[:8]}",
                sync_mode="invalid_mode",
            )
        assert "sync_mode" in str(exc_info.value).lower()

    async def test_update_source_config(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Update should modify source_config."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/update-{uuid4().hex[:8]}",
            source_config={"recursive": False},
        )

        updated = await repository.update(
            source_id=source.id,
            source_config={"recursive": True, "max_depth": 5},
        )

        assert updated.source_config["recursive"] is True
        assert updated.source_config["max_depth"] == 5

    async def test_update_source_sync_mode(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Update should allow changing sync_mode with proper interval."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/updmode-{uuid4().hex[:8]}",
            sync_mode="one_time",
        )

        updated = await repository.update(
            source_id=source.id,
            sync_mode="continuous",
            interval_minutes=60,
        )

        assert updated.sync_mode == "continuous"
        assert updated.interval_minutes == 60
        assert updated.next_run_at is not None  # Should be scheduled

    async def test_update_one_time_ignores_interval_minutes_even_if_too_short(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Updating a one_time source should ignore interval_minutes to avoid DB constraint failures."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/upd-onetime-{uuid4().hex[:8]}",
            sync_mode="one_time",
        )

        updated = await repository.update(source_id=source.id, interval_minutes=5)

        assert updated.sync_mode == "one_time"
        assert updated.interval_minutes is None

    async def test_update_switch_to_one_time_clears_interval_minutes(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """Switching to one_time should clear interval_minutes."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/upd-clear-{uuid4().hex[:8]}",
            sync_mode="continuous",
            interval_minutes=30,
        )

        updated = await repository.update(source_id=source.id, sync_mode="one_time")

        assert updated.sync_mode == "one_time"
        assert updated.interval_minutes is None

    async def test_update_source_raises_for_missing(
        self, repository: CollectionSourceRepository
    ) -> None:
        """Update should raise EntityNotFoundError for missing source."""
        with pytest.raises(EntityNotFoundError):
            await repository.update(source_id=999999, source_config={"test": True})

    async def test_get_due_for_sync_returns_due_sources(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """get_due_for_sync should return sources where next_run_at is in the past."""
        collection = await collection_factory(owner_id=test_user_db.id)

        # Create a source with next_run_at in the past
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/due-{uuid4().hex[:8]}",
            sync_mode="continuous",
            interval_minutes=15,
        )
        # next_run_at is set to now() by create, so it should be due

        due_sources = await repository.get_due_for_sync()

        assert len(due_sources) >= 1
        source_ids = [s.id for s in due_sources]
        assert source.id in source_ids

    async def test_get_due_for_sync_excludes_paused(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """get_due_for_sync should exclude paused sources."""
        collection = await collection_factory(owner_id=test_user_db.id)

        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/paused-{uuid4().hex[:8]}",
            sync_mode="continuous",
            interval_minutes=15,
        )

        # Pause the source
        await repository.pause(source.id)

        due_sources = await repository.get_due_for_sync()
        source_ids = [s.id for s in due_sources]
        assert source.id not in source_ids

    async def test_get_due_for_sync_excludes_one_time(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """get_due_for_sync should exclude one_time sources."""
        collection = await collection_factory(owner_id=test_user_db.id)

        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/onetime-excl-{uuid4().hex[:8]}",
            sync_mode="one_time",
        )

        due_sources = await repository.get_due_for_sync()
        source_ids = [s.id for s in due_sources]
        assert source.id not in source_ids

    async def test_update_sync_status_success(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """update_sync_status should set status and timestamps."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/status-{uuid4().hex[:8]}",
            sync_mode="continuous",
            interval_minutes=15,
        )

        now = datetime.now(UTC)
        updated = await repository.update_sync_status(
            source_id=source.id,
            status="success",
            started_at=now,
            completed_at=now,
        )

        assert updated.last_run_status == "success"
        assert updated.last_run_started_at is not None
        assert updated.last_run_completed_at is not None
        assert updated.last_error is None

    async def test_update_sync_status_failed_with_error(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """update_sync_status should store error message on failure."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/failed-{uuid4().hex[:8]}",
            sync_mode="continuous",
            interval_minutes=15,
        )

        updated = await repository.update_sync_status(
            source_id=source.id,
            status="failed",
            error="Connection timeout",
        )

        assert updated.last_run_status == "failed"
        assert updated.last_error == "Connection timeout"

    async def test_update_sync_status_invalid_status(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """update_sync_status should reject invalid status values."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/invstat-{uuid4().hex[:8]}",
            sync_mode="continuous",
            interval_minutes=15,
        )

        with pytest.raises(ValidationError) as exc_info:
            await repository.update_sync_status(
                source_id=source.id,
                status="invalid_status",
            )
        assert "status" in str(exc_info.value).lower()

    async def test_set_next_run(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """set_next_run should update next_run_at."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/nextrun-{uuid4().hex[:8]}",
            sync_mode="continuous",
            interval_minutes=15,
        )

        future_time = datetime.now(UTC).replace(hour=23, minute=59)
        updated = await repository.set_next_run(source.id, future_time)

        assert updated.next_run_at == future_time

    async def test_set_next_run_clear(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """set_next_run with None on one_time source should keep next_run_at as None."""
        collection = await collection_factory(owner_id=test_user_db.id)
        # Use one_time source with no interval - set_next_run(None) will result in None
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/clrnext-{uuid4().hex[:8]}",
            sync_mode="one_time",
        )

        updated = await repository.set_next_run(source.id, None)

        assert updated.next_run_at is None

    async def test_pause_source(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """pause should set paused_at timestamp."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/pause-{uuid4().hex[:8]}",
            sync_mode="continuous",
            interval_minutes=15,
        )

        paused = await repository.pause(source.id)

        assert paused.paused_at is not None
        # next_run_at is preserved but won't be used while paused

    async def test_pause_one_time_raises(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """pause should raise ValidationError for one_time sources."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/pauseot-{uuid4().hex[:8]}",
            sync_mode="one_time",
        )

        with pytest.raises(ValidationError) as exc_info:
            await repository.pause(source.id)
        assert "continuous" in str(exc_info.value).lower()

    async def test_resume_source(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """resume should clear paused_at and schedule next_run_at."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/resume-{uuid4().hex[:8]}",
            sync_mode="continuous",
            interval_minutes=15,
        )

        await repository.pause(source.id)
        resumed = await repository.resume(source.id)

        assert resumed.paused_at is None
        assert resumed.next_run_at is not None

    async def test_resume_not_paused_is_noop(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """resume on an unpaused source should be a no-op."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/resumenp-{uuid4().hex[:8]}",
            sync_mode="continuous",
            interval_minutes=15,
        )
        original_next_run = source.next_run_at

        # Resume on unpaused source - should return unchanged
        resumed = await repository.resume(source.id)

        assert resumed.paused_at is None
        # next_run_at should be unchanged since it wasn't paused
        assert resumed.next_run_at == original_next_run
