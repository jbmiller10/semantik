"""Integration tests for CollectionSourceRepository using the real database session.

Note: Sync policy (mode, interval, pause/resume) is now managed at collection level.
Sources only track per-source telemetry (last_run_* fields).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from shared.database.exceptions import EntityAlreadyExistsError, EntityNotFoundError, ValidationError
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

    async def test_update_source_raises_for_missing(self, repository: CollectionSourceRepository) -> None:
        """Update should raise EntityNotFoundError for missing source."""
        with pytest.raises(EntityNotFoundError):
            await repository.update(source_id=999999, source_config={"test": True})

    # -------------------------------------------------------------------------
    # Per-source telemetry tests (last_run_* fields are still on source)
    # -------------------------------------------------------------------------

    async def test_update_sync_status_success(
        self, repository: CollectionSourceRepository, collection_factory, test_user_db
    ) -> None:
        """update_sync_status should set status and timestamps."""
        collection = await collection_factory(owner_id=test_user_db.id)
        source = await repository.create(
            collection_id=collection.id,
            source_type="directory",
            source_path=f"/data/status-{uuid4().hex[:8]}",
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
        )

        with pytest.raises(ValidationError) as exc_info:
            await repository.update_sync_status(
                source_id=source.id,
                status="invalid_status",
            )
        assert "status" in str(exc_info.value).lower()
