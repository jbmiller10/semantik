"""Integration tests for ProjectionRunRepository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from shared.database.exceptions import EntityNotFoundError, ValidationError
from shared.database.models import ProjectionRun, ProjectionRunStatus
from shared.database.repositories.projection_run_repository import ProjectionRunRepository
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestProjectionRunRepositoryIntegration:
    """Exercise ProjectionRunRepository against the real database session."""

    @pytest.fixture()
    def repository(self, db_session: AsyncSession) -> ProjectionRunRepository:
        return ProjectionRunRepository(db_session)

    async def test_create_projection_run_persists_row(
        self,
        repository: ProjectionRunRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """Creating a projection run should persist a pending stub."""

        collection = await collection_factory(owner_id=test_user_db.id)

        run = await repository.create(
            collection_id=collection.id,
            reducer="umap",
            dimensionality=3,
            config={"neighbors": 15},
            meta={"color": "document_id"},
            metadata_hash="hash-primary",
        )
        await db_session.commit()

        result = await db_session.execute(select(ProjectionRun).where(ProjectionRun.uuid == run.uuid))
        persisted = result.scalar_one()
        assert persisted.collection_id == collection.id
        assert persisted.status is ProjectionRunStatus.PENDING
        assert persisted.config == {"neighbors": 15}
        assert persisted.meta["color"] == "document_id"
        assert persisted.metadata_hash == "hash-primary"

    async def test_create_projection_run_requires_valid_inputs(
        self,
        repository: ProjectionRunRepository,
        collection_factory,
        test_user_db,
    ) -> None:
        """Validation should reject missing collections or invalid parameters."""

        collection = await collection_factory(owner_id=test_user_db.id)

        with pytest.raises(EntityNotFoundError):
            await repository.create(collection_id=str(uuid4()), reducer="umap", dimensionality=2)

        with pytest.raises(ValidationError):
            await repository.create(collection_id=collection.id, reducer="umap", dimensionality=0)

        with pytest.raises(ValidationError):
            await repository.create(collection_id=collection.id, reducer="", dimensionality=2)

    async def test_get_by_uuid_fetches_projection_run(
        self,
        repository: ProjectionRunRepository,
        collection_factory,
        test_user_db,
    ) -> None:
        """Repository.get_by_uuid should return the stored run when it exists."""

        collection = await collection_factory(owner_id=test_user_db.id)
        created = await repository.create(collection_id=collection.id, reducer="tsne", dimensionality=2)

        fetched = await repository.get_by_uuid(created.uuid)
        missing = await repository.get_by_uuid(str(uuid4()))

        assert fetched is not None
        assert fetched.uuid == created.uuid
        assert missing is None

    async def test_list_for_collection_filters_by_status(
        self,
        repository: ProjectionRunRepository,
        collection_factory,
        test_user_db,
    ) -> None:
        """list_for_collection should honor status filters and ordering."""

        collection = await collection_factory(owner_id=test_user_db.id)
        other_collection = await collection_factory(owner_id=test_user_db.id)

        pending = await repository.create(collection_id=collection.id, reducer="umap", dimensionality=2)
        completed = await repository.create(
            collection_id=collection.id,
            reducer="umap",
            dimensionality=2,
            metadata_hash="hash-x",
        )
        await repository.create(collection_id=other_collection.id, reducer="pca", dimensionality=3)

        await repository.update_status(pending.uuid, ProjectionRunStatus.RUNNING)
        await repository.update_status(completed.uuid, ProjectionRunStatus.COMPLETED)

        runs, total = await repository.list_for_collection(
            collection.id,
            statuses=[ProjectionRunStatus.COMPLETED],
        )

        assert total == 1
        assert runs and runs[0].uuid == completed.uuid

    async def test_update_status_transitions_and_clears_error(
        self,
        repository: ProjectionRunRepository,
        collection_factory,
        test_user_db,
    ) -> None:
        """update_status should manage timestamps and error fields."""

        collection = await collection_factory(owner_id=test_user_db.id)
        run = await repository.create(collection_id=collection.id, reducer="umap", dimensionality=2)

        running = await repository.update_status(run.uuid, ProjectionRunStatus.RUNNING)
        assert running.status is ProjectionRunStatus.RUNNING
        assert running.started_at is not None

        failed = await repository.update_status(run.uuid, ProjectionRunStatus.FAILED, error_message="boom")
        assert failed.status is ProjectionRunStatus.FAILED
        assert failed.error_message == "boom"
        assert failed.completed_at is not None

        completed = await repository.update_status(run.uuid, ProjectionRunStatus.COMPLETED)
        assert completed.status is ProjectionRunStatus.COMPLETED
        assert completed.error_message is None
        assert completed.completed_at is not None

        with pytest.raises(EntityNotFoundError):
            await repository.update_status(str(uuid4()), status=ProjectionRunStatus.RUNNING)

    async def test_set_operation_uuid_links_operation(
        self,
        repository: ProjectionRunRepository,
        db_session: AsyncSession,
        collection_factory,
        operation_factory,
        test_user_db,
    ) -> None:
        """set_operation_uuid should persist the foreign key reference."""

        collection = await collection_factory(owner_id=test_user_db.id)
        run = await repository.create(collection_id=collection.id, reducer="umap", dimensionality=2)
        operation = await operation_factory(collection_id=collection.id, user_id=test_user_db.id)

        updated = await repository.set_operation_uuid(run.uuid, operation.uuid)
        await db_session.commit()

        refreshed = await db_session.get(ProjectionRun, run.id)
        assert updated.operation_uuid == operation.uuid
        assert refreshed.operation_uuid == operation.uuid

        with pytest.raises(EntityNotFoundError):
            await repository.set_operation_uuid(str(uuid4()), operation.uuid)

    async def test_update_metadata_merges_fields(
        self,
        repository: ProjectionRunRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """update_metadata should merge meta dicts and validate point counts."""

        collection = await collection_factory(owner_id=test_user_db.id)
        run = await repository.create(
            collection_id=collection.id,
            reducer="umap",
            dimensionality=2,
            meta={"color": "document_id"},
        )

        updated = await repository.update_metadata(
            run.uuid,
            storage_path="/tmp/run-1.npz",
            point_count=42,
            meta={"projection": "left"},
        )
        await db_session.commit()

        refreshed = await db_session.get(ProjectionRun, run.id)
        assert updated.storage_path == "/tmp/run-1.npz"
        assert refreshed.point_count == 42
        assert refreshed.meta == {"color": "document_id", "projection": "left"}

        with pytest.raises(ValidationError):
            await repository.update_metadata(run.uuid, point_count=-1)

        with pytest.raises(EntityNotFoundError):
            await repository.update_metadata(str(uuid4()), storage_path="/tmp/missing")

    async def test_delete_removes_projection_run(
        self,
        repository: ProjectionRunRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """delete should remove the run and subsequent lookups should fail."""

        collection = await collection_factory(owner_id=test_user_db.id)
        run = await repository.create(collection_id=collection.id, reducer="umap", dimensionality=2)

        await repository.delete(run.uuid)
        await db_session.commit()

        missing = await db_session.get(ProjectionRun, run.id)
        assert missing is None

        with pytest.raises(EntityNotFoundError):
            await repository.delete(str(uuid4()))

    async def test_find_latest_completed_by_metadata_hash(
        self,
        repository: ProjectionRunRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """Repository should return the newest completed run for a metadata hash."""

        collection = await collection_factory(owner_id=test_user_db.id)
        older = await repository.create(
            collection_id=collection.id,
            reducer="umap",
            dimensionality=2,
            metadata_hash="hash-shared",
        )
        newer = await repository.create(
            collection_id=collection.id,
            reducer="umap",
            dimensionality=2,
            metadata_hash="hash-shared",
        )
        running = await repository.create(
            collection_id=collection.id,
            reducer="pca",
            dimensionality=3,
            metadata_hash="hash-shared",
        )

        # Mark completion times explicitly to guarantee ordering
        await repository.update_status(older.uuid, ProjectionRunStatus.COMPLETED)
        await repository.update_status(newer.uuid, ProjectionRunStatus.COMPLETED)
        await repository.update_status(running.uuid, ProjectionRunStatus.RUNNING)

        older.created_at = datetime.now(UTC) - timedelta(hours=2)
        newer.created_at = datetime.now(UTC)
        await db_session.flush()

        latest = await repository.find_latest_completed_by_metadata_hash(collection.id, "hash-shared")
        assert latest is not None and latest.uuid == newer.uuid

        missing = await repository.find_latest_completed_by_metadata_hash(collection.id, "hash-missing")
        assert missing is None
