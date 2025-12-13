"""Integration tests for CollectionService using the real database."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select

from shared.database.exceptions import (
    AccessDeniedError as PackageAccessDeniedError,
    AccessDeniedError as SharedAccessDeniedError,
    InvalidStateError as PackageInvalidStateError,
    InvalidStateError as SharedInvalidStateError,
)
from shared.database.models import Collection, CollectionStatus, Operation, OperationStatus, OperationType, User
from webui.services import collection_service as collection_service_module
from webui.services.factory import create_collection_service


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestCollectionServiceIntegration:
    """Verify CollectionService behavior against the actual repositories."""

    INVALID_STATE_ERRORS = (PackageInvalidStateError, SharedInvalidStateError)
    ACCESS_DENIED_ERRORS = (PackageAccessDeniedError, SharedAccessDeniedError)

    @pytest.fixture()
    def service(self, db_session, fake_qdrant):
        """Provide a CollectionService wired to the real repositories."""
        return create_collection_service(
            db_session,
            qdrant_manager_override=fake_qdrant.manager,
        )

    @pytest.fixture()
    def capture_celery(self, monkeypatch):
        """Capture Celery send_task invocations."""
        calls: list[dict[str, object]] = []

        def fake_send_task(name, args=None, task_id=None, **kwargs):  # noqa: ANN001
            calls.append({"name": name, "args": args, "task_id": task_id, "kwargs": kwargs})
            return SimpleNamespace(id=task_id)

        monkeypatch.setattr(collection_service_module.celery_app, "send_task", fake_send_task)
        return calls

    @pytest.fixture(autouse=True)
    def _auto_patch_celery(self, capture_celery):
        """Ensure Celery is patched for every test in this class."""
        capture_celery.clear()
        yield
        capture_celery.clear()

    @pytest.fixture()
    def fake_qdrant(self):
        """Stub out Qdrant interactions for predictable assertions."""
        deleted: list[str] = []

        class FakeClient:
            def __init__(self):
                self._collections: list[str] = []

            def prime(self, names: list[str]) -> None:
                self._collections = names

            def get_collections(self):
                return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self._collections])

            def delete_collection(self, name: str) -> None:
                deleted.append(name)

        class FakeManager:
            def __init__(self, client: FakeClient):
                self.client = client

            def list_collections(self) -> list[str]:
                collections = self.client.get_collections()
                return [col.name for col in collections.collections]

        client = FakeClient()
        manager = FakeManager(client)
        return SimpleNamespace(client=client, deleted=deleted, manager=manager)

    @pytest.fixture()
    def user_factory(self, db_session):
        """Create persistent users with unique identifiers."""

        async def _create_user(**overrides) -> User:
            user = User(
                id=int(uuid4().int % 2_147_483_647),
                username=f"user_{uuid4().hex[:8]}",
                email=f"user_{uuid4().hex[:8]}@example.com",
                hashed_password="hashed_password",
                is_active=True,
                is_superuser=False,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                last_login=None,
            )
            for field, value in overrides.items():
                setattr(user, field, value)

            db_session.add(user)
            await db_session.commit()
            await db_session.refresh(user)
            return user

        return _create_user

    async def test_create_collection_persists_records_and_dispatches_task(
        self,
        service,
        db_session,
        user_factory,
        capture_celery,
    ) -> None:
        """Collection creation should persist data and enqueue background work."""
        owner = await user_factory()
        collection_dict, operation_dict = await service.create_collection(
            user_id=owner.id,
            name=f"svc-create-{uuid4().hex[:8]}",
            description="integration create",
        )

        # Validate collection was committed
        result = await db_session.execute(select(Collection).where(Collection.id == collection_dict["id"]))
        persisted = result.scalar_one()
        assert persisted.owner_id == owner.id
        assert persisted.status == CollectionStatus.PENDING
        assert persisted.vector_store_name.startswith("col_")

        # Validate operation record persisted
        op_result = await db_session.execute(select(Operation).where(Operation.uuid == operation_dict["uuid"]))
        operation = op_result.scalar_one()
        assert operation.type is OperationType.INDEX
        assert operation.collection_id == collection_dict["id"]

        # Ensure Celery was notified with the operation UUID after commit
        assert len(capture_celery) == 1
        assert capture_celery[0]["name"] == "webui.tasks.process_collection_operation"
        assert capture_celery[0]["args"] == [operation_dict["uuid"]]

    async def test_delete_collection_removes_records_and_invokes_qdrant(
        self,
        service,
        db_session,
        user_factory,
        capture_celery,
        fake_qdrant,
    ) -> None:
        """Collection deletion should cascade and notify Qdrant."""
        client = fake_qdrant.client

        owner = await user_factory()

        # Create collection (this populates DB + operation)
        collection_dict, operation_dict = await service.create_collection(
            user_id=owner.id,
            name=f"svc-delete-{uuid4().hex[:8]}",
        )

        # Mark the indexing operation as completed to allow deletion
        await service.operation_repo.update_status(
            operation_dict["uuid"],
            OperationStatus.COMPLETED,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        await service.collection_repo.update_status(collection_dict["id"], CollectionStatus.READY)
        await db_session.commit()

        # Ensure qdrant stub reports the collection existing
        client.prime([collection_dict["vector_store_name"]])

        await service.delete_collection(collection_dict["id"], owner.id)

        # Verify the collection was removed from the database
        result = await db_session.execute(select(Collection).where(Collection.id == collection_dict["id"]))
        assert result.scalar_one_or_none() is None

        # Operations should cascade-delete as well
        op_result = await db_session.execute(select(Operation).where(Operation.collection_id == collection_dict["id"]))
        assert op_result.scalars().first() is None

        # Qdrant deletion was requested
        assert fake_qdrant.deleted == [collection_dict["vector_store_name"]]

    async def test_update_collection_applies_field_changes(
        self,
        service,
        db_session,
        user_factory,
    ) -> None:
        """Updating a collection should persist new metadata and config values."""
        owner = await user_factory()
        collection_dict, operation_dict = await service.create_collection(
            user_id=owner.id,
            name=f"svc-update-{uuid4().hex[:8]}",
            description="before",
        )

        await self._mark_initial_operation_complete(
            service,
            db_session,
            operation_dict["uuid"],
            collection_dict["id"],
        )

        updates = {
            "description": "after",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "is_public": True,
        }
        updated = await service.update_collection(collection_dict["id"], updates, user_id=owner.id)

        assert updated["description"] == "after"
        assert updated["chunk_size"] == 512
        assert updated["chunk_overlap"] == 64
        assert updated["is_public"] is True
        assert updated["config"]["chunk_size"] == 512

        result = await db_session.execute(select(Collection).where(Collection.id == collection_dict["id"]))
        persisted = result.scalar_one()
        assert persisted.description == "after"
        assert persisted.chunk_size == 512
        assert persisted.chunk_overlap == 64
        assert persisted.is_public is True

    async def test_add_source_creates_append_operation_and_sets_processing_status(
        self,
        service,
        db_session,
        user_factory,
        capture_celery,
    ) -> None:
        """Adding a source should enqueue an APPEND operation and flip collection status."""
        owner = await user_factory()
        collection_dict, operation_dict = await service.create_collection(
            user_id=owner.id,
            name=f"svc-add-source-{uuid4().hex[:8]}",
        )

        await self._mark_initial_operation_complete(
            service,
            db_session,
            operation_dict["uuid"],
            collection_dict["id"],
        )

        capture_celery.clear()

        op_payload = await service.add_source(
            collection_id=collection_dict["id"],
            user_id=owner.id,
            source_type="directory",
            source_config={"path": "/mnt/data/new-docs", "recursive": True},
        )

        assert op_payload["type"] == OperationType.APPEND.value
        assert op_payload["config"]["source_path"] == "/mnt/data/new-docs"
        assert op_payload["config"]["source_config"]["path"] == "/mnt/data/new-docs"
        assert op_payload["config"]["source_config"]["recursive"] is True
        assert op_payload["config"]["source_id"] is not None  # source_id now included

        # Celery task dispatched for the new operation
        assert len(capture_celery) == 1
        assert capture_celery[0]["args"] == [op_payload["uuid"]]

        # Collection status should be PROCESSING after scheduling operation
        result = await db_session.execute(select(Collection).where(Collection.id == collection_dict["id"]))
        persisted = result.scalar_one()
        assert persisted.status == CollectionStatus.PROCESSING

    async def test_remove_source_creates_remove_operation(
        self,
        service,
        db_session,
        user_factory,
        capture_celery,
    ) -> None:
        """Removing a source should schedule a REMOVE_SOURCE operation."""
        from shared.database.repositories.collection_source_repository import CollectionSourceRepository

        owner = await user_factory()
        collection_dict, operation_dict = await service.create_collection(
            user_id=owner.id,
            name=f"svc-remove-source-{uuid4().hex[:8]}",
        )

        await self._mark_initial_operation_complete(
            service,
            db_session,
            operation_dict["uuid"],
            collection_dict["id"],
        )

        # Create a collection source that we'll remove
        source_repo = CollectionSourceRepository(db_session)
        source = await source_repo.create(
            collection_id=collection_dict["id"],
            source_type="directory",
            source_path="/mnt/data/old-docs",
        )
        await db_session.commit()

        capture_celery.clear()

        op_payload = await service.remove_source(
            collection_id=collection_dict["id"],
            user_id=owner.id,
            source_path="/mnt/data/old-docs",
        )

        assert op_payload["type"] == OperationType.REMOVE_SOURCE.value
        assert op_payload["config"]["source_path"] == "/mnt/data/old-docs"
        assert op_payload["config"]["source_id"] == source.id

        assert len(capture_celery) == 1
        assert capture_celery[0]["args"] == [op_payload["uuid"]]

        # Collection status should be PROCESSING while removal runs
        result = await db_session.execute(select(Collection).where(Collection.id == collection_dict["id"]))
        persisted = result.scalar_one()
        assert persisted.status == CollectionStatus.PROCESSING

    async def test_reindex_collection_creates_blue_green_operation(
        self,
        service,
        db_session,
        user_factory,
        capture_celery,
    ) -> None:
        """Reindex should create REINDEX operation with previous/new configs."""
        owner = await user_factory()
        collection_dict, operation_dict = await service.create_collection(
            user_id=owner.id,
            name=f"svc-reindex-{uuid4().hex[:8]}",
        )

        await self._mark_initial_operation_complete(
            service,
            db_session,
            operation_dict["uuid"],
            collection_dict["id"],
        )

        capture_celery.clear()

        payload = await service.reindex_collection(
            collection_id=collection_dict["id"],
            user_id=owner.id,
            config_updates={"chunk_size": 2048, "metadata": {"note": "reindex"}},
        )

        assert payload["type"] == OperationType.REINDEX.value
        assert payload["config"]["blue_green"] is True
        assert payload["config"]["new_config"]["chunk_size"] == 2048
        assert payload["config"]["new_config"]["metadata"] == {"note": "reindex"}
        assert payload["config"]["previous_config"]["chunk_size"] == 1000

        # Operation persisted with updated config
        op_result = await db_session.execute(select(Operation).where(Operation.uuid == payload["uuid"]))
        operation = op_result.scalar_one()
        assert operation.type is OperationType.REINDEX
        assert operation.config["new_config"]["chunk_size"] == 2048

        # Collection status switched to PROCESSING
        result = await db_session.execute(select(Collection).where(Collection.id == collection_dict["id"]))
        persisted = result.scalar_one()
        assert persisted.status == CollectionStatus.PROCESSING

        assert len(capture_celery) == 1
        assert capture_celery[0]["args"] == [payload["uuid"]]

    async def test_create_operation_helper_maps_types_and_commits(
        self,
        service,
        db_session,
        user_factory,
    ) -> None:
        """create_operation should persist a new Operation and map string type."""
        owner = await user_factory()
        collection_dict, operation_dict = await service.create_collection(
            user_id=owner.id,
            name=f"svc-create-op-{uuid4().hex[:8]}",
        )

        await self._mark_initial_operation_complete(
            service,
            db_session,
            operation_dict["uuid"],
            collection_dict["id"],
        )

        payload = await service.create_operation(
            collection_id=collection_dict["id"],
            operation_type="rechunking",
            config={"dry_run": False},
            user_id=owner.id,
        )

        assert payload["type"] == OperationType.REINDEX.value
        assert payload["meta"]["operation_type"] == "rechunking"

        op_result = await db_session.execute(select(Operation).where(Operation.uuid == payload["uuid"]))
        operation = op_result.scalar_one()
        assert operation.type is OperationType.REINDEX
        assert operation.meta["operation_type"] == "rechunking"
        assert operation.config == {"dry_run": False}

    async def test_list_for_user_returns_owned_and_public_collections(
        self,
        service,
        db_session,
        user_factory,
    ) -> None:
        """list_for_user should include owned and public collections when requested."""
        owner = await user_factory()
        other_owner = await user_factory()

        # Owned collection
        owned, _ = await service.create_collection(
            user_id=owner.id,
            name=f"svc-list-owned-{uuid4().hex[:6]}",
        )
        await db_session.commit()

        # Public collection from another user
        public, _ = await service.create_collection(
            user_id=other_owner.id,
            name=f"svc-list-public-{uuid4().hex[:6]}",
            config={"is_public": True},
        )
        await db_session.commit()

        collections, total = await service.list_for_user(user_id=owner.id)

        ids = {collection.id for collection in collections}
        assert owned["id"] in ids
        assert public["id"] in ids
        assert total == len(ids)

    async def test_list_documents_returns_documents_for_collection(
        self,
        service,
        db_session,
        user_factory,
        document_factory,
    ) -> None:
        """list_documents should retrieve persisted documents."""
        owner = await user_factory()
        collection_dict, op_dict = await service.create_collection(
            user_id=owner.id,
            name=f"svc-docs-{uuid4().hex[:6]}",
        )
        await self._mark_initial_operation_complete(service, db_session, op_dict["uuid"], collection_dict["id"])

        # Create sample documents
        docs_created = [
            await document_factory(collection_id=collection_dict["id"], file_name=f"doc-{i}.txt") for i in range(3)
        ]

        documents, total = await service.list_documents(
            collection_id=collection_dict["id"],
            user_id=owner.id,
        )

        assert total == 3
        returned_ids = {doc.id for doc in documents}
        assert returned_ids == {doc.id for doc in docs_created}

    async def test_list_operations_returns_committed_operations(
        self,
        service,
        db_session,
        user_factory,
    ) -> None:
        """list_operations should surface active operations for the collection."""
        owner = await user_factory()
        collection_dict, op_dict = await service.create_collection(
            user_id=owner.id,
            name=f"svc-ops-{uuid4().hex[:6]}",
        )
        await self._mark_initial_operation_complete(service, db_session, op_dict["uuid"], collection_dict["id"])

        # Schedule additional operation
        await service.reindex_collection(collection_dict["id"], owner.id)

        operations, total = await service.list_operations(
            collection_id=collection_dict["id"],
            user_id=owner.id,
        )

        assert total == len(operations)
        assert {op.collection_id for op in operations} == {collection_dict["id"]}

    async def test_add_source_rejects_processing_state(
        self,
        service,
        db_session,
        user_factory,
    ) -> None:
        """Calling add_source while collection is processing should raise InvalidStateError."""
        owner = await user_factory()
        collection_dict, op_dict = await service.create_collection(
            user_id=owner.id,
            name=f"svc-add-invalid-{uuid4().hex[:6]}",
        )
        # Mark initial op processing without completion
        await service.collection_repo.update_status(collection_dict["id"], CollectionStatus.PROCESSING)
        await db_session.commit()

        with pytest.raises(self.INVALID_STATE_ERRORS):
            await service.add_source(collection_dict["id"], owner.id, legacy_source_path="/tmp/new")

    async def test_remove_source_requires_ready_state(
        self,
        service,
        db_session,
        user_factory,
    ) -> None:
        """remove_source should fail if collection is not in READY/DEGRADED."""
        owner = await user_factory()
        collection_dict, op_dict = await service.create_collection(
            user_id=owner.id,
            name=f"svc-remove-invalid-{uuid4().hex[:6]}",
        )
        # Collection is still PENDING (initial state)

        with pytest.raises(self.INVALID_STATE_ERRORS):
            await service.remove_source(collection_dict["id"], owner.id, "/tmp/remove")

    async def test_reindex_collection_rejects_active_operations(
        self,
        service,
        db_session,
        user_factory,
    ) -> None:
        """Reindexing should not proceed if another operation is active."""
        owner = await user_factory()
        collection_dict, op_dict = await service.create_collection(
            user_id=owner.id,
            name=f"svc-reindex-invalid-{uuid4().hex[:6]}",
        )
        # Leave initial operation as PENDING (active)

        with pytest.raises(self.INVALID_STATE_ERRORS):
            await service.reindex_collection(collection_dict["id"], owner.id)

    async def test_delete_collection_requires_owner(
        self,
        service,
        db_session,
        user_factory,
    ) -> None:
        """Deletion should raise when user is not the owner."""
        owner = await user_factory()
        other_owner = await user_factory()
        collection_dict, op_dict = await service.create_collection(
            user_id=owner.id,
            name=f"svc-delete-not-owner-{uuid4().hex[:6]}",
        )
        await self._mark_initial_operation_complete(service, db_session, op_dict["uuid"], collection_dict["id"])

        with pytest.raises(self.ACCESS_DENIED_ERRORS):
            await service.delete_collection(collection_dict["id"], other_owner.id)

    async def _mark_initial_operation_complete(
        self,
        service,
        db_session,
        operation_uuid: str,
        collection_id: str,
    ) -> None:
        """Helper to mark the bootstrapped INDEX operation as finished and collection READY."""
        await service.operation_repo.update_status(
            operation_uuid,
            OperationStatus.COMPLETED,
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )
        await service.collection_repo.update_status(collection_id, CollectionStatus.READY)
        await db_session.commit()
