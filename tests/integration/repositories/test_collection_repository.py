"""Integration tests for CollectionRepository using the real database session."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import select

from shared.database.exceptions import (
    AccessDeniedError,
    DatabaseOperationError,
    EntityAlreadyExistsError,
    ValidationError,
)
from shared.database.models import Collection, CollectionStatus
from shared.database.repositories.collection_repository import CollectionRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestCollectionRepositoryIntegration:
    """Integration coverage for the collection repository."""

    @pytest.fixture()
    def repository(self, db_session: AsyncSession) -> CollectionRepository:
        """Construct the repository with a real async session."""
        return CollectionRepository(db_session)

    async def test_create_collection_persists_expected_defaults(
        self, repository: CollectionRepository, db_session: AsyncSession, test_user_db
    ) -> None:
        """Creating a collection should persist it with default metadata."""
        name = f"repo-create-{uuid4().hex[:8]}"

        collection = await repository.create(
            name=name,
            owner_id=test_user_db.id,
            description="integration create",
            chunk_size=512,
            chunk_overlap=64,
        )

        assert collection.name == name
        assert collection.owner_id == test_user_db.id
        assert collection.status is CollectionStatus.PENDING
        assert collection.vector_store_name.startswith("col_")

        result = await db_session.execute(select(Collection).where(Collection.id == collection.id))
        persisted = result.scalar_one()
        assert persisted.name == name
        assert persisted.chunk_size == 512
        assert persisted.chunk_overlap == 64

    async def test_create_collection_duplicate_name_raises_conflict(
        self,
        repository: CollectionRepository,
        test_user_db,
        collection_factory,
    ) -> None:
        """Duplicate collection names should raise DatabaseOperationError."""
        duplicate_name = f"duplicate-{uuid4().hex[:8]}"
        existing = await collection_factory(owner_id=test_user_db.id, name=duplicate_name)

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repository.create(name=existing.name, owner_id=test_user_db.id)

        assert "already exists" in str(exc_info.value)
        assert duplicate_name in str(exc_info.value)

    async def test_create_collection_invalid_chunk_size_raises_validation_error(
        self, repository: CollectionRepository, test_user_db
    ) -> None:
        """Invalid chunk configuration should bubble up as ValidationError."""
        with pytest.raises(DatabaseOperationError) as exc_info:
            await repository.create(name=f"invalid-{uuid4().hex[:6]}", owner_id=test_user_db.id, chunk_size=0)

        assert "Chunk size must be positive" in str(exc_info.value)

    async def test_get_by_uuid_returns_collection(
        self, repository: CollectionRepository, collection_factory, test_user_db
    ) -> None:
        """Repository should fetch the collection by UUID."""
        collection = await collection_factory(owner_id=test_user_db.id)

        fetched = await repository.get_by_uuid(collection.id)

        assert fetched is not None
        assert fetched.id == collection.id
        assert fetched.name == collection.name

    async def test_get_by_uuid_returns_none_for_missing_collection(self, repository: CollectionRepository) -> None:
        """Missing collections should return None."""
        fetched = await repository.get_by_uuid("00000000-0000-0000-0000-000000000000")

        assert fetched is None

    async def test_get_by_uuid_with_permission_check_rejects_non_owner(
        self, repository: CollectionRepository, collection_factory, test_user_db, other_user_db
    ) -> None:
        """Permission check should raise when user does not own the collection."""
        collection = await collection_factory(owner_id=test_user_db.id)

        with pytest.raises(AccessDeniedError):
            await repository.get_by_uuid_with_permission_check(collection.id, other_user_db.id)

    async def test_get_by_uuid_with_permission_check_allows_owner(
        self, repository: CollectionRepository, collection_factory, test_user_db
    ) -> None:
        """Owner should be allowed to fetch the collection with permission check."""
        collection = await collection_factory(owner_id=test_user_db.id)

        fetched = await repository.get_by_uuid_with_permission_check(collection.id, test_user_db.id)

        assert fetched.id == collection.id

    async def test_list_for_user_includes_owned_and_public_collections(
        self,
        repository: CollectionRepository,
        test_user_db,
        other_user_db,
        collection_factory,
    ) -> None:
        """Listing for a user should include owned and public collections."""
        owned = await collection_factory(owner_id=test_user_db.id, name=f"owned-{uuid4().hex[:8]}")
        public = await collection_factory(owner_id=other_user_db.id, name=f"public-{uuid4().hex[:8]}", is_public=True)

        collections, total = await repository.list_for_user(test_user_db.id)

        returned_ids = {collection.id for collection in collections}
        assert owned.id in returned_ids
        assert public.id in returned_ids
        assert total == len(returned_ids)

        collections_without_public, total_without_public = await repository.list_for_user(
            test_user_db.id, include_public=False
        )

        returned_ids = {collection.id for collection in collections_without_public}
        assert owned.id in returned_ids
        assert public.id not in returned_ids
        assert total_without_public == len(returned_ids)

    async def test_rename_collection_updates_name(
        self,
        repository: CollectionRepository,
        db_session: AsyncSession,
        test_user_db,
        collection_factory,
    ) -> None:
        """Renaming a collection should persist the new name."""
        collection = await collection_factory(owner_id=test_user_db.id, name=f"original-{uuid4().hex[:8]}")
        new_name = f"renamed-{uuid4().hex[:8]}"

        updated = await repository.rename(collection.id, new_name, test_user_db.id)

        assert updated.name == new_name

        result = await db_session.execute(select(Collection).where(Collection.id == collection.id))
        persisted = result.scalar_one()
        assert persisted.name == new_name

    async def test_rename_collection_rejects_duplicate_names(
        self,
        repository: CollectionRepository,
        collection_factory,
        test_user_db,
        other_user_db,
    ) -> None:
        """Renaming fails when the target name already exists."""
        collection = await collection_factory(owner_id=test_user_db.id, name=f"primary-{uuid4().hex[:8]}")
        existing = await collection_factory(owner_id=other_user_db.id, name=f"taken-{uuid4().hex[:8]}", is_public=True)

        with pytest.raises(ValidationError):
            await repository.rename(collection.id, "", test_user_db.id)

        with pytest.raises(EntityAlreadyExistsError):
            await repository.rename(collection.id, existing.name, test_user_db.id)

    async def test_update_stats_overwrites_counts(
        self,
        repository: CollectionRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """Updating collection stats should change the persisted counters."""
        collection = await collection_factory(owner_id=test_user_db.id)

        updated = await repository.update_stats(collection.id, document_count=5, vector_count=10, total_size_bytes=2048)

        assert updated.document_count == 5
        assert updated.vector_count == 10
        assert updated.total_size_bytes == 2048

        result = await db_session.execute(select(Collection).where(Collection.id == collection.id))
        persisted = result.scalar_one()
        assert persisted.document_count == 5
        assert persisted.vector_count == 10
        assert persisted.total_size_bytes == 2048

        with pytest.raises(DatabaseOperationError) as exc_info:
            await repository.update_stats(collection.id, document_count=-1)
        assert "Document count cannot be negative" in str(exc_info.value)

    async def test_update_collection_allows_multiple_fields(
        self,
        repository: CollectionRepository,
        db_session: AsyncSession,
        collection_factory,
        test_user_db,
    ) -> None:
        """Repository.update should apply atomic updates to allowed fields."""
        collection = await collection_factory(owner_id=test_user_db.id)

        updates = {
            "description": "Updated description",
            "chunk_size": 256,
            "chunk_overlap": 32,
            "is_public": True,
        }
        updated = await repository.update(collection.id, updates)

        assert updated.description == "Updated description"
        assert updated.chunk_size == 256
        assert updated.chunk_overlap == 32
        assert updated.is_public is True

        result = await db_session.execute(select(Collection).where(Collection.id == collection.id))
        persisted = result.scalar_one()
        assert persisted.description == "Updated description"
        assert persisted.is_public is True

        with pytest.raises(ValidationError):
            await repository.update(collection.id, {"chunk_overlap": 1024})

        with pytest.raises(ValidationError):
            await repository.update(collection.id, {"unknown_field": "value"})
