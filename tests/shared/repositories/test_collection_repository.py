"""Tests for CollectionRepository.

This module tests collection CRUD operations, permissions, and sync management.
"""

from datetime import UTC, datetime, timedelta

import pytest

from shared.database.exceptions import (
    AccessDeniedError,
    EntityAlreadyExistsError,
    EntityNotFoundError,
    ValidationError,
)
from shared.database.repositories.collection_repository import CollectionRepository


class TestCollectionRepositoryPermissions:
    """Tests for permission checking."""

    @pytest.mark.asyncio()
    async def test_get_by_uuid_with_permission_check_raises_access_denied(
        self, db_session, test_user_db, other_user_db, collection_factory
    ):
        """Test get_by_uuid_with_permission_check() raises AccessDeniedError for non-owner of private collection."""
        owner = test_user_db
        other_user = other_user_db

        # Create private collection owned by 'owner'
        collection = await collection_factory(owner_id=owner.id, is_public=False)

        repo = CollectionRepository(db_session)

        # Other user should not have access
        with pytest.raises(AccessDeniedError):
            await repo.get_by_uuid_with_permission_check(collection.id, other_user.id)

    @pytest.mark.asyncio()
    async def test_get_by_uuid_with_permission_check_allows_public_access(
        self, db_session, test_user_db, other_user_db, collection_factory
    ):
        """Test get_by_uuid_with_permission_check() allows access to public collections."""
        owner = test_user_db
        other_user = other_user_db

        # Create public collection owned by 'owner'
        collection = await collection_factory(owner_id=owner.id, is_public=True)

        repo = CollectionRepository(db_session)

        # Other user should have access to public collection
        result = await repo.get_by_uuid_with_permission_check(collection.id, other_user.id)

        assert result is not None
        assert result.id == collection.id

    @pytest.mark.asyncio()
    async def test_get_by_uuid_with_permission_check_raises_not_found(self, db_session):
        """Test get_by_uuid_with_permission_check() raises EntityNotFoundError for missing collection."""
        repo = CollectionRepository(db_session)

        with pytest.raises(EntityNotFoundError, match="collection"):
            await repo.get_by_uuid_with_permission_check("nonexistent-id", 1)


class TestCollectionRepositoryListForUser:
    """Tests for listing collections."""

    @pytest.mark.asyncio()
    async def test_list_for_user_includes_public_collections(
        self, db_session, test_user_db, other_user_db, collection_factory
    ):
        """Test list_for_user() includes public collections when include_public=True."""
        user = test_user_db
        other_user = other_user_db

        # Create private collection for user
        user_collection = await collection_factory(owner_id=user.id, is_public=False)

        # Create public collection for other user
        public_collection = await collection_factory(owner_id=other_user.id, is_public=True)

        repo = CollectionRepository(db_session)

        collections, total = await repo.list_for_user(user.id, include_public=True)

        # Should include both user's collection and public collection
        collection_ids = [c.id for c in collections]
        assert user_collection.id in collection_ids
        assert public_collection.id in collection_ids

    @pytest.mark.asyncio()
    async def test_list_for_user_excludes_public_when_disabled(
        self, db_session, test_user_db, other_user_db, collection_factory
    ):
        """Test list_for_user() excludes public collections when include_public=False."""
        user = test_user_db
        other_user = other_user_db

        # Create private collection for user
        user_collection = await collection_factory(owner_id=user.id, is_public=False)

        # Create public collection for other user
        public_collection = await collection_factory(owner_id=other_user.id, is_public=True)

        repo = CollectionRepository(db_session)

        collections, total = await repo.list_for_user(user.id, include_public=False)

        # Should only include user's collection
        collection_ids = [c.id for c in collections]
        assert user_collection.id in collection_ids
        assert public_collection.id not in collection_ids


class TestCollectionRepositorySyncManagement:
    """Tests for sync management methods."""

    @pytest.mark.asyncio()
    async def test_pause_sync_raises_error_for_non_continuous_mode(self, db_session, test_user_db, collection_factory):
        """Test pause_sync() raises ValidationError when sync_mode != 'continuous'."""
        user = test_user_db

        # Create one_time sync collection
        collection = await collection_factory(owner_id=user.id, sync_mode="one_time")

        repo = CollectionRepository(db_session)

        with pytest.raises(ValidationError, match="Can only pause continuous sync collections"):
            await repo.pause_sync(collection.id)

    @pytest.mark.asyncio()
    async def test_pause_sync_sets_paused_at_timestamp(self, db_session, test_user_db, collection_factory):
        """Test pause_sync() sets sync_paused_at and clears sync_next_run_at."""
        user = test_user_db

        # Create continuous sync collection
        collection = await collection_factory(
            owner_id=user.id,
            sync_mode="continuous",
            sync_next_run_at=datetime.now(UTC) + timedelta(hours=1),
        )

        repo = CollectionRepository(db_session)

        result = await repo.pause_sync(collection.id)

        assert result.sync_paused_at is not None
        assert result.sync_next_run_at is None

    @pytest.mark.asyncio()
    async def test_resume_sync_raises_error_for_non_continuous_mode(self, db_session, test_user_db, collection_factory):
        """Test resume_sync() raises ValidationError when sync_mode != 'continuous'."""
        user = test_user_db

        # Create one_time sync collection
        collection = await collection_factory(owner_id=user.id, sync_mode="one_time")

        repo = CollectionRepository(db_session)

        with pytest.raises(ValidationError, match="Can only resume continuous sync collections"):
            await repo.resume_sync(collection.id)

    @pytest.mark.asyncio()
    async def test_resume_sync_is_noop_when_not_paused(self, db_session, test_user_db, collection_factory):
        """Test resume_sync() returns collection unchanged when not paused."""
        user = test_user_db

        # Create continuous sync collection that is NOT paused
        collection = await collection_factory(
            owner_id=user.id,
            sync_mode="continuous",
            sync_paused_at=None,
        )

        repo = CollectionRepository(db_session)

        result = await repo.resume_sync(collection.id)

        # Should return unchanged
        assert result.id == collection.id
        assert result.sync_paused_at is None

    @pytest.mark.asyncio()
    async def test_resume_sync_schedules_immediate_run(self, db_session, test_user_db, collection_factory):
        """Test resume_sync() sets sync_next_run_at to now()."""
        user = test_user_db

        # Create paused continuous sync collection
        collection = await collection_factory(
            owner_id=user.id,
            sync_mode="continuous",
            sync_paused_at=datetime.now(UTC) - timedelta(hours=1),
            sync_next_run_at=None,
        )

        repo = CollectionRepository(db_session)

        before_resume = datetime.now(UTC)
        result = await repo.resume_sync(collection.id)
        after_resume = datetime.now(UTC)

        assert result.sync_paused_at is None
        assert result.sync_next_run_at is not None
        assert before_resume <= result.sync_next_run_at <= after_resume


class TestCollectionRepositoryCreate:
    """Tests for collection creation validation."""

    @pytest.mark.asyncio()
    async def test_create_validates_chunk_size_positive(self, db_session, test_user_db):
        """Test create() raises ValidationError for non-positive chunk_size."""
        user = test_user_db
        repo = CollectionRepository(db_session)

        with pytest.raises(ValidationError, match="Chunk size must be positive"):
            await repo.create(
                name="test-collection",
                owner_id=user.id,
                chunk_size=0,
            )

    @pytest.mark.asyncio()
    async def test_create_validates_chunk_overlap_non_negative(self, db_session, test_user_db):
        """Test create() raises ValidationError for negative chunk_overlap."""
        user = test_user_db
        repo = CollectionRepository(db_session)

        with pytest.raises(ValidationError, match="Chunk overlap cannot be negative"):
            await repo.create(
                name="test-collection",
                owner_id=user.id,
                chunk_overlap=-1,
            )

    @pytest.mark.asyncio()
    async def test_create_validates_chunk_overlap_less_than_size(self, db_session, test_user_db):
        """Test create() raises ValidationError when overlap >= chunk_size."""
        user = test_user_db
        repo = CollectionRepository(db_session)

        with pytest.raises(ValidationError, match="Chunk overlap must be less than chunk size"):
            await repo.create(
                name="test-collection",
                owner_id=user.id,
                chunk_size=100,
                chunk_overlap=100,  # Equal to chunk_size
            )


class TestCollectionRepositoryRename:
    """Tests for collection renaming."""

    @pytest.mark.asyncio()
    async def test_rename_raises_error_for_empty_name(self, db_session, test_user_db, collection_factory):
        """Test rename() raises ValidationError for empty name."""
        user = test_user_db
        collection = await collection_factory(owner_id=user.id)

        repo = CollectionRepository(db_session)

        with pytest.raises(ValidationError, match="Collection name cannot be empty"):
            await repo.rename(collection.id, "", user.id)

    @pytest.mark.asyncio()
    async def test_rename_raises_error_for_duplicate_name(self, db_session, test_user_db, collection_factory):
        """Test rename() raises EntityAlreadyExistsError for existing name."""
        user = test_user_db
        await collection_factory(owner_id=user.id, name="collection-1")
        collection2 = await collection_factory(owner_id=user.id, name="collection-2")

        repo = CollectionRepository(db_session)

        with pytest.raises(EntityAlreadyExistsError):
            await repo.rename(collection2.id, "collection-1", user.id)


class TestCollectionRepositoryDelete:
    """Tests for collection deletion."""

    @pytest.mark.asyncio()
    async def test_delete_raises_access_denied_for_non_owner(
        self, db_session, test_user_db, other_user_db, collection_factory
    ):
        """Test delete() raises AccessDeniedError for non-owner."""
        owner = test_user_db
        other_user = other_user_db
        collection = await collection_factory(owner_id=owner.id)

        repo = CollectionRepository(db_session)

        with pytest.raises(AccessDeniedError):
            await repo.delete(collection.id, other_user.id)

    @pytest.mark.asyncio()
    async def test_delete_raises_not_found_for_missing_collection(self, db_session, test_user_db):
        """Test delete() raises EntityNotFoundError for missing collection."""
        user = test_user_db
        repo = CollectionRepository(db_session)

        with pytest.raises(EntityNotFoundError, match="collection"):
            await repo.delete("nonexistent-id", user.id)
