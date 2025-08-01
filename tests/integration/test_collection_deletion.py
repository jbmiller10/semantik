"""Integration tests for collection deletion functionality.

This test suite ensures that collection deletion works correctly end-to-end,
including proper cascade deletion and transaction handling.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import (
    Collection,
    Document,
    Operation,
    CollectionStatus,
    OperationType,
)
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.webui.services.collection_service import CollectionService
from packages.webui.services.factory import create_collection_service


@pytest.mark.asyncio
class TestCollectionDeletion:
    """Test suite for collection deletion functionality."""

    async def test_collection_deletion_removes_from_database(
        self, db_session: AsyncSession, test_user_db, collection_factory
    ):
        """Test that collection deletion actually removes the collection from the database."""
        # Arrange
        collection = await collection_factory(owner_id=test_user_db.id)
        collection_id = collection.id
        collection_uuid = collection.id

        repo = CollectionRepository(db_session)

        # Act
        await repo.delete(collection_uuid, test_user_db.id)
        await db_session.commit()

        # Assert - collection should not exist
        result = await db_session.execute(select(Collection).where(Collection.id == collection_id))
        assert result.scalar_one_or_none() is None

    async def test_collection_deletion_cascades_to_documents(
        self, db_session: AsyncSession, test_user_db, collection_factory, document_factory
    ):
        """Test that deleting a collection cascades to delete all its documents."""
        # Arrange
        collection = await collection_factory(owner_id=test_user_db.id)
        collection_id = collection.id

        # Create multiple documents
        doc1 = await document_factory(collection_id=collection_id)
        doc2 = await document_factory(collection_id=collection_id)
        doc3 = await document_factory(collection_id=collection_id)

        repo = CollectionRepository(db_session)

        # Act
        await repo.delete(collection.id, test_user_db.id)
        await db_session.commit()

        # Assert - all documents should be deleted
        result = await db_session.execute(select(Document).where(Document.collection_id == collection_id))
        documents = result.scalars().all()
        assert len(documents) == 0

    async def test_collection_deletion_cascades_to_operations(
        self, db_session: AsyncSession, test_user_db, collection_factory, operation_factory
    ):
        """Test that deleting a collection cascades to delete all its operations."""
        # Arrange
        collection = await collection_factory(owner_id=test_user_db.id)
        collection_id = collection.id

        # Create multiple operations
        op1 = await operation_factory(collection_id=collection_id, user_id=test_user_db.id, type=OperationType.INDEX)
        op2 = await operation_factory(collection_id=collection_id, user_id=test_user_db.id, type=OperationType.REINDEX)

        repo = CollectionRepository(db_session)

        # Act
        await repo.delete(collection.id, test_user_db.id)
        await db_session.commit()

        # Assert - all operations should be deleted
        result = await db_session.execute(select(Operation).where(Operation.collection_id == collection_id))
        operations = result.scalars().all()
        assert len(operations) == 0

    async def test_collection_deletion_requires_owner_permission(
        self, db_session: AsyncSession, test_user_db, other_user_db, collection_factory
    ):
        """Test that only the owner can delete a collection."""
        # Arrange
        collection = await collection_factory(owner_id=test_user_db.id)
        repo = CollectionRepository(db_session)

        # Act & Assert
        from packages.shared.database.exceptions import AccessDeniedError

        with pytest.raises(AccessDeniedError):
            await repo.delete(collection.id, other_user_db.id)

    async def test_collection_deletion_via_service_commits_transaction(
        self, db_session: AsyncSession, test_user_db, collection_factory, mock_qdrant_deletion
    ):
        """Test that CollectionService.delete_collection properly commits the transaction."""
        # Arrange
        collection = await collection_factory(owner_id=test_user_db.id)
        collection_id = collection.id
        collection_uuid = collection.id

        service = create_collection_service(db_session)

        # Act
        await service.delete_collection(collection_uuid, test_user_db.id)

        # Note: Don't need to commit here because service should do it

        # Assert - collection should be gone
        result = await db_session.execute(select(Collection).where(Collection.id == collection_id))
        assert result.scalar_one_or_none() is None

    async def test_collection_deletion_handles_missing_qdrant_collection(
        self, db_session: AsyncSession, test_user_db, collection_factory, mock_qdrant_deletion
    ):
        """Test that deletion succeeds even if Qdrant collection doesn't exist."""
        # Arrange
        collection = await collection_factory(owner_id=test_user_db.id, vector_store_name="col_test_123")

        # Mock Qdrant to throw exception
        mock_qdrant_deletion.delete_collection.side_effect = Exception("Collection not found")

        service = create_collection_service(db_session)

        # Act - should not raise exception
        await service.delete_collection(collection.id, test_user_db.id)

        # Assert - PostgreSQL collection should still be deleted
        result = await db_session.execute(select(Collection).where(Collection.id == collection.id))
        assert result.scalar_one_or_none() is None

    async def test_collection_deletion_fails_with_active_operations(
        self, db_session: AsyncSession, test_user_db, collection_factory, operation_factory
    ):
        """Test that collection cannot be deleted while operations are active."""
        # Arrange
        collection = await collection_factory(owner_id=test_user_db.id)

        # Create an active operation
        operation = await operation_factory(
            collection_id=collection.id, user_id=test_user_db.id, type=OperationType.INDEX, status="processing"
        )

        service = create_collection_service(db_session)

        # Act & Assert
        from packages.shared.database.exceptions import InvalidStateError

        with pytest.raises(InvalidStateError) as exc_info:
            await service.delete_collection(collection.id, test_user_db.id)

        assert "operations are in progress" in str(exc_info.value)

    async def test_async_delete_pattern_in_repository(self, db_session: AsyncSession, test_user_db, collection_factory):
        """Test that the repository uses correct async SQLAlchemy delete pattern."""
        # Arrange
        collection = await collection_factory(owner_id=test_user_db.id)
        collection_id = collection.id

        repo = CollectionRepository(db_session)

        # Spy on the session to ensure correct pattern is used
        original_execute = db_session.execute
        execute_calls = []

        async def spy_execute(statement):
            execute_calls.append(statement)
            return await original_execute(statement)

        db_session.execute = spy_execute

        # Act
        await repo.delete(collection.id, test_user_db.id)
        await db_session.commit()

        # Assert - should use delete() construct, not session.delete()
        assert len(execute_calls) > 0

        # Check that one of the calls was a DELETE statement
        delete_found = False
        for call in execute_calls:
            if hasattr(call, "is_delete") and call.is_delete:
                delete_found = True
                break

        assert delete_found, "Repository should use delete() construct for async deletion"

        # Restore original method
        db_session.execute = original_execute


@pytest.mark.asyncio
class TestCollectionDeletionEdgeCases:
    """Test edge cases and error scenarios for collection deletion."""

    async def test_delete_nonexistent_collection(self, db_session: AsyncSession, test_user_db):
        """Test deleting a collection that doesn't exist."""
        repo = CollectionRepository(db_session)

        from packages.shared.database.exceptions import EntityNotFoundError

        with pytest.raises(EntityNotFoundError):
            await repo.delete("nonexistent-uuid", test_user_db.id)

    async def test_concurrent_deletion_attempts(self, db_session: AsyncSession, test_user_db, collection_factory):
        """Test that concurrent deletion attempts are handled gracefully."""
        # This test would require more sophisticated setup with multiple sessions
        # For now, we'll just ensure the second delete fails appropriately
        collection = await collection_factory(owner_id=test_user_db.id)
        repo = CollectionRepository(db_session)

        # First deletion
        await repo.delete(collection.id, test_user_db.id)
        await db_session.commit()

        # Second deletion should fail
        from packages.shared.database.exceptions import EntityNotFoundError

        with pytest.raises(EntityNotFoundError):
            await repo.delete(collection.id, test_user_db.id)
