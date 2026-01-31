"""Integration tests for DocumentRepository count_failed methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from shared.database.models import DocumentStatus
from shared.database.repositories.document_repository import DocumentRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestCountFailedIntegration:
    """Integration tests for count_failed methods with real database."""

    @pytest.fixture()
    def repository(self, db_session: AsyncSession) -> DocumentRepository:
        """Construct the repository with a real async session."""
        return DocumentRepository(db_session)

    async def test_count_failed_by_collection(
        self,
        repository: DocumentRepository,
        db_session: AsyncSession,
        collection_factory,
        document_factory,
        test_user_db,
    ) -> None:
        """Test counting failed documents with real database."""
        collection = await collection_factory(owner_id=test_user_db.id)

        # Create mixed status documents
        await document_factory(collection_id=collection.id, status=DocumentStatus.FAILED)
        await document_factory(collection_id=collection.id, status=DocumentStatus.FAILED)
        await document_factory(collection_id=collection.id, status=DocumentStatus.COMPLETED)
        await db_session.commit()

        count = await repository.count_failed_by_collection(collection.id)
        assert count == 2

    async def test_count_failed_by_collection_returns_zero_when_no_failed(
        self,
        repository: DocumentRepository,
        db_session: AsyncSession,
        collection_factory,
        document_factory,
        test_user_db,
    ) -> None:
        """Test returns 0 when no documents have FAILED status."""
        collection = await collection_factory(owner_id=test_user_db.id)

        # Create only non-failed documents
        await document_factory(collection_id=collection.id, status=DocumentStatus.COMPLETED)
        await document_factory(collection_id=collection.id, status=DocumentStatus.PENDING)
        await db_session.commit()

        count = await repository.count_failed_by_collection(collection.id)
        assert count == 0

    async def test_count_failed_isolation(
        self,
        repository: DocumentRepository,
        db_session: AsyncSession,
        collection_factory,
        document_factory,
        test_user_db,
    ) -> None:
        """Test counts are isolated per collection."""
        coll1 = await collection_factory(owner_id=test_user_db.id)
        coll2 = await collection_factory(owner_id=test_user_db.id)

        await document_factory(collection_id=coll1.id, status=DocumentStatus.FAILED)
        await document_factory(collection_id=coll2.id, status=DocumentStatus.FAILED)
        await document_factory(collection_id=coll2.id, status=DocumentStatus.FAILED)
        await db_session.commit()

        assert await repository.count_failed_by_collection(coll1.id) == 1
        assert await repository.count_failed_by_collection(coll2.id) == 2

    async def test_count_failed_by_collections_batch(
        self,
        repository: DocumentRepository,
        db_session: AsyncSession,
        collection_factory,
        document_factory,
        test_user_db,
    ) -> None:
        """Test batch counting returns correct dict."""
        coll1 = await collection_factory(owner_id=test_user_db.id)
        coll2 = await collection_factory(owner_id=test_user_db.id)

        await document_factory(collection_id=coll1.id, status=DocumentStatus.FAILED)
        await document_factory(collection_id=coll2.id, status=DocumentStatus.FAILED)
        await document_factory(collection_id=coll2.id, status=DocumentStatus.FAILED)
        await db_session.commit()

        counts = await repository.count_failed_by_collections([coll1.id, coll2.id])

        assert counts.get(coll1.id) == 1
        assert counts.get(coll2.id) == 2

    async def test_count_failed_by_collections_empty_list(
        self,
        repository: DocumentRepository,
    ) -> None:
        """Test empty input returns empty dict."""
        counts = await repository.count_failed_by_collections([])
        assert counts == {}

    async def test_count_failed_by_collections_missing_collection(
        self,
        repository: DocumentRepository,
        db_session: AsyncSession,
        collection_factory,
        document_factory,
        test_user_db,
    ) -> None:
        """Test batch counting handles collections with no failed docs."""
        coll1 = await collection_factory(owner_id=test_user_db.id)
        coll2 = await collection_factory(owner_id=test_user_db.id)

        # Only coll1 has failed documents
        await document_factory(collection_id=coll1.id, status=DocumentStatus.FAILED)
        await document_factory(collection_id=coll2.id, status=DocumentStatus.COMPLETED)
        await db_session.commit()

        counts = await repository.count_failed_by_collections([coll1.id, coll2.id])

        # coll1 has 1 failed, coll2 may not be in dict or have 0
        assert counts.get(coll1.id) == 1
        # coll2 might not be in the dict since it has no failed docs
        assert counts.get(coll2.id, 0) == 0
