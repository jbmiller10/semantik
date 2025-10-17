"""Integration coverage for ChunkRepository."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import select

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from packages.shared.database.models import Chunk
from packages.shared.database.repositories.chunk_repository import ChunkRepository

pytestmark = [pytest.mark.asyncio(), pytest.mark.usefixtures("_db_isolation")]


class TestChunkRepositoryIntegration:
    """Exercise chunk operations against the real partitioned table."""

    @pytest.fixture()
    def repository(self, db_session: AsyncSession) -> ChunkRepository:
        return ChunkRepository(db_session)

    async def _fetch_chunks(self, db_session: AsyncSession, collection_id: str) -> list[Chunk]:
        result = await db_session.execute(select(Chunk).where(Chunk.collection_id == collection_id))
        return list(result.scalars())

    async def test_create_chunk_and_fetch_by_id(
        self,
        repository: ChunkRepository,
        db_session: AsyncSession,
        collection_factory,
        document_factory,
        test_user_db,
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)

        chunk = await repository.create_chunk(
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": 0,
                "content": "Hello world",
                "metadata": {"origin": "integration"},
            }
        )

        assert chunk.id is not None
        assert chunk.collection_id == collection.id

        fetched = await repository.get_chunk_by_id(chunk.id, collection.id)
        assert fetched is not None
        assert fetched.id == chunk.id
        assert fetched.document_id == document.id

    async def test_create_chunks_bulk_and_retrieve(
        self,
        repository: ChunkRepository,
        db_session: AsyncSession,
        collection_factory,
        document_factory,
        test_user_db,
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)

        chunk_payloads = [
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": idx,
                "content": f"Chunk {idx}",
                "metadata": {"order": idx},
            }
            for idx in range(3)
        ]

        created_count = await repository.create_chunks_bulk(chunk_payloads)
        assert created_count == 3

        chunks = await repository.get_chunks_by_document(document.id, collection.id)
        assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]

        # Fetch by collection with pagination
        subset = await repository.get_chunks_by_collection(collection.id, limit=2)
        assert len(subset) == 2

    async def test_update_embeddings_and_count(
        self,
        repository: ChunkRepository,
        collection_factory,
        document_factory,
        test_user_db,
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)

        first = await repository.create_chunk(
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": 0,
                "content": "embedding chunk",
                "metadata": {},
            }
        )
        second = await repository.create_chunk(
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": 1,
                "content": "embedding chunk 2",
                "metadata": {},
            }
        )

        updated = await repository.update_chunk_embeddings(
            [
                {"id": first.id, "collection_id": collection.id, "embedding_vector_id": str(uuid4())},
                {"id": second.id, "collection_id": collection.id, "embedding_vector_id": str(uuid4())},
            ]
        )
        assert updated == 2

        stats = await repository.get_chunk_statistics(collection.id)
        assert stats["collection_id"] == collection.id
        assert stats["chunk_count"] >= 2

        count = await repository.count_chunks_by_document(document.id, collection.id)
        assert count == 2

    async def test_delete_chunks_by_document_and_collection(
        self,
        repository: ChunkRepository,
        db_session: AsyncSession,
        collection_factory,
        document_factory,
        test_user_db,
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)
        other_collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)
        other_doc = await document_factory(collection_id=other_collection.id)

        for idx in range(2):
            await repository.create_chunk(
                {
                    "collection_id": collection.id,
                    "document_id": document.id,
                    "chunk_index": idx,
                    "content": f"doc chunk {idx}",
                    "metadata": {},
                }
            )
        await repository.create_chunk(
            {
                "collection_id": other_collection.id,
                "document_id": other_doc.id,
                "chunk_index": 0,
                "content": "other",
                "metadata": {},
            }
        )

        deleted = await repository.delete_chunks_by_document(document.id, collection.id)
        assert deleted == 2

        remaining = await self._fetch_chunks(db_session, collection.id)
        assert remaining == []

        deleted_collection = await repository.delete_chunks_by_collection(other_collection.id)
        assert deleted_collection == 1

    async def test_get_chunks_without_embeddings_and_existence_checks(
        self,
        repository: ChunkRepository,
        collection_factory,
        document_factory,
        test_user_db,
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)

        chunk = await repository.create_chunk(
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": 42,
                "content": "no embedding",
                "metadata": {},
            }
        )

        pending = await repository.get_chunks_without_embeddings(collection.id)
        assert any(item.id == chunk.id for item in pending)

        assert await repository.chunk_exists(document.id, collection.id, 42) is True
        assert await repository.chunk_exists(document.id, collection.id, 100) is False

    async def test_get_chunks_batch_returns_multiple_documents(
        self,
        repository: ChunkRepository,
        collection_factory,
        document_factory,
        test_user_db,
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)
        document_one = await document_factory(collection_id=collection.id)
        document_two = await document_factory(collection_id=collection.id)

        for doc in (document_one, document_two):
            for idx in range(2):
                await repository.create_chunk(
                    {
                        "collection_id": collection.id,
                        "document_id": doc.id,
                        "chunk_index": idx,
                        "content": f"chunk {doc.id}-{idx}",
                        "metadata": {},
                    }
                )

        batch = await repository.get_chunks_batch(collection.id, [document_one.id, document_two.id])
        assert len(batch) == 4

    async def test_get_chunks_by_collection_filter_created_after(
        self,
        repository: ChunkRepository,
        collection_factory,
        document_factory,
        test_user_db,
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)

        await repository.create_chunk(
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": 0,
                "content": "earlier",
                "metadata": {},
            }
        )

        cutoff = datetime.now(UTC) + timedelta(milliseconds=1)

        await repository.create_chunk(
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": 1,
                "content": "later",
                "metadata": {},
            }
        )

        recent = await repository.get_chunks_by_collection(collection.id, created_after=cutoff)
        assert all(chunk.chunk_index >= 1 for chunk in recent)
