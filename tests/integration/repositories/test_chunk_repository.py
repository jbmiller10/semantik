"""Integration tests for ChunkRepository with real partition-aware operations."""

from __future__ import annotations

import pytest
from shared.database.models import Chunk
from shared.database.repositories.chunk_repository import ChunkRepository
from sqlalchemy import select


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestChunkRepositoryIntegration:
    """Exercise chunk repository flows against the actual database."""

    @pytest.fixture()
    def repository(self, db_session):
        return ChunkRepository(db_session)

    async def test_create_chunk_persists_record(
        self, repository, db_session, collection_factory, document_factory, test_user_db
    ):
        """Creating a chunk should populate partitioned table."""
        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)

        chunk = await repository.create_chunk(
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": 0,
                "content": "hello world",
                "metadata": {"origin": "integration"},
            }
        )
        await db_session.commit()

        result = await db_session.execute(select(Chunk).where(Chunk.id == chunk.id))
        persisted = result.scalar_one()
        assert persisted.collection_id == collection.id
        assert persisted.document_id == document.id
        assert persisted.chunk_index == 0
        assert persisted.metadata["origin"] == "integration"

    async def test_create_chunks_bulk_inserts_all(
        self, repository, db_session, collection_factory, document_factory, test_user_db
    ):
        """Bulk insert should create all chunks grouped per collection."""
        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)
        chunk_payload = [
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": idx,
                "content": f"bulk {idx}",
            }
            for idx in range(3)
        ]

        created_count = await repository.create_chunks_bulk(chunk_payload)
        await db_session.commit()

        assert created_count == 3
        result = await db_session.execute(select(Chunk).where(Chunk.collection_id == collection.id))
        persisted = result.scalars().all()
        assert len(persisted) == 3
        assert {chunk.chunk_index for chunk in persisted} == {0, 1, 2}

    async def test_get_chunk_by_id_fetches_partition(
        self, repository, db_session, collection_factory, document_factory, test_user_db
    ):
        """Fetching by id should return the created chunk when using correct collection id."""
        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)
        chunk = await repository.create_chunk(
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": 5,
                "content": "indexed chunk",
            }
        )
        await db_session.commit()

        fetched = await repository.get_chunk_by_id(chunk.id, collection.id)
        assert fetched is not None
        assert fetched.chunk_index == 5

    async def test_get_chunk_by_embedding_vector_id_fetches_chunk(
        self, repository, db_session, collection_factory, document_factory, test_user_db
    ):
        """Fetching by embedding_vector_id should return the created chunk when mapping exists."""
        from uuid import uuid4

        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)
        chunk = await repository.create_chunk(
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": 7,
                "content": "vector-mapped chunk",
            }
        )

        # Assign a synthetic embedding_vector_id and persist it.
        vector_id = str(uuid4())
        chunk.embedding_vector_id = vector_id
        await db_session.commit()

        fetched = await repository.get_chunk_by_embedding_vector_id(vector_id, collection.id)
        assert fetched is not None
        assert fetched.id == chunk.id
        assert fetched.embedding_vector_id == vector_id

    async def test_get_chunks_by_document_orders_results(
        self, repository, db_session, collection_factory, document_factory, test_user_db
    ):
        """Retrieving by document should return chunks in index order."""
        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)
        for idx in range(3):
            await repository.create_chunk(
                {
                    "collection_id": collection.id,
                    "document_id": document.id,
                    "chunk_index": idx,
                    "content": f"chunk {idx}",
                }
            )
        await db_session.commit()

        chunks = await repository.get_chunks_by_document(document.id, collection.id)
        assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]

    async def test_delete_chunks_by_document_removes_rows(
        self, repository, db_session, collection_factory, document_factory, test_user_db
    ):
        """Deleting by document id should remove rows from the partition."""
        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)
        await repository.create_chunks_bulk(
            [
                {
                    "collection_id": collection.id,
                    "document_id": document.id,
                    "chunk_index": idx,
                    "content": f"delete {idx}",
                }
                for idx in range(2)
            ]
        )
        await db_session.commit()

        deleted = await repository.delete_chunks_by_document(document.id, collection.id)
        await db_session.commit()
        assert deleted == 2

        count = await repository.count_chunks_by_document(document.id, collection.id)
        assert count == 0

    async def test_chunk_exists_checks_presence(
        self, repository, db_session, collection_factory, document_factory, test_user_db
    ):
        """chunk_exists should reflect whether a chunk is stored."""
        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)
        await repository.create_chunk(
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": 42,
                "content": "exists",
            }
        )
        await db_session.commit()

        assert await repository.chunk_exists(document.id, collection.id, 42) is True
        assert await repository.chunk_exists(document.id, collection.id, 99) is False
