"""Integration tests for partition helpers operating on the chunks table."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import select

from packages.shared.database.models import Chunk
from packages.shared.database.partition_utils import ChunkPartitionHelper, PartitionAwareMixin, PartitionValidation


@pytest.mark.asyncio()
@pytest.mark.usefixtures("_db_isolation")
class TestPartitionUtilitiesIntegration:
    """Ensure partition helpers interact correctly with the real database."""

    async def test_bulk_insert_partitioned_writes_grouped_batches(
        self, db_session, collection_factory, document_factory, test_user_db
    ) -> None:
        mixin = PartitionAwareMixin()
        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)

        created_at = datetime.now(UTC)
        items = [
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": idx,
                "content": f"partitioned {idx}",
                "created_at": created_at,
            }
            for idx in range(4)
        ]

        await mixin.bulk_insert_partitioned(db_session, Chunk, items)
        await db_session.commit()

        result = await db_session.execute(select(Chunk).where(Chunk.collection_id == collection.id))
        persisted = result.scalars().all()
        assert len(persisted) == 4
        assert {chunk.chunk_index for chunk in persisted} == {0, 1, 2, 3}

    async def test_delete_by_partition_filter_removes_rows(
        self, db_session, collection_factory, document_factory, test_user_db
    ) -> None:
        mixin = PartitionAwareMixin()
        collection = await collection_factory(owner_id=test_user_db.id)
        doc_one = await document_factory(collection_id=collection.id)
        doc_two = await document_factory(collection_id=collection.id)

        await mixin.bulk_insert_partitioned(
            db_session,
            Chunk,
            [
                {
                    "collection_id": collection.id,
                    "document_id": doc_one.id,
                    "chunk_index": 0,
                    "content": "retain",
                },
                {
                    "collection_id": collection.id,
                    "document_id": doc_two.id,
                    "chunk_index": 1,
                    "content": "remove",
                },
            ],
        )
        await db_session.commit()

        deleted = await mixin.delete_by_partition_filter(
            db_session,
            Chunk,
            Chunk.collection_id,
            collection.id,
            [Chunk.document_id == doc_two.id],
        )
        await db_session.commit()
        assert deleted == 1

        remaining = await db_session.execute(select(Chunk).where(Chunk.collection_id == collection.id))
        assert len(remaining.scalars().all()) == 1

    async def test_partition_statistics_reports_counts(
        self, db_session, collection_factory, document_factory, test_user_db
    ) -> None:
        collection = await collection_factory(owner_id=test_user_db.id)
        document = await document_factory(collection_id=collection.id)
        mixin = PartitionAwareMixin()

        await mixin.bulk_insert_partitioned(
            db_session,
            Chunk,
            [
                {
                    "collection_id": collection.id,
                    "document_id": document.id,
                    "chunk_index": idx,
                    "content": f"stats {idx}",
                }
                for idx in range(2)
            ],
        )
        await db_session.commit()

        stats = await ChunkPartitionHelper.get_partition_statistics(db_session, collection.id)
        assert stats["collection_id"] == collection.id
        assert stats["chunk_count"] >= 2
        assert stats["total_content_length"] >= 0

    async def test_partition_validation_sanitises_input(self) -> None:
        """PartitionValidation should normalise UUIDs and content strings."""
        invalid_uuid = str(uuid4()).upper()
        assert PartitionValidation.validate_uuid(invalid_uuid) == invalid_uuid.lower()

        chunk = PartitionValidation.validate_chunk_data(
            {
                "collection_id": str(uuid4()),
                "document_id": str(uuid4()),
                "chunk_index": 0,
                "content": "hello",
            }
        )
        assert chunk["chunk_index"] == 0
        assert chunk["collection_id"] == chunk["collection_id"].lower()
