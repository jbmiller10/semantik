"""Integration tests for partition utility helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from packages.shared.database.models import Chunk
from packages.shared.database.partition_utils import ChunkPartitionHelper, PartitionAwareMixin
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = [pytest.mark.asyncio(), pytest.mark.usefixtures("_db_isolation")]


class TestPartitionUtilitiesIntegration:
    """Validate partition helpers against the real database."""

    @pytest.fixture()
    def mixin(self) -> PartitionAwareMixin:
        return PartitionAwareMixin()

    async def _load_chunks(self, db_session: AsyncSession, collection_id: str) -> list[Chunk]:
        result = await db_session.execute(select(Chunk).where(Chunk.collection_id == collection_id))
        return list(result.scalars())

    async def test_bulk_insert_partitioned_and_delete_filter(
        self,
        mixin: PartitionAwareMixin,
        db_session: AsyncSession,
        collection_factory,
        document_factory,
    ) -> None:
        collection = await collection_factory()
        document = await document_factory(collection_id=collection.id)

        other_collection = await collection_factory()
        other_document = await document_factory(collection_id=other_collection.id)

        payload = [
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": idx,
                "content": f"chunk-{idx}",
                "metadata": {"partition": "primary"},
            }
            for idx in range(2)
        ] + [
            {
                "collection_id": other_collection.id,
                "document_id": other_document.id,
                "chunk_index": 0,
                "content": "other",
                "metadata": {"partition": "secondary"},
            }
        ]

        await mixin.bulk_insert_partitioned(db_session, Chunk, payload, partition_key_field="collection_id")

        primary_chunks = await self._load_chunks(db_session, collection.id)
        assert len(primary_chunks) == 2

        deleted = await mixin.delete_by_partition_filter(
            db_session,
            Chunk,
            Chunk.collection_id,
            collection.id,
            [Chunk.document_id == document.id],
        )
        assert deleted == 2

        remaining_primary = await self._load_chunks(db_session, collection.id)
        assert remaining_primary == []

    async def test_get_partition_statistics_with_data(
        self,
        mixin: PartitionAwareMixin,
        db_session: AsyncSession,
        collection_factory,
        document_factory,
    ) -> None:
        collection = await collection_factory()
        document = await document_factory(collection_id=collection.id)

        payload = [
            {
                "collection_id": collection.id,
                "document_id": document.id,
                "chunk_index": idx,
                "content": f"chunk-{idx}",
                "metadata": {},
                "created_at": datetime.now(UTC) - timedelta(minutes=idx),
            }
            for idx in range(3)
        ]
        await mixin.bulk_insert_partitioned(db_session, Chunk, payload)

        stats = await ChunkPartitionHelper.get_partition_statistics(db_session, collection.id)
        assert stats["collection_id"] == collection.id
        assert stats["chunk_count"] == 3
        assert stats["avg_content_length"] > 0
        assert stats["newest_chunk"] >= stats["oldest_chunk"]

    async def test_partition_helper_handles_empty_dataset(self, db_session: AsyncSession) -> None:
        empty_collection_id = str(uuid4())
        stats = await ChunkPartitionHelper.get_partition_statistics(db_session, empty_collection_id)
        assert stats["chunk_count"] == 0
        assert stats["total_content_length"] == 0
