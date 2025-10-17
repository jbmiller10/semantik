#!/usr/bin/env python3
"""Integration tests verifying document chunk count updates."""

from __future__ import annotations

import pytest

from packages.shared.database.factory import create_collection_repository, create_document_repository
from packages.shared.database.models import DocumentStatus
from packages.webui.services.chunking_service import ChunkingService

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def test_chunk_count_updates_persist_to_database(
    db_session,
    collection_factory,
    document_factory,
) -> None:
    """Chunk counts produced by ChunkingService should persist via the repository layer."""
    collection = await collection_factory()
    document = await document_factory(collection_id=collection.id, chunk_count=0, status=DocumentStatus.PENDING)

    service = ChunkingService(
        db_session=db_session,
        collection_repo=create_collection_repository(db_session),
        document_repo=create_document_repository(db_session),
        redis_client=None,
    )

    chunking_result = await service.execute_ingestion_chunking(
        text=" ".join(f"Sentence {i} for chunk persistence." for i in range(25)),
        document_id=document.id,
        collection={
            "id": collection.id,
            "name": collection.name,
            "chunking_strategy": collection.chunking_strategy,
            "chunking_config": collection.chunking_config,
            "chunk_size": collection.chunk_size,
            "chunk_overlap": collection.chunk_overlap,
        },
    )

    chunk_count = chunking_result["stats"]["chunk_count"]
    repo = create_document_repository(db_session)
    await repo.update_status(document.id, DocumentStatus.COMPLETED, chunk_count=chunk_count)
    await db_session.commit()

    refreshed = await repo.get_by_id(document.id)
    assert refreshed is not None
    assert refreshed.chunk_count == chunk_count
    assert refreshed.status == DocumentStatus.COMPLETED.value
