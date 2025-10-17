"""Integration coverage for the v2 chunking API."""

from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient

from packages.shared.database.models import Chunk, OperationStatus
from packages.webui.services.chunking_service import ChunkingService
from packages.webui.services.dtos.chunking_dtos import ServiceStrategyInfo


@pytest.mark.asyncio()
async def test_list_chunking_strategies_integration(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strategies endpoint should return the built-in strategies."""

    async def fake_get_available_strategies_for_api(self):
        return [
            ServiceStrategyInfo(
                id="fixed_size",
                name="Fixed Size",
                description="Splits text into fixed-size windows.",
                best_for=["general text"],
                default_config={"strategy": "fixed_size", "chunk_size": 512, "chunk_overlap": 32},
                performance_characteristics={"speed": "fast"},
            ),
            ServiceStrategyInfo(
                id="recursive",
                name="Recursive",
                description="Structure-aware recursive strategy.",
                best_for=["structured documents"],
                default_config={"strategy": "recursive", "chunk_size": 1000, "chunk_overlap": 100},
                performance_characteristics={"speed": "medium"},
            ),
        ]

    monkeypatch.setattr(ChunkingService, "get_available_strategies_for_api", fake_get_available_strategies_for_api)

    response = await api_client.get("/api/v2/chunking/strategies", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    payload = response.json()
    strategy_ids = {entry["id"] for entry in payload}
    assert {"fixed_size", "recursive"}.issubset(strategy_ids)


@pytest.mark.asyncio()
async def test_get_collection_chunks_returns_seeded_data(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    document_factory,
    db_session,
) -> None:
    """Persisted chunk rows should be returned via the API."""

    collection = await collection_factory(owner_id=test_user_db.id, chunking_strategy="recursive")
    document = await document_factory(collection_id=collection.id, chunk_count=1)

    content = "Integration chunk content"
    chunk = Chunk(
        collection_id=collection.id,
        partition_key=0,
        document_id=document.id,
        chunk_index=0,
        content=content,
        token_count=42,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db_session.add(chunk)
    await db_session.commit()

    response = await api_client.get(
        f"/api/v2/chunking/collections/{collection.id}/chunks",
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 1
    assert payload["page"] == 1
    assert payload["has_next"] is False

    returned = payload["chunks"][0]
    assert returned["document_id"] == document.id
    assert returned["token_count"] == 42
    assert returned["char_count"] == len(content)


@pytest.mark.asyncio()
async def test_get_chunking_stats_reflects_recent_operation(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    document_factory,
    operation_factory,
    db_session,
) -> None:
    """Stats endpoint should summarise chunk and operation data."""

    collection = await collection_factory(owner_id=test_user_db.id, chunking_strategy="recursive")
    document = await document_factory(collection_id=collection.id, chunk_count=2)

    now = datetime.now(UTC)
    contents = ["alpha" * 5, "beta" * 5]
    for index, text in enumerate(contents):
        db_session.add(
            Chunk(
                collection_id=collection.id,
                partition_key=0,
                document_id=document.id,
                chunk_index=index,
                content=text,
                token_count=len(text) // 2,
                created_at=now,
                updated_at=now,
            )
        )

    document.chunk_count = len(contents)
    document.chunks_count = len(contents)
    document.chunking_completed_at = now
    collection.chunking_strategy = "recursive"
    collection.updated_at = now

    operation = await operation_factory(
        collection_id=collection.id,
        user_id=test_user_db.id,
        status=OperationStatus.COMPLETED,
    )
    operation.started_at = now - timedelta(seconds=12)
    operation.completed_at = now
    operation.config = {"strategy": "recursive"}

    await db_session.commit()

    response = await api_client.get(
        f"/api/v2/chunking/collections/{collection.id}/chunking-stats",
        params={"collection_uuid": collection.id},
        headers=api_auth_headers,
    )

    assert response.status_code == 200, response.text
    stats = response.json()
    assert stats["total_chunks"] == len(contents)
    assert stats["total_documents"] == 1
    assert stats["strategy_used"] == "recursive"
    assert stats["processing_time_seconds"] >= 12
    assert stats["avg_chunk_size"] >= min(len(text) for text in contents)


@pytest.mark.asyncio()
async def test_get_global_metrics_counts_recent_activity(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
    document_factory,
    operation_factory,
    db_session,
) -> None:
    """Global metrics should include recent chunking activity."""

    collection = await collection_factory(owner_id=test_user_db.id, chunking_strategy="recursive")
    document = await document_factory(collection_id=collection.id, chunk_count=1)

    timestamp = datetime.now(UTC)
    db_session.add(
        Chunk(
            collection_id=collection.id,
            partition_key=0,
            document_id=document.id,
            chunk_index=0,
            content="metrics chunk",
            token_count=12,
            created_at=timestamp,
            updated_at=timestamp,
        )
    )

    document.chunk_count = 1
    document.chunks_count = 1
    document.chunking_completed_at = timestamp
    collection.chunking_strategy = "recursive"
    collection.updated_at = timestamp

    operation = await operation_factory(
        collection_id=collection.id,
        user_id=test_user_db.id,
        status=OperationStatus.COMPLETED,
    )
    operation.started_at = timestamp - timedelta(seconds=5)
    operation.completed_at = timestamp
    await db_session.commit()

    response = await api_client.get("/api/v2/chunking/metrics", headers=api_auth_headers)

    assert response.status_code == 200
    metrics = response.json()
    assert metrics["total_chunks_created"] >= 1
    assert metrics["total_documents_processed"] >= 1
    assert metrics["total_collections_processed"] >= 1
    assert metrics["most_used_strategy"] in {"recursive", "fixed_size"}
