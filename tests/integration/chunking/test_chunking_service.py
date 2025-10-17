#!/usr/bin/env python3
"""Integration tests covering high-value ChunkingService flows."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from packages.shared.database.factory import create_collection_repository, create_document_repository
from packages.webui.services import chunking_metrics
from packages.webui.services.chunking_service import ChunkingService

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture()
def metrics_registry() -> CollectorRegistry:
    """Isolated metrics registry to validate Prometheus updates."""
    registry = CollectorRegistry()
    chunking_metrics.init_metrics(registry)
    yield registry
    chunking_metrics.init_metrics()


async def _create_service(db_session) -> ChunkingService:
    """Construct a ChunkingService bound to the async session."""
    return ChunkingService(
        db_session=db_session,
        collection_repo=create_collection_repository(db_session),
        document_repo=create_document_repository(db_session),
        redis_client=None,
    )


async def test_execute_ingestion_chunking_returns_chunk_stats(
    metrics_registry: CollectorRegistry,
    db_session,
) -> None:
    """Basic recursive chunking should produce deterministic stats and metrics."""
    service = await _create_service(db_session)

    collection_payload = {
        "id": "collection-recursive",
        "name": "Recursive Integration",
        "chunking_strategy": "recursive",
        "chunking_config": {"chunk_size": 150, "chunk_overlap": 30},
        "chunk_size": 150,
        "chunk_overlap": 30,
    }
    text = " ".join(f"Recursive sentence {i} for integration coverage." for i in range(40))

    result = await service.execute_ingestion_chunking(text=text, document_id="doc-recursive", collection=collection_payload)

    assert result["stats"]["strategy_used"] in {"recursive", "ChunkingStrategy.RECURSIVE"}
    assert result["stats"]["fallback"] is False
    assert result["stats"]["chunk_count"] == len(result["chunks"])
    assert all("chunk_id" in chunk for chunk in result["chunks"])

    chunk_counter = metrics_registry.get_sample_value(
        "ingestion_chunks_total_total",
        {"strategy": "recursive"},
    )
    assert chunk_counter == pytest.approx(len(result["chunks"]))


async def test_execute_ingestion_chunking_with_invalid_config_falls_back(
    metrics_registry: CollectorRegistry,
    db_session,
) -> None:
    """Invalid strategy configuration should trigger the token chunker fallback."""
    service = await _create_service(db_session)
    text = " ".join(f"Fallback sentence {i} for coverage." for i in range(20))

    collection_payload = {
        "id": "collection-invalid",
        "name": "Invalid Config",
        "chunking_strategy": "recursive",
        # Intentionally invalid configuration: negative size triggers validation errors
        "chunking_config": {"chunk_size": -10, "chunk_overlap": 500},
        "chunk_size": 100,
        "chunk_overlap": 20,
    }

    result = await service.execute_ingestion_chunking(text=text, document_id="doc-invalid", collection=collection_payload)

    assert result["stats"]["fallback"] is True
    assert result["stats"]["fallback_reason"] == "config_error"
    assert result["stats"]["strategy_used"] in {"TokenChunker", "character"}

    fallback_counter = metrics_registry.get_sample_value(
        "ingestion_chunking_fallback_total_total",
        {"strategy": "recursive", "reason": "config_error"},
    )
    assert fallback_counter == 1.0
