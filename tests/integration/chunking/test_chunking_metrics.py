#!/usr/bin/env python3
"""Integration tests validating chunking metrics with real service flows."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from packages.shared.database.factory import create_collection_repository, create_document_repository
from packages.webui.services import chunking_metrics
from packages.webui.services.chunking_service import ChunkingService

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture()
def metrics_registry() -> CollectorRegistry:
    """Provide an isolated Prometheus registry for integration assertions."""
    registry = CollectorRegistry()
    chunking_metrics.init_metrics(registry)
    yield registry
    chunking_metrics.init_metrics()


async def test_execute_ingestion_chunking_emits_metrics(
    metrics_registry: CollectorRegistry,
    db_session,
) -> None:
    """ChunkingService should record duration, chunk counts, and size metrics."""
    service = ChunkingService(
        db_session=db_session,
        collection_repo=create_collection_repository(db_session),
        document_repo=create_document_repository(db_session),
        redis_client=None,
    )

    collection_payload = {
        "id": "col-metrics-1",
        "name": "Metrics Integration Collection",
        "chunking_strategy": "recursive",
        "chunking_config": {"chunk_size": 120, "chunk_overlap": 24},
        "chunk_size": 120,
        "chunk_overlap": 24,
    }
    text = " ".join(f"Sentence {i} for recursive chunking." for i in range(30))

    result = await service.execute_ingestion_chunking(
        text=text,
        document_id="doc-metrics-1",
        collection=collection_payload,
    )

    chunk_count = len(result["chunks"])
    assert chunk_count > 0

    chunks_total = metrics_registry.get_sample_value(
        "ingestion_chunks_total_total",
        {"strategy": "recursive"},
    )
    duration_count = metrics_registry.get_sample_value(
        "ingestion_chunking_duration_seconds_count",
        {"strategy": "recursive"},
    )
    avg_size_count = metrics_registry.get_sample_value(
        "ingestion_avg_chunk_size_bytes_count",
        {"strategy": "recursive"},
    )

    assert chunks_total == pytest.approx(chunk_count)
    assert duration_count == 1.0
    assert avg_size_count == 1.0

    avg_size_sum = metrics_registry.get_sample_value(
        "ingestion_avg_chunk_size_bytes_sum",
        {"strategy": "recursive"},
    )
    assert avg_size_sum is not None and avg_size_sum > 0
