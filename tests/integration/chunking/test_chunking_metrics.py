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

    chunk_samples = [
        sample
        for metric in metrics_registry.collect()
        for sample in metric.samples
        if sample.name == "ingestion_chunks_total"
    ]
    assert chunk_samples
    assert any(sample.value == pytest.approx(chunk_count) for sample in chunk_samples)

    duration_counts = [
        sample.value
        for metric in metrics_registry.collect()
        for sample in metric.samples
        if sample.name == "ingestion_chunking_duration_seconds_count"
    ]
    assert duration_counts
    assert max(duration_counts) >= 1.0

    avg_size_counts = [
        sample.value
        for metric in metrics_registry.collect()
        for sample in metric.samples
        if sample.name == "ingestion_avg_chunk_size_bytes_count"
    ]
    assert avg_size_counts
    assert max(avg_size_counts) >= 1.0

    avg_size_sums = [
        sample.value
        for metric in metrics_registry.collect()
        for sample in metric.samples
        if sample.name == "ingestion_avg_chunk_size_bytes_sum"
    ]
    assert avg_size_sums
    assert max(avg_size_sums) > 0
