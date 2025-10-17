"""Integration tests for chunking metrics and ingestion flows."""

from __future__ import annotations

import importlib
import sys
from hashlib import sha256
from typing import Generator

import pytest
from prometheus_client import CollectorRegistry

from packages.shared.database.models import DocumentStatus
from packages.shared.database.repositories.collection_repository import CollectionRepository
from packages.shared.database.repositories.document_repository import DocumentRepository
from packages.webui.services.chunking_service import ChunkingService

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

TEXT_SAMPLE = " ".join(f"Sentence {i} about chunking." for i in range(64))


@pytest.fixture()
def isolated_chunking_metrics(monkeypatch: pytest.MonkeyPatch) -> Generator[tuple[CollectorRegistry, object, object], None, None]:
    """Provide an isolated CollectorRegistry for chunking metric modules."""

    registry = CollectorRegistry()
    shared_metrics = importlib.import_module("packages.shared.metrics.prometheus")
    original_registry = shared_metrics.registry

    monkeypatch.setattr(shared_metrics, "registry", registry, raising=False)

    prometheus_client = importlib.import_module("prometheus_client")
    prometheus_registry = importlib.import_module("prometheus_client.registry")
    prometheus_metrics = importlib.import_module("prometheus_client.metrics")
    prometheus_core = importlib.import_module("prometheus_client.core")

    monkeypatch.setattr(prometheus_client, "REGISTRY", registry, raising=False)
    monkeypatch.setattr(prometheus_registry, "REGISTRY", registry, raising=False)
    monkeypatch.setattr(prometheus_metrics, "REGISTRY", registry, raising=False)
    monkeypatch.setattr(prometheus_core, "REGISTRY", registry, raising=False)

    module_names = (
        "packages.webui.services.chunking_metrics",
        "packages.webui.services.chunking_error_metrics",
    )
    original_modules = {name: sys.modules.get(name) for name in module_names}
    for module_name in module_names:
        sys.modules.pop(module_name, None)

    chunking_metrics = importlib.import_module("packages.webui.services.chunking_metrics")
    chunking_error_metrics = importlib.import_module("packages.webui.services.chunking_error_metrics")

    chunking_error_metrics.chunking_errors_total = prometheus_client.Counter(
        "chunking_errors_total",
        "Total number of chunking errors by type and strategy",
        ["error_type", "strategy", "recoverable"],
        registry=registry,
    )
    chunking_error_metrics.chunking_error_recovery_attempts = prometheus_client.Counter(
        "chunking_error_recovery_attempts_total",
        "Total number of error recovery attempts",
        ["error_type", "recovery_strategy", "success"],
        registry=registry,
    )
    chunking_error_metrics.chunking_error_recovery_duration = prometheus_client.Histogram(
        "chunking_error_recovery_duration_seconds",
        "Time taken to recover from errors",
        ["error_type", "recovery_strategy"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
        registry=registry,
    )
    chunking_error_metrics.chunking_operations_active = prometheus_client.Gauge(
        "chunking_operations_active",
        "Number of currently active chunking operations",
        ["strategy", "operation_type"],
        registry=registry,
    )
    chunking_error_metrics.chunking_operations_failed = prometheus_client.Gauge(
        "chunking_operations_failed",
        "Current number of failed operations pending retry",
        ["strategy", "error_type"],
        registry=registry,
    )
    chunking_error_metrics.chunking_partial_failures_total = prometheus_client.Counter(
        "chunking_partial_failures_total",
        "Total number of partial failures",
        ["strategy"],
        registry=registry,
    )
    chunking_error_metrics.chunking_partial_failure_document_ratio = prometheus_client.Histogram(
        "chunking_partial_failure_document_ratio",
        "Ratio of failed documents in partial failures",
        ["strategy"],
        buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        registry=registry,
    )
    chunking_error_metrics.chunking_circuit_breaker_state = prometheus_client.Gauge(
        "chunking_circuit_breaker_state",
        "Circuit breaker state",
        ["service"],
        registry=registry,
    )
    chunking_error_metrics.chunking_circuit_breaker_trips = prometheus_client.Counter(
        "chunking_circuit_breaker_trips_total",
        "Number of times circuit breaker has tripped",
        ["service", "reason"],
        registry=registry,
    )
    chunking_error_metrics.chunking_memory_usage_bytes = prometheus_client.Histogram(
        "chunking_memory_usage_bytes",
        "Memory usage per chunking operation",
        ["strategy", "status"],
        buckets=[1_000_000, 10_000_000, 50_000_000, 100_000_000, 250_000_000, 500_000_000, 1_000_000_000],
        registry=registry,
    )
    chunking_error_metrics.chunking_cpu_usage_seconds = prometheus_client.Histogram(
        "chunking_cpu_usage_seconds",
        "CPU time used per chunking operation",
        ["strategy", "status"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300],
        registry=registry,
    )

    try:
        yield registry, chunking_metrics, chunking_error_metrics
    finally:
        for module_name in module_names:
            sys.modules.pop(module_name, None)
        monkeypatch.setattr(shared_metrics, "registry", original_registry, raising=False)
        for module_name, original_module in original_modules.items():
            if original_module is not None:
                sys.modules[module_name] = original_module


@pytest.fixture()
def chunking_service(db_session, use_fakeredis) -> ChunkingService:
    """Instantiate the chunking service with real repositories and fakeredis."""
    _, async_redis = use_fakeredis
    collection_repo = CollectionRepository(db_session)
    document_repo = DocumentRepository(db_session)
    return ChunkingService(db_session, collection_repo, document_repo, async_redis)


async def _create_document(
    db_session,
    document_repo: DocumentRepository,
    collection_id: str,
    text: str,
) -> str:
    """Persist a document record for chunking tests."""
    document = await document_repo.create(
        collection_id=collection_id,
        file_path="/tmp/chunking-doc.txt",
        file_name="chunking-doc.txt",
        file_size=len(text.encode("utf-8")),
        content_hash=sha256(text.encode("utf-8")).hexdigest(),
        mime_type="text/plain",
    )
    await db_session.commit()
    return document.id


async def test_successful_ingestion_records_metrics(
    isolated_chunking_metrics,
    chunking_service: ChunkingService,
    collection_factory,
    test_user_db,
    db_session,
) -> None:
    """Chunking a document should record Prometheus metrics and persist chunk counts."""
    registry, _, _ = isolated_chunking_metrics
    document_repo = chunking_service.document_repo

    collection = await collection_factory(
        owner_id=test_user_db.id,
        chunking_strategy="recursive",
        chunk_size=180,
        chunk_overlap=24,
    )

    collection_payload = {
        "id": collection.id,
        "chunking_strategy": "recursive",
        "chunking_config": {"chunk_size": 180, "chunk_overlap": 24},
        "chunk_size": 180,
        "chunk_overlap": 24,
    }

    document_id = await _create_document(db_session, document_repo, collection.id, TEXT_SAMPLE)

    result = await chunking_service.execute_ingestion_chunking(
        text=TEXT_SAMPLE,
        document_id=document_id,
        collection=collection_payload,
        metadata={"source": "integration"},
    )

    await document_repo.update_status(
        document_id,
        DocumentStatus.COMPLETED,
        chunk_count=result["stats"]["chunk_count"],
    )
    await db_session.commit()

    updated_document = await document_repo.get_by_id(document_id)
    assert updated_document is not None
    assert updated_document.chunk_count == result["stats"]["chunk_count"]
    assert updated_document.status == DocumentStatus.COMPLETED

    assert result["stats"]["fallback"] is False
    assert result["stats"]["chunk_count"] > 0

    recorded_chunks = registry.get_sample_value(
        "ingestion_chunks_total",
        labels={"strategy": "recursive"},
    )
    assert recorded_chunks == float(result["stats"]["chunk_count"])

    histogram_count = registry.get_sample_value(
        "ingestion_chunking_duration_seconds_count",
        labels={"strategy": "recursive"},
    )
    assert histogram_count == 1.0

    summary_count = registry.get_sample_value(
        "ingestion_avg_chunk_size_bytes_count",
        labels={"strategy": "recursive"},
    )
    assert summary_count == float(result["stats"]["chunk_count"])

    fallback_counter = registry.get_sample_value(
        "ingestion_chunking_fallback_total",
        labels={"strategy": "recursive", "reason": "runtime_error"},
    )
    assert fallback_counter is None


async def test_invalid_strategy_falls_back_to_token_chunker(
    isolated_chunking_metrics,
    chunking_service: ChunkingService,
    collection_factory,
    test_user_db,
    db_session,
) -> None:
    """Invalid strategy choices should fall back and emit Prometheus counters."""
    registry, _, _ = isolated_chunking_metrics
    document_repo = chunking_service.document_repo

    collection = await collection_factory(
        owner_id=test_user_db.id,
        chunking_strategy="nonsense",
        chunk_size=128,
        chunk_overlap=16,
    )

    collection_payload = {
        "id": collection.id,
        "chunking_strategy": "nonsense",
        "chunking_config": {"chunk_size": 128, "chunk_overlap": 16},
        "chunk_size": 128,
        "chunk_overlap": 16,
    }

    document_id = await _create_document(db_session, document_repo, collection.id, TEXT_SAMPLE)

    result = await chunking_service.execute_ingestion_chunking(
        text=TEXT_SAMPLE,
        document_id=document_id,
        collection=collection_payload,
        metadata={"source": "integration-invalid"},
    )

    assert result["stats"]["fallback"] is True
    assert result["stats"]["fallback_reason"] in {"config_error", "invalid_config"}
    assert result["stats"]["chunk_count"] > 0

    fallback_value = registry.get_sample_value(
        "ingestion_chunking_fallback_total",
        labels={"strategy": "nonsense", "reason": result["stats"]["fallback_reason"]},
    )
    assert fallback_value == 1.0

    token_chunks = registry.get_sample_value(
        "ingestion_chunks_total",
        labels={"strategy": "character"},
    )
    assert token_chunks == float(result["stats"]["chunk_count"])


def test_error_metrics_recording_uses_isolated_registry(isolated_chunking_metrics) -> None:
    """Error metrics helpers should write to the injected registry without touching globals."""
    registry, _, error_metrics = isolated_chunking_metrics

    error_metrics.record_chunking_error("timeout", "semantic", recoverable=True)
    error_metrics.record_recovery_attempt("timeout", "retry", success=False, duration=1.5)
    error_metrics.record_partial_failure("semantic", total_documents=10, failed_documents=2)
    error_metrics.update_operation_status("recursive", "APPEND", active_delta=1)
    error_metrics.update_operation_status("recursive", "APPEND", failed_delta=1, error_type="timeout")
    error_metrics.update_circuit_breaker_state("chunking", state=1, trip_reason="timeout")
    error_metrics.record_resource_usage(
        strategy="semantic",
        status="failure",
        memory_bytes=1024 * 1024,
        cpu_seconds=0.5,
    )

    error_counter = registry.get_sample_value(
        "chunking_errors_total",
        labels={"error_type": "timeout", "strategy": "semantic", "recoverable": "True"},
    )
    assert error_counter == 1.0

    retry_counter = registry.get_sample_value(
        "chunking_error_recovery_attempts_total",
        labels={"error_type": "timeout", "recovery_strategy": "retry", "success": "False"},
    )
    assert retry_counter == 1.0

    partial_failures = registry.get_sample_value(
        "chunking_partial_failures_total",
        labels={"strategy": "semantic"},
    )
    assert partial_failures == 1.0

    circuit_trips = registry.get_sample_value(
        "chunking_circuit_breaker_trips_total",
        labels={"service": "chunking", "reason": "timeout"},
    )
    assert circuit_trips == 1.0

    active_operations = registry.get_sample_value(
        "chunking_operations_active",
        labels={"strategy": "recursive", "operation_type": "APPEND"},
    )
    assert active_operations == 1.0

    failed_operations = registry.get_sample_value(
        "chunking_operations_failed",
        labels={"strategy": "recursive", "error_type": "timeout"},
    )
    assert failed_operations == 1.0

    cpu_sum = registry.get_sample_value(
        "chunking_cpu_usage_seconds_sum",
        labels={"strategy": "semantic", "status": "failure"},
    )
    assert cpu_sum == pytest.approx(0.5)
