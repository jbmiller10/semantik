import json
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy import select

from shared.database.exceptions import ValidationError
from shared.database.models import Document, DocumentStatus, Operation, OperationType
from shared.database.repositories.benchmark_dataset_repository import BenchmarkDatasetRepository
from shared.database.repositories.collection_repository import CollectionRepository
from shared.database.repositories.document_repository import DocumentRepository
from shared.database.repositories.operation_repository import OperationRepository
from webui.services.benchmark_dataset_service import BenchmarkDatasetService


@pytest.mark.asyncio()
async def test_upload_dataset_enforces_max_upload_bytes(db_session, test_user_db, monkeypatch) -> None:
    monkeypatch.setattr("shared.config.settings.BENCHMARK_DATASET_MAX_UPLOAD_BYTES", 1)

    service = BenchmarkDatasetService(
        db_session=db_session,
        benchmark_dataset_repo=BenchmarkDatasetRepository(db_session),
        collection_repo=CollectionRepository(db_session),
        document_repo=DocumentRepository(db_session),
        operation_repo=OperationRepository(db_session),
    )

    with pytest.raises(ValidationError):
        await service.upload_dataset(
            user_id=test_user_db.id,
            name="too-big",
            description=None,
            file_content=b"{}",
        )


@pytest.mark.asyncio()
async def test_upload_dataset_enforces_max_queries(db_session, test_user_db, monkeypatch) -> None:
    monkeypatch.setattr("shared.config.settings.BENCHMARK_DATASET_MAX_UPLOAD_BYTES", 10_000)
    monkeypatch.setattr("shared.config.settings.BENCHMARK_DATASET_MAX_QUERIES", 1)

    service = BenchmarkDatasetService(
        db_session=db_session,
        benchmark_dataset_repo=BenchmarkDatasetRepository(db_session),
        collection_repo=CollectionRepository(db_session),
        document_repo=DocumentRepository(db_session),
        operation_repo=OperationRepository(db_session),
    )

    payload = {
        "schema_version": "1.0",
        "queries": [
            {"query_key": "q1", "query_text": "hello", "relevant_docs": []},
            {"query_key": "q2", "query_text": "world", "relevant_docs": []},
        ],
    }

    with pytest.raises(ValidationError):
        await service.upload_dataset(
            user_id=test_user_db.id,
            name="too-many",
            description=None,
            file_content=json.dumps(payload).encode("utf-8"),
        )


@pytest.mark.asyncio()
async def test_upload_dataset_enforces_max_judgments_per_query(db_session, test_user_db, monkeypatch) -> None:
    monkeypatch.setattr("shared.config.settings.BENCHMARK_DATASET_MAX_UPLOAD_BYTES", 10_000)
    monkeypatch.setattr("shared.config.settings.BENCHMARK_DATASET_MAX_QUERIES", 10)
    monkeypatch.setattr("shared.config.settings.BENCHMARK_DATASET_MAX_JUDGMENTS_PER_QUERY", 1)

    service = BenchmarkDatasetService(
        db_session=db_session,
        benchmark_dataset_repo=BenchmarkDatasetRepository(db_session),
        collection_repo=CollectionRepository(db_session),
        document_repo=DocumentRepository(db_session),
        operation_repo=OperationRepository(db_session),
    )

    payload = {
        "schema_version": "1.0",
        "queries": [
            {
                "query_key": "q1",
                "query_text": "hello",
                "relevant_docs": [
                    {"doc_ref": {"uri": "a"}, "relevance_grade": 2},
                    {"doc_ref": {"uri": "b"}, "relevance_grade": 2},
                ],
            }
        ],
    }

    with pytest.raises(ValidationError):
        await service.upload_dataset(
            user_id=test_user_db.id,
            name="too-many-judgments",
            description=None,
            file_content=json.dumps(payload).encode("utf-8"),
        )


@pytest.mark.asyncio()
async def test_resolve_mapping_handles_ambiguous_file_name(
    db_session, test_user_db, collection_factory, monkeypatch
) -> None:
    monkeypatch.setattr("shared.config.settings.BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_REFS", 10_000)
    monkeypatch.setattr("shared.config.settings.BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_DOCS", 50_000)
    monkeypatch.setattr("shared.config.settings.BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_WALL_MS", 8_000)

    collection = await collection_factory(owner_id=test_user_db.id)

    # Two docs share a file_name (ambiguous fallback)
    shared_name = "shared.txt"
    doc_a = Document(
        id=str(uuid4()),
        collection_id=collection.id,
        file_path="/tmp/a.txt",
        file_name=shared_name,
        file_size=1,
        mime_type="text/plain",
        content_hash=uuid4().hex,
        status=DocumentStatus.COMPLETED,
        chunk_count=0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        uri="file:///tmp/a.txt",
    )
    doc_b = Document(
        id=str(uuid4()),
        collection_id=collection.id,
        file_path="/tmp/b.txt",
        file_name=shared_name,
        file_size=1,
        mime_type="text/plain",
        content_hash=uuid4().hex,
        status=DocumentStatus.COMPLETED,
        chunk_count=0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        uri="file:///tmp/b.txt",
    )
    db_session.add_all([doc_a, doc_b])
    await db_session.commit()

    dataset_service = BenchmarkDatasetService(
        db_session=db_session,
        benchmark_dataset_repo=BenchmarkDatasetRepository(db_session),
        collection_repo=CollectionRepository(db_session),
        document_repo=DocumentRepository(db_session),
        operation_repo=OperationRepository(db_session),
    )

    dataset_payload = {
        "schema_version": "1.0",
        "queries": [
            {
                "query_key": "q1",
                "query_text": "test",
                "relevant_docs": [
                    {"doc_ref": {"file_name": shared_name}, "relevance_grade": 2},
                ],
            }
        ],
    }

    dataset = await dataset_service.upload_dataset(
        user_id=test_user_db.id,
        name="ambiguous-file-name",
        description=None,
        file_content=json.dumps(dataset_payload).encode("utf-8"),
    )

    mapping = await dataset_service.create_mapping(
        dataset_id=str(dataset["id"]),
        collection_id=collection.id,
        user_id=test_user_db.id,
    )

    result = await dataset_service.resolve_mapping(mapping_id=int(mapping["id"]), user_id=test_user_db.id)

    assert result["operation_uuid"] is None
    assert result["total_count"] == 1
    assert result["mapped_count"] == 0
    assert result["mapping_status"] == "pending"
    assert result["unresolved"]
    assert result["unresolved"][0]["reason"] == "ambiguous"


@pytest.mark.asyncio()
async def test_resolve_mapping_routes_large_jobs_to_async(
    db_session, test_user_db, collection_factory, monkeypatch
) -> None:
    # Force async routing by setting thresholds to zero
    monkeypatch.setattr("shared.config.settings.BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_REFS", 0)
    monkeypatch.setattr("shared.config.settings.BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_DOCS", 0)

    # Prevent actual Celery dispatch
    sent: dict[str, object] = {}

    def _fake_send_task(name: str, *, kwargs: dict[str, object]) -> None:  # noqa: ANN001
        sent["name"] = name
        sent["kwargs"] = kwargs

    monkeypatch.setattr("webui.celery_app.celery_app.send_task", _fake_send_task)

    collection = await collection_factory(owner_id=test_user_db.id)

    # Single document to satisfy collection existence
    doc = Document(
        id=str(uuid4()),
        collection_id=collection.id,
        file_path="/tmp/x.txt",
        file_name="x.txt",
        file_size=1,
        mime_type="text/plain",
        content_hash=uuid4().hex,
        status=DocumentStatus.COMPLETED,
        chunk_count=0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        uri="file:///tmp/x.txt",
    )
    db_session.add(doc)
    await db_session.commit()

    dataset_service = BenchmarkDatasetService(
        db_session=db_session,
        benchmark_dataset_repo=BenchmarkDatasetRepository(db_session),
        collection_repo=CollectionRepository(db_session),
        document_repo=DocumentRepository(db_session),
        operation_repo=OperationRepository(db_session),
    )

    dataset_payload = {
        "schema_version": "1.0",
        "queries": [
            {
                "query_key": "q1",
                "query_text": "test",
                "relevant_docs": [
                    {"doc_ref": {"uri": "file:///tmp/x.txt"}, "relevance_grade": 2},
                ],
            }
        ],
    }

    dataset = await dataset_service.upload_dataset(
        user_id=test_user_db.id,
        name="async-route",
        description=None,
        file_content=json.dumps(dataset_payload).encode("utf-8"),
    )

    mapping = await dataset_service.create_mapping(
        dataset_id=str(dataset["id"]),
        collection_id=collection.id,
        user_id=test_user_db.id,
    )

    result = await dataset_service.resolve_mapping(mapping_id=int(mapping["id"]), user_id=test_user_db.id)

    assert isinstance(result["operation_uuid"], str)
    assert sent["name"] == "webui.tasks.benchmark_mapping.resolve_mapping"

    operation_uuid = cast(str, result["operation_uuid"])
    operation = (await db_session.execute(select(Operation).where(Operation.uuid == operation_uuid))).scalar_one()
    assert operation.type == OperationType.BENCHMARK
    assert cast(dict[str, object], operation.config)["kind"] == "mapping_resolve"
