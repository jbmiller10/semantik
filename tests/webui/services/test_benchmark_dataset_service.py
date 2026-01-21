import hashlib
import json
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy import select

from shared.database.exceptions import ValidationError
from shared.database.models import BenchmarkRelevance, Document, DocumentStatus, Operation, OperationType
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


@pytest.mark.asyncio()
async def test_create_mapping_copies_pending_relevance_to_mapping(db_session, test_user_db, collection_factory) -> None:
    collection = await collection_factory(owner_id=test_user_db.id)

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
                    {"doc_ref": {"uri": "file:///tmp/a.txt"}, "relevance_grade": 2},
                    {"doc_ref": {"uri": "file:///tmp/b.txt"}, "relevance_grade": 1},
                ],
            }
        ],
    }

    dataset = await dataset_service.upload_dataset(
        user_id=test_user_db.id,
        name="copy-relevance",
        description=None,
        file_content=json.dumps(dataset_payload).encode("utf-8"),
    )

    mapping = await dataset_service.create_mapping(
        dataset_id=str(dataset["id"]),
        collection_id=collection.id,
        user_id=test_user_db.id,
    )

    assert mapping["total_count"] == 2

    mapping_id = int(mapping["id"])
    relevances = await BenchmarkDatasetRepository(db_session).get_relevance_for_mapping(mapping_id)
    assert len(relevances) == 2
    assert all(r.doc_ref_hash for r in relevances)


@pytest.mark.asyncio()
async def test_resolve_mapping_enqueues_when_wall_clock_exceeded(
    db_session, test_user_db, collection_factory, monkeypatch
) -> None:
    monkeypatch.setattr("shared.config.settings.BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_REFS", 10_000)
    monkeypatch.setattr("shared.config.settings.BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_DOCS", 50_000)
    monkeypatch.setattr("shared.config.settings.BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_WALL_MS", 0)

    sent: dict[str, object] = {}

    def _fake_send_task(name: str, *, kwargs: dict[str, object]) -> None:  # noqa: ANN001
        sent["name"] = name
        sent["kwargs"] = kwargs

    monkeypatch.setattr("webui.celery_app.celery_app.send_task", _fake_send_task)

    collection = await collection_factory(owner_id=test_user_db.id)

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
        name="wall-budget",
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
    assert result["mapping_status"] == "pending"
    assert sent["name"] == "webui.tasks.benchmark_mapping.resolve_mapping"


@pytest.mark.asyncio()
async def test_upload_dataset_supports_legacy_fields_and_scalar_judgments(
    db_session, test_user_db, monkeypatch
) -> None:
    monkeypatch.setattr("shared.config.settings.BENCHMARK_DATASET_MAX_UPLOAD_BYTES", 10_000)
    monkeypatch.setattr("shared.config.settings.BENCHMARK_DATASET_MAX_QUERIES", 10)
    monkeypatch.setattr("shared.config.settings.BENCHMARK_DATASET_MAX_JUDGMENTS_PER_QUERY", 10)

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
                "query_id": "q-1",
                "query": "hello",
                "relevant_doc_refs": [
                    "file:///tmp/a.txt",
                    {"doc_ref": "file:///tmp/b.txt", "relevance_grade": "3"},
                ],
            }
        ],
    }

    uploaded = await service.upload_dataset(
        user_id=test_user_db.id,
        name="legacy",
        description=None,
        file_content=json.dumps(payload).encode("utf-8"),
    )
    assert uploaded["query_count"] == 1

    queries = await BenchmarkDatasetRepository(db_session).get_queries_for_dataset(str(uploaded["id"]))
    assert len(queries) == 1
    qmeta = queries[0].query_metadata or {}
    pending = qmeta.get("_pending_relevance", [])
    assert len(pending) == 2
    assert pending[0]["doc_ref"] == {"uri": "file:///tmp/a.txt"}
    assert pending[1]["doc_ref"] == {"uri": "file:///tmp/b.txt"}
    assert pending[1]["relevance_grade"] == 3


@pytest.mark.asyncio()
async def test_resolve_mapping_with_progress_streams_updates_and_marks_partial(
    db_session, test_user_db, collection_factory
) -> None:
    collection = await collection_factory(owner_id=test_user_db.id)

    # Documents for resolution (unique uri/content_hash per collection)
    doc_a = Document(
        id=str(uuid4()),
        collection_id=collection.id,
        file_path="/tmp/a.txt",
        file_name="a.txt",
        file_size=1,
        mime_type="text/plain",
        content_hash=uuid4().hex,
        status=DocumentStatus.COMPLETED,
        chunk_count=0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        uri="file:///tmp/a.txt",
    )
    doc_unique = Document(
        id=str(uuid4()),
        collection_id=collection.id,
        file_path="/tmp/unique.txt",
        file_name="unique.txt",
        file_size=1,
        mime_type="text/plain",
        content_hash=uuid4().hex,
        status=DocumentStatus.COMPLETED,
        chunk_count=0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        uri="file:///tmp/unique.txt",
    )
    dup_name = "dup.txt"
    doc_dup_1 = Document(
        id=str(uuid4()),
        collection_id=collection.id,
        file_path="/tmp/dup1.txt",
        file_name=dup_name,
        file_size=1,
        mime_type="text/plain",
        content_hash=uuid4().hex,
        status=DocumentStatus.COMPLETED,
        chunk_count=0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        uri="file:///tmp/dup1.txt",
    )
    doc_dup_2 = Document(
        id=str(uuid4()),
        collection_id=collection.id,
        file_path="/tmp/dup2.txt",
        file_name=dup_name,
        file_size=1,
        mime_type="text/plain",
        content_hash=uuid4().hex,
        status=DocumentStatus.COMPLETED,
        chunk_count=0,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        uri="file:///tmp/dup2.txt",
    )
    db_session.add_all([doc_a, doc_unique, doc_dup_1, doc_dup_2])
    await db_session.commit()

    dataset_repo = BenchmarkDatasetRepository(db_session)
    dataset = await dataset_repo.create(
        name="progress-ds",
        owner_id=test_user_db.id,
        query_count=0,
        description=None,
        schema_version="1.0",
        metadata={},
    )
    mapping = await dataset_repo.create_mapping(dataset_id=str(dataset.id), collection_id=collection.id)
    query = await dataset_repo.add_query(dataset_id=str(dataset.id), query_key="q1", query_text="hello", metadata={})

    # Resolved via document_id
    await dataset_repo.add_relevance(
        query_id=int(query.id),
        mapping_id=int(mapping.id),
        doc_ref={"document_id": doc_a.id},
        relevance_grade=2,
    )
    # Resolved via uri
    await dataset_repo.add_relevance(
        query_id=int(query.id),
        mapping_id=int(mapping.id),
        doc_ref={"uri": doc_a.uri},
        relevance_grade=2,
    )
    # Resolved via file_name unique
    await dataset_repo.add_relevance(
        query_id=int(query.id),
        mapping_id=int(mapping.id),
        doc_ref={"file_name": doc_unique.file_name},
        relevance_grade=2,
    )
    # Ambiguous via duplicated file_name
    await dataset_repo.add_relevance(
        query_id=int(query.id),
        mapping_id=int(mapping.id),
        doc_ref={"file_name": dup_name},
        relevance_grade=2,
    )
    # Unresolved: invalid doc_ref JSON type
    raw_doc_ref: object = "not-a-dict"
    doc_ref_hash = hashlib.sha256(json.dumps(raw_doc_ref, sort_keys=True).encode()).hexdigest()
    db_session.add(
        BenchmarkRelevance(
            benchmark_query_id=int(query.id),
            mapping_id=int(mapping.id),
            doc_ref_hash=doc_ref_hash,
            doc_ref=raw_doc_ref,
            relevance_grade=2,
            relevance_metadata=None,
        )
    )
    await db_session.commit()

    class FakeProgress:
        def __init__(self):
            self.collection_id: str | None = None
            self.updates: list[tuple[str, dict]] = []

        def set_collection_id(self, collection_id: str | None) -> None:
            self.collection_id = collection_id

        async def send_update(self, update_type: str, data: dict) -> None:
            self.updates.append((update_type, data))

    progress = FakeProgress()

    service = BenchmarkDatasetService(
        db_session=db_session,
        benchmark_dataset_repo=dataset_repo,
        collection_repo=CollectionRepository(db_session),
        document_repo=DocumentRepository(db_session),
        operation_repo=OperationRepository(db_session),
    )

    result = await service.resolve_mapping_with_progress(
        mapping_id=int(mapping.id),
        user_id=test_user_db.id,
        operation_uuid="op-1",
        progress_reporter=cast(Any, progress),
    )

    assert result["mapping_id"] == int(mapping.id)
    assert result["operation_uuid"] == "op-1"
    assert result["total_count"] == 5
    assert result["mapped_count"] == 3
    assert result["mapping_status"] == "partial"

    stages = [data["stage"] for (_t, data) in progress.updates]
    assert stages[0] == "starting"
    assert "loading_documents" in stages
    assert "resolving" in stages
    assert stages[-1] == "completed"
