import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import settings
from shared.database.exceptions import AccessDeniedError, ValidationError
from shared.database.models import MappingStatus
from webui.services.benchmark_dataset_service import BenchmarkDatasetService


@pytest.mark.asyncio()
async def test_upload_dataset_rejects_oversize(monkeypatch) -> None:
    monkeypatch.setattr(settings, "BENCHMARK_DATASET_MAX_UPLOAD_BYTES", 1, raising=False)

    service = BenchmarkDatasetService(
        db_session=AsyncMock(),
        benchmark_dataset_repo=AsyncMock(),
        collection_repo=AsyncMock(),
        document_repo=AsyncMock(),
        operation_repo=AsyncMock(),
    )

    with pytest.raises(ValidationError) as excinfo:
        await service.upload_dataset(
            user_id=1,
            name="ds",
            description=None,
            file_content=b"too-big",
        )

    assert "too large" in str(excinfo.value).lower()


@pytest.mark.asyncio()
async def test_upload_dataset_parses_and_stores_pending_relevance(monkeypatch) -> None:
    monkeypatch.setattr(settings, "BENCHMARK_DATASET_MAX_UPLOAD_BYTES", 10_000, raising=False)
    monkeypatch.setattr(settings, "BENCHMARK_DATASET_MAX_QUERIES", 1000, raising=False)
    monkeypatch.setattr(settings, "BENCHMARK_DATASET_MAX_JUDGMENTS_PER_QUERY", 100, raising=False)

    dataset_obj = SimpleNamespace(
        id="ds-1",
        name="My Dataset",
        description=None,
        query_count=2,
        schema_version="1.0",
        created_at="2025-01-01T00:00:00Z",
    )

    repo = AsyncMock()
    repo.create.return_value = dataset_obj
    repo.add_query = AsyncMock()

    service = BenchmarkDatasetService(
        db_session=AsyncMock(),
        benchmark_dataset_repo=repo,
        collection_repo=AsyncMock(),
        document_repo=AsyncMock(),
        operation_repo=AsyncMock(),
    )

    payload = {
        "schema_version": "1.0",
        "queries": [
            {
                "query_key": "q1",
                "query_text": "hello",
                "relevant_docs": [
                    "doc-a",
                    {"doc_ref": "doc-b", "relevance_grade": 3},
                    {"doc_ref": {"uri": "doc-c"}, "relevance_grade": "2"},
                ],
            },
            {
                # Backwards-compat with early UI iteration
                "query_id": "q2",
                "query": "world",
                "relevant_doc_refs": [{"uri": "doc-d"}],
            },
        ],
        "metadata": {"source": "unit-test"},
    }

    result = await service.upload_dataset(
        user_id=1,
        name="My Dataset",
        description=None,
        file_content=json.dumps(payload).encode("utf-8"),
    )

    assert result["id"] == "ds-1"
    assert result["query_count"] == 2

    assert repo.add_query.await_count == 2
    first_call = repo.add_query.await_args_list[0].kwargs
    assert first_call["query_key"] == "q1"
    assert first_call["query_text"] == "hello"
    pending = first_call["metadata"]["_pending_relevance"]
    assert pending == [
        {"doc_ref": {"uri": "doc-a"}, "relevance_grade": 2},
        {"doc_ref": {"uri": "doc-b"}, "relevance_grade": 3},
        {"doc_ref": {"uri": "doc-c"}, "relevance_grade": 2},
    ]


@pytest.mark.asyncio()
async def test_upload_dataset_rejects_missing_query_fields() -> None:
    repo = AsyncMock()
    repo.create.return_value = SimpleNamespace(
        id="ds-1", name="ds", description=None, query_count=1, schema_version="1.0"
    )

    service = BenchmarkDatasetService(
        db_session=AsyncMock(),
        benchmark_dataset_repo=repo,
        collection_repo=AsyncMock(),
        document_repo=AsyncMock(),
        operation_repo=AsyncMock(),
    )

    payload = {"schema_version": "1.0", "queries": [{"query_key": "q1"}]}

    with pytest.raises(ValidationError) as excinfo:
        await service.upload_dataset(
            user_id=1,
            name="ds",
            description=None,
            file_content=json.dumps(payload).encode("utf-8"),
        )
    assert "query_key" in str(excinfo.value)


@pytest.mark.asyncio()
async def test_create_mapping_copies_pending_relevance_and_updates_counts() -> None:
    dataset_obj = SimpleNamespace(id="ds-1")
    collection_obj = SimpleNamespace(id="col-1")
    mapping_obj = SimpleNamespace(
        id=10,
        dataset_id="ds-1",
        collection_id="col-1",
        mapping_status=MappingStatus.PENDING.value,
        mapped_count=0,
        total_count=0,
        created_at="2025-01-01T00:00:00Z",
    )

    repo = AsyncMock()
    repo.get_by_uuid_for_user.return_value = dataset_obj
    repo.create_mapping.return_value = mapping_obj
    repo.get_queries_for_dataset.return_value = [
        SimpleNamespace(
            id=1,
            query_metadata={
                "_pending_relevance": [
                    {"doc_ref": {"uri": "doc-a"}, "relevance_grade": 2},
                    {"doc_ref": {"uri": "doc-b"}, "relevance_grade": 3},
                ]
            },
        )
    ]
    repo.add_relevance = AsyncMock()
    repo.update_mapping_status = AsyncMock()

    collection_repo = AsyncMock()
    collection_repo.get_by_uuid_with_permission_check.return_value = collection_obj

    service = BenchmarkDatasetService(
        db_session=AsyncMock(),
        benchmark_dataset_repo=repo,
        collection_repo=collection_repo,
        document_repo=AsyncMock(),
        operation_repo=AsyncMock(),
    )

    result = await service.create_mapping(dataset_id="ds-1", collection_id="col-1", user_id=1)

    assert result["id"] == 10
    assert result["total_count"] == 2

    assert repo.add_relevance.await_count == 2
    first_relevance = repo.add_relevance.await_args_list[0].kwargs
    assert first_relevance["query_id"] == 1
    assert first_relevance["mapping_id"] == 10
    expected_hash = hashlib.sha256(json.dumps({"uri": "doc-a"}, sort_keys=True).encode()).hexdigest()
    assert first_relevance["doc_ref_hash"] == expected_hash

    repo.update_mapping_status.assert_awaited_once()
    assert repo.update_mapping_status.await_args.kwargs["total_count"] == 2


@pytest.mark.asyncio()
async def test_create_mapping_translates_collection_permission_errors() -> None:
    repo = AsyncMock()
    repo.get_by_uuid_for_user.return_value = SimpleNamespace(id="ds-1")

    collection_repo = AsyncMock()
    collection_repo.get_by_uuid_with_permission_check.side_effect = AccessDeniedError("1", "collection", "col-1")

    service = BenchmarkDatasetService(
        db_session=AsyncMock(),
        benchmark_dataset_repo=repo,
        collection_repo=collection_repo,
        document_repo=AsyncMock(),
        operation_repo=AsyncMock(),
    )

    with pytest.raises(AccessDeniedError) as excinfo:
        await service.create_mapping(dataset_id="ds-1", collection_id="col-1", user_id=1)
    assert "collection" in str(excinfo.value)


@pytest.mark.asyncio()
async def test_resolve_mapping_enqueues_large_jobs(monkeypatch) -> None:
    monkeypatch.setattr(settings, "BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_REFS", 0, raising=False)
    monkeypatch.setattr(settings, "BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_DOCS", 0, raising=False)

    mapping = SimpleNamespace(
        id=10,
        dataset_id="ds-1",
        collection_id="col-1",
        mapping_status=MappingStatus.PENDING.value,
        mapped_count=0,
        total_count=123,
    )

    repo = AsyncMock()
    repo.get_mapping.return_value = mapping
    repo.get_by_uuid_for_user.return_value = SimpleNamespace(id="ds-1")
    repo.count_relevance_for_mapping.return_value = 999

    document_repo = AsyncMock()
    document_repo.count_by_collection.return_value = 999

    service = BenchmarkDatasetService(
        db_session=AsyncMock(),
        benchmark_dataset_repo=repo,
        collection_repo=AsyncMock(),
        document_repo=document_repo,
        operation_repo=AsyncMock(),
    )
    service._enqueue_mapping_resolution_operation = AsyncMock(return_value="op-uuid")  # type: ignore[attr-defined]

    result = await service.resolve_mapping(mapping_id=10, user_id=1)

    assert result["operation_uuid"] == "op-uuid"
    assert result["unresolved"] == []


class _Relevance:
    def __init__(self, relevance_id: int, query_id: int, doc_ref, doc_ref_hash: str | None = None) -> None:
        self.id = relevance_id
        self.benchmark_query_id = query_id
        self.doc_ref = doc_ref
        self.doc_ref_hash = doc_ref_hash
        self.resolved_document_id: str | None = None


@pytest.mark.asyncio()
async def test_resolve_mapping_sync_path_resolves_and_collects_samples(monkeypatch) -> None:
    monkeypatch.setattr(settings, "BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_REFS", 10_000, raising=False)
    monkeypatch.setattr(settings, "BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_DOCS", 50_000, raising=False)
    monkeypatch.setattr(settings, "BENCHMARK_MAPPING_RESOLVE_SYNC_MAX_WALL_MS", 60_000, raising=False)

    mapping = SimpleNamespace(
        id=10,
        dataset_id="ds-1",
        collection_id="col-1",
        mapping_status=MappingStatus.PENDING.value,
        mapped_count=0,
        total_count=7,
    )

    relevances = [
        _Relevance(1, 1, {"document_id": "doc-1"}),
        _Relevance(2, 1, {"uri": "u-1"}),
        _Relevance(3, 1, {"content_hash": "h-1"}),
        _Relevance(4, 1, {"content_hash": "h-2"}),  # ambiguous
        _Relevance(5, 1, {"path": "p-1"}),  # path treated like uri
        _Relevance(6, 1, {"file_name": "f-1"}),
        _Relevance(7, 1, {}),  # invalid doc_ref
    ]

    repo = AsyncMock()
    repo.get_mapping.return_value = mapping
    repo.get_by_uuid_for_user.return_value = SimpleNamespace(id="ds-1")
    repo.count_relevance_for_mapping.return_value = len(relevances)
    repo.list_unresolved_relevance_for_mapping.side_effect = [relevances, []]
    repo.count_resolved_relevance_for_mapping.return_value = 5
    repo.update_mapping_status = AsyncMock()

    document_repo = AsyncMock()
    document_repo.count_by_collection.return_value = 10
    document_repo.get_existing_ids_in_collection.return_value = {"doc-1"}
    document_repo.get_doc_ids_by_uri_bulk.return_value = {"u-1": "doc-2", "p-1": "doc-5"}
    document_repo.get_doc_ids_by_content_hash_bulk.return_value = {"h-1": ["doc-3"], "h-2": ["doc-4", "doc-x"]}
    document_repo.get_doc_ids_by_file_name_bulk.return_value = {"f-1": ["doc-6"]}

    db_session = AsyncMock()

    service = BenchmarkDatasetService(
        db_session=db_session,
        benchmark_dataset_repo=repo,
        collection_repo=AsyncMock(),
        document_repo=document_repo,
        operation_repo=AsyncMock(),
    )

    result = await service.resolve_mapping(mapping_id=10, user_id=1)

    assert result["mapping_status"] == MappingStatus.PARTIAL.value
    assert result["mapped_count"] == 5
    assert result["total_count"] == len(relevances)
    assert any(sample["reason"] == "ambiguous" for sample in result["unresolved"])
    assert any(sample["reason"] == "invalid_doc_ref" for sample in result["unresolved"])

    assert relevances[0].resolved_document_id == "doc-1"
    assert relevances[1].resolved_document_id == "doc-2"
    assert relevances[2].resolved_document_id == "doc-3"
    assert relevances[4].resolved_document_id == "doc-5"
    assert relevances[5].resolved_document_id == "doc-6"

    assert db_session.flush.await_count >= 1
    assert db_session.commit.await_count >= 1


@pytest.mark.asyncio()
async def test_resolve_mapping_with_progress_sends_stages() -> None:
    mapping = SimpleNamespace(
        id=10,
        dataset_id="ds-1",
        collection_id="col-1",
        mapping_status=MappingStatus.PENDING.value,
        mapped_count=0,
        total_count=2,
    )

    relevances = [
        _Relevance(1, 1, {"uri": "u-1"}),
        _Relevance(2, 1, {"content_hash": "h-2"}),
    ]

    repo = AsyncMock()
    repo.get_mapping.return_value = mapping
    repo.get_by_uuid_for_user.return_value = SimpleNamespace(id="ds-1")
    repo.count_relevance_for_mapping.return_value = len(relevances)
    repo.list_unresolved_relevance_for_mapping.side_effect = [relevances, []]
    repo.count_resolved_relevance_for_mapping.return_value = 1
    repo.update_mapping_status = AsyncMock()

    document_repo = AsyncMock()
    document_repo.get_existing_ids_in_collection.return_value = set()
    document_repo.get_doc_ids_by_uri_bulk.return_value = {"u-1": "doc-1"}
    document_repo.get_doc_ids_by_content_hash_bulk.return_value = {"h-2": ["doc-a", "doc-b"]}  # ambiguous
    document_repo.get_doc_ids_by_file_name_bulk.return_value = {}

    db_session = AsyncMock()
    progress = SimpleNamespace(set_collection_id=MagicMock(), send_update=AsyncMock())

    service = BenchmarkDatasetService(
        db_session=db_session,
        benchmark_dataset_repo=repo,
        collection_repo=AsyncMock(),
        document_repo=document_repo,
        operation_repo=AsyncMock(),
    )

    result = await service.resolve_mapping_with_progress(
        mapping_id=10,
        user_id=1,
        operation_uuid="op-1",
        progress_reporter=progress,
    )

    assert result["mapping_status"] == MappingStatus.PARTIAL.value

    stages = [call.args[1]["stage"] for call in progress.send_update.await_args_list]
    assert stages[0] == "starting"
    assert "loading_documents" in stages
    assert "resolving" in stages
    assert stages[-1] == "completed"
