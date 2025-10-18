"""Async coverage for :mod:`packages.webui.chunking_tasks` orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from packages.shared.database.models import CollectionStatus, DocumentStatus, OperationStatus, OperationType
from packages.webui.api.chunking_exceptions import (
    ChunkingMemoryError,
    ChunkingPartialFailureError,
    ChunkingTimeoutError,
)
from packages.webui.chunking_tasks import _process_chunking_operation_async
from packages.webui.services.progress_manager import ProgressSendResult


class _ManagerStub:
    def __init__(self) -> None:
        self.batch_size = 2
        self.check_calls: list[tuple[str, str]] = []
        self.monitor_calls: list[dict[str, Any]] = []
        self.monitor_exc: Exception | None = None

    async def check_resource_limits(self, *, operation_id: str, correlation_id: str) -> None:
        self.check_calls.append((operation_id, correlation_id))

    async def calculate_batch_size(self) -> int:
        return self.batch_size

    async def monitor_resources(
        self,
        *,
        process: Any,
        operation_id: str,
        initial_memory: int,
        initial_cpu_time: float,
        correlation_id: str,
    ) -> None:
        if self.monitor_exc:
            raise self.monitor_exc
        self.monitor_calls.append(
            {
                "operation_id": operation_id,
                "correlation_id": correlation_id,
                "initial_memory": initial_memory,
                "initial_cpu_time": initial_cpu_time,
                "process": process,
            }
        )


class _OperationRepoStub:
    def __init__(self, operation: SimpleNamespace) -> None:
        self._operation = operation
        self.status_calls: list[tuple[str, OperationStatus, dict[str, Any]]] = []
        self.task_ids: list[tuple[str, str]] = []

    async def get_by_uuid(self, operation_id: str) -> SimpleNamespace | None:
        return self._operation if operation_id == self._operation.id else None

    async def set_task_id(self, operation_id: str, task_id: str) -> None:
        self.task_ids.append((operation_id, task_id))

    async def update_status(self, operation_id: str, status: OperationStatus, **kwargs: Any) -> None:
        self.status_calls.append((operation_id, status, kwargs))


class _CollectionRepoStub:
    def __init__(self, collection: SimpleNamespace) -> None:
        self._collection = collection
        self.status_updates: list[tuple[str, CollectionStatus, dict[str, Any]]] = []

    async def get_by_uuid(self, collection_id: str) -> SimpleNamespace | None:
        return self._collection if collection_id == self._collection.id else None

    async def update_status(self, collection_id: str, status: CollectionStatus, **kwargs: Any) -> None:
        self.status_updates.append((collection_id, status, kwargs))


class _DocumentRepoStub:
    def __init__(self) -> None:
        self.by_id: dict[str, SimpleNamespace] = {}
        self.collection_docs: list[SimpleNamespace] = []
        self.status_updates: list[tuple[str, DocumentStatus, dict[str, Any]]] = []

    async def get_by_id(self, document_id: str) -> SimpleNamespace | None:
        return self.by_id.get(document_id)

    async def list_by_collection(
        self,
        collection_id: str,
        *,
        status: DocumentStatus | None,
        limit: int,
    ) -> tuple[list[SimpleNamespace], int]:
        _ = (collection_id, status, limit)
        return self.collection_docs, len(self.collection_docs)

    async def update_status(self, document_id: str, status: DocumentStatus, **kwargs: Any) -> None:
        self.status_updates.append((document_id, status, kwargs))


class _ChunkRepoStub:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def create_chunks_bulk(self, rows: list[dict[str, Any]]) -> None:
        self.rows.extend(rows)


class _ChunkingServiceStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result: dict[str, Any] = {
            "chunks": [
                {"text": "chunk-a", "metadata": {"part": 1}, "chunk_index": 0},
                {"content": "chunk-b", "metadata": "tag"},
            ],
            "stats": {"total": 2},
        }
        self.failures_by_document: dict[str, Exception] = {}
        self.after_call: Callable[[str], None] | None = None

    async def execute_ingestion_chunking(self, *, document_id: str, **kwargs: Any) -> dict[str, Any]:
        if document_id in self.failures_by_document:
            raise self.failures_by_document[document_id]
        self.calls.append({"document_id": document_id, **kwargs})
        if self.after_call:
            self.after_call(document_id)
        return self.result


class _FakeSession:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0
        self.flushes = 0

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollbacks += 1

    async def flush(self) -> None:
        self.flushes += 1


class _SessionContext:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        _ = (exc_type, exc, tb)
        return False


class _TaskStub:
    def __init__(self, manager: _ManagerStub) -> None:
        self._operation_manager = manager
        self._graceful_shutdown = False

    def _ensure_operation_manager(self) -> _ManagerStub:
        return self._operation_manager


@dataclass
class _ChunkingTestContext:
    operation_id: str
    celery_task: _TaskStub
    manager: _ManagerStub
    operation: SimpleNamespace
    collection: SimpleNamespace
    operation_repo: _OperationRepoStub
    collection_repo: _CollectionRepoStub
    document_repo: _DocumentRepoStub
    chunk_repo: _ChunkRepoStub
    chunking_service: _ChunkingServiceStub
    db_session: _FakeSession
    extract_outputs: dict[str, list[tuple[str, dict[str, Any]]]] = field(default_factory=dict)
    progress_events: list[tuple[int, str]] = field(default_factory=list)


@pytest.fixture
def chunking_test_context(monkeypatch: pytest.MonkeyPatch) -> _ChunkingTestContext:
    manager = _ManagerStub()
    operation = SimpleNamespace(
        id="op-test",
        collection_id="col-test",
        meta={},
        config={},
        type=OperationType.INDEX,
    )
    collection = SimpleNamespace(
        id="col-test",
        name="Demo",
        chunking_strategy="basic",
        chunking_config={"chunk_size": 1024},
        chunk_size=1024,
        chunk_overlap=64,
        embedding_model="test-model",
    )

    operation_repo = _OperationRepoStub(operation)
    collection_repo = _CollectionRepoStub(collection)
    document_repo = _DocumentRepoStub()
    chunk_repo = _ChunkRepoStub()
    chunking_service = _ChunkingServiceStub()
    db_session = _FakeSession()
    task = _TaskStub(manager)

    progress_events: list[tuple[int, str]] = []

    class _ProgressManagerStub:
        def send_sync_update(self, payload, **kwargs):  # noqa: ANN001
            _ = kwargs
            progress_events.append((payload.progress, payload.message))
            return ProgressSendResult.SENT

    monkeypatch.setattr("packages.webui.chunking_tasks._progress_update_manager", None, raising=False)
    monkeypatch.setattr("packages.webui.chunking_tasks.get_progress_update_manager", lambda: _ProgressManagerStub())
    monkeypatch.setattr("packages.webui.chunking_tasks.get_redis_client", lambda: SimpleNamespace())

    class _ProcessStub:
        @staticmethod
        def memory_info() -> SimpleNamespace:
            return SimpleNamespace(rss=42_000)

        @staticmethod
        def cpu_times() -> SimpleNamespace:
            return SimpleNamespace(user=1.0, system=0.5)

    monkeypatch.setattr("packages.webui.chunking_tasks.psutil.Process", lambda: _ProcessStub())

    async def _fake_initialize() -> None:
        return None

    monkeypatch.setattr("packages.webui.chunking_tasks.pg_connection_manager._sessionmaker", object(), raising=False)
    monkeypatch.setattr("packages.webui.chunking_tasks.pg_connection_manager.initialize", _fake_initialize)

    session_context = _SessionContext(db_session)

    def _session_factory() -> _SessionContext:
        return session_context

    monkeypatch.setattr("packages.webui.chunking_tasks.AsyncSessionLocal", _session_factory)
    monkeypatch.setattr("packages.webui.chunking_tasks.OperationRepository", lambda *args, **kwargs: operation_repo)
    monkeypatch.setattr("packages.webui.chunking_tasks.CollectionRepository", lambda *args, **kwargs: collection_repo)
    monkeypatch.setattr("packages.webui.chunking_tasks.DocumentRepository", lambda *args, **kwargs: document_repo)
    monkeypatch.setattr("packages.webui.chunking_tasks.ChunkRepository", lambda *args, **kwargs: chunk_repo)

    def _default_extract(file_path: str) -> list[tuple[str, dict[str, Any]]]:
        return [(f"content:{file_path}", {"path": file_path})]

    extract_outputs: dict[str, list[tuple[str, dict[str, Any]]]] = {}

    def _extract(file_path: str) -> list[tuple[str, dict[str, Any]]]:
        return extract_outputs.get(file_path, _default_extract(file_path))

    monkeypatch.setattr("packages.webui.chunking_tasks.extract_and_serialize_thread_safe", _extract)

    async def _resolve_service(*args, **kwargs) -> _ChunkingServiceStub:  # noqa: ANN001
        _ = (args, kwargs)
        return chunking_service

    monkeypatch.setattr("packages.webui.chunking_tasks.resolve_celery_chunking_service", _resolve_service)

    context = _ChunkingTestContext(
        operation_id=operation.id,
        celery_task=task,
        manager=manager,
        operation=operation,
        collection=collection,
        operation_repo=operation_repo,
        collection_repo=collection_repo,
        document_repo=document_repo,
        chunk_repo=chunk_repo,
        chunking_service=chunking_service,
        db_session=db_session,
        extract_outputs=extract_outputs,
        progress_events=progress_events,
    )

    return context


def _make_document(doc_id: str, *, status: DocumentStatus = DocumentStatus.PENDING, chunk_count: int = 0) -> SimpleNamespace:
    return SimpleNamespace(id=doc_id, file_path=f"/tmp/{doc_id}.txt", status=status, chunk_count=chunk_count)


@pytest.mark.asyncio
async def test_process_chunking_operation_async_success(chunking_test_context: _ChunkingTestContext) -> None:
    ctx = chunking_test_context

    doc_one = _make_document("doc-1")
    doc_two = _make_document("doc-2", status=DocumentStatus.COMPLETED, chunk_count=0)
    doc_skip = _make_document("doc-3", status=DocumentStatus.COMPLETED, chunk_count=3)

    ctx.document_repo.by_id = {d.id: d for d in (doc_one, doc_two, doc_skip)}
    ctx.operation.meta = {"document_ids": [doc_one.id, doc_two.id, doc_skip.id]}

    ctx.extract_outputs[doc_one.file_path] = [("text 1", {"section": 1})]
    ctx.extract_outputs[doc_two.file_path] = [("text 2", {"section": 2})]
    ctx.chunking_service.result = {
        "chunks": [
            {"text": "chunk-one", "metadata": {"tag": "a"}, "chunk_index": 5, "token_count": 10},
            {"content": "chunk-two", "metadata": "raw"},
        ],
        "stats": {"tokens": 20},
    }

    result = await _process_chunking_operation_async(
        operation_id=ctx.operation_id,
        correlation_id="corr-success",
        celery_task=ctx.celery_task,
    )

    assert result["operation_id"] == ctx.operation_id
    assert result["status"] == "success"
    assert result["chunks_created"] == 4
    assert result["documents_processed"] == 2
    assert result["documents_failed"] == 0
    assert result["duration_seconds"] >= 0

    assert len(ctx.chunk_repo.rows) == 4
    assert all(row["collection_id"] == "col-test" for row in ctx.chunk_repo.rows)
    assert ctx.chunk_repo.rows[1]["metadata"] == {"value": "raw"}
    assert ctx.document_repo.status_updates == [
        (doc_one.id, DocumentStatus.COMPLETED, {"chunk_count": 2}),
        (doc_two.id, DocumentStatus.COMPLETED, {"chunk_count": 2}),
    ]
    assert ctx.manager.check_calls == [(ctx.operation_id, "corr-success")]
    assert ctx.db_session.commits >= 2
    assert ctx.progress_events[0][0] == 0
    assert ctx.progress_events[-1][0] == 100
    assert ctx.operation.meta["chunks_created"] == 4
    assert not ctx.operation.meta["partial_failure"]


@pytest.mark.asyncio
async def test_process_chunking_operation_async_returns_when_no_documents(chunking_test_context: _ChunkingTestContext) -> None:
    ctx = chunking_test_context
    ctx.operation.meta = {}
    ctx.document_repo.collection_docs = []

    result = await _process_chunking_operation_async(
        operation_id=ctx.operation_id,
        correlation_id="corr-empty",
        celery_task=ctx.celery_task,
    )

    assert result["operation_id"] == ctx.operation_id
    assert result["status"] == "success"
    assert result["chunks_created"] == 0
    assert result["documents_processed"] == 0
    assert result["documents_failed"] == 0
    assert result["duration_seconds"] >= 0

    assert ctx.db_session.commits == 1
    assert ctx.chunk_repo.rows == []


@pytest.mark.asyncio
async def test_process_chunking_operation_async_handles_partial_failure(chunking_test_context: _ChunkingTestContext) -> None:
    ctx = chunking_test_context

    doc_one = _make_document("doc-1")
    doc_two = _make_document("doc-2")

    ctx.document_repo.by_id = {doc_one.id: doc_one, doc_two.id: doc_two}
    ctx.operation.meta = {"document_ids": [doc_one.id, doc_two.id]}

    ctx.chunking_service.failures_by_document[doc_one.id] = ChunkingPartialFailureError(
        detail="partial",
        operation_id=ctx.operation_id,
        total_documents=2,
        failed_documents=[doc_one.id],
        successful_chunks=1,
    )

    result = await _process_chunking_operation_async(
        operation_id=ctx.operation_id,
        correlation_id="corr-partial",
        celery_task=ctx.celery_task,
    )

    assert result["status"] == "partial_success"
    assert result["documents_failed"] == 1
    assert ctx.operation.meta["failed_documents"] == [doc_one.id]
    assert ctx.operation.meta["partial_failure"] is True
    assert ctx.document_repo.status_updates[-1][0] == doc_two.id
    assert all(update[0] != doc_one.id for update in ctx.document_repo.status_updates)


@pytest.mark.asyncio
async def test_process_chunking_operation_async_honors_graceful_shutdown(chunking_test_context: _ChunkingTestContext) -> None:
    ctx = chunking_test_context

    doc_one = _make_document("doc-1")
    doc_two = _make_document("doc-2")
    ctx.document_repo.by_id = {doc_one.id: doc_one, doc_two.id: doc_two}
    ctx.operation.meta = {"document_ids": [doc_one.id, doc_two.id]}

    def _trigger_shutdown(_: str) -> None:
        ctx.chunking_service.after_call = None
        ctx.celery_task._graceful_shutdown = True

    ctx.chunking_service.after_call = _trigger_shutdown

    result = await _process_chunking_operation_async(
        operation_id=ctx.operation_id,
        correlation_id="corr-shutdown",
        celery_task=ctx.celery_task,
    )

    assert result["status"] == "partial_success"
    assert result["documents_processed"] == 1
    assert ctx.operation.meta["partial_failure"] is True
    assert ctx.progress_events[-1][0] == 100


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_factory",
    [
        lambda op_id: ChunkingMemoryError("memory", operation_id=op_id),
        lambda op_id: ChunkingTimeoutError("timeout", operation_id=op_id, elapsed_time=1, timeout_limit=1),
    ],
)
async def test_process_chunking_operation_async_raises_on_resource_monitoring(
    chunking_test_context: _ChunkingTestContext,
    exception_factory: Callable[[str], Exception],
) -> None:
    ctx = chunking_test_context

    doc_one = _make_document("doc-1")
    ctx.document_repo.by_id = {doc_one.id: doc_one}
    ctx.operation.meta = {"document_ids": [doc_one.id]}

    ctx.manager.monitor_exc = exception_factory(ctx.operation_id)

    with pytest.raises(type(ctx.manager.monitor_exc)):
        await _process_chunking_operation_async(
            operation_id=ctx.operation_id,
            correlation_id="corr-resource",
            celery_task=ctx.celery_task,
        )

    assert ctx.db_session.rollbacks == 1
    assert ctx.chunk_repo.rows == []


@pytest.mark.asyncio
async def test_process_chunking_operation_async_marks_failed_documents(chunking_test_context: _ChunkingTestContext) -> None:
    ctx = chunking_test_context

    doc_one = _make_document("doc-1")
    doc_two = _make_document("doc-2")
    ctx.document_repo.by_id = {doc_one.id: doc_one, doc_two.id: doc_two}
    ctx.operation.meta = {"document_ids": [doc_one.id, doc_two.id]}

    ctx.chunking_service.failures_by_document[doc_one.id] = ValueError("boom")

    result = await _process_chunking_operation_async(
        operation_id=ctx.operation_id,
        correlation_id="corr-error",
        celery_task=ctx.celery_task,
    )

    assert result["status"] == "partial_success"
    assert result["documents_failed"] == 1
    assert ctx.operation.meta["failed_documents"] == [doc_one.id]
    assert len(ctx.chunk_repo.rows) == 2
    assert (
        (doc_one.id, DocumentStatus.FAILED)
        in [(doc_id, status) for doc_id, status, _ in ctx.document_repo.status_updates]
    )
    assert (
        (doc_two.id, DocumentStatus.COMPLETED)
        in [(doc_id, status) for doc_id, status, _ in ctx.document_repo.status_updates]
    )
    failed_entry = next(
        kwargs for doc_id, status, kwargs in ctx.document_repo.status_updates if doc_id == doc_one.id
    )
    assert failed_entry["error_message"] == "boom"
