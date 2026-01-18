"""Unit tests for parallel ingestion helpers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shared.database.models import DocumentStatus
from shared.text_processing.parsers import ParseResult
from webui.tasks import parallel_ingestion as pi
from webui.tasks.parallel_ingestion import (
    ChunkBatch,
    EmbeddingResult,
    _best_effort,
    _queue_put_with_shutdown,
    extract_and_chunk_document,
)


class _FakeResponse:
    def __init__(self, embeddings):
        self._embeddings = embeddings
        self.status_code = 200
        self.text = "ok"

    def raise_for_status(self) -> None:  # pragma: no cover - behavior is no-op
        return None

    def json(self):
        return {"embeddings": self._embeddings}


class _FakeAsyncClient:
    def __init__(self, embeddings):
        self._response = _FakeResponse(embeddings)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *_args, **_kwargs):
        return self._response


class _DummySession:
    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


class _DummyDocumentRepo:
    def __init__(self) -> None:
        self.session = _DummySession()
        self.update_status = AsyncMock()


@pytest.mark.asyncio()
async def test_queue_put_with_shutdown_stops() -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    downstream = asyncio.Event()
    downstream.set()

    with pytest.raises(RuntimeError, match="Downstream consumer stopped"):
        await _queue_put_with_shutdown(queue, "item", downstream_stopped=downstream)


@pytest.mark.asyncio()
async def test_queue_put_with_shutdown_waits_for_space(monkeypatch: pytest.MonkeyPatch) -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    await queue.put("existing")

    monkeypatch.setattr(pi, "QUEUE_PUT_POLL_INTERVAL_SECONDS", 0.01)

    async def _release() -> None:
        await asyncio.sleep(0.02)
        await queue.get()
        queue.task_done()

    put_task = asyncio.create_task(_queue_put_with_shutdown(queue, "new"))
    await asyncio.gather(put_task, _release())

    assert queue.qsize() == 1
    assert await queue.get() == "new"


@pytest.mark.asyncio()
async def test_best_effort_swallows_exceptions() -> None:
    async def _boom() -> None:
        raise ValueError("boom")

    await _best_effort("cleanup", _boom())


@pytest.mark.asyncio()
async def test_best_effort_reraises_fatal() -> None:
    async def _boom() -> None:
        raise SystemExit(1)

    with pytest.raises(SystemExit):
        await _best_effort("cleanup", _boom())


def test_chunkbatch_invariant_mismatch() -> None:
    with pytest.raises(ValueError, match="ChunkBatch invariant violated"):
        ChunkBatch(doc_id="doc", doc_identifier="doc", chunks=[{"text": "a"}], texts=[])


def test_embeddingresult_invariant_success_with_error() -> None:
    with pytest.raises(ValueError, match="success=True but error is set"):
        EmbeddingResult(
            doc_id="doc",
            doc_identifier="doc",
            chunks=[{"text": "a"}],
            embeddings=[[0.1]],
            success=True,
            error="boom",
        )


def test_embeddingresult_invariant_failure_without_error() -> None:
    with pytest.raises(ValueError, match="success=False but error is empty"):
        EmbeddingResult(
            doc_id="doc",
            doc_identifier="doc",
            chunks=[{"text": "a"}],
            embeddings=[],
            success=False,
            error=None,
        )


@pytest.mark.asyncio()
async def test_extract_and_chunk_uses_cached_content() -> None:
    doc = SimpleNamespace(id="doc-1", file_path="/tmp/doc.txt", uri=None)
    new_doc_contents = {"doc-1": "Hello world"}
    chunking_service = AsyncMock()
    chunking_service.execute_ingestion_chunking.return_value = [
        {"text": "Hello", "chunk_id": "c1", "metadata": {"k": "v"}}
    ]

    result = await extract_and_chunk_document(
        doc=doc,
        extract_fn=lambda _: [],
        chunking_service=chunking_service,
        collection={"chunking_strategy": "recursive", "chunking_config": {}},
        executor_pool=None,
        new_doc_contents=new_doc_contents,
    )

    assert result.success is True
    assert result.batch is not None
    assert result.batch.texts == ["Hello"]


@pytest.mark.asyncio()
async def test_extract_and_chunk_handles_no_text_blocks() -> None:
    doc = SimpleNamespace(id="doc-2", file_path="/tmp/doc.txt", uri=None)

    result = await extract_and_chunk_document(
        doc=doc,
        extract_fn=lambda _: [],
        chunking_service=AsyncMock(),
        collection={},
        executor_pool=None,
        new_doc_contents={},
    )

    assert result.success is True
    assert result.skip_reason == "no_text_extracted"


@pytest.mark.asyncio()
async def test_extract_and_chunk_handles_no_chunks() -> None:
    doc = SimpleNamespace(id="doc-3", file_path="/tmp/doc.txt", uri=None)
    chunking_service = AsyncMock()
    chunking_service.execute_ingestion_chunking.return_value = []

    result = await extract_and_chunk_document(
        doc=doc,
        extract_fn=lambda _: ParseResult(text="content", metadata={}),
        chunking_service=chunking_service,
        collection={},
        executor_pool=None,
        new_doc_contents={},
    )

    assert result.success is True
    assert result.skip_reason == "no_chunks_created"


@pytest.mark.asyncio()
async def test_extract_and_chunk_rejects_missing_text() -> None:
    doc = SimpleNamespace(id="doc-4", file_path="/tmp/doc.txt", uri=None)
    chunking_service = AsyncMock()
    chunking_service.execute_ingestion_chunking.return_value = [{"chunk_id": "c1"}]

    result = await extract_and_chunk_document(
        doc=doc,
        extract_fn=lambda _: ParseResult(text="content", metadata={}),
        chunking_service=chunking_service,
        collection={},
        executor_pool=None,
        new_doc_contents={},
    )

    assert result.success is False
    assert "Chunk missing text" in (result.error or "")


@pytest.mark.asyncio()
async def test_embedding_worker_success(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = {"chunk_id": "c1", "text": "hello", "metadata": {}}
    batch = ChunkBatch(
        doc_id="doc-1",
        doc_identifier="doc-1",
        chunks=[chunk],
        texts=["hello"],
    )

    chunk_queue: asyncio.Queue = asyncio.Queue()
    result_queue: asyncio.Queue = asyncio.Queue()
    stopped = asyncio.Event()

    monkeypatch.setattr(pi, "_build_internal_api_headers", dict)
    monkeypatch.setattr(pi.httpx, "AsyncClient", lambda *args, **kwargs: _FakeAsyncClient([[0.1, 0.2]]))

    task = asyncio.create_task(
        pi.embedding_worker(
            chunk_queue=chunk_queue,
            result_queue=result_queue,
            embedding_model="model",
            quantization="float16",
            instruction=None,
            batch_size=1,
            num_producers=1,
            embedding_stopped=stopped,
        )
    )

    await chunk_queue.put(batch)
    await chunk_queue.put(None)

    await task

    result = await result_queue.get()
    assert result.success is True
    assert result.embeddings == [[0.1, 0.2]]
    assert result.doc_id == "doc-1"

    assert await result_queue.get() is None


@pytest.mark.asyncio()
async def test_embedding_worker_error_on_mismatched_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = {"chunk_id": "c1", "text": "hello", "metadata": {}}
    batch = ChunkBatch(
        doc_id="doc-2",
        doc_identifier="doc-2",
        chunks=[chunk],
        texts=["hello"],
    )

    chunk_queue: asyncio.Queue = asyncio.Queue()
    result_queue: asyncio.Queue = asyncio.Queue()
    stopped = asyncio.Event()

    monkeypatch.setattr(pi, "_build_internal_api_headers", dict)
    # Return zero embeddings to trigger mismatch error.
    monkeypatch.setattr(pi.httpx, "AsyncClient", lambda *args, **kwargs: _FakeAsyncClient([]))

    task = asyncio.create_task(
        pi.embedding_worker(
            chunk_queue=chunk_queue,
            result_queue=result_queue,
            embedding_model="model",
            quantization="float16",
            instruction=None,
            batch_size=1,
            num_producers=1,
            embedding_stopped=stopped,
        )
    )

    await chunk_queue.put(batch)
    await chunk_queue.put(None)

    await task

    result = await result_queue.get()
    assert result.success is False
    assert "Embedding response size mismatch" in (result.error or "")
    assert await result_queue.get() is None


@pytest.mark.asyncio()
async def test_result_processor_success(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = {"chunk_id": "c1", "text": "hello", "metadata": {}}
    result = EmbeddingResult(
        doc_id="doc-3",
        doc_identifier="doc-3",
        chunks=[chunk],
        embeddings=[[0.1, 0.2]],
        success=True,
    )

    result_queue: asyncio.Queue = asyncio.Queue()
    stats = {"processed": 0, "failed": 0, "skipped": 0, "vectors": 0}
    doc_repo = _DummyDocumentRepo()

    monkeypatch.setattr(pi, "_build_internal_api_headers", dict)
    monkeypatch.setattr(pi.httpx, "AsyncClient", lambda *args, **kwargs: _FakeAsyncClient({"ok": True}))

    task = asyncio.create_task(
        pi.result_processor(
            result_queue=result_queue,
            qdrant_collection_name="collection",
            collection_id="collection-id",
            collection={"id": "collection-id", "document_count": 0, "vector_count": 0, "total_size_bytes": 0},
            document_repo=doc_repo,
            collection_repo=None,
            stats=stats,
            stats_lock=asyncio.Lock(),
            db_lock=asyncio.Lock(),
            updater=None,
        )
    )

    await result_queue.put(result)
    await result_queue.put(None)

    await task

    doc_repo.update_status.assert_awaited()
    assert stats["processed"] == 1
    assert stats["vectors"] == 1

    call = doc_repo.update_status.await_args
    assert call.args[1] == DocumentStatus.COMPLETED


@pytest.mark.asyncio()
async def test_extraction_worker_marks_failed_on_extraction_result_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = SimpleNamespace(id="doc-fail", file_path="/tmp/doc.txt", uri=None)
    doc_queue: asyncio.Queue = asyncio.Queue()
    chunk_queue: asyncio.Queue = asyncio.Queue()
    doc_repo = _DummyDocumentRepo()
    stats = {"processed": 0, "failed": 0, "skipped": 0, "vectors": 0}

    async def _fake_extract_and_chunk_document(**_kwargs):
        return pi.ExtractionResult(success=False, error="boom")

    await doc_queue.put(doc)
    await doc_queue.put(None)

    monkeypatch.setattr(pi, "extract_and_chunk_document", _fake_extract_and_chunk_document)
    await pi.extraction_worker(
        worker_id=0,
        doc_queue=doc_queue,
        chunk_queue=chunk_queue,
        extract_fn=lambda _: [],
        chunking_service=AsyncMock(),
        collection={},
        executor_pool=None,
        new_doc_contents={},
        document_repo=doc_repo,
        stats=stats,
        stats_lock=asyncio.Lock(),
        db_lock=asyncio.Lock(),
        embedding_stopped=asyncio.Event(),
    )

    doc_repo.update_status.assert_awaited()
    assert stats["failed"] == 1

    update_call = doc_repo.update_status.await_args
    assert update_call.args[0] == "doc-fail"
    assert update_call.args[1] == DocumentStatus.FAILED

    # Worker signals downstream completion with a poison pill.
    assert await chunk_queue.get() is None


@pytest.mark.asyncio()
async def test_extraction_worker_marks_completed_on_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = SimpleNamespace(id="doc-skip", file_path="/tmp/doc.txt", uri=None)
    doc_queue: asyncio.Queue = asyncio.Queue()
    chunk_queue: asyncio.Queue = asyncio.Queue()
    doc_repo = _DummyDocumentRepo()
    stats = {"processed": 0, "failed": 0, "skipped": 0, "vectors": 0}

    async def _fake_extract_and_chunk_document(**_kwargs):
        return pi.ExtractionResult(success=True, skip_reason="no_text_extracted")

    await doc_queue.put(doc)
    await doc_queue.put(None)

    monkeypatch.setattr(pi, "extract_and_chunk_document", _fake_extract_and_chunk_document)
    await pi.extraction_worker(
        worker_id=0,
        doc_queue=doc_queue,
        chunk_queue=chunk_queue,
        extract_fn=lambda _: [],
        chunking_service=AsyncMock(),
        collection={},
        executor_pool=None,
        new_doc_contents={},
        document_repo=doc_repo,
        stats=stats,
        stats_lock=asyncio.Lock(),
        db_lock=asyncio.Lock(),
        embedding_stopped=asyncio.Event(),
    )

    doc_repo.update_status.assert_awaited()
    assert stats["skipped"] == 1

    update_call = doc_repo.update_status.await_args
    assert update_call.args[0] == "doc-skip"
    assert update_call.args[1] == DocumentStatus.COMPLETED
    assert update_call.kwargs["chunk_count"] == 0

    assert await chunk_queue.get() is None


@pytest.mark.asyncio()
async def test_extraction_worker_marks_failed_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = SimpleNamespace(id="doc-exc", file_path="/tmp/doc.txt", uri=None)
    doc_queue: asyncio.Queue = asyncio.Queue()
    chunk_queue: asyncio.Queue = asyncio.Queue()
    doc_repo = _DummyDocumentRepo()
    stats = {"processed": 0, "failed": 0, "skipped": 0, "vectors": 0}

    async def _boom(**_kwargs):
        raise ValueError("boom")

    await doc_queue.put(doc)
    await doc_queue.put(None)

    monkeypatch.setattr(pi, "extract_and_chunk_document", _boom)
    await pi.extraction_worker(
        worker_id=0,
        doc_queue=doc_queue,
        chunk_queue=chunk_queue,
        extract_fn=lambda _: [],
        chunking_service=AsyncMock(),
        collection={},
        executor_pool=None,
        new_doc_contents={},
        document_repo=doc_repo,
        stats=stats,
        stats_lock=asyncio.Lock(),
        db_lock=asyncio.Lock(),
        embedding_stopped=asyncio.Event(),
    )

    doc_repo.update_status.assert_awaited()
    assert stats["failed"] == 1

    update_call = doc_repo.update_status.await_args
    assert update_call.args[0] == "doc-exc"
    assert update_call.args[1] == DocumentStatus.FAILED
    assert "Extraction worker error:" in (update_call.kwargs.get("error_message") or "")

    assert await chunk_queue.get() is None


@pytest.mark.asyncio()
async def test_process_documents_parallel_orchestrates_workers_and_caps_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed: list[str] = []
    worker_started: set[int] = set()

    async def _fake_extraction_worker(
        worker_id: int,
        doc_queue: asyncio.Queue,
        chunk_queue: asyncio.Queue,
        **_kwargs,
    ) -> None:
        worker_started.add(worker_id)
        while True:
            doc = await doc_queue.get()
            doc_queue.task_done()
            if doc is None:
                break
            processed.append(doc.id)
        await chunk_queue.put(None)

    async def _fake_embedding_worker(
        chunk_queue: asyncio.Queue,
        result_queue: asyncio.Queue,
        num_producers: int,
        **_kwargs,
    ) -> None:
        producers_done = 0
        while producers_done < num_producers:
            item = await chunk_queue.get()
            chunk_queue.task_done()
            if item is None:
                producers_done += 1
        await result_queue.put(None)

    async def _fake_result_processor(result_queue: asyncio.Queue, **_kwargs) -> None:
        item = await result_queue.get()
        result_queue.task_done()
        assert item is None

    monkeypatch.setattr(pi, "extraction_worker", _fake_extraction_worker)
    monkeypatch.setattr(pi, "embedding_worker", _fake_embedding_worker)
    monkeypatch.setattr(pi, "result_processor", _fake_result_processor)

    docs = [
        SimpleNamespace(id="d1", file_path="/tmp/d1.txt", uri=None),
        SimpleNamespace(id="d2", file_path="/tmp/d2.txt", uri=None),
        SimpleNamespace(id="d3", file_path="/tmp/d3.txt", uri=None),
    ]

    stats = await pi.process_documents_parallel(
        documents=docs,
        extract_fn=lambda _: [],
        chunking_service=AsyncMock(),
        collection={"id": "collection-id"},
        executor_pool=None,
        new_doc_contents={},
        document_repo=_DummyDocumentRepo(),
        collection_repo=None,
        qdrant_collection_name="collection",
        embedding_model="model",
        quantization="float16",
        instruction=None,
        batch_size=2,
        num_extraction_workers=5,
        max_extraction_workers=2,
    )

    assert stats == {"processed": 0, "failed": 0, "skipped": 0, "vectors": 0}
    assert set(processed) == {"d1", "d2", "d3"}
    assert worker_started == {0, 1}
