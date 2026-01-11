from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from webui.tasks import parallel_ingestion as pi


def test_calculate_optimal_workers_uses_cpu_and_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    mem = SimpleNamespace(
        available=4096 * 1024 * 1024,
        total=8192 * 1024 * 1024,
        used=4096 * 1024 * 1024,
    )

    monkeypatch.setattr(pi.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(pi.psutil, "virtual_memory", lambda: mem)
    monkeypatch.setattr(pi, "_get_container_memory_limit_mb", lambda: None)
    monkeypatch.setattr(pi.os, "getloadavg", lambda: (0.0, 0.0, 0.0))

    assert pi.calculate_optimal_workers(max_workers=10, min_workers=2) == 4


def test_calculate_optimal_workers_reduces_on_high_load(monkeypatch: pytest.MonkeyPatch) -> None:
    mem = SimpleNamespace(
        available=4096 * 1024 * 1024,
        total=8192 * 1024 * 1024,
        used=4096 * 1024 * 1024,
    )

    monkeypatch.setattr(pi.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(pi.psutil, "virtual_memory", lambda: mem)
    monkeypatch.setattr(pi, "_get_container_memory_limit_mb", lambda: None)
    monkeypatch.setattr(pi.os, "getloadavg", lambda: (4.0, 0.0, 0.0))

    assert pi.calculate_optimal_workers(max_workers=10, min_workers=1) == 1


def test_get_container_memory_limit_mb_reads_cgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_open(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        if str(self) == "/sys/fs/cgroup/memory.max":
            return io.StringIO("104857600")
        raise FileNotFoundError

    monkeypatch.setattr(pi.Path, "open", fake_open)

    assert pi._get_container_memory_limit_mb() == pytest.approx(100.0)


def test_get_container_memory_limit_mb_unlimited(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_open(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        if str(self) == "/sys/fs/cgroup/memory.max":
            return io.StringIO("max")
        raise FileNotFoundError

    monkeypatch.setattr(pi.Path, "open", fake_open)

    assert pi._get_container_memory_limit_mb() is None


def test_chunkbatch_invariant() -> None:
    with pytest.raises(ValueError, match="ChunkBatch invariant violated"):
        pi.ChunkBatch(doc_id="doc", doc_identifier="doc", chunks=[{"text": "x"}], texts=[])


def test_embedding_result_invariants() -> None:
    with pytest.raises(ValueError, match="success=True but error is set"):
        pi.EmbeddingResult(
            doc_id="doc",
            doc_identifier="doc",
            chunks=[{"text": "x"}],
            embeddings=[[0.1]],
            success=True,
            error="boom",
        )

    with pytest.raises(ValueError, match="embeddings != .*chunks"):
        pi.EmbeddingResult(
            doc_id="doc",
            doc_identifier="doc",
            chunks=[{"text": "x"}],
            embeddings=[],
            success=True,
        )

    with pytest.raises(ValueError, match="success=False but error is empty"):
        pi.EmbeddingResult(
            doc_id="doc",
            doc_identifier="doc",
            chunks=[{"text": "x"}],
            embeddings=[],
            success=False,
        )


def test_extraction_result_invariants() -> None:
    with pytest.raises(ValueError, match="success=True must have exactly one of batch/skip_reason"):
        pi.ExtractionResult(success=True, batch=None, skip_reason=None)

    with pytest.raises(ValueError, match="success=False but error is empty"):
        pi.ExtractionResult(success=False, error=None)


@pytest.mark.asyncio()
async def test_queue_put_with_shutdown_raises() -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    await queue.put("occupied")
    stopped = asyncio.Event()
    stopped.set()

    with pytest.raises(RuntimeError, match="Downstream consumer stopped"):
        await pi._queue_put_with_shutdown(queue, "item", downstream_stopped=stopped)


@pytest.mark.asyncio()
async def test_best_effort_ignores_nonfatal() -> None:
    async def _boom():  # type: ignore[no-untyped-def]
        raise ValueError("nope")

    await pi._best_effort("cleanup", _boom())


@pytest.mark.asyncio()
async def test_extract_and_chunk_uses_preparsed_content() -> None:
    doc = SimpleNamespace(id="doc-1", file_path="/tmp/doc.txt", uri=None)
    new_doc_contents = {"doc-1": "Hello world"}
    extract_fn = MagicMock()

    chunking_service = AsyncMock()
    chunking_service.execute_ingestion_chunking.return_value = [{"text": "chunk 1"}]

    result = await pi.extract_and_chunk_document(
        doc=doc,
        extract_fn=extract_fn,
        chunking_service=chunking_service,
        collection={},
        executor_pool=None,
        new_doc_contents=new_doc_contents,
    )

    assert result.success is True
    assert result.batch is not None
    assert result.batch.texts == ["chunk 1"]
    extract_fn.assert_not_called()


@pytest.mark.asyncio()
async def test_extract_and_chunk_handles_no_text_extracted() -> None:
    doc = SimpleNamespace(id="doc-2", file_path="/tmp/doc.txt", uri=None)
    extract_fn = MagicMock(return_value=[])
    chunking_service = AsyncMock()

    result = await pi.extract_and_chunk_document(
        doc=doc,
        extract_fn=extract_fn,
        chunking_service=chunking_service,
        collection={},
        executor_pool=None,
        new_doc_contents={},
    )

    assert result.success is True
    assert result.skip_reason == "no_text_extracted"
    chunking_service.execute_ingestion_chunking.assert_not_called()


@pytest.mark.asyncio()
async def test_extract_and_chunk_handles_empty_content() -> None:
    doc = SimpleNamespace(id="doc-3", file_path="/tmp/doc.txt", uri=None)
    extract_fn = MagicMock(return_value=[("   ", {})])
    chunking_service = AsyncMock()

    result = await pi.extract_and_chunk_document(
        doc=doc,
        extract_fn=extract_fn,
        chunking_service=chunking_service,
        collection={},
        executor_pool=None,
        new_doc_contents={},
    )

    assert result.success is True
    assert result.skip_reason == "empty_content"


@pytest.mark.asyncio()
async def test_extract_and_chunk_handles_no_chunks_created() -> None:
    doc = SimpleNamespace(id="doc-4", file_path="/tmp/doc.txt", uri=None)
    extract_fn = MagicMock(return_value=[("hello", {})])
    chunking_service = AsyncMock()
    chunking_service.execute_ingestion_chunking.return_value = []

    result = await pi.extract_and_chunk_document(
        doc=doc,
        extract_fn=extract_fn,
        chunking_service=chunking_service,
        collection={},
        executor_pool=None,
        new_doc_contents={},
    )

    assert result.success is True
    assert result.skip_reason == "no_chunks_created"


@pytest.mark.asyncio()
async def test_extract_and_chunk_handles_chunk_missing_text() -> None:
    doc = SimpleNamespace(id="doc-5", file_path="/tmp/doc.txt", uri=None)
    extract_fn = MagicMock(return_value=[("hello", {})])
    chunking_service = AsyncMock()
    chunking_service.execute_ingestion_chunking.return_value = [{"content": None}]

    result = await pi.extract_and_chunk_document(
        doc=doc,
        extract_fn=extract_fn,
        chunking_service=chunking_service,
        collection={},
        executor_pool=None,
        new_doc_contents={},
    )

    assert result.success is False
    assert result.error is not None
