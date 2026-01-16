"""Cover the document-level chunking helper used by Celery tasks."""

import pytest

from shared.text_processing.parsers import ParseResult
from webui.chunking_tasks import _process_document_chunking


class _FakeChunkingService:
    def __init__(self, chunks):
        self._chunks = chunks

    async def execute_ingestion_chunking(self, **_kwargs):  # noqa: D401
        return self._chunks


class _FakeChunkRepo:
    def __init__(self):
        self.rows = None

    async def create_chunks_bulk(self, rows):  # noqa: D401
        self.rows = rows


class _FakeDocumentRepo:
    def __init__(self):
        self.updated = None

    async def update_status(self, doc_id, status, chunk_count=0):  # noqa: D401
        self.updated = (doc_id, status, chunk_count)


class _Doc:
    def __init__(self):
        self.id = "doc-1"
        self.file_path = "/tmp/doc-1.txt"


@pytest.mark.asyncio()
async def test_process_document_chunking_builds_stats(monkeypatch):
    chunks = [
        {"text": "hello", "metadata": {"fallback": True, "fallback_reason": "size"}},
        {"text": "world", "metadata": {}},
    ]
    chunk_repo = _FakeChunkRepo()
    document_repo = _FakeDocumentRepo()

    # Avoid file system access by patching extractor
    monkeypatch.setattr(
        "webui.chunking_tasks.parse_file_thread_safe",
        lambda _path: ParseResult(text="hello world", metadata={"source": "test"}),
    )

    created, stats = await _process_document_chunking(
        chunking_service=_FakeChunkingService(chunks),
        chunk_repo=chunk_repo,
        document_repo=document_repo,
        collection_payload={"id": "col-1", "chunking_strategy": "recursive", "chunking_config": {}},
        collection_id="col-1",
        document=_Doc(),
        correlation_id="corr-1",
    )

    assert created == 2
    assert stats["fallback"] is True
    assert stats["strategy_used"] == "recursive"
    assert chunk_repo.rows is not None
    assert document_repo.updated[2] == 2
