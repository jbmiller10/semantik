"""Unit tests for deterministic helpers in :mod:`packages.webui.chunking_tasks`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from packages.shared.database.models import DocumentStatus
from packages.webui.chunking_tasks import (
    _build_chunk_rows,
    _collection_to_payload,
    _combine_text_blocks,
    _extract_document_ids,
    _resolve_documents_for_operation,
)


def test_collection_to_payload_merges_defaults() -> None:
    collection = {
        "id": "col-1",
        "name": "Example",
        "chunking_strategy": "basic",
        "chunking_config": {"size": 512},
        "vector_collection_id": "vc-123",
    }

    payload = _collection_to_payload(collection)

    assert payload["id"] == "col-1"
    assert payload["chunking_config"] == {"size": 512}
    assert payload["chunk_overlap"] == 200  # default
    assert payload["vector_store_name"] == "vc-123"


def test_extract_document_ids_deduplicates_from_config_and_meta() -> None:
    operation = SimpleNamespace(
        config={"document_ids": ["doc-1", {"id": "doc-2"}]},
        meta={
            "documents": [
                {"document_id": "doc-3"},
                "doc-4",
            ],
            "pending_document_ids": ["doc-2", "doc-5"],
        },
    )

    document_ids = _extract_document_ids(operation)

    assert document_ids == ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"]


def test_combine_text_blocks_merges_and_preserves_metadata() -> None:
    blocks = [
        ("First paragraph", {"lang": "en"}),
        ("   ", {"ignored": True}),
        ("Second paragraph", {"sentiment": "neutral"}),
    ]

    text, metadata = _combine_text_blocks(blocks)

    assert text == "First paragraph\n\nSecond paragraph"
    assert metadata == {"lang": "en", "ignored": True, "sentiment": "neutral"}


def test_build_chunk_rows_filters_blank_rows_and_normalizes_metadata() -> None:
    chunks = [
        {"text": "chunk one", "metadata": {"source": "pdf"}, "chunk_index": 3},
        {"text": "   ", "metadata": {"ignored": True}},
        {"content": "chunk two", "metadata": "tag"},
    ]

    rows = _build_chunk_rows("col-1", "doc-9", chunks)

    assert len(rows) == 2
    assert rows[0]["chunk_index"] == 3
    assert rows[1]["metadata"] == {"value": "tag"}
    assert all("document_id" in row and "collection_id" in row for row in rows)


class _DocumentRepoStub:
    def __init__(self) -> None:
        self.by_id: dict[str, SimpleNamespace] = {}
        self.collection_docs: list[SimpleNamespace] = []

    async def get_by_id(self, document_id: str) -> SimpleNamespace | None:
        return self.by_id.get(document_id)

    async def list_by_collection(self, *_: object, **__: object) -> tuple[list[SimpleNamespace], int]:
        return self.collection_docs, len(self.collection_docs)


@pytest.mark.asyncio
async def test_resolve_documents_for_operation_with_ids_filters_processed() -> None:
    doc_ready = SimpleNamespace(id="doc-1", status=DocumentStatus.PENDING, chunk_count=0)
    doc_completed = SimpleNamespace(id="doc-2", status=DocumentStatus.COMPLETED, chunk_count=0)
    doc_skipped = SimpleNamespace(id="doc-3", status=DocumentStatus.COMPLETED, chunk_count=4)

    repo = _DocumentRepoStub()
    repo.by_id = {
        doc_ready.id: doc_ready,
        doc_completed.id: doc_completed,
        doc_skipped.id: doc_skipped,
    }

    operation = SimpleNamespace(
        config={"document_ids": ["doc-1", "doc-2", "doc-3"]},
        meta={},
        collection_id="col-1",
    )

    documents = await _resolve_documents_for_operation(operation, repo)

    assert [doc.id for doc in documents] == ["doc-1", "doc-2"]


@pytest.mark.asyncio
async def test_resolve_documents_for_operation_uses_collection_listing() -> None:
    doc_pending = SimpleNamespace(id="doc-10", status=DocumentStatus.PENDING, chunk_count=0)
    doc_deleted = SimpleNamespace(id="doc-11", status=DocumentStatus.DELETED, chunk_count=0)

    repo = _DocumentRepoStub()
    repo.collection_docs = [doc_pending, doc_deleted]

    operation = SimpleNamespace(config={}, meta={}, collection_id="collection-7")

    documents = await _resolve_documents_for_operation(operation, repo)

    assert [doc.id for doc in documents] == ["doc-10"]
