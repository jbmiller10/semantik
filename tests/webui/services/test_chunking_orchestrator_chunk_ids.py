"""Regression tests for ingestion chunk_id assignment."""

import copy

import pytest
from webui.services.chunking import (
    ChunkingCache,
    ChunkingConfigManager,
    ChunkingMetrics,
    ChunkingProcessor,
    ChunkingValidator,
)
from webui.services.chunking.orchestrator import ChunkingOrchestrator


def _orchestrator() -> ChunkingOrchestrator:
    return ChunkingOrchestrator(
        processor=ChunkingProcessor(),
        cache=ChunkingCache(redis_client=None),
        metrics=ChunkingMetrics(),
        validator=ChunkingValidator(),
        config_manager=ChunkingConfigManager(profile_repo=None),
    )


@pytest.mark.asyncio()
async def test_execute_ingestion_chunking_assigns_doc_scoped_ids():
    orchestrator = _orchestrator()

    chunks = await orchestrator.execute_ingestion_chunking(
        content="hello world this is a chunk id test",
        strategy="character",
        config={"chunk_size": 10, "chunk_overlap": 0},
        metadata={"document_id": "doc123"},
    )

    assert chunks, "expected chunking to yield results"
    for i, chunk in enumerate(chunks):
        assert "chunk_id" in chunk, "chunk_id should be present on each chunk"
        assert chunk["chunk_id"].startswith("doc123_")
        assert chunk["chunk_index"] == i
        assert chunk["metadata"]["chunk_id"] == chunk["chunk_id"]
        assert chunk["metadata"]["chunk_index"] == i
        assert chunk["metadata"]["document_id"] == "doc123"


@pytest.mark.asyncio()
async def test_hierarchical_ids_rewritten_consistently(monkeypatch):
    orchestrator = _orchestrator()

    sample_chunks = [
        {
            "chunk_id": "old_parent",
            "chunk_index": 0,
            "text": "parent",
            "metadata": {"chunk_id": "old_parent"},
        },
        {
            "chunk_id": "child_a",
            "chunk_index": 1,
            "text": "child",
            "metadata": {
                "chunk_id": "child_a",
                "parent_chunk_id": "old_parent",
                "child_chunk_ids": ["leaf_a"],
                "custom_attributes": {"parent_chunk_id": "old_parent", "child_chunk_ids": ["leaf_a"]},
            },
        },
        {
            "chunk_id": "leaf_a",
            "chunk_index": 2,
            "text": "leaf",
            "metadata": {"chunk_id": "leaf_a", "parent_chunk_id": "child_a"},
        },
    ]

    async def fake_process_document(*_args, **_kwargs):
        return copy.deepcopy(sample_chunks)

    monkeypatch.setattr(orchestrator.processor, "process_document", fake_process_document)

    chunks = await orchestrator.execute_ingestion_chunking(
        content="ignored",
        strategy="hierarchical",
        config={},
        metadata={"document_id": "doc99"},
    )

    assert {c["chunk_id"] for c in chunks} == {
        "doc99_0000",
        "doc99_0001",
        "doc99_0002",
    }

    child = chunks[1]["metadata"]
    leaf = chunks[2]["metadata"]

    assert child["parent_chunk_id"] == "doc99_0000"
    assert child["child_chunk_ids"] == ["doc99_0002"]
    assert child["custom_attributes"]["parent_chunk_id"] == "doc99_0000"
    assert child["custom_attributes"]["child_chunk_ids"] == ["doc99_0002"]
    assert leaf["parent_chunk_id"] == "doc99_0001"
