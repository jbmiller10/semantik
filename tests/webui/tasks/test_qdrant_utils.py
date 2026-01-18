"""Tests for Qdrant utility functions."""

from webui.tasks.qdrant_utils import build_chunk_point


def test_build_chunk_point_basic():
    """Test basic point construction."""
    chunk = {
        "chunk_id": "doc1_0001",
        "text": "Hello world",
        "metadata": {"source": "test.txt"},
    }

    point = build_chunk_point(
        collection_id="col-123",
        doc_id="doc-456",
        chunk=chunk,
        chunk_index=0,
        total_chunks=5,
        path="/test/file.txt",
        embedding=[0.1, 0.2, 0.3],
    )

    assert point.vector == [0.1, 0.2, 0.3]
    assert point.payload["collection_id"] == "col-123"
    assert point.payload["doc_id"] == "doc-456"
    assert point.payload["chunk_id"] == "doc1_0001"
    assert point.payload["path"] == "/test/file.txt"
    assert point.payload["content"] == "Hello world"
    assert point.payload["metadata"] == {"source": "test.txt"}
    assert point.payload["chunk_index"] == 0
    assert point.payload["total_chunks"] == 5


def test_build_chunk_point_content_fallback():
    """Test content extraction from 'content' key when 'text' missing."""
    chunk = {
        "chunk_id": "doc1_0001",
        "content": "Fallback content",
    }

    point = build_chunk_point(
        collection_id="col-123",
        doc_id="doc-456",
        chunk=chunk,
        chunk_index=0,
        total_chunks=1,
        path="/test.txt",
        embedding=[0.1],
    )

    assert point.payload["content"] == "Fallback content"


def test_build_chunk_point_empty_content():
    """Test empty string default when no content."""
    chunk = {"chunk_id": "doc1_0001"}

    point = build_chunk_point(
        collection_id="col-123",
        doc_id="doc-456",
        chunk=chunk,
        chunk_index=0,
        total_chunks=1,
        path="/test.txt",
        embedding=[0.1],
    )

    assert point.payload["content"] == ""
    assert point.payload["metadata"] == {}


def test_build_chunk_point_generates_unique_ids():
    """Test that each point gets a unique UUID."""
    chunk = {"chunk_id": "doc1_0001", "text": "test"}

    point1 = build_chunk_point(
        collection_id="col-123",
        doc_id="doc-456",
        chunk=chunk,
        chunk_index=0,
        total_chunks=1,
        path="/test.txt",
        embedding=[0.1],
    )

    point2 = build_chunk_point(
        collection_id="col-123",
        doc_id="doc-456",
        chunk=chunk,
        chunk_index=0,
        total_chunks=1,
        path="/test.txt",
        embedding=[0.1],
    )

    assert point1.id != point2.id


def test_build_chunk_point_text_takes_priority_over_content():
    """Test that 'text' key is preferred over 'content' key."""
    chunk = {
        "chunk_id": "doc1_0001",
        "text": "Primary text",
        "content": "Secondary content",
    }

    point = build_chunk_point(
        collection_id="col-123",
        doc_id="doc-456",
        chunk=chunk,
        chunk_index=0,
        total_chunks=1,
        path="/test.txt",
        embedding=[0.1],
    )

    assert point.payload["content"] == "Primary text"
