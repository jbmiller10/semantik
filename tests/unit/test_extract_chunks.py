"""Unit tests for extract_chunks.py document chunking and extraction logic."""

import pytest

from shared.text_processing.chunking import TokenChunker


@pytest.fixture()
def sample_text() -> str:
    """Provide sample text for testing."""
    return (
        "This is the first sentence of our test document. "
        "It contains multiple sentences that will be used for testing. "
        "The chunker should split this text appropriately. "
        "We need enough content to test overlapping chunks. "
        "This is additional content to ensure we have sufficient text. "
        "More sentences are being added here. "
        "The quick brown fox jumps over the lazy dog. "
        "Python is a great programming language for many tasks. "
        "Testing is an important part of software development. "
        "We should always write comprehensive tests for our code."
    )


@pytest.fixture()
def short_text() -> str:
    """Provide short text that fits in a single chunk."""
    return "This is a short text that should fit in a single chunk."


@pytest.fixture()
def empty_text() -> str:
    """Provide empty text for edge case testing."""
    return ""


class TestTokenChunker:
    """Test suite for TokenChunker class."""

    def test_chunker_basic_functionality(self, sample_text: str) -> None:
        """Test basic chunking functionality with default parameters."""
        chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
        doc_id = "test_doc_001"

        chunks = chunker.chunk_text(sample_text, doc_id)

        # Verify we get multiple chunks
        assert len(chunks) > 1, "Should produce multiple chunks for long text"

        # Verify first chunk structure
        first_chunk = chunks[0]
        assert first_chunk["doc_id"] == doc_id
        assert first_chunk["chunk_id"] == f"{doc_id}_0000"
        assert isinstance(first_chunk["text"], str)
        assert len(first_chunk["text"]) > 0
        assert first_chunk["token_count"] > 0
        assert first_chunk["start_token"] == 0
        assert first_chunk["end_token"] > 0

        # Verify chunk IDs are sequential
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_id"] == f"{doc_id}_{i:04d}"

    def test_chunker_overlap(self, sample_text: str) -> None:
        """Test that chunks have the specified overlap."""
        chunk_size = 30
        chunk_overlap = 10
        chunker = TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        doc_id = "test_doc_002"

        chunks = chunker.chunk_text(sample_text, doc_id)

        # Verify overlap between consecutive chunks
        if len(chunks) > 1:
            # The overlap should be approximately chunk_overlap tokens
            # Check that the start of the second chunk is before the end of the first chunk
            first_chunk_end = chunks[0]["end_token"]
            second_chunk_start = chunks[1]["start_token"]

            # The overlap should be close to chunk_overlap
            actual_overlap = first_chunk_end - second_chunk_start
            assert actual_overlap > 0, "Chunks should overlap"
            # Allow some flexibility due to sentence boundary adjustments
            assert actual_overlap <= chunk_overlap + 5, f"Overlap {actual_overlap} exceeds expected {chunk_overlap}"

    def test_chunker_edge_cases(self, short_text: str, empty_text: str) -> None:
        """Test edge cases: text shorter than chunk_size and empty text."""
        chunker = TokenChunker(chunk_size=100, chunk_overlap=20)
        doc_id = "test_doc_003"

        # Test with short text (should produce one chunk)
        short_chunks = chunker.chunk_text(short_text, doc_id)
        assert len(short_chunks) == 1, "Short text should produce exactly one chunk"
        assert short_chunks[0]["text"] == short_text.strip()
        assert short_chunks[0]["chunk_id"] == f"{doc_id}_0000"

        # Test with empty text (should produce zero chunks)
        empty_chunks = chunker.chunk_text(empty_text, doc_id)
        assert len(empty_chunks) == 0, "Empty text should produce zero chunks"

        # Test with whitespace-only text
        whitespace_chunks = chunker.chunk_text("   \n\t  ", doc_id)
        assert len(whitespace_chunks) == 0, "Whitespace-only text should produce zero chunks"

    def test_chunker_with_metadata(self, sample_text: str) -> None:
        """Test that metadata is preserved in chunks."""
        chunker = TokenChunker(chunk_size=50, chunk_overlap=10)
        doc_id = "test_doc_004"
        metadata = {"source": "test", "page_number": 1, "author": "Test Author"}

        chunks = chunker.chunk_text(sample_text, doc_id, metadata)

        # Verify all chunks have the metadata
        for chunk in chunks:
            assert "metadata" in chunk
            assert chunk["metadata"] == metadata

    def test_chunker_invalid_parameters(self) -> None:
        """Test chunker behavior with invalid parameters."""
        # Test with chunk_overlap >= chunk_size (should adjust overlap)
        chunker = TokenChunker(chunk_size=50, chunk_overlap=60)
        assert chunker.chunk_overlap == 25, "Overlap should be adjusted to chunk_size/2"

        # Test with negative chunk_size (should raise error)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TokenChunker(chunk_size=-10, chunk_overlap=5)

        # Test with zero chunk_size (should raise error)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TokenChunker(chunk_size=0, chunk_overlap=5)
