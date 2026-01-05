"""Unit tests for extract_chunks.py document chunking and extraction logic."""

from unittest.mock import patch

import pytest

from shared.text_processing.chunking import TokenChunker
from shared.text_processing.extraction import extract_and_serialize


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


class TestExtractAndSerialize:
    """Test suite for extract_and_serialize function.

    Note: These tests mock parse_document_content since extract_and_serialize
    now reads the file first before calling the parsing function.
    """

    def test_metadata_preservation(self, tmp_path) -> None:
        """Test that metadata from parse_document_content is preserved."""
        # Create a temporary file
        test_file = tmp_path / "document.pdf"
        test_file.write_bytes(b"fake pdf content")

        # Mock parse_document_content to return expected results
        with patch("shared.text_processing.extraction.parse_document_content") as mock_parse:
            mock_parse.return_value = [
                (
                    "This is page 1 content.",
                    {
                        "filename": "document.pdf",
                        "file_type": "pdf",
                        "page_number": 1,
                        "element_type": "Title",
                    },
                ),
                (
                    "This is page 2 content.",
                    {
                        "filename": "document.pdf",
                        "file_type": "pdf",
                        "page_number": 2,
                        "element_type": "NarrativeText",
                        "has_coordinates": "True",
                    },
                ),
            ]

            results = extract_and_serialize(str(test_file))

            # Verify results
            assert len(results) == 2, "Should have 2 non-empty results"

            # Check first element
            text1, metadata1 = results[0]
            assert text1 == "This is page 1 content."
            assert metadata1["filename"] == "document.pdf"
            assert metadata1["file_type"] == "pdf"
            assert metadata1["page_number"] == 1
            assert metadata1["element_type"] == "Title"
            assert "has_coordinates" not in metadata1

            # Check second element
            text2, metadata2 = results[1]
            assert text2 == "This is page 2 content."
            assert metadata2["page_number"] == 2
            assert metadata2["element_type"] == "NarrativeText"
            assert metadata2["has_coordinates"] == "True"

    def test_text_concatenation(self, tmp_path) -> None:
        """Test that text from all elements is correctly extracted."""
        # Create a temporary file - use .pdf to ensure parse_document_content is called
        test_file = tmp_path / "document.pdf"
        test_file.write_bytes(b"fake pdf content")

        with patch("shared.text_processing.extraction.parse_document_content") as mock_parse:
            # Create mock return values
            expected_results = [(f"This is content from element {i}.", {"page_number": i + 1}) for i in range(5)]
            mock_parse.return_value = expected_results

            # Call extract_and_serialize
            results = extract_and_serialize(str(test_file))

            # Verify all texts are extracted
            assert len(results) == 5
            for i, (text, metadata) in enumerate(results):
                assert text == f"This is content from element {i}."
                assert metadata["page_number"] == i + 1

    def test_element_without_metadata(self, tmp_path) -> None:
        """Test handling of elements without full metadata."""
        # Create a temporary file - use .pdf to ensure parse_document_content is called
        test_file = tmp_path / "document.pdf"
        test_file.write_bytes(b"fake pdf content")

        with patch("shared.text_processing.extraction.parse_document_content") as mock_parse:
            # Return element with minimal metadata (no page_number)
            mock_parse.return_value = [
                (
                    "Content without page metadata",
                    {"filename": "document.pdf", "file_type": "pdf"},
                )
            ]

            # Call extract_and_serialize
            results = extract_and_serialize(str(test_file))

            # Should still work but with minimal metadata
            assert len(results) == 1
            text, metadata = results[0]
            assert text == "Content without page metadata"
            assert metadata["filename"] == "document.pdf"
            assert metadata["file_type"] == "pdf"
            # When element has no page_number, it's not in the metadata
            assert "page_number" not in metadata

    def test_partition_error_handling(self, tmp_path) -> None:
        """Test error handling when parse_document_content fails."""
        # Create a temporary file
        test_file = tmp_path / "document.pdf"
        test_file.write_bytes(b"fake content")

        with patch("shared.text_processing.extraction.parse_document_content") as mock_parse:
            mock_parse.side_effect = Exception("Partition failed")

            # Should raise the exception
            with pytest.raises(Exception, match="Partition failed"):
                extract_and_serialize(str(test_file))

    def test_file_type_extraction(self, tmp_path) -> None:
        """Test that file type is correctly extracted from filepath."""
        # Test various file extensions using mock
        with patch("shared.text_processing.extraction.parse_document_content") as mock_parse:
            test_cases = [
                ("document.pdf", "pdf"),
                ("file.docx", "docx"),
                ("README.md", "md"),
                ("noextension", "unknown"),
            ]

            for filename, expected_type in test_cases:
                # Create temporary file for each test case
                test_file = tmp_path / filename
                test_file.write_bytes(b"test content")

                # Mock returns metadata with expected file_type
                mock_parse.return_value = [("Test content", {"file_type": expected_type})]

                results = extract_and_serialize(str(test_file))
                _, metadata = results[0]
                assert metadata["file_type"] == expected_type, f"Failed for {filename}"

    def test_page_number_continuity(self, tmp_path) -> None:
        """Test that page numbers maintain continuity when elements lack page metadata."""
        # Create a temporary file
        test_file = tmp_path / "document.pdf"
        test_file.write_bytes(b"fake content")

        with patch("shared.text_processing.extraction.parse_document_content") as mock_parse:
            # Mock returns elements with mixed page numbers
            # The page continuity logic is now in parse_document_content
            mock_parse.return_value = [
                ("Content 1", {"page_number": 5}),
                ("Content 2", {"page_number": 5}),  # Inherited from previous
                ("Content 3", {"page_number": 7}),
            ]

            results = extract_and_serialize(str(test_file))

            # Check page numbers
            assert results[0][1]["page_number"] == 5
            assert results[1][1]["page_number"] == 5  # Should inherit from previous
            assert results[2][1]["page_number"] == 7
