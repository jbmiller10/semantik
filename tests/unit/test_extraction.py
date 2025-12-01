"""Unit tests for document extraction module."""

from unittest.mock import MagicMock, patch

import pytest

from shared.text_processing.extraction import (
    _extension_to_content_type,
    extract_and_serialize,
    extract_text,
    parse_document_content,
)


class TestExtensionToContentType:
    """Tests for _extension_to_content_type helper."""

    def test_txt_extension(self) -> None:
        """Test .txt maps to text/plain."""
        assert _extension_to_content_type(".txt") == "text/plain"

    def test_text_extension(self) -> None:
        """Test .text maps to text/plain."""
        assert _extension_to_content_type(".text") == "text/plain"

    def test_md_extension(self) -> None:
        """Test .md maps to text/markdown."""
        assert _extension_to_content_type(".md") == "text/markdown"

    def test_html_extension(self) -> None:
        """Test .html maps to text/html."""
        assert _extension_to_content_type(".html") == "text/html"

    def test_htm_extension(self) -> None:
        """Test .htm maps to text/html."""
        assert _extension_to_content_type(".htm") == "text/html"

    def test_unknown_extension(self) -> None:
        """Test unknown extension returns None."""
        assert _extension_to_content_type(".pdf") is None
        assert _extension_to_content_type(".docx") is None


class TestParseDocumentContent:
    """Tests for parse_document_content function."""

    def test_string_content_with_txt_extension(self) -> None:
        """Test parsing text string with .txt extension."""
        with patch("shared.text_processing.extraction.partition") as mock_partition:
            # Create mock element
            mock_element = MagicMock()
            mock_element.__str__ = MagicMock(return_value="Hello, world!")
            mock_element.metadata = MagicMock()
            mock_element.metadata.page_number = 1
            mock_element.metadata.category = "NarrativeText"
            mock_partition.return_value = [mock_element]

            result = parse_document_content("Hello, world!", ".txt")

            assert len(result) == 1
            text, metadata = result[0]
            assert text == "Hello, world!"
            assert metadata["file_type"] == "txt"
            assert metadata["page_number"] == 1
            assert metadata["element_type"] == "NarrativeText"

            # Verify partition was called with text parameter
            mock_partition.assert_called_once()
            call_kwargs = mock_partition.call_args.kwargs
            assert call_kwargs["text"] == "Hello, world!"
            assert call_kwargs["content_type"] == "text/plain"

    def test_bytes_content_with_pdf_extension(self) -> None:
        """Test parsing bytes with .pdf extension."""
        with patch("shared.text_processing.extraction.partition") as mock_partition:
            mock_element = MagicMock()
            mock_element.__str__ = MagicMock(return_value="PDF content")
            mock_element.metadata = MagicMock()
            mock_element.metadata.page_number = 1
            mock_partition.return_value = [mock_element]

            content = b"fake pdf bytes"
            result = parse_document_content(content, ".pdf")

            assert len(result) == 1
            text, metadata = result[0]
            assert text == "PDF content"
            assert metadata["file_type"] == "pdf"

            # Verify partition was called with file parameter (BytesIO)
            mock_partition.assert_called_once()
            call_kwargs = mock_partition.call_args.kwargs
            assert "file" in call_kwargs
            assert call_kwargs["metadata_filename"] == "document.pdf"

    def test_metadata_merging(self) -> None:
        """Test that provided metadata is merged into results."""
        with patch("shared.text_processing.extraction.partition") as mock_partition:
            mock_element = MagicMock()
            mock_element.__str__ = MagicMock(return_value="Content")
            mock_element.metadata = MagicMock()
            mock_element.metadata.page_number = 1
            mock_partition.return_value = [mock_element]

            custom_metadata = {"source": "test", "filename": "test.txt"}
            result = parse_document_content("Content", ".txt", metadata=custom_metadata)

            text, metadata = result[0]
            assert metadata["source"] == "test"
            assert metadata["filename"] == "test.txt"
            assert metadata["file_type"] == "txt"

    def test_extension_normalization_without_dot(self) -> None:
        """Test extension without leading dot is normalized."""
        with patch("shared.text_processing.extraction.partition") as mock_partition:
            mock_element = MagicMock()
            mock_element.__str__ = MagicMock(return_value="Content")
            mock_element.metadata = MagicMock()
            mock_partition.return_value = [mock_element]

            result = parse_document_content("Content", "txt")  # No dot

            text, metadata = result[0]
            assert metadata["file_type"] == "txt"

    def test_empty_elements_filtered(self) -> None:
        """Test that empty text elements are filtered out."""
        with patch("shared.text_processing.extraction.partition") as mock_partition:
            mock_element1 = MagicMock()
            mock_element1.__str__ = MagicMock(return_value="")  # Empty
            mock_element2 = MagicMock()
            mock_element2.__str__ = MagicMock(return_value="   ")  # Whitespace only
            mock_element3 = MagicMock()
            mock_element3.__str__ = MagicMock(return_value="Valid content")
            mock_element3.metadata = MagicMock()
            mock_element3.metadata.page_number = 1
            mock_partition.return_value = [mock_element1, mock_element2, mock_element3]

            result = parse_document_content("text", ".txt")

            assert len(result) == 1
            assert result[0][0] == "Valid content"

    def test_page_number_tracking(self) -> None:
        """Test that page numbers are tracked across elements."""
        with patch("shared.text_processing.extraction.partition") as mock_partition:
            # Element without page number (should use current_page=1)
            mock_element1 = MagicMock()
            mock_element1.__str__ = MagicMock(return_value="Page 1 content")
            mock_element1.metadata = MagicMock()
            mock_element1.metadata.page_number = None  # No explicit page

            # Element with page number 2
            mock_element2 = MagicMock()
            mock_element2.__str__ = MagicMock(return_value="Page 2 content")
            mock_element2.metadata = MagicMock()
            mock_element2.metadata.page_number = 2

            # Element without page (should inherit page 2)
            mock_element3 = MagicMock()
            mock_element3.__str__ = MagicMock(return_value="Still page 2")
            mock_element3.metadata = MagicMock()
            mock_element3.metadata.page_number = None

            mock_partition.return_value = [mock_element1, mock_element2, mock_element3]

            result = parse_document_content("text", ".txt")

            assert result[0][1]["page_number"] == 1
            assert result[1][1]["page_number"] == 2
            assert result[2][1]["page_number"] == 2

    def test_coordinates_metadata(self) -> None:
        """Test that coordinates presence is noted in metadata."""
        with patch("shared.text_processing.extraction.partition") as mock_partition:
            mock_element = MagicMock()
            mock_element.__str__ = MagicMock(return_value="Content")
            mock_element.metadata = MagicMock()
            mock_element.metadata.page_number = 1
            mock_element.metadata.coordinates = {"x": 0, "y": 0}
            mock_partition.return_value = [mock_element]

            result = parse_document_content("text", ".txt")

            assert result[0][1]["has_coordinates"] == "True"


class TestExtractAndSerialize:
    """Tests for extract_and_serialize function (regression tests)."""

    def test_reads_file_and_delegates(self, tmp_path) -> None:
        """Test that extract_and_serialize reads file and delegates to parse_document_content."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        with patch("shared.text_processing.extraction.parse_document_content") as mock_parse:
            mock_parse.return_value = [("Test content", {"file_type": "txt"})]

            result = extract_and_serialize(str(test_file))

            assert result == [("Test content", {"file_type": "txt"})]
            mock_parse.assert_called_once()
            # Verify content was read as bytes
            call_args = mock_parse.call_args
            assert call_args.kwargs["content"] == b"Test content"
            assert call_args.kwargs["file_extension"] == ".txt"
            assert call_args.kwargs["metadata"] == {"filename": "test.txt"}

    def test_file_not_found_raises(self) -> None:
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            extract_and_serialize("/nonexistent/file.txt")


class TestExtractText:
    """Tests for extract_text function (regression tests)."""

    def test_joins_text_parts(self, tmp_path) -> None:
        """Test that extract_text joins text parts with double newlines."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")

        with patch("shared.text_processing.extraction.extract_and_serialize") as mock_extract:
            mock_extract.return_value = [
                ("Part 1", {}),
                ("Part 2", {}),
                ("Part 3", {}),
            ]

            result = extract_text(str(test_file))

            assert result == "Part 1\n\nPart 2\n\nPart 3"

    def test_timeout_parameter_accepted(self, tmp_path) -> None:
        """Test that timeout parameter is accepted (for backward compatibility)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")

        with patch("shared.text_processing.extraction.extract_and_serialize") as mock_extract:
            mock_extract.return_value = [("Content", {})]

            # Should not raise - timeout is accepted but not used
            result = extract_text(str(test_file), timeout=60)

            assert result == "Content"
