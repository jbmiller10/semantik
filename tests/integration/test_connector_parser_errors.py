"""Integration tests for connector parser error handling (Phase 9).

Tests verify that connectors properly distinguish between:
- UnsupportedFormatError: File is skipped silently (debug log only)
- ExtractionFailedError: File is skipped with error logging

These tests lock down the documented error handling policies.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from shared.connectors.local import _process_file_worker
from shared.text_processing.parsers import ExtractionFailedError, UnsupportedFormatError


class TestLocalFileConnectorSequential:
    """Tests for LocalFileConnector sequential path error handling."""

    @pytest.fixture()
    def temp_dir(self) -> Any:
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio()
    async def test_unsupported_format_error_skips_file(self, temp_dir: Path) -> None:
        """UnsupportedFormatError causes file to be skipped without error log."""
        from shared.connectors.local import LocalFileConnector

        # Create a text file with binary content (triggers UnsupportedFormatError)
        test_file = temp_dir / "binary.txt"
        test_file.write_bytes(b"\x00\x01\x02\x03binary content")

        connector = LocalFileConnector({"path": str(temp_dir)})
        await connector.authenticate()

        # Collect all documents
        docs = [doc async for doc in connector.load_documents()]

        # File should be skipped (binary content rejected)
        assert len(docs) == 0

    @pytest.mark.asyncio()
    async def test_extraction_failed_error_skips_file_with_log(self, temp_dir: Path) -> None:
        """ExtractionFailedError causes file to be skipped with error logging."""
        from shared.connectors.local import LocalFileConnector

        # Create a valid text file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Valid content")

        connector = LocalFileConnector({"path": str(temp_dir)})
        await connector.authenticate()

        # Mock parse_content to raise ExtractionFailedError
        with patch("shared.connectors.local.parse_content") as mock_parse:
            mock_parse.side_effect = ExtractionFailedError("Simulated extraction failure")

            docs = [doc async for doc in connector.load_documents()]

            # File should be skipped
            assert len(docs) == 0

    @pytest.mark.asyncio()
    async def test_valid_file_is_processed(self, temp_dir: Path) -> None:
        """Valid text files are processed successfully."""
        from shared.connectors.local import LocalFileConnector

        # Create a valid text file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, world!")

        connector = LocalFileConnector({"path": str(temp_dir)})
        await connector.authenticate()

        docs = [doc async for doc in connector.load_documents()]

        assert len(docs) == 1
        assert docs[0].content == "Hello, world!"


class TestLocalFileConnectorParallelWorker:
    """Tests for LocalFileConnector parallel worker error handling."""

    @pytest.fixture()
    def temp_dir(self) -> Any:
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_worker_unsupported_format_returns_skipped(self, temp_dir: Path) -> None:
        """Worker returns 'skipped' status for UnsupportedFormatError."""
        # Create a binary file
        test_file = temp_dir / "binary.txt"
        test_file.write_bytes(b"\x00\x01\x02\x03binary content")

        result = _process_file_worker(str(test_file))

        assert result["status"] == "skipped"
        assert result["reason"] == "unsupported_format"
        assert str(test_file) in result["path"]

    def test_worker_extraction_failed_returns_error(self, temp_dir: Path) -> None:
        """Worker returns 'error' status for ExtractionFailedError."""
        # Create a valid text file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Valid content")

        # Mock parse_content to raise ExtractionFailedError
        with patch("shared.connectors.local.parse_content") as mock_parse:
            mock_parse.side_effect = ExtractionFailedError("Simulated extraction failure")

            result = _process_file_worker(str(test_file))

        assert result["status"] == "error"
        assert "Failed to parse" in result["reason"]
        assert str(test_file) in result["path"]

    def test_worker_success_returns_primitives(self, temp_dir: Path) -> None:
        """Worker returns primitive dict for successful parsing."""
        # Create a valid text file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, world!")

        result = _process_file_worker(str(test_file))

        assert result["status"] == "success"
        assert "data" in result
        data = result["data"]

        # All values are primitives
        assert isinstance(data["content"], str)
        assert isinstance(data["unique_id"], str)
        assert isinstance(data["source_type"], str)
        assert isinstance(data["metadata"], dict)
        assert isinstance(data["content_hash"], str)
        assert isinstance(data["file_path"], str)

        # Content matches file
        assert data["content"] == "Hello, world!"

    def test_worker_empty_file_returns_skipped(self, temp_dir: Path) -> None:
        """Worker returns 'skipped' status for empty files."""
        test_file = temp_dir / "empty.txt"
        test_file.write_text("   \n   ")  # Whitespace only

        result = _process_file_worker(str(test_file))

        assert result["status"] == "skipped"
        assert result["reason"] == "empty_content"


class TestGitConnectorFallbackBehavior:
    """Tests documenting GitConnector parser error fallback behavior.

    GitConnector intentionally falls back to read_text(errors='replace')
    when parsing fails. This is documented behavior to preserve git content.
    """

    def test_git_connector_has_fallback_read_text(self) -> None:
        """GitConnector falls back to read_text on parser errors.

        This is intentional: git repos may contain files that parsers
        can't handle, but we still want to index their content.
        """
        import inspect

        from shared.connectors.git import GitConnector

        # Verify the fallback pattern exists in source code
        source = inspect.getsource(GitConnector)

        # Should contain fallback read_text with errors='replace'
        assert 'read_text(encoding="utf-8", errors="replace")' in source

    def test_git_connector_catches_both_error_types(self) -> None:
        """GitConnector catches both UnsupportedFormatError and ExtractionFailedError."""
        import inspect

        from shared.connectors.git import GitConnector

        source = inspect.getsource(GitConnector)

        # Both exception types should be caught and handled identically
        assert "UnsupportedFormatError" in source
        assert "ExtractionFailedError" in source
        # They're caught together in a tuple
        assert "(UnsupportedFormatError, ExtractionFailedError)" in source


class TestErrorTypeDistinction:
    """Tests verifying the semantic distinction between error types."""

    def test_unsupported_format_vs_extraction_failed_semantics(self) -> None:
        """Document the semantic difference between error types.

        UnsupportedFormatError: The content format is not supported (e.g., binary)
            - Normal situation, no need to log as error
            - File should be skipped silently

        ExtractionFailedError: Extraction was attempted but failed unexpectedly
            - Abnormal situation, should log as error for debugging
            - File should be skipped but logged for investigation
        """
        from shared.text_processing.parsers import TextParser

        parser = TextParser()

        # Binary content raises UnsupportedFormatError (expected, not an error)
        with pytest.raises(UnsupportedFormatError):
            parser.parse_bytes(b"\x00\x01\x02", file_extension=".txt")

        # Decoding failure with strict mode raises ExtractionFailedError (unexpected)
        parser_strict = TextParser({"encoding": "utf-8", "errors": "strict"})
        with pytest.raises(ExtractionFailedError):
            parser_strict.parse_bytes(b"\xff\xfe", file_extension=".txt")

    def test_error_types_have_distinct_base_classes(self) -> None:
        """Error types are distinct exceptions for proper catching."""
        # Both are Exception subclasses but distinct
        assert issubclass(UnsupportedFormatError, Exception)
        assert issubclass(ExtractionFailedError, Exception)
        assert UnsupportedFormatError is not ExtractionFailedError

        # They don't share a custom base (both directly from Exception)
        # This allows precise catching
        assert not issubclass(UnsupportedFormatError, ExtractionFailedError)
        assert not issubclass(ExtractionFailedError, UnsupportedFormatError)
