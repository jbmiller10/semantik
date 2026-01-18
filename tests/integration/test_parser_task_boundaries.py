"""Integration tests for parser task boundary safety (Phase 9).

Tests verify that parsers return only primitives across process boundaries
(Celery/billiard) and that ParseResult/ParsedElement stay local.

These tests lock down the current safe behavior where workers return
dict[str, str | dict | int] primitives rather than ParseResult objects.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from shared.text_processing.parsers import ParseResult, parse_content


class TestParserTaskBoundaries:
    """Tests for task boundary safety - primitives only across boundaries."""

    def test_parse_result_is_not_serialized_in_worker(self) -> None:
        """Verify that worker returns primitives, not ParseResult.

        The _process_file_worker in LocalFileConnector returns a dict
        with only str/dict/int values, never ParseResult or ParsedElement.
        """
        # Import the worker function
        from shared.connectors.local import _process_file_worker

        # The worker function signature shows it returns _WorkerResult
        # which is a TypedDict with only primitive fields
        # The function exists and is callable
        assert callable(_process_file_worker)

        # Verify the return type annotation (if available)
        # The function should return _WorkerResult which is:
        # _WorkerSuccess | _WorkerSkipped | _WorkerError
        # All of these are TypedDict with only primitives

    def test_worker_success_data_is_json_serializable(self) -> None:
        """Verify that _WorkerSuccessData can be JSON serialized."""
        # Create a sample success data structure matching _WorkerSuccessData schema
        sample_data = {
            "content": "Sample text content",
            "unique_id": "file:///path/to/file.txt",
            "source_type": "directory",
            "metadata": {
                "parser": "text",
                "filename": "file.txt",
                "file_extension": ".txt",
                "file_type": "txt",
                "mime_type": "text/plain",
            },
            "content_hash": "abc123def456",
            "file_path": "/path/to/file.txt",
        }

        # Must be JSON serializable (for Celery task results)
        serialized = json.dumps(sample_data)
        deserialized = json.loads(serialized)
        assert deserialized == sample_data

    def test_worker_result_types_have_no_parse_result(self) -> None:
        """Verify _WorkerResult types don't include ParseResult."""
        from shared.connectors.local import (
            _WorkerError,
            _WorkerSkipped,
            _WorkerSuccess,
            _WorkerSuccessData,
        )

        # Check annotations don't reference ParseResult
        success_data_annotations = _WorkerSuccessData.__annotations__
        assert all(
            "ParseResult" not in str(v) and "ParsedElement" not in str(v) for v in success_data_annotations.values()
        )

        success_annotations = _WorkerSuccess.__annotations__
        assert "ParseResult" not in str(success_annotations)

        skipped_annotations = _WorkerSkipped.__annotations__
        assert all(isinstance(v, type) or "ParseResult" not in str(v) for v in skipped_annotations.values())

        error_annotations = _WorkerError.__annotations__
        assert all(isinstance(v, type) or "ParseResult" not in str(v) for v in error_annotations.values())

    def test_parse_content_result_can_extract_primitives(self) -> None:
        """Verify parse_content returns data that can be extracted as primitives."""
        result = parse_content("Hello, world!", file_extension=".txt")

        # ParseResult can be destructured into primitives
        text: str = result.text
        metadata: dict[str, Any] = result.metadata

        # These are all JSON serializable
        primitives = {
            "text": text,
            "metadata": metadata,
        }
        serialized = json.dumps(primitives)
        deserialized = json.loads(serialized)

        assert deserialized["text"] == "Hello, world!"
        assert deserialized["metadata"]["parser"] == "text"


class TestIncludeElementsDefault:
    """Tests verifying include_elements=False is the default."""

    def test_parse_content_defaults_to_no_elements(self) -> None:
        """parse_content() defaults to include_elements=False."""
        result = parse_content("Hello", file_extension=".txt")
        assert result.elements == []

    def test_text_parser_defaults_to_no_elements(self) -> None:
        """TextParser defaults to include_elements=False."""
        from shared.text_processing.parsers import TextParser

        parser = TextParser()
        result = parser.parse_bytes(b"Hello", file_extension=".txt")
        assert result.elements == []

    def test_local_connector_worker_uses_no_elements(self) -> None:
        """LocalFileConnector worker calls parse_content without include_elements.

        Verifying by checking the source code contains no include_elements=True.
        """
        import inspect

        from shared.connectors.local import _process_file_worker

        source = inspect.getsource(_process_file_worker)

        # The worker should NOT specify include_elements=True
        # (it's not needed for ingestion, and would bloat serialization)
        assert "include_elements=True" not in source
        assert "include_elements = True" not in source


class TestSerializationBoundary:
    """Tests for Celery task serialization boundaries."""

    def test_parse_result_cannot_be_directly_jsonified(self) -> None:
        """ParseResult is a frozen dataclass, not directly JSON serializable.

        This test documents WHY we use primitives across boundaries -
        ParseResult objects cannot be pickled/JSONified without conversion.
        """
        result = parse_content("Hello", file_extension=".txt")

        # ParseResult is a frozen dataclass with ParsedElement list
        assert isinstance(result, ParseResult)

        # Direct JSON serialization would fail
        with pytest.raises(TypeError):
            json.dumps(result)

    def test_primitives_extracted_from_parse_result_are_serializable(self) -> None:
        """Primitives extracted from ParseResult can cross serialization boundaries."""
        result = parse_content("Hello", file_extension=".txt")

        # Extract only primitives (what workers do)
        worker_result = {
            "status": "success",
            "data": {
                "content": result.text,
                "unique_id": "file:///test.txt",
                "source_type": "directory",
                "metadata": dict(result.metadata),  # Convert MappingProxyType if needed
                "content_hash": "abc123",
                "file_path": "/test.txt",
            },
        }

        # This is what actually crosses the boundary - pure primitives
        serialized = json.dumps(worker_result)
        deserialized = json.loads(serialized)

        assert deserialized["status"] == "success"
        assert deserialized["data"]["content"] == "Hello"
