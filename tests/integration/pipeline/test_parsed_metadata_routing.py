"""Integration tests for mid-pipeline routing based on parsed metadata.

Tests that parser metadata (has_code_blocks, approx_token_count, etc.) is
properly enriched to FileReference and can be used for routing decisions.
"""

from __future__ import annotations

import pytest

from shared.pipeline.types import FileReference
from shared.plugins.builtins.text_parser import TextParserPlugin


class TestParserMetadataEnrichment:
    """Tests for parser metadata enrichment to FileReference."""

    def test_text_parser_enriches_file_ref_metadata(self) -> None:
        """Verify TextParser emits metadata that can be used for routing."""
        # Create FileReference (simulates loader output)
        file_ref = FileReference(
            uri="file:///test/code_example.md",
            source_type="directory",
            content_type="document",
            mime_type="text/markdown",
            metadata={"source": {}},
        )

        # Parse content with code blocks
        content = b"# Example\n```python\nprint('hello')\n```\nSome text here"
        parser = TextParserPlugin()
        result = parser.parse_bytes(content, filename="code_example.md", mime_type="text/markdown")

        # Verify parser emits parsed metadata fields
        assert "has_code_blocks" in result.metadata
        assert result.metadata["has_code_blocks"] is True
        assert "approx_token_count" in result.metadata
        assert result.metadata["approx_token_count"] > 0
        assert "line_count" in result.metadata
        assert "detected_language" in result.metadata

        # Simulate executor enriching file_ref (what _enrich_parsed_metadata does)
        parsed_fields = {
            "page_count",
            "has_tables",
            "has_images",
            "has_code_blocks",
            "detected_language",
            "approx_token_count",
            "line_count",
            "element_types",
            "text_quality",
        }
        if "parsed" not in file_ref.metadata:
            file_ref.metadata["parsed"] = {}

        for key, value in result.metadata.items():
            if key in parsed_fields:
                file_ref.metadata["parsed"][key] = value

        # Verify file_ref now has parsed metadata for routing
        assert "parsed" in file_ref.metadata
        assert file_ref.metadata["parsed"]["has_code_blocks"] is True
        assert file_ref.metadata["parsed"]["approx_token_count"] > 0

    def test_unstructured_parser_enriches_file_ref_metadata(self) -> None:
        """Verify UnstructuredParser emits metadata for routing."""
        try:
            from shared.plugins.builtins.unstructured_parser import UnstructuredParserPlugin
        except ImportError:
            pytest.skip("unstructured not installed")

        file_ref = FileReference(
            uri="file:///test/document.txt",
            source_type="directory",
            content_type="document",
            mime_type="text/plain",
            metadata={"source": {}},
        )

        # Parse simple text
        content = b"Title\n\nParagraph text"
        parser = UnstructuredParserPlugin()
        result = parser.parse_bytes(content, mime_type="text/plain")

        # Verify parser emits metadata
        assert "page_count" in result.metadata
        assert "has_tables" in result.metadata
        assert "has_images" in result.metadata
        assert "element_types" in result.metadata
        assert "approx_token_count" in result.metadata

        # Simulate enrichment
        parsed_fields = {
            "page_count",
            "has_tables",
            "has_images",
            "has_code_blocks",
            "detected_language",
            "approx_token_count",
            "line_count",
            "element_types",
            "text_quality",
        }
        if "parsed" not in file_ref.metadata:
            file_ref.metadata["parsed"] = {}

        for key, value in result.metadata.items():
            if key in parsed_fields:
                file_ref.metadata["parsed"][key] = value

        # Verify enrichment
        assert "parsed" in file_ref.metadata
        assert "page_count" in file_ref.metadata["parsed"]
        assert "has_tables" in file_ref.metadata["parsed"]
        assert "element_types" in file_ref.metadata["parsed"]


class TestRoutingPredicateExamples:
    """Examples of how routing predicates would use parsed metadata."""

    def test_routing_on_has_code_blocks(self) -> None:
        """Example: Route documents with code blocks to code-aware chunker."""
        # Simulate file_ref after parser enrichment
        file_ref = FileReference(
            uri="file:///test/code.md",
            source_type="directory",
            content_type="document",
            mime_type="text/markdown",
            metadata={
                "source": {},
                "parsed": {
                    "has_code_blocks": True,
                    "approx_token_count": 500,
                    "line_count": 25,
                },
            },
        )

        # Routing predicate: metadata.parsed.has_code_blocks == true
        should_route_to_code_chunker = file_ref.metadata.get("parsed", {}).get("has_code_blocks") is True
        assert should_route_to_code_chunker is True

    def test_routing_on_large_token_count(self) -> None:
        """Example: Route large documents to semantic chunker."""
        file_ref = FileReference(
            uri="file:///test/large_doc.txt",
            source_type="directory",
            content_type="document",
            mime_type="text/plain",
            metadata={
                "source": {},
                "parsed": {
                    "approx_token_count": 15000,
                    "line_count": 500,
                },
            },
        )

        # Routing predicate: metadata.parsed.approx_token_count > 10000
        token_count = file_ref.metadata.get("parsed", {}).get("approx_token_count", 0)
        should_route_to_semantic = token_count > 10000
        assert should_route_to_semantic is True

    def test_routing_on_language(self) -> None:
        """Example: Route Chinese documents to multilingual embedder."""
        file_ref = FileReference(
            uri="file:///test/chinese_doc.txt",
            source_type="directory",
            content_type="document",
            mime_type="text/plain",
            metadata={
                "source": {},
                "parsed": {
                    "detected_language": "zh",
                    "approx_token_count": 1000,
                },
            },
        )

        # Routing predicate: metadata.parsed.detected_language == "zh"
        detected_lang = file_ref.metadata.get("parsed", {}).get("detected_language")
        should_route_to_multilingual = detected_lang == "zh"
        assert should_route_to_multilingual is True

    def test_routing_on_has_tables(self) -> None:
        """Example: Route documents with tables to table-aware chunker."""
        file_ref = FileReference(
            uri="file:///test/spreadsheet.pdf",
            source_type="directory",
            content_type="document",
            mime_type="application/pdf",
            metadata={
                "source": {},
                "parsed": {
                    "page_count": 10,
                    "has_tables": True,
                    "has_images": False,
                    "element_types": ["Table", "Title", "Text"],
                },
            },
        )

        # Routing predicate: metadata.parsed.has_tables == true
        should_route_to_table_chunker = file_ref.metadata.get("parsed", {}).get("has_tables") is True
        assert should_route_to_table_chunker is True

    def test_routing_fallback_when_field_missing(self) -> None:
        """Verify routing handles missing fields gracefully."""
        file_ref = FileReference(
            uri="file:///test/doc.txt",
            source_type="directory",
            content_type="document",
            mime_type="text/plain",
            metadata={"source": {}},  # No parsed metadata
        )

        # Predicate should handle missing field gracefully (treat as False/None)
        has_code_blocks = file_ref.metadata.get("parsed", {}).get("has_code_blocks")
        assert has_code_blocks is None  # Missing field returns None

        # Routing should fall through to catch-all edge
        should_route_to_code_chunker = has_code_blocks is True
        assert should_route_to_code_chunker is False
