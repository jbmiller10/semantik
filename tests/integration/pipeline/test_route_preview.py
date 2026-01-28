"""Integration tests for pipeline route preview functionality."""

from __future__ import annotations

import pytest

from shared.pipeline.sniff import ContentSniffer, SniffConfig
from shared.pipeline.types import FileReference
from webui.services.pipeline_preview_service import PipelinePreviewService


@pytest.fixture
def preview_service() -> PipelinePreviewService:
    """Create a preview service instance."""
    return PipelinePreviewService(SniffConfig(timeout_seconds=5.0))


@pytest.fixture
def simple_dag() -> dict:
    """Create a simple pipeline DAG for testing."""
    return {
        "id": "test-pipeline",
        "version": "1.0",
        "nodes": [
            {"id": "parser1", "type": "parser", "plugin_id": "pdf_parser", "config": {}},
            {"id": "parser2", "type": "parser", "plugin_id": "text_parser", "config": {}},
            {"id": "chunker", "type": "chunker", "plugin_id": "recursive", "config": {}},
            {"id": "embedder", "type": "embedder", "plugin_id": "default", "config": {}},
        ],
        "edges": [
            {"from_node": "_source", "to_node": "parser1", "when": {"mime_type": "application/pdf"}},
            {"from_node": "_source", "to_node": "parser2", "when": None},  # catch-all
            {"from_node": "parser1", "to_node": "chunker", "when": None},
            {"from_node": "parser2", "to_node": "chunker", "when": None},
            {"from_node": "chunker", "to_node": "embedder", "when": None},
        ],
    }


@pytest.fixture
def dag_with_metadata_predicates() -> dict:
    """Create a DAG with predicates on detected metadata."""
    return {
        "id": "metadata-test-pipeline",
        "version": "1.0",
        "nodes": [
            {"id": "ocr_parser", "type": "parser", "plugin_id": "ocr_parser", "config": {}},
            {"id": "pdf_parser", "type": "parser", "plugin_id": "pdf_parser", "config": {}},
            {"id": "code_chunker", "type": "chunker", "plugin_id": "code_chunker", "config": {}},
            {"id": "text_chunker", "type": "chunker", "plugin_id": "text_chunker", "config": {}},
            {"id": "embedder", "type": "embedder", "plugin_id": "default", "config": {}},
        ],
        "edges": [
            # Route scanned PDFs to OCR parser
            {
                "from_node": "_source",
                "to_node": "ocr_parser",
                "when": {"mime_type": "application/pdf", "metadata.detected.is_scanned_pdf": True},
            },
            # Route native PDFs to regular parser
            {"from_node": "_source", "to_node": "pdf_parser", "when": {"mime_type": "application/pdf"}},
            # Catch-all for other files
            {"from_node": "_source", "to_node": "text_chunker", "when": None},
            # Route code files to code chunker
            {"from_node": "ocr_parser", "to_node": "text_chunker", "when": None},
            {"from_node": "pdf_parser", "to_node": "code_chunker", "when": {"metadata.detected.is_code": True}},
            {"from_node": "pdf_parser", "to_node": "text_chunker", "when": None},
            {"from_node": "code_chunker", "to_node": "embedder", "when": None},
            {"from_node": "text_chunker", "to_node": "embedder", "when": None},
        ],
    }


class TestPipelinePreviewService:
    """Integration tests for PipelinePreviewService."""

    @pytest.mark.asyncio
    async def test_preview_route_pdf_file(
        self,
        preview_service: PipelinePreviewService,
        simple_dag: dict,
    ) -> None:
        """Test routing a PDF file through the pipeline."""
        # Minimal PDF content
        pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [] /Count 0 >> endobj
xref
0 3
trailer << /Size 3 /Root 1 0 R >>
startxref
0
%%EOF"""

        result = await preview_service.preview_route(
            file_content=pdf_content,
            filename="test.pdf",
            dag=simple_dag,
            include_parser_metadata=False,
        )

        # Verify basic response structure
        assert result.file_info["filename"] == "test.pdf"
        assert result.file_info["mime_type"] == "application/pdf"
        assert result.file_info["extension"] == ".pdf"

        # Verify path includes PDF parser
        assert "_source" in result.path
        assert "parser1" in result.path  # PDF parser
        assert "chunker" in result.path
        assert "embedder" in result.path

        # Verify routing stages
        assert len(result.routing_stages) >= 1
        entry_stage = result.routing_stages[0]
        assert entry_stage.stage == "entry_routing"
        assert entry_stage.selected_node == "parser1"

    @pytest.mark.asyncio
    async def test_preview_route_text_file(
        self,
        preview_service: PipelinePreviewService,
        simple_dag: dict,
    ) -> None:
        """Test routing a text file through the pipeline (uses catch-all)."""
        text_content = b"This is plain text content for testing."

        result = await preview_service.preview_route(
            file_content=text_content,
            filename="readme.txt",
            dag=simple_dag,
            include_parser_metadata=False,
        )

        # Verify file info
        assert result.file_info["filename"] == "readme.txt"
        assert result.file_info["extension"] == ".txt"

        # Verify path uses catch-all (parser2)
        assert "_source" in result.path
        assert "parser2" in result.path  # catch-all text parser
        assert "parser1" not in result.path  # NOT PDF parser
        assert "chunker" in result.path
        assert "embedder" in result.path

    @pytest.mark.asyncio
    async def test_preview_route_shows_field_evaluations(
        self,
        preview_service: PipelinePreviewService,
        simple_dag: dict,
    ) -> None:
        """Test that field evaluations are included for edges with predicates."""
        text_content = b"Plain text file"

        result = await preview_service.preview_route(
            file_content=text_content,
            filename="test.txt",
            dag=simple_dag,
            include_parser_metadata=False,
        )

        # Find entry routing stage
        entry_stage = result.routing_stages[0]

        # Find the PDF edge (should not match)
        pdf_edge = next(
            (e for e in entry_stage.evaluated_edges if e.to_node == "parser1"),
            None,
        )

        assert pdf_edge is not None
        assert pdf_edge.status == "not_matched"
        assert pdf_edge.field_evaluations is not None
        assert len(pdf_edge.field_evaluations) > 0

        # Verify mime_type field evaluation
        mime_eval = next(
            (f for f in pdf_edge.field_evaluations if f.field == "mime_type"),
            None,
        )
        assert mime_eval is not None
        assert mime_eval.matched is False
        assert mime_eval.pattern == "application/pdf"

    @pytest.mark.asyncio
    async def test_preview_route_timing_included(
        self,
        preview_service: PipelinePreviewService,
        simple_dag: dict,
    ) -> None:
        """Test that timing information is included."""
        result = await preview_service.preview_route(
            file_content=b"test",
            filename="test.txt",
            dag=simple_dag,
            include_parser_metadata=False,
        )

        assert result.total_duration_ms > 0

    @pytest.mark.asyncio
    async def test_preview_route_empty_dag_returns_warnings(
        self,
        preview_service: PipelinePreviewService,
    ) -> None:
        """Test that an empty DAG returns warnings, not errors."""
        empty_dag = {
            "id": "empty",
            "version": "1.0",
            "nodes": [],  # No nodes - empty but parseable
            "edges": [],
        }

        result = await preview_service.preview_route(
            file_content=b"test",
            filename="test.txt",
            dag=empty_dag,
        )

        # Should succeed but indicate no route found
        assert result is not None
        assert "_source" in result.path
        # Should have warning about no matching route
        assert any("No matching route" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_preview_route_no_matching_route(
        self,
        preview_service: PipelinePreviewService,
    ) -> None:
        """Test handling when no route matches the file."""
        # DAG with only a very specific predicate, no catch-all
        strict_dag = {
            "id": "strict-pipeline",
            "version": "1.0",
            "nodes": [
                {"id": "parser", "type": "parser", "plugin_id": "parser", "config": {}},
                {"id": "chunker", "type": "chunker", "plugin_id": "chunker", "config": {}},
                {"id": "embedder", "type": "embedder", "plugin_id": "embedder", "config": {}},
            ],
            "edges": [
                # Only matches a very specific mime type
                {"from_node": "_source", "to_node": "parser", "when": {"mime_type": "application/x-very-specific"}},
                # Need a catch-all for DAG validation, but route to different path
                {"from_node": "_source", "to_node": "chunker", "when": None},
                {"from_node": "parser", "to_node": "chunker", "when": None},
                {"from_node": "chunker", "to_node": "embedder", "when": None},
            ],
        }

        result = await preview_service.preview_route(
            file_content=b"test",
            filename="test.txt",
            dag=strict_dag,
            include_parser_metadata=False,
        )

        # Should still find a route via catch-all
        assert len(result.path) > 1
        assert "_source" in result.path

    @pytest.mark.asyncio
    async def test_preview_route_detects_code_files(
        self,
        preview_service: PipelinePreviewService,
        simple_dag: dict,
    ) -> None:
        """Test that code files are detected by sniffer."""
        python_code = b"""#!/usr/bin/env python
import os

def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
"""

        result = await preview_service.preview_route(
            file_content=python_code,
            filename="script.py",
            dag=simple_dag,
            include_parser_metadata=False,
        )

        # Sniff result should detect code
        assert result.sniff_result is not None
        assert result.sniff_result.get("is_code") is True

    @pytest.mark.asyncio
    async def test_preview_route_detects_json_files(
        self,
        preview_service: PipelinePreviewService,
        simple_dag: dict,
    ) -> None:
        """Test that JSON files are detected as structured data."""
        json_content = b'{"name": "test", "value": 123, "nested": {"key": "value"}}'

        result = await preview_service.preview_route(
            file_content=json_content,
            filename="data.json",
            dag=simple_dag,
            include_parser_metadata=False,
        )

        # Sniff result should detect structured data
        assert result.sniff_result is not None
        assert result.sniff_result.get("is_structured_data") is True
        assert result.sniff_result.get("structured_format") == "json"


class TestContentSniffer:
    """Tests for content sniffing functionality."""

    @pytest.mark.asyncio
    async def test_sniff_code_by_extension(self) -> None:
        """Test code detection by file extension."""
        sniffer = ContentSniffer()
        file_ref = FileReference(
            uri="file://test.py",
            source_type="test",
            content_type="document",
            filename="test.py",
            extension=".py",
        )

        result = await sniffer.sniff(b"print('hello')", file_ref)
        assert result.is_code is True

    @pytest.mark.asyncio
    async def test_sniff_code_by_shebang(self) -> None:
        """Test code detection by shebang line."""
        sniffer = ContentSniffer()
        file_ref = FileReference(
            uri="file://script",
            source_type="test",
            content_type="document",
            filename="script",
            extension=None,
        )

        content = b"#!/usr/bin/env python\nprint('hello')"
        result = await sniffer.sniff(content, file_ref)
        assert result.is_code is True

    @pytest.mark.asyncio
    async def test_sniff_json_content(self) -> None:
        """Test JSON detection."""
        sniffer = ContentSniffer()
        file_ref = FileReference(
            uri="file://data.json",
            source_type="test",
            content_type="document",
            filename="data.json",
            extension=".json",
        )

        content = b'{"key": "value"}'
        result = await sniffer.sniff(content, file_ref)
        assert result.is_structured_data is True
        assert result.structured_format == "json"

    @pytest.mark.asyncio
    async def test_sniff_csv_content(self) -> None:
        """Test CSV detection."""
        sniffer = ContentSniffer()
        file_ref = FileReference(
            uri="file://data.csv",
            source_type="test",
            content_type="document",
            filename="data.csv",
            extension=".csv",
        )

        content = b"name,age,city\nAlice,30,NYC\nBob,25,LA\n"
        result = await sniffer.sniff(content, file_ref)
        assert result.is_structured_data is True
        assert result.structured_format == "csv"
