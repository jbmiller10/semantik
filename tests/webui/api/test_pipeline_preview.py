"""Tests for pipeline route preview API endpoint."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from webui.api.v2.pipeline_schemas import RoutePreviewResponse


@pytest.fixture
def sample_dag() -> dict[str, Any]:
    """Sample pipeline DAG for testing."""
    return {
        "id": "test-dag",
        "version": "1.0",
        "nodes": [
            {"id": "pdf_parser", "type": "parser", "plugin_id": "pdf_parser", "config": {}},
            {"id": "recursive_chunker", "type": "chunker", "plugin_id": "recursive", "config": {}},
            {"id": "embedder", "type": "embedder", "plugin_id": "default_embedder", "config": {}},
        ],
        "edges": [
            {"from_node": "_source", "to_node": "pdf_parser", "when": {"mime_type": "application/pdf"}},
            {"from_node": "_source", "to_node": "recursive_chunker", "when": None},  # catch-all
            {"from_node": "pdf_parser", "to_node": "recursive_chunker", "when": None},
            {"from_node": "recursive_chunker", "to_node": "embedder", "when": None},
        ],
    }


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Minimal PDF content for testing."""
    # This is a minimal valid PDF structure
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<< /Size 4 /Root 1 0 R >>
startxref
196
%%EOF"""


@pytest.fixture
def sample_text_content() -> bytes:
    """Sample text file content for testing."""
    return b"This is a sample text file for testing routing."


class TestPipelinePreviewEndpoint:
    """Tests for POST /api/v2/pipeline/preview-route."""

    def test_preview_route_pdf_matches_pdf_edge(
        self,
        test_client: TestClient,
        sample_dag: dict[str, Any],
        sample_pdf_content: bytes,
    ) -> None:
        """Test that PDF file matches the PDF-specific edge."""
        response = test_client.post(
            "/api/v2/pipeline/preview-route",
            files={"file": ("test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")},
            data={
                "dag": json.dumps(sample_dag),
                "include_parser_metadata": "false",  # Skip parser for faster test
            },
        )

        assert response.status_code == status.HTTP_200_OK
        result = response.json()

        # Verify file info
        assert result["file_info"]["filename"] == "test.pdf"
        assert result["file_info"]["mime_type"] == "application/pdf"
        assert result["file_info"]["extension"] == ".pdf"

        # Verify path includes PDF parser
        assert "_source" in result["path"]
        assert "pdf_parser" in result["path"]

        # Verify routing stages exist
        assert len(result["routing_stages"]) > 0

        # First stage should be entry routing
        entry_stage = result["routing_stages"][0]
        assert entry_stage["stage"] == "entry_routing"
        assert entry_stage["from_node"] == "_source"

        # Should have matched the PDF edge
        pdf_edge = next(
            (e for e in entry_stage["evaluated_edges"] if e["to_node"] == "pdf_parser"),
            None,
        )
        assert pdf_edge is not None
        assert pdf_edge["status"] == "matched"

    def test_preview_route_text_uses_catchall(
        self,
        test_client: TestClient,
        sample_dag: dict[str, Any],
        sample_text_content: bytes,
    ) -> None:
        """Test that text file uses catch-all edge."""
        response = test_client.post(
            "/api/v2/pipeline/preview-route",
            files={"file": ("test.txt", io.BytesIO(sample_text_content), "text/plain")},
            data={
                "dag": json.dumps(sample_dag),
                "include_parser_metadata": "false",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        result = response.json()

        # Verify file info
        assert result["file_info"]["filename"] == "test.txt"

        # Verify path goes directly to chunker (catch-all)
        assert "_source" in result["path"]
        assert "recursive_chunker" in result["path"]

        # Should NOT include pdf_parser (PDF predicate doesn't match)
        assert "pdf_parser" not in result["path"]

    def test_preview_route_invalid_dag(
        self,
        test_client: TestClient,
        sample_text_content: bytes,
    ) -> None:
        """Test error handling for invalid DAG."""
        response = test_client.post(
            "/api/v2/pipeline/preview-route",
            files={"file": ("test.txt", io.BytesIO(sample_text_content), "text/plain")},
            data={"dag": "not valid json"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid DAG JSON" in response.json()["detail"]

    def test_preview_route_returns_field_evaluations(
        self,
        test_client: TestClient,
        sample_dag: dict[str, Any],
        sample_text_content: bytes,
    ) -> None:
        """Test that field evaluations are returned for edges with predicates."""
        response = test_client.post(
            "/api/v2/pipeline/preview-route",
            files={"file": ("test.txt", io.BytesIO(sample_text_content), "text/plain")},
            data={
                "dag": json.dumps(sample_dag),
                "include_parser_metadata": "false",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        result = response.json()

        # Find the entry routing stage
        entry_stage = result["routing_stages"][0]

        # The PDF edge should have field evaluations showing why it didn't match
        pdf_edge = next(
            (e for e in entry_stage["evaluated_edges"] if e["to_node"] == "pdf_parser"),
            None,
        )

        if pdf_edge and pdf_edge["status"] == "not_matched":
            # Should have field evaluations
            assert pdf_edge["field_evaluations"] is not None
            assert len(pdf_edge["field_evaluations"]) > 0

            # Should show mime_type evaluation
            mime_eval = next(
                (f for f in pdf_edge["field_evaluations"] if f["field"] == "mime_type"),
                None,
            )
            assert mime_eval is not None
            assert mime_eval["matched"] is False

    def test_preview_route_file_too_large(
        self,
        test_client: TestClient,
        sample_dag: dict[str, Any],
    ) -> None:
        """Test error handling for oversized files."""
        # Create a file larger than 10MB limit
        large_content = b"x" * (11 * 1024 * 1024)

        response = test_client.post(
            "/api/v2/pipeline/preview-route",
            files={"file": ("large.txt", io.BytesIO(large_content), "text/plain")},
            data={"dag": json.dumps(sample_dag)},
        )

        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

    def test_preview_route_includes_timing(
        self,
        test_client: TestClient,
        sample_dag: dict[str, Any],
        sample_text_content: bytes,
    ) -> None:
        """Test that timing information is included in response."""
        response = test_client.post(
            "/api/v2/pipeline/preview-route",
            files={"file": ("test.txt", io.BytesIO(sample_text_content), "text/plain")},
            data={
                "dag": json.dumps(sample_dag),
                "include_parser_metadata": "false",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        result = response.json()

        assert "total_duration_ms" in result
        assert result["total_duration_ms"] >= 0


class TestRoutePreviewSchemas:
    """Tests for Pydantic schemas."""

    def test_route_preview_response_schema(self) -> None:
        """Test RoutePreviewResponse schema validation."""
        data = {
            "file_info": {
                "filename": "test.pdf",
                "extension": ".pdf",
                "mime_type": "application/pdf",
                "size_bytes": 1024,
                "uri": "preview://test.pdf",
            },
            "sniff_result": {"is_scanned_pdf": False},
            "routing_stages": [
                {
                    "stage": "entry_routing",
                    "from_node": "_source",
                    "evaluated_edges": [],
                    "selected_node": "pdf_parser",
                    "metadata_snapshot": {},
                }
            ],
            "path": ["_source", "pdf_parser", "chunker", "embedder"],
            "parsed_metadata": None,
            "total_duration_ms": 42.5,
            "warnings": [],
        }

        response = RoutePreviewResponse(**data)
        assert response.file_info["filename"] == "test.pdf"
        assert response.path == ["_source", "pdf_parser", "chunker", "embedder"]
        assert response.total_duration_ms == 42.5
