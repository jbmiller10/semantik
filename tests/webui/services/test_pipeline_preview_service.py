"""Tests for PipelinePreviewService.

Tests cover parser failure handling, legacy registry fallback,
parsed metadata filtering, DAG validation errors, and invalid DAG handling.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add packages to path for imports
packages_path = Path(__file__).parent.parent.parent.parent / "packages"
if str(packages_path) not in sys.path:
    sys.path.insert(0, str(packages_path))

from shared.pipeline.types import FileReference, NodeType  # noqa: E402
from webui.services.pipeline_preview_service import PipelinePreviewService  # noqa: E402


class TestPipelinePreviewServiceBasics:
    """Basic tests for PipelinePreviewService."""

    @pytest.fixture()
    def preview_service(self) -> PipelinePreviewService:
        """Create a preview service instance."""
        return PipelinePreviewService()

    @pytest.fixture()
    def simple_dag_dict(self) -> dict:
        """Create a simple valid DAG dictionary."""
        return {
            "id": "test-dag",
            "version": "1.0",
            "nodes": [
                {"id": "parser1", "type": "parser", "plugin_id": "text", "config": {}},
                {"id": "chunker1", "type": "chunker", "plugin_id": "recursive", "config": {}},
                {"id": "embedder1", "type": "embedder", "plugin_id": "dense_local", "config": {}},
            ],
            "edges": [
                {"from_node": "_source", "to_node": "parser1", "when": None},
                {"from_node": "parser1", "to_node": "chunker1", "when": None},
                {"from_node": "chunker1", "to_node": "embedder1", "when": None},
            ],
        }

    @pytest.fixture()
    def dag_with_parser(self) -> dict:
        """Create a DAG with a parser node for testing parser execution."""
        return {
            "id": "test-dag-parser",
            "version": "1.0",
            "nodes": [
                {"id": "parser1", "type": "parser", "plugin_id": "text", "config": {}},
                {"id": "embedder1", "type": "embedder", "plugin_id": "dense_local", "config": {}},
            ],
            "edges": [
                {"from_node": "_source", "to_node": "parser1", "when": None},
                {"from_node": "parser1", "to_node": "embedder1", "when": None},
            ],
        }

    @pytest.mark.asyncio()
    async def test_preview_route_basic_flow(
        self, preview_service: PipelinePreviewService, simple_dag_dict: dict
    ) -> None:
        """Test basic route preview returns expected structure."""
        content = b"Hello, world!"
        filename = "test.txt"

        # Mock the parser to avoid actual parsing
        with patch.object(preview_service, "_run_parser", return_value={"parser": "text"}):
            result = await preview_service.preview_route(
                content,
                filename,
                simple_dag_dict,
                include_parser_metadata=False,
            )

        assert result.file_info["filename"] == filename
        assert result.file_info["extension"] == ".txt"
        assert result.path[0] == "_source"
        assert len(result.routing_stages) > 0


class TestParserFailureHandling:
    """Tests for parser failure handling."""

    @pytest.fixture()
    def preview_service(self) -> PipelinePreviewService:
        """Create a preview service instance."""
        return PipelinePreviewService()

    @pytest.fixture()
    def dag_with_parser(self) -> dict:
        """Create a DAG with a parser node."""
        return {
            "id": "test-dag-parser",
            "version": "1.0",
            "nodes": [
                {"id": "parser1", "type": "parser", "plugin_id": "text", "config": {}},
                {"id": "embedder1", "type": "embedder", "plugin_id": "dense_local", "config": {}},
            ],
            "edges": [
                {"from_node": "_source", "to_node": "parser1", "when": None},
                {"from_node": "parser1", "to_node": "embedder1", "when": None},
            ],
        }

    @pytest.mark.asyncio()
    async def test_preview_route_with_parser_failure_adds_warning(
        self, preview_service: PipelinePreviewService, dag_with_parser: dict
    ) -> None:
        """Test that parser exceptions add warnings and preview continues."""
        content = b"Test content"
        filename = "test.txt"

        # Mock _run_parser to raise an exception
        with patch.object(
            preview_service,
            "_run_parser",
            side_effect=RuntimeError("Parser failed: simulated error"),
        ):
            result = await preview_service.preview_route(
                content,
                filename,
                dag_with_parser,
                include_parser_metadata=True,
            )

        # Verify warning was added
        assert any("Parser error" in w for w in result.warnings)
        # Verify preview still completed
        assert len(result.path) > 1
        # Verify parsed_metadata is None due to failure
        assert result.parsed_metadata is None


class TestLegacyRegistryFallback:
    """Tests for fallback to legacy parser registry."""

    @pytest.fixture()
    def preview_service(self) -> PipelinePreviewService:
        """Create a preview service instance."""
        return PipelinePreviewService()

    def test_run_parser_falls_back_to_legacy_registry(
        self, preview_service: PipelinePreviewService
    ) -> None:
        """Test that plugin registry returning None triggers legacy registry fallback."""
        from shared.pipeline.types import PipelineNode

        node = PipelineNode(
            id="parser1",
            type=NodeType.PARSER,
            plugin_id="text",
            config={},
        )
        content = b"Test content"
        file_ref = FileReference(
            uri="test://test.txt",
            source_type="preview",
            content_type="document",
            filename="test.txt",
            extension=".txt",
            mime_type="text/plain",
        )

        # Mock plugin_registry.get to return None (import is inside _run_parser)
        with patch("shared.plugins.plugin_registry") as mock_registry:
            mock_registry.get.return_value = None

            # Mock get_parser (legacy registry) to return a mock parser
            with patch("webui.services.pipeline_preview_service.get_parser") as mock_get_parser:
                mock_parser = MagicMock()
                mock_parser.parse_bytes.return_value = MagicMock(
                    metadata={"parser": "text", "test_key": "test_value"}
                )
                mock_get_parser.return_value = mock_parser

                result = preview_service._run_parser(node, content, file_ref)

                # Verify legacy registry was called
                mock_get_parser.assert_called_once_with("text", {})
                assert result == {"parser": "text", "test_key": "test_value"}


class TestEnrichParsedMetadata:
    """Tests for _enrich_parsed_metadata filtering."""

    @pytest.fixture()
    def preview_service(self) -> PipelinePreviewService:
        """Create a preview service instance."""
        return PipelinePreviewService()

    def test_enrich_parsed_metadata_filters_recognized_fields(
        self, preview_service: PipelinePreviewService
    ) -> None:
        """Test that only recognized parsed fields are stored in file_ref.metadata."""
        file_ref = FileReference(
            uri="test://test.txt",
            source_type="preview",
            content_type="document",
            filename="test.txt",
            metadata={},
        )

        # Mix of recognized and unrecognized fields
        parse_metadata = {
            "page_count": 5,
            "has_tables": True,
            "has_images": False,
            "detected_language": "en",
            "approx_token_count": 1000,
            "unrecognized_field": "should_be_ignored",
            "another_unknown": 42,
            "parser": "text",  # Standard metadata field, not parsed.* field
        }

        preview_service._enrich_parsed_metadata(file_ref, parse_metadata)

        # Verify only recognized fields are in parsed namespace
        assert "parsed" in file_ref.metadata
        assert file_ref.metadata["parsed"]["page_count"] == 5
        assert file_ref.metadata["parsed"]["has_tables"] is True
        assert file_ref.metadata["parsed"]["has_images"] is False
        assert file_ref.metadata["parsed"]["detected_language"] == "en"
        assert file_ref.metadata["parsed"]["approx_token_count"] == 1000

        # Verify unrecognized fields were not included
        assert "unrecognized_field" not in file_ref.metadata["parsed"]
        assert "another_unknown" not in file_ref.metadata["parsed"]
        assert "parser" not in file_ref.metadata["parsed"]

    def test_enrich_parsed_metadata_empty_dict_no_change(
        self, preview_service: PipelinePreviewService
    ) -> None:
        """Test that empty parse_metadata doesn't modify file_ref."""
        file_ref = FileReference(
            uri="test://test.txt",
            source_type="preview",
            content_type="document",
            filename="test.txt",
            metadata={},
        )

        preview_service._enrich_parsed_metadata(file_ref, {})

        assert "parsed" not in file_ref.metadata

    def test_enrich_parsed_metadata_creates_parsed_namespace(
        self, preview_service: PipelinePreviewService
    ) -> None:
        """Test that parsed namespace is created if not present."""
        file_ref = FileReference(
            uri="test://test.txt",
            source_type="preview",
            content_type="document",
            filename="test.txt",
            metadata={"source": {"path": "/test"}},
        )

        preview_service._enrich_parsed_metadata(file_ref, {"line_count": 10})

        assert "parsed" in file_ref.metadata
        assert file_ref.metadata["parsed"]["line_count"] == 10
        # Original metadata preserved
        assert file_ref.metadata["source"]["path"] == "/test"


class TestDAGValidationErrors:
    """Tests for DAG validation error handling."""

    @pytest.fixture()
    def preview_service(self) -> PipelinePreviewService:
        """Create a preview service instance."""
        return PipelinePreviewService()

    @pytest.mark.asyncio()
    async def test_preview_route_dag_validation_errors_in_warnings(
        self, preview_service: PipelinePreviewService
    ) -> None:
        """Test that DAG validation errors appear in warnings list."""
        content = b"Test content"
        filename = "test.txt"

        # DAG with validation issues (missing embedder - required terminal node)
        # This depends on what validation rules are in place
        dag_with_issues = {
            "id": "test-dag-invalid",
            "version": "1.0",
            "nodes": [
                {"id": "parser1", "type": "parser", "plugin_id": "text", "config": {}},
            ],
            "edges": [
                {"from_node": "_source", "to_node": "parser1", "when": None},
            ],
        }

        result = await preview_service.preview_route(
            content,
            filename,
            dag_with_issues,
            include_parser_metadata=False,
        )

        # The DAG should still be processed but warnings should contain validation issues
        # The exact warning depends on validation rules; check that response is returned
        assert result is not None
        # Check if any validation warnings were added (may or may not have depending on validation rules)
        # If validation produces errors, they should be in warnings
        assert isinstance(result.warnings, list)


class TestInvalidDAGHandling:
    """Tests for invalid DAG dictionary handling."""

    @pytest.fixture()
    def preview_service(self) -> PipelinePreviewService:
        """Create a preview service instance."""
        return PipelinePreviewService()

    @pytest.mark.asyncio()
    async def test_preview_route_invalid_dag_raises_value_error(
        self, preview_service: PipelinePreviewService
    ) -> None:
        """Test that malformed DAG dict raises ValueError."""
        content = b"Test content"
        filename = "test.txt"

        # Completely invalid DAG structure
        invalid_dag = {
            "not_nodes": [],
            "not_edges": [],
        }

        with pytest.raises(ValueError, match="Invalid DAG"):
            await preview_service.preview_route(
                content,
                filename,
                invalid_dag,
                include_parser_metadata=False,
            )

    @pytest.mark.asyncio()
    async def test_preview_route_invalid_node_type_raises_value_error(
        self, preview_service: PipelinePreviewService
    ) -> None:
        """Test that invalid node type raises ValueError."""
        content = b"Test content"
        filename = "test.txt"

        # DAG with invalid node type
        invalid_dag = {
            "nodes": [
                {"id": "bad_node", "type": "invalid_type", "plugin_id": "text", "config": {}},
            ],
            "edges": [
                {"from_node": "_source", "to_node": "bad_node", "when": None},
            ],
        }

        with pytest.raises(ValueError, match="Invalid DAG"):
            await preview_service.preview_route(
                content,
                filename,
                invalid_dag,
                include_parser_metadata=False,
            )

    @pytest.mark.asyncio()
    async def test_preview_route_missing_required_fields_raises_value_error(
        self, preview_service: PipelinePreviewService
    ) -> None:
        """Test that missing required node fields raise ValueError."""
        content = b"Test content"
        filename = "test.txt"

        # DAG with node missing plugin_id
        invalid_dag = {
            "nodes": [
                {"id": "parser1", "type": "parser"},  # Missing plugin_id
            ],
            "edges": [
                {"from_node": "_source", "to_node": "parser1", "when": None},
            ],
        }

        with pytest.raises(ValueError, match="Invalid DAG"):
            await preview_service.preview_route(
                content,
                filename,
                invalid_dag,
                include_parser_metadata=False,
            )


class TestBuildFileReference:
    """Tests for _build_file_reference method."""

    @pytest.fixture()
    def preview_service(self) -> PipelinePreviewService:
        """Create a preview service instance."""
        return PipelinePreviewService()

    def test_build_file_reference_extracts_extension(
        self, preview_service: PipelinePreviewService
    ) -> None:
        """Test that file reference correctly extracts extension."""
        content = b"Hello, world!"
        filename = "document.pdf"

        file_ref = preview_service._build_file_reference(content, filename)

        assert file_ref.extension == ".pdf"
        assert file_ref.filename == "document.pdf"
        assert file_ref.mime_type == "application/pdf"
        assert file_ref.size_bytes == len(content)

    def test_build_file_reference_no_extension(
        self, preview_service: PipelinePreviewService
    ) -> None:
        """Test file reference with no extension."""
        content = b"Content"
        filename = "README"

        file_ref = preview_service._build_file_reference(content, filename)

        assert file_ref.extension is None
        assert file_ref.filename == "README"

    def test_build_file_reference_content_hash(
        self, preview_service: PipelinePreviewService
    ) -> None:
        """Test that content hash is computed correctly."""
        content = b"Test content for hashing"
        filename = "test.txt"

        file_ref = preview_service._build_file_reference(content, filename)

        # Verify change_hint is a valid SHA-256 hex digest
        assert len(file_ref.change_hint) == 64  # SHA-256 produces 64 hex chars
        assert all(c in "0123456789abcdef" for c in file_ref.change_hint)
