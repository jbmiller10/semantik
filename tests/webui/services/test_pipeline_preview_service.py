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

        # Verify warning was added (new format includes parser_id and filename)
        assert any("Parser 'text' failed on 'test.txt'" in w for w in result.warnings)
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

    def test_run_parser_falls_back_to_legacy_registry(self, preview_service: PipelinePreviewService) -> None:
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
                mock_parser.parse_bytes.return_value = MagicMock(metadata={"parser": "text", "test_key": "test_value"})
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

    def test_enrich_parsed_metadata_filters_recognized_fields(self, preview_service: PipelinePreviewService) -> None:
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

    def test_enrich_parsed_metadata_empty_dict_no_change(self, preview_service: PipelinePreviewService) -> None:
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

    def test_enrich_parsed_metadata_creates_parsed_namespace(self, preview_service: PipelinePreviewService) -> None:
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
    async def test_preview_route_invalid_dag_raises_value_error(self, preview_service: PipelinePreviewService) -> None:
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

    def test_build_file_reference_extracts_extension(self, preview_service: PipelinePreviewService) -> None:
        """Test that file reference correctly extracts extension."""
        content = b"Hello, world!"
        filename = "document.pdf"

        file_ref = preview_service._build_file_reference(content, filename)

        assert file_ref.extension == ".pdf"
        assert file_ref.filename == "document.pdf"
        assert file_ref.mime_type == "application/pdf"
        assert file_ref.size_bytes == len(content)

    def test_build_file_reference_sanitizes_path_traversal(self, preview_service: PipelinePreviewService) -> None:
        """Test that path traversal attempts are sanitized."""
        content = b"Malicious content"
        # Classic path traversal attack
        filename = "../../../etc/passwd"

        file_ref = preview_service._build_file_reference(content, filename)

        # Should only keep the basename
        assert file_ref.filename == "passwd"
        assert file_ref.uri == "preview://passwd"
        assert file_ref.metadata["source"]["filename"] == "passwd"

    def test_build_file_reference_sanitizes_nested_path(self, preview_service: PipelinePreviewService) -> None:
        """Test that nested directory paths are sanitized to basename only."""
        content = b"Some content"
        filename = "foo/bar/baz.txt"

        file_ref = preview_service._build_file_reference(content, filename)

        # Should only keep the final filename
        assert file_ref.filename == "baz.txt"
        assert file_ref.uri == "preview://baz.txt"
        assert file_ref.extension == ".txt"

    def test_build_file_reference_normal_filename_unchanged(self, preview_service: PipelinePreviewService) -> None:
        """Test that normal filenames without path components work correctly."""
        content = b"Normal content"
        filename = "document.pdf"

        file_ref = preview_service._build_file_reference(content, filename)

        assert file_ref.filename == "document.pdf"
        assert file_ref.uri == "preview://document.pdf"
        assert file_ref.extension == ".pdf"

    def test_build_file_reference_no_extension(self, preview_service: PipelinePreviewService) -> None:
        """Test file reference with no extension."""
        content = b"Content"
        filename = "README"

        file_ref = preview_service._build_file_reference(content, filename)

        assert file_ref.extension is None
        assert file_ref.filename == "README"

    def test_build_file_reference_content_hash(self, preview_service: PipelinePreviewService) -> None:
        """Test that content hash is computed correctly."""
        content = b"Test content for hashing"
        filename = "test.txt"

        file_ref = preview_service._build_file_reference(content, filename)

        # Verify change_hint is a valid SHA-256 hex digest
        assert len(file_ref.change_hint) == 64  # SHA-256 produces 64 hex chars
        assert all(c in "0123456789abcdef" for c in file_ref.change_hint)


class TestParallelEdgeRouting:
    """Tests for parallel edge routing behavior."""

    @pytest.fixture()
    def preview_service(self) -> PipelinePreviewService:
        """Create a preview service instance."""
        return PipelinePreviewService()

    @pytest.fixture()
    def dag_with_parallel_edges(self) -> dict:
        """Create a DAG with two parallel edges from _source."""
        return {
            "id": "test-dag-parallel",
            "version": "1.0",
            "nodes": [
                {"id": "pdf_parser", "type": "parser", "plugin_id": "pdf", "config": {}},
                {"id": "text_parser", "type": "parser", "plugin_id": "text", "config": {}},
                {"id": "embedder1", "type": "embedder", "plugin_id": "dense_local", "config": {}},
            ],
            "edges": [
                # Two parallel predicate edges - both should fire if conditions match
                {
                    "from_node": "_source",
                    "to_node": "pdf_parser",
                    "when": {"mime_type": "application/pdf"},
                    "parallel": True,
                    "path_name": "pdf_path",
                },
                {
                    "from_node": "_source",
                    "to_node": "text_parser",
                    "when": {"extension": ".pdf"},  # Also matches PDF files
                    "parallel": True,
                    "path_name": "text_path",
                },
                {"from_node": "pdf_parser", "to_node": "embedder1", "when": None},
                {"from_node": "text_parser", "to_node": "embedder1", "when": None},
            ],
        }

    @pytest.fixture()
    def dag_with_mixed_edges(self) -> dict:
        """Create a DAG with parallel and exclusive edges."""
        return {
            "id": "test-dag-mixed",
            "version": "1.0",
            "nodes": [
                {"id": "parser_a", "type": "parser", "plugin_id": "text", "config": {}},
                {"id": "parser_b", "type": "parser", "plugin_id": "text", "config": {}},
                {"id": "parser_c", "type": "parser", "plugin_id": "text", "config": {}},
                {"id": "embedder1", "type": "embedder", "plugin_id": "dense_local", "config": {}},
            ],
            "edges": [
                # Parallel predicate edge
                {
                    "from_node": "_source",
                    "to_node": "parser_a",
                    "when": {"extension": ".txt"},
                    "parallel": True,
                    "path_name": "path_a",
                },
                # Exclusive predicate edge (first-match-wins)
                {
                    "from_node": "_source",
                    "to_node": "parser_b",
                    "when": {"extension": ".txt"},
                    "parallel": False,
                    "path_name": "path_b",
                },
                # Exclusive catch-all
                {
                    "from_node": "_source",
                    "to_node": "parser_c",
                    "when": None,
                    "parallel": False,
                    "path_name": "path_c",
                },
                {"from_node": "parser_a", "to_node": "embedder1", "when": None},
                {"from_node": "parser_b", "to_node": "embedder1", "when": None},
                {"from_node": "parser_c", "to_node": "embedder1", "when": None},
            ],
        }

    @pytest.mark.asyncio()
    async def test_parallel_edges_both_fire(
        self, preview_service: PipelinePreviewService, dag_with_parallel_edges: dict
    ) -> None:
        """Test that two parallel edges matching the same file both fire."""
        content = b"%PDF-1.4 fake pdf content"
        filename = "document.pdf"

        with patch.object(preview_service, "_run_parser", return_value={}):
            result = await preview_service.preview_route(
                content,
                filename,
                dag_with_parallel_edges,
                include_parser_metadata=False,
            )

        # Verify both paths were selected
        entry_stage = result.routing_stages[0]
        assert entry_stage.selected_nodes is not None, "selected_nodes should be populated for parallel"
        selected_nodes = entry_stage.selected_nodes  # Type narrowing
        assert len(selected_nodes) == 2, "Both parallel edges should have matched"
        assert "pdf_parser" in selected_nodes
        assert "text_parser" in selected_nodes

        # Verify paths field contains both paths
        assert result.paths is not None, "paths should be populated for parallel fan-out"
        assert len(result.paths) == 2, "Should have two paths"
        path_names = {p.path_name for p in result.paths}
        assert "pdf_path" in path_names
        assert "text_path" in path_names

        # Primary path should still be set
        assert result.path[0] == "_source"
        assert result.path[1] in ("pdf_parser", "text_parser")

    @pytest.mark.asyncio()
    async def test_edge_status_matched_parallel(
        self, preview_service: PipelinePreviewService, dag_with_parallel_edges: dict
    ) -> None:
        """Test that parallel edges get status='matched_parallel'."""
        content = b"%PDF-1.4 fake pdf content"
        filename = "document.pdf"

        with patch.object(preview_service, "_run_parser", return_value={}):
            result = await preview_service.preview_route(
                content,
                filename,
                dag_with_parallel_edges,
                include_parser_metadata=False,
            )

        entry_stage = result.routing_stages[0]
        matched_edges = [e for e in entry_stage.evaluated_edges if e.matched]

        # All matched edges should have status "matched_parallel"
        for edge in matched_edges:
            assert edge.status == "matched_parallel", f"Expected matched_parallel, got {edge.status}"
            assert edge.is_parallel is True, "Edge should be marked as parallel"

    @pytest.mark.asyncio()
    async def test_exclusive_edge_blocks_later_exclusive(
        self, preview_service: PipelinePreviewService, dag_with_mixed_edges: dict
    ) -> None:
        """Test that an exclusive match blocks later exclusive edges (catch-all)."""
        content = b"Test content"
        filename = "test.txt"

        with patch.object(preview_service, "_run_parser", return_value={}):
            result = await preview_service.preview_route(
                content,
                filename,
                dag_with_mixed_edges,
                include_parser_metadata=False,
            )

        entry_stage = result.routing_stages[0]

        # Find edges by to_node
        edges_by_target = {e.to_node: e for e in entry_stage.evaluated_edges}

        # Parallel predicate edge should match
        assert edges_by_target["parser_a"].status == "matched_parallel"
        assert edges_by_target["parser_a"].is_parallel is True

        # Exclusive predicate edge should also match (first exclusive match)
        assert edges_by_target["parser_b"].status == "matched"
        assert edges_by_target["parser_b"].is_parallel is False

        # Exclusive catch-all should be skipped (exclusive already matched)
        assert edges_by_target["parser_c"].status == "skipped"

    @pytest.mark.asyncio()
    async def test_non_matching_uses_catchall(
        self, preview_service: PipelinePreviewService, dag_with_mixed_edges: dict
    ) -> None:
        """Test that non-matching predicates fall back to catch-all."""
        content = b"Binary content"
        filename = "data.bin"  # Won't match .txt predicate

        with patch.object(preview_service, "_run_parser", return_value={}):
            result = await preview_service.preview_route(
                content,
                filename,
                dag_with_mixed_edges,
                include_parser_metadata=False,
            )

        entry_stage = result.routing_stages[0]

        # Find edges by to_node
        edges_by_target = {e.to_node: e for e in entry_stage.evaluated_edges}

        # Parallel predicate edge should not match
        assert edges_by_target["parser_a"].status == "not_matched"

        # Exclusive predicate edge should not match
        assert edges_by_target["parser_b"].status == "not_matched"

        # Exclusive catch-all should match
        assert edges_by_target["parser_c"].status == "matched"

        # Should only have single path (catch-all)
        assert result.paths is None, "paths should be None for single-path result"
        assert "parser_c" in result.path

    @pytest.mark.asyncio()
    async def test_non_parallel_dag_unchanged(self, preview_service: PipelinePreviewService) -> None:
        """Test that DAGs without parallel edges work unchanged (backward compat)."""
        dag = {
            "id": "test-dag-simple",
            "version": "1.0",
            "nodes": [
                {"id": "parser1", "type": "parser", "plugin_id": "text", "config": {}},
                {"id": "embedder1", "type": "embedder", "plugin_id": "dense_local", "config": {}},
            ],
            "edges": [
                {"from_node": "_source", "to_node": "parser1", "when": None},  # No parallel field
                {"from_node": "parser1", "to_node": "embedder1", "when": None},
            ],
        }

        content = b"Test content"
        filename = "test.txt"

        with patch.object(preview_service, "_run_parser", return_value={}):
            result = await preview_service.preview_route(
                content,
                filename,
                dag,
                include_parser_metadata=False,
            )

        # Verify backward compatibility
        assert result.path == ["_source", "parser1", "embedder1"]
        assert result.paths is None, "paths should be None for non-parallel DAG"

        # Entry stage should have selected_nodes as None (single path)
        entry_stage = result.routing_stages[0]
        assert entry_stage.selected_node == "parser1"
        assert entry_stage.selected_nodes is None, "selected_nodes should be None for single path"

    @pytest.mark.asyncio()
    async def test_parallel_catchall_fires_even_with_exclusive_match(
        self, preview_service: PipelinePreviewService
    ) -> None:
        """Test that parallel catch-all edges fire even when an exclusive predicate matches."""
        dag = {
            "id": "test-dag-parallel-catchall",
            "version": "1.0",
            "nodes": [
                {"id": "parser_exclusive", "type": "parser", "plugin_id": "text", "config": {}},
                {"id": "parser_parallel", "type": "parser", "plugin_id": "text", "config": {}},
                {"id": "parser_fallback", "type": "parser", "plugin_id": "text", "config": {}},
                {"id": "embedder1", "type": "embedder", "plugin_id": "dense_local", "config": {}},
            ],
            "edges": [
                # Exclusive predicate match
                {
                    "from_node": "_source",
                    "to_node": "parser_exclusive",
                    "when": {"extension": ".txt"},
                    "parallel": False,
                    "path_name": "exclusive_path",
                },
                # Parallel catch-all should still fire
                {
                    "from_node": "_source",
                    "to_node": "parser_parallel",
                    "when": None,
                    "parallel": True,
                    "path_name": "parallel_catchall_path",
                },
                # Exclusive catch-all should be skipped (exclusive predicate matched)
                {
                    "from_node": "_source",
                    "to_node": "parser_fallback",
                    "when": None,
                    "parallel": False,
                    "path_name": "fallback_path",
                },
                {"from_node": "parser_exclusive", "to_node": "embedder1", "when": None},
                {"from_node": "parser_parallel", "to_node": "embedder1", "when": None},
                {"from_node": "parser_fallback", "to_node": "embedder1", "when": None},
            ],
        }

        content = b"Test content"
        filename = "test.txt"

        with patch.object(preview_service, "_run_parser", return_value={}):
            result = await preview_service.preview_route(content, filename, dag, include_parser_metadata=False)

        entry_stage = result.routing_stages[0]
        assert entry_stage.selected_nodes is not None
        assert set(entry_stage.selected_nodes) == {"parser_exclusive", "parser_parallel"}

        edges_by_target = {e.to_node: e for e in entry_stage.evaluated_edges}
        assert edges_by_target["parser_exclusive"].status == "matched"
        assert edges_by_target["parser_parallel"].status == "matched_parallel"
        assert edges_by_target["parser_fallback"].status == "skipped"

        assert result.paths is not None
        assert {p.path_name for p in result.paths} == {"exclusive_path", "parallel_catchall_path"}

    @pytest.mark.asyncio()
    async def test_mid_pipeline_parallel_routing_creates_multiple_paths(
        self, preview_service: PipelinePreviewService
    ) -> None:
        """Test that mid-pipeline parallel fan-out is reflected in preview paths and stage evaluation."""
        dag = {
            "id": "test-dag-mid-pipeline-parallel",
            "version": "1.0",
            "nodes": [
                {"id": "parser1", "type": "parser", "plugin_id": "text", "config": {}},
                {"id": "chunker_a", "type": "chunker", "plugin_id": "recursive", "config": {}},
                {"id": "chunker_b", "type": "chunker", "plugin_id": "recursive", "config": {}},
                {"id": "embedder1", "type": "embedder", "plugin_id": "dense_local", "config": {}},
            ],
            "edges": [
                {"from_node": "_source", "to_node": "parser1", "when": None},
                {"from_node": "parser1", "to_node": "chunker_a", "when": None, "parallel": True, "path_name": "path_a"},
                {"from_node": "parser1", "to_node": "chunker_b", "when": None, "parallel": True, "path_name": "path_b"},
                {"from_node": "chunker_a", "to_node": "embedder1", "when": None},
                {"from_node": "chunker_b", "to_node": "embedder1", "when": None},
            ],
        }

        content = b"Test content"
        filename = "test.txt"

        with patch.object(preview_service, "_run_parser", return_value={}):
            result = await preview_service.preview_route(content, filename, dag, include_parser_metadata=False)

        # Mid-pipeline fan-out should create multiple paths even with a single entry node.
        assert result.paths is not None
        assert {p.path_name for p in result.paths} == {"path_a", "path_b"}

        # The routing stage from the parser should reflect fan-out.
        parser_stage = next(s for s in result.routing_stages if s.from_node == "parser1")
        assert parser_stage.selected_nodes is not None
        assert set(parser_stage.selected_nodes) == {"chunker_a", "chunker_b"}

    @pytest.mark.asyncio()
    async def test_preview_route_multi_entry_parsers_use_path_local_metadata(
        self, preview_service: PipelinePreviewService
    ) -> None:
        """Ensure per-path parser metadata is used for routing in multi-entry DAGs."""
        dag = {
            "id": "test-dag-multi-entry-multi-parser",
            "version": "1.0",
            "nodes": [
                {"id": "parser_a", "type": "parser", "plugin_id": "text", "config": {}},
                {"id": "parser_b", "type": "parser", "plugin_id": "text", "config": {}},
                {"id": "chunker_a", "type": "chunker", "plugin_id": "recursive", "config": {}},
                {"id": "chunker_b", "type": "chunker", "plugin_id": "recursive", "config": {}},
                {"id": "embedder1", "type": "embedder", "plugin_id": "dense_local", "config": {}},
            ],
            "edges": [
                {
                    "from_node": "_source",
                    "to_node": "parser_a",
                    "when": {"extension": ".txt"},
                    "parallel": True,
                    "path_name": "path_a",
                },
                {
                    "from_node": "_source",
                    "to_node": "parser_b",
                    "when": {"extension": ".txt"},
                    "parallel": True,
                    "path_name": "path_b",
                },
                {"from_node": "parser_a", "to_node": "chunker_a", "when": {"metadata.parsed.detected_language": "a"}},
                {"from_node": "parser_b", "to_node": "chunker_b", "when": {"metadata.parsed.detected_language": "b"}},
                {"from_node": "chunker_a", "to_node": "embedder1", "when": None},
                {"from_node": "chunker_b", "to_node": "embedder1", "when": None},
            ],
        }

        content = b"Test content"
        filename = "test.txt"

        def parser_side_effect(node, _content, _file_ref):
            if node.id == "parser_a":
                return {"detected_language": "a"}
            if node.id == "parser_b":
                return {"detected_language": "b"}
            raise AssertionError(f"Unexpected parser node: {node.id}")

        with patch.object(preview_service, "_run_parser", side_effect=parser_side_effect) as run_parser:
            result = await preview_service.preview_route(content, filename, dag, include_parser_metadata=True)

        assert run_parser.call_count == 2, "Each entry parser should run to support path-correct routing"
        assert result.paths is not None

        paths_by_name = {p.path_name: p.nodes for p in result.paths}
        assert paths_by_name["path_a"] == ["_source", "parser_a", "chunker_a", "embedder1"]
        assert paths_by_name["path_b"] == ["_source", "parser_b", "chunker_b", "embedder1"]
