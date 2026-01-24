"""Unit tests for pipeline router."""

import pytest

from shared.pipeline.router import PipelineRouter
from shared.pipeline.types import FileReference, NodeType, PipelineDAG, PipelineEdge, PipelineNode


class TestPipelineRouter:
    """Tests for PipelineRouter class."""

    @pytest.fixture()
    def simple_dag(self) -> PipelineDAG:
        """Create a simple linear DAG for testing."""
        return PipelineDAG(
            id="simple",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="text"),
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense-local"),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="parser"),
                PipelineEdge(from_node="parser", to_node="chunker"),
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        )

    @pytest.fixture()
    def branching_dag(self) -> PipelineDAG:
        """Create a DAG with predicate-based branching."""
        return PipelineDAG(
            id="branching",
            version="1.0",
            nodes=[
                PipelineNode(id="pdf-parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="text-parser", type=NodeType.PARSER, plugin_id="text"),
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense-local"),
            ],
            edges=[
                # PDF files go to pdf-parser
                PipelineEdge(
                    from_node="_source",
                    to_node="pdf-parser",
                    when={"extension": ".pdf"},
                ),
                # Everything else goes to text-parser
                PipelineEdge(
                    from_node="_source",
                    to_node="text-parser",
                ),
                # Both parsers feed into chunker
                PipelineEdge(from_node="pdf-parser", to_node="chunker"),
                PipelineEdge(from_node="text-parser", to_node="chunker"),
                # Chunker feeds into embedder
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        )

    @pytest.fixture()
    def pdf_file(self) -> FileReference:
        """Create a PDF file reference."""
        return FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
            extension=".pdf",
            mime_type="application/pdf",
            size_bytes=1024,
        )

    @pytest.fixture()
    def text_file(self) -> FileReference:
        """Create a text file reference."""
        return FileReference(
            uri="file:///doc.txt",
            source_type="directory",
            content_type="document",
            extension=".txt",
            mime_type="text/plain",
            size_bytes=512,
        )

    def test_init_builds_node_index(self, simple_dag: PipelineDAG) -> None:
        """Test router builds node index on init."""
        router = PipelineRouter(simple_dag)

        assert len(router._node_index) == 3
        assert "parser" in router._node_index
        assert "chunker" in router._node_index
        assert "embedder" in router._node_index

    def test_init_builds_edge_index(self, simple_dag: PipelineDAG) -> None:
        """Test router builds outgoing edge index on init."""
        router = PipelineRouter(simple_dag)

        assert "_source" in router._outgoing_edges
        assert "parser" in router._outgoing_edges
        assert "chunker" in router._outgoing_edges

    def test_get_entry_node_simple(self, simple_dag: PipelineDAG, text_file: FileReference) -> None:
        """Test getting entry node from simple DAG."""
        router = PipelineRouter(simple_dag)

        entry = router.get_entry_node(text_file)

        assert entry is not None
        assert entry.id == "parser"

    def test_get_entry_node_with_predicate(self, branching_dag: PipelineDAG, pdf_file: FileReference) -> None:
        """Test entry node selection with predicate matching."""
        router = PipelineRouter(branching_dag)

        entry = router.get_entry_node(pdf_file)

        assert entry is not None
        assert entry.id == "pdf-parser"

    def test_get_entry_node_catchall(self, branching_dag: PipelineDAG, text_file: FileReference) -> None:
        """Test entry node falls back to catch-all."""
        router = PipelineRouter(branching_dag)

        entry = router.get_entry_node(text_file)

        assert entry is not None
        assert entry.id == "text-parser"

    def test_get_entry_node_no_source_edges(self) -> None:
        """Test get_entry_node returns None when no _source edges."""
        dag = PipelineDAG(
            id="no-source",
            version="1.0",
            nodes=[
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[],
        )
        router = PipelineRouter(dag)
        file_ref = FileReference(
            uri="file:///test.txt",
            source_type="directory",
            content_type="document",
            size_bytes=100,
        )

        entry = router.get_entry_node(file_ref)

        assert entry is None

    def test_get_next_nodes_simple(self, simple_dag: PipelineDAG, text_file: FileReference) -> None:
        """Test getting next nodes in simple linear DAG."""
        router = PipelineRouter(simple_dag)
        parser_node = router._node_index["parser"]

        next_nodes = router.get_next_nodes(parser_node, text_file)

        assert len(next_nodes) == 1
        assert next_nodes[0].id == "chunker"

    def test_get_next_nodes_terminal(self, simple_dag: PipelineDAG, text_file: FileReference) -> None:
        """Test getting next nodes returns empty at terminal node."""
        router = PipelineRouter(simple_dag)
        embedder_node = router._node_index["embedder"]

        next_nodes = router.get_next_nodes(embedder_node, text_file)

        assert next_nodes == []

    def test_get_node_exists(self, simple_dag: PipelineDAG) -> None:
        """Test getting existing node by ID."""
        router = PipelineRouter(simple_dag)

        node = router.get_node("parser")

        assert node is not None
        assert node.id == "parser"
        assert node.type == NodeType.PARSER

    def test_get_node_not_found(self, simple_dag: PipelineDAG) -> None:
        """Test getting non-existent node returns None."""
        router = PipelineRouter(simple_dag)

        node = router.get_node("nonexistent")

        assert node is None

    def test_get_all_paths_simple(self, simple_dag: PipelineDAG, text_file: FileReference) -> None:
        """Test getting all paths in simple DAG."""
        router = PipelineRouter(simple_dag)

        paths = router.get_all_paths(text_file)

        assert len(paths) == 1
        assert len(paths[0]) == 3
        assert paths[0][0].id == "parser"
        assert paths[0][1].id == "chunker"
        assert paths[0][2].id == "embedder"

    def test_get_all_paths_no_match(self) -> None:
        """Test get_all_paths returns empty when no matching entry."""
        dag = PipelineDAG(
            id="predicate-only",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="text"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                # Only matches .pdf files
                PipelineEdge(from_node="_source", to_node="parser", when={"extension": ".pdf"}),
                PipelineEdge(from_node="parser", to_node="embedder"),
            ],
        )
        router = PipelineRouter(dag)
        # This file is .txt, not .pdf
        txt_file = FileReference(
            uri="file:///doc.txt",
            source_type="directory",
            content_type="document",
            extension=".txt",
            size_bytes=100,
        )

        paths = router.get_all_paths(txt_file)

        assert paths == []

    def test_predicate_edges_checked_before_catchall(self) -> None:
        """Test that predicate edges are checked before catch-all."""
        dag = PipelineDAG(
            id="order-test",
            version="1.0",
            nodes=[
                PipelineNode(id="special", type=NodeType.PARSER, plugin_id="special"),
                PipelineNode(id="default", type=NodeType.PARSER, plugin_id="default"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                # Catch-all first in list (should not be selected for .py files)
                PipelineEdge(from_node="_source", to_node="default"),
                # Predicate-based edge (should be selected for .py files)
                PipelineEdge(from_node="_source", to_node="special", when={"extension": ".py"}),
                PipelineEdge(from_node="special", to_node="embedder"),
                PipelineEdge(from_node="default", to_node="embedder"),
            ],
        )
        router = PipelineRouter(dag)
        py_file = FileReference(
            uri="file:///code.py",
            source_type="directory",
            content_type="code",
            extension=".py",
            size_bytes=100,
        )

        entry = router.get_entry_node(py_file)

        # Predicate edge should be selected even though catch-all is first
        assert entry is not None
        assert entry.id == "special"

    def test_empty_when_is_catchall(self) -> None:
        """Test that empty when dict is treated as catch-all."""
        dag = PipelineDAG(
            id="empty-when",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="text"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="parser", when={}),
                PipelineEdge(from_node="parser", to_node="embedder"),
            ],
        )
        router = PipelineRouter(dag)
        file_ref = FileReference(
            uri="file:///any.txt",
            source_type="directory",
            content_type="document",
            size_bytes=100,
        )

        entry = router.get_entry_node(file_ref)

        assert entry is not None
        assert entry.id == "parser"
