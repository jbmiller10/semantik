"""Unit tests for pipeline type definitions."""

import json

import pytest

from shared.pipeline.types import (
    DAGValidationError,
    FileReference,
    LoadResult,
    NodeType,
    ParseResult,
    PipelineDAG,
    PipelineEdge,
    PipelineNode,
)


class TestNodeType:
    """Tests for NodeType enum."""

    def test_enum_values(self) -> None:
        """Test that enum has expected values."""
        assert NodeType.PARSER.value == "parser"
        assert NodeType.CHUNKER.value == "chunker"
        assert NodeType.EXTRACTOR.value == "extractor"
        assert NodeType.EMBEDDER.value == "embedder"

    def test_string_conversion(self) -> None:
        """Test that enum can be created from string."""
        assert NodeType("parser") == NodeType.PARSER
        assert NodeType("chunker") == NodeType.CHUNKER
        assert NodeType("extractor") == NodeType.EXTRACTOR
        assert NodeType("embedder") == NodeType.EMBEDDER

    def test_invalid_string_raises(self) -> None:
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError, match="'invalid' is not a valid NodeType"):
            NodeType("invalid")

    def test_is_string_subclass(self) -> None:
        """Test that NodeType is a string subclass for JSON serialization."""
        assert isinstance(NodeType.PARSER, str)
        assert NodeType.PARSER == "parser"


class TestFileReference:
    """Tests for FileReference dataclass."""

    def test_valid_construction_minimal(self) -> None:
        """Test creating FileReference with minimal fields."""
        ref = FileReference(
            uri="file:///path/to/doc.pdf",
            source_type="directory",
            content_type="document",
        )
        assert ref.uri == "file:///path/to/doc.pdf"
        assert ref.source_type == "directory"
        assert ref.content_type == "document"
        assert ref.filename is None
        assert ref.extension is None
        assert ref.mime_type is None
        assert ref.size_bytes == 0
        assert ref.change_hint is None
        assert ref.metadata == {}

    def test_valid_construction_all_fields(self) -> None:
        """Test creating FileReference with all fields."""
        ref = FileReference(
            uri="file:///path/to/doc.pdf",
            source_type="directory",
            content_type="document",
            filename="doc.pdf",
            extension=".pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            change_hint="mtime:1234567890",
            metadata={"source": {"author": "test", "local_path": "/path/to/doc.pdf"}},
        )
        assert ref.uri == "file:///path/to/doc.pdf"
        assert ref.filename == "doc.pdf"
        assert ref.extension == ".pdf"
        assert ref.mime_type == "application/pdf"
        assert ref.size_bytes == 1024
        assert ref.change_hint == "mtime:1234567890"
        assert ref.metadata == {"source": {"author": "test", "local_path": "/path/to/doc.pdf"}}

    def test_extension_normalization_lowercase(self) -> None:
        """Test that extension is normalized to lowercase."""
        ref = FileReference(
            uri="file:///doc.PDF",
            source_type="directory",
            content_type="document",
            extension="PDF",
        )
        assert ref.extension == ".pdf"

    def test_extension_normalization_adds_dot(self) -> None:
        """Test that extension gets leading dot added."""
        ref = FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
            extension="pdf",
        )
        assert ref.extension == ".pdf"

    def test_extension_normalization_preserves_dot(self) -> None:
        """Test that extension with dot is preserved."""
        ref = FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
            extension=".pdf",
        )
        assert ref.extension == ".pdf"

    def test_extension_empty_string_becomes_empty(self) -> None:
        """Test that empty extension remains empty."""
        ref = FileReference(
            uri="file:///doc",
            source_type="directory",
            content_type="document",
            extension="",
        )
        assert ref.extension == ""

    def test_empty_uri_raises(self) -> None:
        """Test that empty uri raises ValueError."""
        with pytest.raises(ValueError, match="uri cannot be empty"):
            FileReference(
                uri="",
                source_type="directory",
                content_type="document",
            )

    def test_empty_source_type_raises(self) -> None:
        """Test that empty source_type raises ValueError."""
        with pytest.raises(ValueError, match="source_type cannot be empty"):
            FileReference(
                uri="file:///doc.pdf",
                source_type="",
                content_type="document",
            )

    def test_empty_content_type_raises(self) -> None:
        """Test that empty content_type raises ValueError."""
        with pytest.raises(ValueError, match="content_type cannot be empty"):
            FileReference(
                uri="file:///doc.pdf",
                source_type="directory",
                content_type="",
            )

    def test_negative_size_bytes_raises(self) -> None:
        """Test that negative size_bytes raises ValueError."""
        with pytest.raises(ValueError, match="size_bytes must be >= 0"):
            FileReference(
                uri="file:///doc.pdf",
                source_type="directory",
                content_type="document",
                size_bytes=-1,
            )

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        ref = FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
            filename="doc.pdf",
            extension=".pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            change_hint="mtime:123",
            metadata={"source": {"key": "value"}},
        )
        d = ref.to_dict()
        assert d["uri"] == "file:///doc.pdf"
        assert d["source_type"] == "directory"
        assert d["content_type"] == "document"
        assert d["filename"] == "doc.pdf"
        assert d["extension"] == ".pdf"
        assert d["mime_type"] == "application/pdf"
        assert d["size_bytes"] == 1024
        assert d["change_hint"] == "mtime:123"
        # New format
        assert d["metadata"] == {"source": {"key": "value"}}
        # Legacy format for backward compat
        assert d["source_metadata"] == {"key": "value"}

    def test_from_dict_new_format(self) -> None:
        """Test deserialization from dictionary with new metadata format."""
        d = {
            "uri": "file:///doc.pdf",
            "source_type": "directory",
            "content_type": "document",
            "filename": "doc.pdf",
            "extension": ".pdf",
            "mime_type": "application/pdf",
            "size_bytes": 1024,
            "change_hint": "mtime:123",
            "metadata": {"source": {"key": "value"}},
        }
        ref = FileReference.from_dict(d)
        assert ref.uri == "file:///doc.pdf"
        assert ref.source_type == "directory"
        assert ref.content_type == "document"
        assert ref.filename == "doc.pdf"
        assert ref.extension == ".pdf"
        assert ref.mime_type == "application/pdf"
        assert ref.size_bytes == 1024
        assert ref.change_hint == "mtime:123"
        assert ref.metadata == {"source": {"key": "value"}}

    def test_from_dict_legacy_format(self) -> None:
        """Test deserialization from dictionary with legacy source_metadata format."""
        d = {
            "uri": "file:///doc.pdf",
            "source_type": "directory",
            "content_type": "document",
            "filename": "doc.pdf",
            "extension": ".pdf",
            "mime_type": "application/pdf",
            "size_bytes": 1024,
            "change_hint": "mtime:123",
            "source_metadata": {"key": "value"},
        }
        ref = FileReference.from_dict(d)
        assert ref.uri == "file:///doc.pdf"
        assert ref.source_type == "directory"
        assert ref.content_type == "document"
        # Legacy source_metadata is normalized to metadata.source
        assert ref.metadata == {"source": {"key": "value"}}

    def test_roundtrip_serialization(self) -> None:
        """Test to_dict -> from_dict preserves all fields."""
        original = FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
            filename="doc.pdf",
            extension=".pdf",
            mime_type="application/pdf",
            size_bytes=1024,
            change_hint="mtime:123",
            metadata={"source": {"nested": {"key": "value"}}},
        )
        restored = FileReference.from_dict(original.to_dict())
        assert restored.uri == original.uri
        assert restored.source_type == original.source_type
        assert restored.content_type == original.content_type
        assert restored.filename == original.filename
        assert restored.extension == original.extension
        assert restored.mime_type == original.mime_type
        assert restored.size_bytes == original.size_bytes
        assert restored.change_hint == original.change_hint
        assert restored.metadata == original.metadata

    def test_json_serializable(self) -> None:
        """Test that to_dict output is JSON serializable."""
        ref = FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
            metadata={"source": {"nested": {"key": [1, 2, 3]}}},
        )
        # Should not raise
        json_str = json.dumps(ref.to_dict())
        restored = json.loads(json_str)
        assert restored["uri"] == "file:///doc.pdf"
        assert restored["metadata"] == {"source": {"nested": {"key": [1, 2, 3]}}}

    def test_source_metadata_property_deprecated(self) -> None:
        """Test that source_metadata property emits deprecation warning."""
        import warnings

        ref = FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
            metadata={"source": {"key": "value"}},
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ref.source_metadata
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "source_metadata is deprecated" in str(w[0].message)
            assert result == {"key": "value"}

    def test_source_metadata_property_returns_empty_dict(self) -> None:
        """Test that source_metadata returns empty dict when source namespace missing."""
        import warnings

        ref = FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
            metadata={},  # No source namespace
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ref.source_metadata
            assert len(w) == 1
            assert result == {}


class TestLoadResult:
    """Tests for LoadResult dataclass."""

    @pytest.fixture()
    def file_ref(self) -> FileReference:
        """Create a test FileReference."""
        return FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
        )

    def test_valid_construction(self, file_ref: FileReference) -> None:
        """Test creating LoadResult with required fields."""
        result = LoadResult(
            file_ref=file_ref,
            content=b"test content",
            content_hash="abc123",
        )
        assert result.file_ref == file_ref
        assert result.content == b"test content"
        assert result.content_hash == "abc123"
        assert result.retention == "ephemeral"
        assert result.local_path is None
        assert result.artifact_id is None

    def test_to_dict(self, file_ref: FileReference) -> None:
        """Test serialization to dictionary."""
        result = LoadResult(
            file_ref=file_ref,
            content=b"\x00\x01\x02",
            content_hash="abc123",
            retention="cached",
            local_path="/tmp/doc.pdf",
            artifact_id="art-123",
        )
        d = result.to_dict()
        assert d["content"] == "000102"  # Hex encoded
        assert d["content_hash"] == "abc123"
        assert d["retention"] == "cached"
        assert d["local_path"] == "/tmp/doc.pdf"
        assert d["artifact_id"] == "art-123"

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        d = {
            "file_ref": {
                "uri": "file:///doc.pdf",
                "source_type": "directory",
                "content_type": "document",
            },
            "content": "000102",
            "content_hash": "abc123",
            "retention": "cached",
        }
        result = LoadResult.from_dict(d)
        assert result.content == b"\x00\x01\x02"
        assert result.content_hash == "abc123"
        assert result.retention == "cached"

    def test_roundtrip_serialization(self, file_ref: FileReference) -> None:
        """Test to_dict -> from_dict preserves all fields."""
        original = LoadResult(
            file_ref=file_ref,
            content=b"binary content",
            content_hash="abc123",
            retention="permanent",
            local_path="/tmp/doc.pdf",
            artifact_id="art-123",
        )
        restored = LoadResult.from_dict(original.to_dict())
        assert restored.content == original.content
        assert restored.content_hash == original.content_hash
        assert restored.retention == original.retention


class TestParseResult:
    """Tests for ParseResult dataclass."""

    @pytest.fixture()
    def file_ref(self) -> FileReference:
        """Create a test FileReference."""
        return FileReference(
            uri="file:///doc.pdf",
            source_type="directory",
            content_type="document",
        )

    def test_valid_construction(self, file_ref: FileReference) -> None:
        """Test creating ParseResult with required fields."""
        result = ParseResult(
            file_ref=file_ref,
            content_hash="abc123",
            text="Extracted text content",
        )
        assert result.file_ref == file_ref
        assert result.content_hash == "abc123"
        assert result.text == "Extracted text content"
        assert result.parse_metadata == {}
        assert result.local_path is None
        assert result.artifact_id is None

    def test_to_dict(self, file_ref: FileReference) -> None:
        """Test serialization to dictionary."""
        result = ParseResult(
            file_ref=file_ref,
            content_hash="abc123",
            text="Extracted text",
            parse_metadata={"pages": 10, "language": "en"},
        )
        d = result.to_dict()
        assert d["content_hash"] == "abc123"
        assert d["text"] == "Extracted text"
        assert d["parse_metadata"] == {"pages": 10, "language": "en"}

    def test_roundtrip_serialization(self, file_ref: FileReference) -> None:
        """Test to_dict -> from_dict preserves all fields."""
        original = ParseResult(
            file_ref=file_ref,
            content_hash="abc123",
            text="Content here",
            parse_metadata={"pages": 5},
            local_path="/tmp/parsed.txt",
            artifact_id="art-456",
        )
        restored = ParseResult.from_dict(original.to_dict())
        assert restored.text == original.text
        assert restored.parse_metadata == original.parse_metadata


class TestPipelineNode:
    """Tests for PipelineNode dataclass."""

    def test_valid_construction(self) -> None:
        """Test creating PipelineNode with required fields."""
        node = PipelineNode(
            id="parser-1",
            type=NodeType.PARSER,
            plugin_id="pdf-parser",
        )
        assert node.id == "parser-1"
        assert node.type == NodeType.PARSER
        assert node.plugin_id == "pdf-parser"
        assert node.config == {}

    def test_valid_construction_with_config(self) -> None:
        """Test creating PipelineNode with config."""
        node = PipelineNode(
            id="chunker-1",
            type=NodeType.CHUNKER,
            plugin_id="recursive",
            config={"chunk_size": 1000, "overlap": 100},
        )
        assert node.config == {"chunk_size": 1000, "overlap": 100}

    def test_string_type_converted_to_enum(self) -> None:
        """Test that string type is converted to enum."""
        node = PipelineNode(
            id="parser-1",
            type="parser",  # type: ignore[arg-type]
            plugin_id="pdf-parser",
        )
        assert node.type == NodeType.PARSER
        assert isinstance(node.type, NodeType)

    def test_empty_id_raises(self) -> None:
        """Test that empty id raises ValueError."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            PipelineNode(
                id="",
                type=NodeType.PARSER,
                plugin_id="pdf-parser",
            )

    def test_empty_plugin_id_raises(self) -> None:
        """Test that empty plugin_id raises ValueError."""
        with pytest.raises(ValueError, match="plugin_id cannot be empty"):
            PipelineNode(
                id="parser-1",
                type=NodeType.PARSER,
                plugin_id="",
            )

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        node = PipelineNode(
            id="parser-1",
            type=NodeType.PARSER,
            plugin_id="pdf-parser",
            config={"option": True},
        )
        d = node.to_dict()
        assert d["id"] == "parser-1"
        assert d["type"] == "parser"  # String value
        assert d["plugin_id"] == "pdf-parser"
        assert d["config"] == {"option": True}

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        d = {
            "id": "parser-1",
            "type": "parser",
            "plugin_id": "pdf-parser",
            "config": {"option": True},
        }
        node = PipelineNode.from_dict(d)
        assert node.id == "parser-1"
        assert node.type == NodeType.PARSER
        assert node.plugin_id == "pdf-parser"
        assert node.config == {"option": True}

    def test_roundtrip_serialization(self) -> None:
        """Test to_dict -> from_dict preserves all fields."""
        original = PipelineNode(
            id="embedder-1",
            type=NodeType.EMBEDDER,
            plugin_id="dense-local",
            config={"model": "qwen"},
        )
        restored = PipelineNode.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.plugin_id == original.plugin_id
        assert restored.config == original.config

    def test_json_serializable(self) -> None:
        """Test that to_dict output is JSON serializable."""
        node = PipelineNode(
            id="parser-1",
            type=NodeType.PARSER,
            plugin_id="pdf-parser",
        )
        json_str = json.dumps(node.to_dict())
        restored = json.loads(json_str)
        assert restored["type"] == "parser"


class TestPipelineEdge:
    """Tests for PipelineEdge dataclass."""

    def test_valid_construction_no_predicate(self) -> None:
        """Test creating PipelineEdge without predicate (catch-all)."""
        edge = PipelineEdge(
            from_node="_source",
            to_node="parser-1",
        )
        assert edge.from_node == "_source"
        assert edge.to_node == "parser-1"
        assert edge.when is None

    def test_valid_construction_with_predicate(self) -> None:
        """Test creating PipelineEdge with predicate."""
        edge = PipelineEdge(
            from_node="_source",
            to_node="parser-1",
            when={"mime_type": "application/pdf"},
        )
        assert edge.when == {"mime_type": "application/pdf"}

    def test_empty_from_node_raises(self) -> None:
        """Test that empty from_node raises ValueError."""
        with pytest.raises(ValueError, match="from_node cannot be empty"):
            PipelineEdge(
                from_node="",
                to_node="parser-1",
            )

    def test_empty_to_node_raises(self) -> None:
        """Test that empty to_node raises ValueError."""
        with pytest.raises(ValueError, match="to_node cannot be empty"):
            PipelineEdge(
                from_node="_source",
                to_node="",
            )

    def test_self_loop_raises(self) -> None:
        """Test that self-loop raises ValueError."""
        with pytest.raises(ValueError, match="self-loops are not allowed"):
            PipelineEdge(
                from_node="parser-1",
                to_node="parser-1",
            )

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        edge = PipelineEdge(
            from_node="_source",
            to_node="parser-1",
            when={"mime_type": "application/pdf"},
        )
        d = edge.to_dict()
        assert d["from_node"] == "_source"
        assert d["to_node"] == "parser-1"
        assert d["when"] == {"mime_type": "application/pdf"}

    def test_to_dict_none_predicate(self) -> None:
        """Test serialization with None predicate."""
        edge = PipelineEdge(
            from_node="_source",
            to_node="parser-1",
        )
        d = edge.to_dict()
        assert d["when"] is None

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        d = {
            "from_node": "_source",
            "to_node": "parser-1",
            "when": {"size_bytes": ">1000"},
        }
        edge = PipelineEdge.from_dict(d)
        assert edge.from_node == "_source"
        assert edge.to_node == "parser-1"
        assert edge.when == {"size_bytes": ">1000"}

    def test_roundtrip_serialization(self) -> None:
        """Test to_dict -> from_dict preserves all fields."""
        original = PipelineEdge(
            from_node="parser-1",
            to_node="chunker-1",
            when={"extension": [".md", ".txt"]},
        )
        restored = PipelineEdge.from_dict(original.to_dict())
        assert restored.from_node == original.from_node
        assert restored.to_node == original.to_node
        assert restored.when == original.when


class TestDAGValidationError:
    """Tests for DAGValidationError dataclass."""

    def test_valid_construction_minimal(self) -> None:
        """Test creating error with minimal fields."""
        error = DAGValidationError(
            rule="no_embedder",
            message="DAG must have exactly one EMBEDDER node",
        )
        assert error.rule == "no_embedder"
        assert error.message == "DAG must have exactly one EMBEDDER node"
        assert error.node_id is None
        assert error.edge_index is None

    def test_valid_construction_with_node_id(self) -> None:
        """Test creating error with node_id."""
        error = DAGValidationError(
            rule="unknown_plugin",
            message="Unknown plugin: bad-plugin",
            node_id="parser-1",
        )
        assert error.node_id == "parser-1"

    def test_valid_construction_with_edge_index(self) -> None:
        """Test creating error with edge_index."""
        error = DAGValidationError(
            rule="invalid_to_node",
            message="Edge references non-existent node",
            edge_index=2,
        )
        assert error.edge_index == 2

    def test_frozen(self) -> None:
        """Test that DAGValidationError is frozen (immutable)."""
        error = DAGValidationError(
            rule="test",
            message="Test error",
        )
        with pytest.raises(AttributeError):
            error.rule = "modified"  # type: ignore[misc]


class TestPipelineDAG:
    """Tests for PipelineDAG dataclass."""

    def test_valid_construction(self) -> None:
        """Test creating PipelineDAG."""
        dag = PipelineDAG(
            id="test-dag",
            version="1.0",
            nodes=[
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="embedder"),
            ],
        )
        assert dag.id == "test-dag"
        assert dag.version == "1.0"
        assert len(dag.nodes) == 1
        assert len(dag.edges) == 1

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        dag = PipelineDAG(
            id="test-dag",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="parser"),
                PipelineEdge(from_node="parser", to_node="embedder"),
            ],
        )
        d = dag.to_dict()
        assert d["id"] == "test-dag"
        assert d["version"] == "1.0"
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 2
        assert d["nodes"][0]["id"] == "parser"
        assert d["edges"][0]["from_node"] == "_source"

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        d = {
            "id": "test-dag",
            "version": "1.0",
            "nodes": [
                {"id": "embedder", "type": "embedder", "plugin_id": "dense"},
            ],
            "edges": [
                {"from_node": "_source", "to_node": "embedder"},
            ],
        }
        dag = PipelineDAG.from_dict(d)
        assert dag.id == "test-dag"
        assert dag.version == "1.0"
        assert len(dag.nodes) == 1
        assert dag.nodes[0].type == NodeType.EMBEDDER
        assert len(dag.edges) == 1

    def test_roundtrip_serialization(self) -> None:
        """Test to_dict -> from_dict preserves all fields."""
        original = PipelineDAG(
            id="complex-dag",
            version="2.0",
            nodes=[
                PipelineNode(
                    id="parser",
                    type=NodeType.PARSER,
                    plugin_id="pdf",
                    config={"ocr": True},
                ),
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(
                    from_node="_source",
                    to_node="parser",
                    when={"mime_type": "application/pdf"},
                ),
                PipelineEdge(from_node="_source", to_node="chunker"),
                PipelineEdge(from_node="parser", to_node="chunker"),
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        )
        restored = PipelineDAG.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.version == original.version
        assert len(restored.nodes) == len(original.nodes)
        assert len(restored.edges) == len(original.edges)
        assert restored.nodes[0].config == original.nodes[0].config
        assert restored.edges[0].when == original.edges[0].when

    def test_json_serializable(self) -> None:
        """Test that to_dict output is JSON serializable."""
        dag = PipelineDAG(
            id="test-dag",
            version="1.0",
            nodes=[
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="embedder"),
            ],
        )
        json_str = json.dumps(dag.to_dict())
        restored = json.loads(json_str)
        assert restored["id"] == "test-dag"

    def test_validate_returns_list(self) -> None:
        """Test that validate returns a list of errors."""
        dag = PipelineDAG(
            id="test-dag",
            version="1.0",
            nodes=[
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node="_source", to_node="embedder"),
            ],
        )
        errors = dag.validate()
        assert isinstance(errors, list)
        # This DAG should be valid
        assert len(errors) == 0

    def test_empty_nodes_list(self) -> None:
        """Test DAG with empty nodes list."""
        dag = PipelineDAG(
            id="empty-dag",
            version="1.0",
            nodes=[],
            edges=[],
        )
        errors = dag.validate()
        # Should have "no_embedder" error
        assert any(e.rule == "no_embedder" for e in errors)
