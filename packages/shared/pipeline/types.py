"""Core type definitions for the Pipeline DAG abstraction.

This module defines the data structures for representing document processing
pipelines as directed acyclic graphs (DAGs). Each node represents a processing
step (parser, chunker, extractor, embedder), and edges define the flow of
documents through the pipeline with optional predicate-based routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    """Type of processing node in the pipeline DAG.

    Each node type represents a specific stage in document processing:
    - PARSER: Extracts text content from raw documents (PDF, DOCX, etc.)
    - CHUNKER: Splits text into smaller chunks for embedding
    - EXTRACTOR: Extracts metadata or entities from content
    - EMBEDDER: Generates vector embeddings for semantic search
    """

    PARSER = "parser"
    CHUNKER = "chunker"
    EXTRACTOR = "extractor"
    EMBEDDER = "embedder"


@dataclass
class FileReference:
    """Reference to a file being processed in the pipeline.

    This is the primary input to the pipeline DAG, representing a document
    from any source (local file, web, Slack, etc.) with its metadata.

    Attributes:
        uri: Unique identifier for the file (file://, https://, slack://, etc.)
        source_type: Source connector type (e.g., "directory", "web", "slack")
        content_type: Semantic content type (e.g., "document", "message", "code")
        filename: Original filename if available
        extension: File extension (normalized to lowercase with dot)
        mime_type: MIME type if known (e.g., "application/pdf")
        size_bytes: File size in bytes (must be >= 0)
        change_hint: Optional hint for change detection (mtime, etag, hash)
        source_metadata: Additional source-specific metadata
    """

    uri: str
    source_type: str
    content_type: str
    filename: str | None = None
    extension: str | None = None
    mime_type: str | None = None
    size_bytes: int = 0
    change_hint: str | None = None
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not self.uri:
            raise ValueError("uri cannot be empty")
        if not self.source_type:
            raise ValueError("source_type cannot be empty")
        if not self.content_type:
            raise ValueError("content_type cannot be empty")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be >= 0")

        # Normalize extension to lowercase with leading dot
        if self.extension is not None:
            ext = self.extension.lower()
            if ext and not ext.startswith("."):
                ext = f".{ext}"
            self.extension = ext

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "uri": self.uri,
            "source_type": self.source_type,
            "content_type": self.content_type,
            "filename": self.filename,
            "extension": self.extension,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "change_hint": self.change_hint,
            "source_metadata": dict(self.source_metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileReference:
        """Create a FileReference from a dictionary."""
        return cls(
            uri=data["uri"],
            source_type=data["source_type"],
            content_type=data["content_type"],
            filename=data.get("filename"),
            extension=data.get("extension"),
            mime_type=data.get("mime_type"),
            size_bytes=data.get("size_bytes", 0),
            change_hint=data.get("change_hint"),
            source_metadata=data.get("source_metadata", {}),
        )


@dataclass
class LoadResult:
    """Result of loading a file's content.

    Produced by the loading stage (before parsing), containing the raw
    content bytes and metadata about the loaded file.

    Attributes:
        file_ref: Original file reference
        content: Raw file content as bytes
        content_hash: SHA-256 hash of content (for deduplication)
        retention: Content retention policy ("ephemeral", "cached", "permanent")
        local_path: Optional local filesystem path for cached content
        artifact_id: Optional artifact ID for content stored in artifact system
    """

    file_ref: FileReference
    content: bytes
    content_hash: str
    retention: str = "ephemeral"
    local_path: str | None = None
    artifact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "file_ref": self.file_ref.to_dict(),
            "content": self.content.hex(),  # Encode bytes as hex string
            "content_hash": self.content_hash,
            "retention": self.retention,
            "local_path": self.local_path,
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoadResult:
        """Create a LoadResult from a dictionary."""
        return cls(
            file_ref=FileReference.from_dict(data["file_ref"]),
            content=bytes.fromhex(data["content"]),
            content_hash=data["content_hash"],
            retention=data.get("retention", "ephemeral"),
            local_path=data.get("local_path"),
            artifact_id=data.get("artifact_id"),
        )


@dataclass
class ParseResult:
    """Result of parsing a loaded file.

    Produced by parser nodes, containing the extracted text content
    and any parsing metadata.

    Attributes:
        file_ref: Original file reference
        content_hash: SHA-256 hash of the parsed text
        text: Extracted text content
        parse_metadata: Parser-specific metadata (pages, language, etc.)
        local_path: Optional local path for cached parsed content
        artifact_id: Optional artifact ID for parsed content
    """

    file_ref: FileReference
    content_hash: str
    text: str
    parse_metadata: dict[str, Any] = field(default_factory=dict)
    local_path: str | None = None
    artifact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "file_ref": self.file_ref.to_dict(),
            "content_hash": self.content_hash,
            "text": self.text,
            "parse_metadata": dict(self.parse_metadata),
            "local_path": self.local_path,
            "artifact_id": self.artifact_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParseResult:
        """Create a ParseResult from a dictionary."""
        return cls(
            file_ref=FileReference.from_dict(data["file_ref"]),
            content_hash=data["content_hash"],
            text=data["text"],
            parse_metadata=data.get("parse_metadata", {}),
            local_path=data.get("local_path"),
            artifact_id=data.get("artifact_id"),
        )


@dataclass
class PipelineNode:
    """A processing node in the pipeline DAG.

    Each node represents a specific processing step with a plugin that
    implements the actual processing logic.

    Attributes:
        id: Unique identifier for this node within the DAG
        type: Type of processing (parser, chunker, extractor, embedder)
        plugin_id: ID of the plugin that implements this node's processing
        config: Node-specific configuration passed to the plugin
    """

    id: str
    type: NodeType
    plugin_id: str
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not self.id:
            raise ValueError("id cannot be empty")
        if not self.plugin_id:
            raise ValueError("plugin_id cannot be empty")

        # Convert string type to enum if needed
        if isinstance(self.type, str):
            self.type = NodeType(self.type)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "id": self.id,
            "type": self.type.value,
            "plugin_id": self.plugin_id,
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineNode:
        """Create a PipelineNode from a dictionary."""
        return cls(
            id=data["id"],
            type=NodeType(data["type"]),
            plugin_id=data["plugin_id"],
            config=data.get("config", {}),
        )


@dataclass
class PipelineEdge:
    """An edge connecting two nodes in the pipeline DAG.

    Edges define the flow of data through the pipeline. Each edge can
    have an optional predicate (`when`) that determines whether this
    edge should be taken based on the input file's attributes.

    Attributes:
        from_node: Source node ID (or "_source" for entry edges)
        to_node: Target node ID
        when: Optional predicate for conditional routing (None = catch-all)
    """

    from_node: str
    to_node: str
    when: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not self.from_node:
            raise ValueError("from_node cannot be empty")
        if not self.to_node:
            raise ValueError("to_node cannot be empty")
        if self.from_node == self.to_node:
            raise ValueError("self-loops are not allowed")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "when": self.when,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineEdge:
        """Create a PipelineEdge from a dictionary."""
        return cls(
            from_node=data["from_node"],
            to_node=data["to_node"],
            when=data.get("when"),
        )


@dataclass(frozen=True)
class DAGValidationError:
    """An error found during DAG validation.

    Attributes:
        rule: Error code identifying the validation rule that failed
        message: Human-readable description of the error
        node_id: ID of the node involved (if applicable)
        edge_index: Index of the edge involved (if applicable)
    """

    rule: str
    message: str
    node_id: str | None = None
    edge_index: int | None = None


class DAGValidationException(Exception):
    """Exception raised when DAG validation fails.

    Attributes:
        errors: List of validation errors encountered
    """

    def __init__(self, errors: list[DAGValidationError]) -> None:
        self.errors = errors
        error_messages = "; ".join(e.message for e in errors)
        super().__init__(f"DAG validation failed: {error_messages}")


@dataclass
class PipelineDAG:
    """A complete pipeline DAG definition.

    The DAG represents a document processing pipeline with nodes for each
    processing step and edges defining the data flow. The special "_source"
    node represents the entry point for files entering the pipeline.

    Attributes:
        id: Unique identifier for this DAG definition
        version: Schema version for forward compatibility
        nodes: List of processing nodes in the DAG
        edges: List of edges defining the data flow
    """

    id: str
    version: str
    nodes: list[PipelineNode]
    edges: list[PipelineEdge]

    def validate(self, known_plugins: set[str] | None = None) -> list[DAGValidationError]:
        """Validate the DAG structure and return any errors.

        Validation rules:
        1. Exactly one EMBEDDER node
        2. All edge node refs exist (or are "_source")
        3. Every node is reachable from _source
        4. Every node has a path to the embedder
        5. No cycles
        6. At least one catch-all edge from _source
        7. Node IDs are unique
        8. Plugin IDs are registered (if known_plugins provided)

        Args:
            known_plugins: Optional set of registered plugin IDs for validation

        Returns:
            List of validation errors (empty if valid)
        """
        # Import here to avoid circular dependency
        from shared.pipeline.validation import validate_dag

        result: list[DAGValidationError] = validate_dag(self, known_plugins)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "id": self.id,
            "version": self.version,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineDAG:
        """Create a PipelineDAG from a dictionary."""
        return cls(
            id=data["id"],
            version=data["version"],
            nodes=[PipelineNode.from_dict(n) for n in data.get("nodes", [])],
            edges=[PipelineEdge.from_dict(e) for e in data.get("edges", [])],
        )

    @classmethod
    def create_validated(
        cls,
        id: str,
        version: str,
        nodes: list[PipelineNode],
        edges: list[PipelineEdge],
        known_plugins: set[str] | None = None,
    ) -> PipelineDAG:
        """Create a PipelineDAG with validation.

        Factory method that creates a DAG and validates it immediately,
        raising an exception if validation fails. Use this to ensure
        DAG-level invariants are enforced at construction time.

        Args:
            id: Unique identifier for this DAG definition
            version: Schema version for forward compatibility
            nodes: List of processing nodes in the DAG
            edges: List of edges defining the data flow
            known_plugins: Optional set of registered plugin IDs for validation

        Returns:
            A validated PipelineDAG instance

        Raises:
            DAGValidationException: If the DAG fails validation
        """
        dag = cls(id=id, version=version, nodes=nodes, edges=edges)
        errors = dag.validate(known_plugins)
        if errors:
            raise DAGValidationException(errors)
        return dag


__all__ = [
    "NodeType",
    "FileReference",
    "LoadResult",
    "ParseResult",
    "PipelineNode",
    "PipelineEdge",
    "DAGValidationError",
    "DAGValidationException",
    "PipelineDAG",
]
