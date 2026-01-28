"""Pipeline DAG abstraction for document processing.

This package provides data structures, validation logic, and execution engine
for representing and running document processing pipelines as directed acyclic
graphs (DAGs).

Core Types:
    - NodeType: Enum of processing node types (PARSER, CHUNKER, EXTRACTOR, EMBEDDER)
    - FileReference: Reference to a file being processed
    - LoadResult: Result of loading a file's content
    - ParseResult: Result of parsing a loaded file
    - PipelineNode: A processing node in the DAG
    - PipelineEdge: An edge connecting nodes with optional predicates
    - PipelineDAG: Complete DAG definition with validation
    - DAGValidationError: Error found during validation

Execution Types:
    - ExecutionMode: Mode of execution (FULL or DRY_RUN)
    - ExecutionResult: Result of pipeline execution
    - ProgressEvent: Progress events during execution
    - ChunkStats: Statistics about chunks created
    - SampleOutput: Sample output from DRY_RUN mode
    - StageFailure: Details of a failure at a stage

Execution Components:
    - PipelineExecutor: Main execution engine
    - PipelineLoader: Content loading with hash computation
    - PipelineRouter: DAG edge matching and traversal
    - ConsecutiveFailureTracker: Failure detection for halt logic

Predicate Matching:
    - matches_predicate: Check if a file matches an edge's predicate
    - match_value: Match a single value against a pattern
    - get_nested_value: Get nested values using dot notation

Validation:
    - validate_dag: Validate DAG structure
    - SOURCE_NODE: Special ID for the entry point ("_source")

Example:
    >>> from shared.pipeline import (
    ...     PipelineDAG, PipelineNode, PipelineEdge,
    ...     NodeType, FileReference, matches_predicate
    ... )
    >>>
    >>> # Create a simple pipeline
    >>> dag = PipelineDAG(
    ...     id="simple-pipeline",
    ...     version="1.0",
    ...     nodes=[
    ...         PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf-parser"),
    ...         PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="recursive"),
    ...         PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense-local"),
    ...     ],
    ...     edges=[
    ...         PipelineEdge(from_node="_source", to_node="parser"),
    ...         PipelineEdge(from_node="parser", to_node="chunker"),
    ...         PipelineEdge(from_node="chunker", to_node="embedder"),
    ...     ],
    ... )
    >>>
    >>> # Validate the DAG
    >>> errors = dag.validate()
    >>> if not errors:
    ...     print("DAG is valid!")
    >>>
    >>> # Check predicate matching
    >>> file_ref = FileReference(
    ...     uri="file:///doc.pdf",
    ...     source_type="directory",
    ...     content_type="document",
    ...     mime_type="application/pdf",
    ...     size_bytes=1024,
    ... )
    >>> matches_predicate(file_ref, {"mime_type": "application/pdf"})
    True
"""

from shared.pipeline.defaults import get_default_pipeline
from shared.pipeline.executor import PipelineExecutionError, PipelineExecutor
from shared.pipeline.executor_types import (
    ChunkStats,
    ExecutionMode,
    ExecutionResult,
    ProgressEvent,
    SampleOutput,
    StageFailure,
)
from shared.pipeline.failure_tracker import ConsecutiveFailureTracker, FailureRecord
from shared.pipeline.loader import LoadError, PipelineLoader
from shared.pipeline.predicates import get_nested_value, match_value, matches_predicate
from shared.pipeline.router import PipelineRouter
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
from shared.pipeline.validation import SOURCE_NODE, validate_dag

__all__ = [
    # Core Types
    "NodeType",
    "FileReference",
    "LoadResult",
    "ParseResult",
    "PipelineNode",
    "PipelineEdge",
    "DAGValidationError",
    "PipelineDAG",
    # Execution Types
    "ExecutionMode",
    "ExecutionResult",
    "ProgressEvent",
    "ChunkStats",
    "SampleOutput",
    "StageFailure",
    # Execution Components
    "PipelineExecutor",
    "PipelineExecutionError",
    "PipelineLoader",
    "LoadError",
    "PipelineRouter",
    "ConsecutiveFailureTracker",
    "FailureRecord",
    # Predicates
    "matches_predicate",
    "match_value",
    "get_nested_value",
    # Validation
    "validate_dag",
    "SOURCE_NODE",
    # Defaults
    "get_default_pipeline",
]
