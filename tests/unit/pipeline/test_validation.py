"""Unit tests for pipeline DAG validation."""

from shared.pipeline.types import NodeType, PipelineDAG, PipelineEdge, PipelineNode
from shared.pipeline.validation import SOURCE_NODE, validate_dag


class TestValidSimpleDAG:
    """Tests for valid simple DAG configurations."""

    def test_minimal_valid_dag(self) -> None:
        """Test minimal valid DAG: _source -> embedder."""
        dag = PipelineDAG(
            id="minimal",
            version="1.0",
            nodes=[
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        assert len(errors) == 0

    def test_linear_pipeline_valid(self) -> None:
        """Test valid linear pipeline: _source -> parser -> chunker -> embedder."""
        dag = PipelineDAG(
            id="linear",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="parser", to_node="chunker"),
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        assert len(errors) == 0


class TestValidBranchingDAG:
    """Tests for valid branching DAG configurations."""

    def test_branching_by_predicate(self) -> None:
        """Test valid DAG with predicate-based branching."""
        dag = PipelineDAG(
            id="branching",
            version="1.0",
            nodes=[
                PipelineNode(id="pdf-parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="txt-parser", type=NodeType.PARSER, plugin_id="txt"),
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                # Branch based on mime_type
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="pdf-parser",
                    when={"mime_type": "application/pdf"},
                ),
                PipelineEdge(from_node=SOURCE_NODE, to_node="txt-parser"),  # Catch-all
                PipelineEdge(from_node="pdf-parser", to_node="chunker"),
                PipelineEdge(from_node="txt-parser", to_node="chunker"),
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        assert len(errors) == 0

    def test_multiple_paths_to_embedder(self) -> None:
        """Test valid DAG where multiple paths converge to embedder."""
        dag = PipelineDAG(
            id="converging",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="chunker-a", type=NodeType.CHUNKER, plugin_id="semantic"),
                PipelineNode(id="chunker-b", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(
                    from_node="parser",
                    to_node="chunker-a",
                    when={"source_metadata.use_semantic": True},
                ),
                PipelineEdge(from_node="parser", to_node="chunker-b"),  # Catch-all
                PipelineEdge(from_node="chunker-a", to_node="embedder"),
                PipelineEdge(from_node="chunker-b", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        assert len(errors) == 0


class TestRule1NoEmbedder:
    """Tests for Rule 1: Exactly one EMBEDDER node."""

    def test_no_embedder_error(self) -> None:
        """Test error when DAG has no EMBEDDER node."""
        dag = PipelineDAG(
            id="no-embedder",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="recursive"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="parser", to_node="chunker"),
            ],
        )
        errors = validate_dag(dag)
        assert any(e.rule == "no_embedder" for e in errors)

    def test_multiple_embedders_error(self) -> None:
        """Test error when DAG has multiple EMBEDDER nodes."""
        dag = PipelineDAG(
            id="multi-embedder",
            version="1.0",
            nodes=[
                PipelineNode(id="embedder-1", type=NodeType.EMBEDDER, plugin_id="dense"),
                PipelineNode(id="embedder-2", type=NodeType.EMBEDDER, plugin_id="sparse"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="embedder-1"),
                PipelineEdge(from_node=SOURCE_NODE, to_node="embedder-2"),
            ],
        )
        errors = validate_dag(dag)
        embedder_errors = [e for e in errors if e.rule == "multiple_embedders"]
        assert len(embedder_errors) == 2
        assert {e.node_id for e in embedder_errors} == {"embedder-1", "embedder-2"}


class TestRule2InvalidNodeRefs:
    """Tests for Rule 2: All edge node refs exist."""

    def test_invalid_from_node_error(self) -> None:
        """Test error when edge references non-existent from_node."""
        dag = PipelineDAG(
            id="bad-from",
            version="1.0",
            nodes=[
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="embedder"),
                PipelineEdge(from_node="nonexistent", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        assert any(e.rule == "invalid_from_node" for e in errors)

    def test_invalid_to_node_error(self) -> None:
        """Test error when edge references non-existent to_node."""
        dag = PipelineDAG(
            id="bad-to",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="parser", to_node="nonexistent"),
                PipelineEdge(from_node=SOURCE_NODE, to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        assert any(e.rule == "invalid_to_node" for e in errors)

    def test_source_node_valid_as_from(self) -> None:
        """Test that _source is valid as from_node."""
        dag = PipelineDAG(
            id="source-valid",
            version="1.0",
            nodes=[
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        # _source should not cause invalid_from_node error
        assert not any(e.rule == "invalid_from_node" for e in errors)


class TestRule3UnreachableNodes:
    """Tests for Rule 3: Every node reachable from _source."""

    def test_unreachable_node_error(self) -> None:
        """Test error when node is not reachable from _source."""
        dag = PipelineDAG(
            id="unreachable",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="orphan", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="parser", to_node="embedder"),
                # orphan has no incoming edge from source-reachable nodes
            ],
        )
        errors = validate_dag(dag)
        unreachable = [e for e in errors if e.rule == "unreachable_node"]
        assert len(unreachable) == 1
        assert unreachable[0].node_id == "orphan"

    def test_all_nodes_reachable(self) -> None:
        """Test no unreachable error when all nodes are reachable."""
        dag = PipelineDAG(
            id="all-reachable",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="parser", to_node="chunker"),
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        assert not any(e.rule == "unreachable_node" for e in errors)


class TestRule4NoPathToTerminal:
    """Tests for Rule 4: Every node has path to a valid terminal (embedder or extractor)."""

    def test_no_path_to_terminal_error(self) -> None:
        """Test error when node has no path to embedder or extractor."""
        dag = PipelineDAG(
            id="dead-end",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(
                    id="dead-end", type=NodeType.CHUNKER, plugin_id="orphan"
                ),  # Chunker is NOT a valid terminal
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="parser", to_node="chunker"),
                PipelineEdge(from_node="parser", to_node="dead-end"),  # Branch to dead-end chunker
                PipelineEdge(from_node="chunker", to_node="embedder"),
                # dead-end chunker has no outgoing edge - invalid because chunkers aren't terminals
            ],
        )
        errors = validate_dag(dag)
        no_path = [e for e in errors if e.rule == "no_path_to_terminal"]
        assert len(no_path) == 1
        assert no_path[0].node_id == "dead-end"

    def test_extractor_is_valid_terminal(self) -> None:
        """Test that extractor nodes are valid terminals (no error for paths ending at extractor)."""
        dag = PipelineDAG(
            id="extractor-terminal",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="extractor", type=NodeType.EXTRACTOR, plugin_id="meta"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="parser", to_node="chunker"),
                PipelineEdge(from_node="parser", to_node="extractor", parallel=True),  # Parallel path to extractor
                PipelineEdge(from_node="chunker", to_node="embedder"),
                # extractor has no outgoing edge - valid because extractors ARE terminals
            ],
        )
        errors = validate_dag(dag)
        no_path = [e for e in errors if e.rule == "no_path_to_terminal"]
        assert len(no_path) == 0  # No error - extractor is a valid terminal

    def test_all_nodes_reach_embedder(self) -> None:
        """Test no error when all nodes can reach embedder."""
        dag = PipelineDAG(
            id="all-reach",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="extractor", type=NodeType.EXTRACTOR, plugin_id="meta"),
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="parser", to_node="extractor"),
                PipelineEdge(from_node="extractor", to_node="chunker"),
                PipelineEdge(from_node="chunker", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        assert not any(e.rule == "no_path_to_terminal" for e in errors)


class TestRule5NoCycles:
    """Tests for Rule 5: No cycles."""

    def test_cycle_detected_error(self) -> None:
        """Test error when DAG contains a cycle."""
        dag = PipelineDAG(
            id="cyclic",
            version="1.0",
            nodes=[
                PipelineNode(id="a", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="b", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="c", type=NodeType.EXTRACTOR, plugin_id="meta"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="a"),
                PipelineEdge(from_node="a", to_node="b"),
                PipelineEdge(from_node="b", to_node="c"),
                PipelineEdge(from_node="c", to_node="a"),  # Creates cycle: a -> b -> c -> a
                PipelineEdge(from_node="b", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        assert any(e.rule == "cycle_detected" for e in errors)

    def test_no_cycle_linear(self) -> None:
        """Test no cycle error for linear DAG."""
        dag = PipelineDAG(
            id="linear",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="parser", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        assert not any(e.rule == "cycle_detected" for e in errors)

    def test_no_cycle_diamond(self) -> None:
        """Test no cycle error for diamond-shaped DAG (converging paths)."""
        dag = PipelineDAG(
            id="diamond",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="chunker-a", type=NodeType.CHUNKER, plugin_id="semantic"),
                PipelineNode(id="chunker-b", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="parser", to_node="chunker-a"),
                PipelineEdge(from_node="parser", to_node="chunker-b"),
                PipelineEdge(from_node="chunker-a", to_node="embedder"),
                PipelineEdge(from_node="chunker-b", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        assert not any(e.rule == "cycle_detected" for e in errors)


class TestRule6NoSourceCatchall:
    """Tests for Rule 6: At least one catch-all from _source."""

    def test_no_source_catchall_error(self) -> None:
        """Test error when no catch-all edge from _source."""
        dag = PipelineDAG(
            id="no-catchall",
            version="1.0",
            nodes=[
                PipelineNode(id="pdf-parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                # All edges from _source have predicates (no catch-all)
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="pdf-parser",
                    when={"mime_type": "application/pdf"},
                ),
                PipelineEdge(from_node="pdf-parser", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        assert any(e.rule == "no_source_catchall" for e in errors)

    def test_source_catchall_none(self) -> None:
        """Test no error when catch-all edge has when=None."""
        dag = PipelineDAG(
            id="catchall-none",
            version="1.0",
            nodes=[
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="embedder", when=None),
            ],
        )
        errors = validate_dag(dag)
        assert not any(e.rule == "no_source_catchall" for e in errors)

    def test_source_catchall_empty_dict(self) -> None:
        """Test no error when catch-all edge has when={}."""
        dag = PipelineDAG(
            id="catchall-empty",
            version="1.0",
            nodes=[
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="embedder", when={}),
            ],
        )
        errors = validate_dag(dag)
        assert not any(e.rule == "no_source_catchall" for e in errors)


class TestRule7DuplicateNodeId:
    """Tests for Rule 7: Node IDs unique."""

    def test_duplicate_node_id_error(self) -> None:
        """Test error when DAG has duplicate node IDs."""
        dag = PipelineDAG(
            id="duplicate-ids",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="txt"),  # Duplicate
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="parser", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        dup_errors = [e for e in errors if e.rule == "duplicate_node_id"]
        assert len(dup_errors) == 1
        assert dup_errors[0].node_id == "parser"

    def test_unique_node_ids(self) -> None:
        """Test no error when all node IDs are unique."""
        dag = PipelineDAG(
            id="unique-ids",
            version="1.0",
            nodes=[
                PipelineNode(id="parser-1", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="parser-2", type=NodeType.PARSER, plugin_id="txt"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser-1"),
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser-2"),
                PipelineEdge(from_node="parser-1", to_node="embedder"),
                PipelineEdge(from_node="parser-2", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        assert not any(e.rule == "duplicate_node_id" for e in errors)


class TestRule8UnknownPlugin:
    """Tests for Rule 8: Plugin IDs registered."""

    def test_unknown_plugin_error(self) -> None:
        """Test error when plugin ID not in known_plugins."""
        dag = PipelineDAG(
            id="unknown-plugin",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="unknown-parser"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="parser", to_node="embedder"),
            ],
        )
        known_plugins = {"dense", "recursive"}
        errors = validate_dag(dag, known_plugins)
        unknown = [e for e in errors if e.rule == "unknown_plugin"]
        assert len(unknown) == 1
        assert unknown[0].node_id == "parser"

    def test_all_plugins_known(self) -> None:
        """Test no error when all plugins are known."""
        dag = PipelineDAG(
            id="all-known",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="parser", to_node="embedder"),
            ],
        )
        known_plugins = {"pdf", "dense", "recursive"}
        errors = validate_dag(dag, known_plugins)
        assert not any(e.rule == "unknown_plugin" for e in errors)

    def test_no_validation_without_known_plugins(self) -> None:
        """Test no plugin validation when known_plugins is None."""
        dag = PipelineDAG(
            id="no-plugin-check",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="unknown"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="also-unknown"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="parser", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag, known_plugins=None)
        assert not any(e.rule == "unknown_plugin" for e in errors)


class TestMultipleErrors:
    """Tests for DAGs with multiple validation errors."""

    def test_multiple_errors_reported(self) -> None:
        """Test that multiple errors are reported together."""
        dag = PipelineDAG(
            id="multi-error",
            version="1.0",
            nodes=[
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="pdf"),
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="txt"),  # Duplicate
                PipelineNode(id="orphan", type=NodeType.CHUNKER, plugin_id="recursive"),
                # No embedder
            ],
            edges=[
                # No catch-all from _source
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="parser",
                    when={"mime_type": "application/pdf"},
                ),
                PipelineEdge(from_node="parser", to_node="nonexistent"),  # Invalid to_node
            ],
        )
        errors = validate_dag(dag)
        error_rules = {e.rule for e in errors}
        assert "no_embedder" in error_rules
        assert "duplicate_node_id" in error_rules
        assert "invalid_to_node" in error_rules
        assert "no_source_catchall" in error_rules
        assert "unreachable_node" in error_rules


class TestPipelineDAGValidateMethod:
    """Tests for PipelineDAG.validate() method."""

    def test_validate_method_returns_errors(self) -> None:
        """Test that PipelineDAG.validate() returns errors."""
        dag = PipelineDAG(
            id="empty",
            version="1.0",
            nodes=[],
            edges=[],
        )
        errors = dag.validate()
        assert isinstance(errors, list)
        assert len(errors) > 0
        assert any(e.rule == "no_embedder" for e in errors)

    def test_validate_method_with_known_plugins(self) -> None:
        """Test that PipelineDAG.validate() accepts known_plugins."""
        dag = PipelineDAG(
            id="test",
            version="1.0",
            nodes=[
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="unknown"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="embedder"),
            ],
        )
        errors = dag.validate(known_plugins={"known"})
        assert any(e.rule == "unknown_plugin" for e in errors)

    def test_validate_method_valid_dag(self) -> None:
        """Test that valid DAG returns empty error list."""
        dag = PipelineDAG(
            id="valid",
            version="1.0",
            nodes=[
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(from_node=SOURCE_NODE, to_node="embedder"),
            ],
        )
        errors = dag.validate()
        assert errors == []


class TestSourceNodeConstant:
    """Tests for SOURCE_NODE constant."""

    def test_source_node_value(self) -> None:
        """Test SOURCE_NODE has expected value."""
        assert SOURCE_NODE == "_source"


class TestRule9DuplicatePathNames:
    """Tests for Rule 9: Parallel edges must have unique path_names."""

    def test_duplicate_path_name_error(self) -> None:
        """Test error when parallel edges have duplicate path_names."""
        dag = PipelineDAG(
            id="duplicate-path",
            version="1.0",
            nodes=[
                PipelineNode(id="chunker-a", type=NodeType.CHUNKER, plugin_id="a"),
                PipelineNode(id="chunker-b", type=NodeType.CHUNKER, plugin_id="b"),
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="text"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                # Two parallel edges with same explicit path_name
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="chunker-a",
                    parallel=True,
                    path_name="same-path",
                ),
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="chunker-b",
                    parallel=True,
                    path_name="same-path",  # Duplicate!
                ),
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="chunker-a", to_node="embedder"),
                PipelineEdge(from_node="chunker-b", to_node="embedder"),
                PipelineEdge(from_node="parser", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        dup_errors = [e for e in errors if e.rule == "duplicate_path_name"]
        assert len(dup_errors) == 1
        assert "same-path" in dup_errors[0].message

    def test_duplicate_implicit_path_name_error(self) -> None:
        """Test error when parallel edges have duplicate implicit path_names (to_node)."""
        # This would require two parallel edges going to the same node,
        # which would be weird but possible if someone creates it incorrectly.
        # The path_name defaults to to_node, so two edges to same node = duplicate.
        dag = PipelineDAG(
            id="implicit-duplicate",
            version="1.0",
            nodes=[
                PipelineNode(id="chunker", type=NodeType.CHUNKER, plugin_id="recursive"),
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="text"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                # Two parallel edges to same node (implicit path_name = "chunker")
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="chunker",
                    when={"extension": ".txt"},
                    parallel=True,
                ),
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="chunker",
                    when={"extension": ".md"},
                    parallel=True,
                ),
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="chunker", to_node="embedder"),
                PipelineEdge(from_node="parser", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        dup_errors = [e for e in errors if e.rule == "duplicate_path_name"]
        # Should flag duplicate "chunker" path_name
        assert len(dup_errors) == 1
        assert "chunker" in dup_errors[0].message

    def test_unique_path_names_no_error(self) -> None:
        """Test no error when parallel edges have unique path_names."""
        dag = PipelineDAG(
            id="unique-paths",
            version="1.0",
            nodes=[
                PipelineNode(id="chunker-a", type=NodeType.CHUNKER, plugin_id="a"),
                PipelineNode(id="chunker-b", type=NodeType.CHUNKER, plugin_id="b"),
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="text"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="chunker-a",
                    parallel=True,
                    path_name="detailed",
                ),
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="chunker-b",
                    parallel=True,
                    path_name="summary",
                ),
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="chunker-a", to_node="embedder"),
                PipelineEdge(from_node="chunker-b", to_node="embedder"),
                PipelineEdge(from_node="parser", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        dup_errors = [e for e in errors if e.rule == "duplicate_path_name"]
        assert len(dup_errors) == 0

    def test_single_parallel_edge_no_error(self) -> None:
        """Test no error when only one parallel edge from a node."""
        dag = PipelineDAG(
            id="single-parallel",
            version="1.0",
            nodes=[
                PipelineNode(id="chunker-parallel", type=NodeType.CHUNKER, plugin_id="a"),
                PipelineNode(id="parser", type=NodeType.PARSER, plugin_id="text"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="chunker-parallel",
                    parallel=True,
                    path_name="parallel-path",
                ),
                PipelineEdge(from_node=SOURCE_NODE, to_node="parser"),
                PipelineEdge(from_node="chunker-parallel", to_node="embedder"),
                PipelineEdge(from_node="parser", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        dup_errors = [e for e in errors if e.rule == "duplicate_path_name"]
        assert len(dup_errors) == 0

    def test_non_parallel_edges_not_checked(self) -> None:
        """Test that non-parallel edges are not checked for path_name uniqueness."""
        dag = PipelineDAG(
            id="non-parallel",
            version="1.0",
            nodes=[
                PipelineNode(id="chunker-a", type=NodeType.CHUNKER, plugin_id="a"),
                PipelineNode(id="chunker-b", type=NodeType.CHUNKER, plugin_id="b"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                # Two non-parallel edges (exclusive) - path_name doesn't matter
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="chunker-a",
                    when={"extension": ".txt"},
                    parallel=False,
                ),
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="chunker-b",
                    when={"extension": ".md"},
                    parallel=False,
                ),
                PipelineEdge(from_node=SOURCE_NODE, to_node="chunker-a"),  # Catch-all
                PipelineEdge(from_node="chunker-a", to_node="embedder"),
                PipelineEdge(from_node="chunker-b", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        dup_errors = [e for e in errors if e.rule == "duplicate_path_name"]
        # No duplicate_path_name errors because none are parallel
        assert len(dup_errors) == 0


class TestRule6SourceCatchallWithParallel:
    """Tests for Rule 6: Catch-all requirements with parallel edges."""

    def test_only_parallel_catchall_error(self) -> None:
        """Test error when only parallel catch-all edges exist from _source."""
        dag = PipelineDAG(
            id="only-parallel-catchall",
            version="1.0",
            nodes=[
                PipelineNode(id="chunker-a", type=NodeType.CHUNKER, plugin_id="a"),
                PipelineNode(id="chunker-b", type=NodeType.CHUNKER, plugin_id="b"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                # Both catch-all edges are parallel - no exclusive catch-all!
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="chunker-a",
                    parallel=True,
                    path_name="path-a",
                ),
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="chunker-b",
                    parallel=True,
                    path_name="path-b",
                ),
                PipelineEdge(from_node="chunker-a", to_node="embedder"),
                PipelineEdge(from_node="chunker-b", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        catchall_errors = [e for e in errors if e.rule == "no_source_catchall"]
        # Should have error because there's no non-parallel catch-all
        assert len(catchall_errors) == 1
        assert "non-parallel" in catchall_errors[0].message

    def test_parallel_with_exclusive_catchall_no_error(self) -> None:
        """Test no error when parallel edges exist with exclusive catch-all."""
        dag = PipelineDAG(
            id="parallel-with-catchall",
            version="1.0",
            nodes=[
                PipelineNode(id="chunker-parallel", type=NodeType.CHUNKER, plugin_id="parallel"),
                PipelineNode(id="parser-default", type=NodeType.PARSER, plugin_id="text"),
                PipelineNode(id="embedder", type=NodeType.EMBEDDER, plugin_id="dense"),
            ],
            edges=[
                # Parallel edge
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="chunker-parallel",
                    parallel=True,
                    path_name="parallel-path",
                ),
                # Non-parallel catch-all
                PipelineEdge(
                    from_node=SOURCE_NODE,
                    to_node="parser-default",
                    parallel=False,
                ),
                PipelineEdge(from_node="chunker-parallel", to_node="embedder"),
                PipelineEdge(from_node="parser-default", to_node="embedder"),
            ],
        )
        errors = validate_dag(dag)
        catchall_errors = [e for e in errors if e.rule == "no_source_catchall"]
        assert len(catchall_errors) == 0
