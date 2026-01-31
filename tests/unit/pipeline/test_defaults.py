"""Unit tests for default pipeline generation."""

from shared.pipeline.defaults import get_default_pipeline
from shared.pipeline.types import NodeType


class TestGetDefaultPipeline:
    """Tests for the get_default_pipeline function."""

    def test_creates_valid_dag(self) -> None:
        """Test that a valid DAG is created with minimal config."""
        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={"chunk_size": 1000, "chunk_overlap": 200},
        )
        errors = dag.validate()
        assert not errors, f"Unexpected validation errors: {errors}"
        assert len(dag.nodes) == 4  # 2 parsers + chunker + embedder

    def test_respects_chunking_strategy(self) -> None:
        """Test that chunking_strategy is applied to the chunker node."""
        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={"chunking_strategy": "semantic", "chunk_size": 500},
        )
        chunker = next(n for n in dag.nodes if n.type == NodeType.CHUNKER)
        assert chunker.plugin_id == "semantic"
        assert chunker.config["max_tokens"] == 500

    def test_defaults_to_recursive_strategy(self) -> None:
        """Test that recursive strategy is used when not specified."""
        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={"chunk_size": 1000},
        )
        chunker = next(n for n in dag.nodes if n.type == NodeType.CHUNKER)
        assert chunker.plugin_id == "recursive"

    def test_has_catch_all_edge(self) -> None:
        """Test that there is a catch-all edge from _source."""
        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={},
        )
        catch_all = [e for e in dag.edges if e.from_node == "_source" and e.when is None]
        assert len(catch_all) == 1

    def test_has_conditional_edge_for_pdf(self) -> None:
        """Test that there is a conditional edge for PDF files."""
        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={},
        )
        pdf_edges = [e for e in dag.edges if e.from_node == "_source" and e.when and "mime_type" in e.when]
        assert len(pdf_edges) == 1
        assert "application/pdf" in pdf_edges[0].when["mime_type"]

    def test_embedding_model_in_config(self) -> None:
        """Test that embedding model is stored in embedder node config."""
        dag = get_default_pipeline(
            embedding_model="custom/embedding-model",
            chunk_config={},
        )
        embedder = next(n for n in dag.nodes if n.type == NodeType.EMBEDDER)
        assert embedder.config["model"] == "custom/embedding-model"

    def test_chunk_overlap_in_config(self) -> None:
        """Test that chunk_overlap is converted to overlap_tokens."""
        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={"chunk_size": 1000, "chunk_overlap": 150},
        )
        chunker = next(n for n in dag.nodes if n.type == NodeType.CHUNKER)
        assert chunker.config["overlap_tokens"] == 150

    def test_min_tokens_calculated_from_chunk_size(self) -> None:
        """Test that min_tokens is derived from chunk_size and overlap constraint.

        min_tokens = max(base_min, overlap + 1) where base_min = max(100, chunk_size // 10)
        This ensures min_tokens > overlap_tokens for chunker validation.
        """
        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={"chunk_size": 2000},  # No explicit overlap, defaults to 200
        )
        chunker = next(n for n in dag.nodes if n.type == NodeType.CHUNKER)
        # base_min = max(100, 2000 // 10) = 200
        # min_tokens = max(200, 200 + 1) = 201 (must exceed overlap)
        assert chunker.config["min_tokens"] == 201

    def test_min_tokens_always_exceeds_overlap(self) -> None:
        """Test that min_tokens is always > overlap_tokens to satisfy chunker constraint."""
        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={"chunk_size": 500},  # No explicit overlap, defaults to 200
        )
        chunker = next(n for n in dag.nodes if n.type == NodeType.CHUNKER)
        # base_min = max(100, 500 // 10) = 100
        # min_tokens = max(100, 200 + 1) = 201 (must exceed overlap of 200)
        assert chunker.config["min_tokens"] == 201
        assert chunker.config["min_tokens"] > chunker.config["overlap_tokens"]

    def test_merges_chunking_config(self) -> None:
        """Test that strategy-specific config is merged."""
        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={
                "chunk_size": 1000,
                "chunking_config": {"preserve_sentences": True, "custom_key": "value"},
            },
        )
        chunker = next(n for n in dag.nodes if n.type == NodeType.CHUNKER)
        assert chunker.config["preserve_sentences"] is True
        assert chunker.config["custom_key"] == "value"
        # Standard fields should still be present
        assert chunker.config["max_tokens"] == 1000

    def test_dag_has_correct_node_types(self) -> None:
        """Test that all expected node types are present."""
        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={},
        )
        node_types = {n.type for n in dag.nodes}
        assert NodeType.PARSER in node_types
        assert NodeType.CHUNKER in node_types
        assert NodeType.EMBEDDER in node_types

    def test_parsers_connect_to_chunker(self) -> None:
        """Test that both parsers connect to the chunker."""
        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={},
        )
        parser_to_chunker_edges = [
            e for e in dag.edges if e.to_node == "chunker" and e.from_node in ("unstructured_parser", "text_parser")
        ]
        assert len(parser_to_chunker_edges) == 2

    def test_chunker_connects_to_embedder(self) -> None:
        """Test that chunker connects to embedder."""
        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={},
        )
        chunker_to_embedder_edges = [e for e in dag.edges if e.from_node == "chunker" and e.to_node == "embedder"]
        assert len(chunker_to_embedder_edges) == 1

    def test_serialization_roundtrip(self) -> None:
        """Test that the DAG can be serialized and deserialized."""
        from shared.pipeline.types import PipelineDAG

        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={"chunk_size": 800, "chunking_strategy": "markdown"},
        )

        # Serialize to dict
        dag_dict = dag.to_dict()

        # Deserialize back
        restored_dag = PipelineDAG.from_dict(dag_dict)

        # Validate restored DAG
        errors = restored_dag.validate()
        assert not errors

        # Check key properties preserved
        assert restored_dag.id == dag.id
        assert restored_dag.version == dag.version
        assert len(restored_dag.nodes) == len(dag.nodes)
        assert len(restored_dag.edges) == len(dag.edges)

    def test_default_chunk_values(self) -> None:
        """Test that default values are used when not specified."""
        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={},  # Empty config
        )
        chunker = next(n for n in dag.nodes if n.type == NodeType.CHUNKER)
        # Defaults: chunk_size=1000, chunk_overlap=200
        assert chunker.config["max_tokens"] == 1000
        assert chunker.config["overlap_tokens"] == 200

    def test_none_chunking_strategy_defaults_to_recursive(self) -> None:
        """Test that None chunking_strategy defaults to recursive."""
        dag = get_default_pipeline(
            embedding_model="test-model",
            chunk_config={"chunking_strategy": None},
        )
        chunker = next(n for n in dag.nodes if n.type == NodeType.CHUNKER)
        assert chunker.plugin_id == "recursive"
