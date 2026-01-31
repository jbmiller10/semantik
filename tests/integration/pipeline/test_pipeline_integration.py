"""Integration tests for pipeline integration with collection service."""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.repositories.collection_repository import CollectionRepository
from shared.pipeline.defaults import get_default_pipeline
from shared.pipeline.types import NodeType, PipelineDAG


def _unique_name(prefix: str) -> str:
    """Generate a unique collection name for tests."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio()
async def test_get_default_pipeline_creates_valid_dag() -> None:
    """Test that get_default_pipeline creates a valid, serializable DAG."""
    # Create pipeline with typical collection config
    dag = get_default_pipeline(
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        chunk_config={
            "chunking_strategy": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
    )

    # Validate DAG structure
    errors = dag.validate()
    assert not errors, f"Validation errors: {errors}"

    # Check serialization works
    dag_dict = dag.to_dict()
    assert "nodes" in dag_dict
    assert "edges" in dag_dict
    assert dag_dict["id"] == "default-v1"
    assert dag_dict["version"] == "1"

    # Check deserialization works
    restored = PipelineDAG.from_dict(dag_dict)
    assert len(restored.nodes) == len(dag.nodes)
    assert len(restored.edges) == len(dag.edges)


@pytest.mark.asyncio()
async def test_collection_repository_stores_pipeline_config(
    db_session: AsyncSession,
    test_user_db: object,
) -> None:
    """Test that CollectionRepository correctly stores pipeline_config."""
    repo = CollectionRepository(db_session)

    # Generate pipeline config
    dag = get_default_pipeline(
        embedding_model="test-model",
        chunk_config={"chunk_size": 800},
    )
    pipeline_config = dag.to_dict()

    # Create collection with pipeline config
    collection = await repo.create(
        name=_unique_name("test-pipeline-collection"),
        owner_id=test_user_db.id,  # type: ignore[attr-defined]
        embedding_model="test-model",
        chunk_size=800,
        chunk_overlap=100,
        pipeline_config=pipeline_config,
        persist_originals=True,
    )

    assert collection.pipeline_config is not None
    assert collection.pipeline_config == pipeline_config
    assert collection.pipeline_version == 1
    assert collection.persist_originals is True


@pytest.mark.asyncio()
async def test_collection_repository_stores_persist_originals_default(
    db_session: AsyncSession,
    test_user_db: object,
) -> None:
    """Test that persist_originals defaults to False."""
    repo = CollectionRepository(db_session)

    collection = await repo.create(
        name=_unique_name("test-no-originals"),
        owner_id=test_user_db.id,  # type: ignore[attr-defined]
        embedding_model="test-model",
    )

    assert collection.persist_originals is False


@pytest.mark.asyncio()
async def test_pipeline_config_roundtrip_from_repository(
    db_session: AsyncSession,
    test_user_db: object,
) -> None:
    """Test that pipeline_config survives roundtrip to/from database."""
    repo = CollectionRepository(db_session)

    # Create pipeline with specific config
    dag = get_default_pipeline(
        embedding_model="custom/model",
        chunk_config={
            "chunking_strategy": "semantic",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "chunking_config": {"preserve_sentences": True},
        },
    )

    # Store in database
    collection = await repo.create(
        name=_unique_name("test-roundtrip"),
        owner_id=test_user_db.id,  # type: ignore[attr-defined]
        embedding_model="custom/model",
        pipeline_config=dag.to_dict(),
    )
    await db_session.commit()

    # Fetch back
    fetched = await repo.get_by_uuid(collection.id)
    assert fetched is not None
    assert fetched.pipeline_config is not None

    # Reconstruct DAG and verify
    restored_dag = PipelineDAG.from_dict(fetched.pipeline_config)
    errors = restored_dag.validate()
    assert not errors

    # Verify chunker config preserved
    chunker = next(n for n in restored_dag.nodes if n.type == NodeType.CHUNKER)
    assert chunker.plugin_id == "semantic"
    assert chunker.config["max_tokens"] == 500
    assert chunker.config.get("preserve_sentences") is True

    # Verify embedder config preserved
    embedder = next(n for n in restored_dag.nodes if n.type == NodeType.EMBEDDER)
    assert embedder.config["model"] == "custom/model"


@pytest.mark.asyncio()
async def test_collection_without_pipeline_config(
    db_session: AsyncSession,
    test_user_db: object,
) -> None:
    """Test that collections can be created without pipeline_config (backward compat)."""
    repo = CollectionRepository(db_session)

    collection = await repo.create(
        name=_unique_name("test-no-pipeline"),
        owner_id=test_user_db.id,  # type: ignore[attr-defined]
        embedding_model="test-model",
        # No pipeline_config provided
    )

    assert collection.pipeline_config is None
    assert collection.pipeline_version == 1  # Default value


class TestPipelineDAGValidation:
    """Tests for PipelineDAG validation in integration context."""

    def test_default_dag_passes_all_validation_rules(self) -> None:
        """Verify the default DAG passes all validation rules."""
        dag = get_default_pipeline(
            embedding_model="test",
            chunk_config={},
        )
        errors = dag.validate()
        assert not errors

        # Verify specific validation rules pass
        # Rule 1: Exactly one embedder
        embedders = [n for n in dag.nodes if n.type == NodeType.EMBEDDER]
        assert len(embedders) == 1

        # Rule 6: Catch-all edge from _source
        catch_all = [e for e in dag.edges if e.from_node == "_source" and e.when is None]
        assert len(catch_all) >= 1

        # Rule 7: Unique node IDs
        node_ids = [n.id for n in dag.nodes]
        assert len(node_ids) == len(set(node_ids))

    def test_default_dag_all_nodes_reachable(self) -> None:
        """Verify all nodes in default DAG are reachable from _source."""
        dag = get_default_pipeline(
            embedding_model="test",
            chunk_config={},
        )

        # Build adjacency list
        from collections import defaultdict

        adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in dag.edges:
            adjacency[edge.from_node].append(edge.to_node)

        # BFS from _source
        visited: set[str] = set()
        queue = list(adjacency.get("_source", []))
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            queue.extend(adjacency.get(node, []))

        # All nodes should be reachable
        node_ids = {n.id for n in dag.nodes}
        assert node_ids == visited, f"Unreachable nodes: {node_ids - visited}"

    def test_default_dag_all_nodes_reach_embedder(self) -> None:
        """Verify all nodes can reach the embedder."""
        dag = get_default_pipeline(
            embedding_model="test",
            chunk_config={},
        )

        # Build reverse adjacency (for backward traversal)
        from collections import defaultdict

        reverse_adj: dict[str, list[str]] = defaultdict(list)
        for edge in dag.edges:
            reverse_adj[edge.to_node].append(edge.from_node)

        # BFS backward from embedder
        embedder_id = next(n.id for n in dag.nodes if n.type == NodeType.EMBEDDER)
        can_reach_embedder: set[str] = set()
        queue = [embedder_id]
        while queue:
            node = queue.pop(0)
            if node in can_reach_embedder:
                continue
            can_reach_embedder.add(node)
            queue.extend(reverse_adj.get(node, []))

        # All non-embedder nodes should be able to reach embedder
        for node in dag.nodes:
            if node.id != embedder_id:
                assert node.id in can_reach_embedder, f"Node {node.id} cannot reach embedder"
