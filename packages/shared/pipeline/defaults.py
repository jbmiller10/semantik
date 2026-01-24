"""Default pipeline generation for collections.

This module provides factory functions for generating default pipeline DAGs
from collection configuration. The default pipeline routes files based on
MIME type and handles parsing, chunking, and embedding stages.
"""

from __future__ import annotations

from typing import Any

from shared.pipeline.types import NodeType, PipelineDAG, PipelineEdge, PipelineNode


def get_default_pipeline(
    embedding_model: str,
    chunk_config: dict[str, Any],
) -> PipelineDAG:
    """Generate a default pipeline DAG from collection configuration.

    The default pipeline routes files based on MIME type:
    - PDFs and Office docs → unstructured parser
    - All other files → text parser (catch-all)
    - Then → chunker → embedder

    Args:
        embedding_model: The embedding model ID (e.g., "Qwen/Qwen3-Embedding-0.6B")
        chunk_config: Dict with chunking_strategy, chunk_size, chunk_overlap, chunking_config

    Returns:
        A validated PipelineDAG ready for serialization/execution

    Raises:
        ValueError: If the generated DAG is invalid
    """
    # Extract chunking settings with defaults
    chunking_strategy = chunk_config.get("chunking_strategy") or "recursive"
    chunk_size = chunk_config.get("chunk_size", 1000)
    chunk_overlap = chunk_config.get("chunk_overlap", 200)

    # Build chunker config, ensuring overlap_tokens < min_tokens
    base_min_tokens = max(100, chunk_size // 10)
    # Ensure min_tokens is always greater than overlap_tokens
    min_tokens = max(base_min_tokens, chunk_overlap + 1)
    chunker_config: dict[str, Any] = {
        "max_tokens": chunk_size,
        "min_tokens": min_tokens,
        "overlap_tokens": chunk_overlap,
    }
    # Merge any strategy-specific config
    if chunk_config.get("chunking_config"):
        chunker_config.update(chunk_config["chunking_config"])

    dag = PipelineDAG(
        id="default-v1",
        version="1",
        nodes=[
            PipelineNode(
                id="unstructured_parser",
                type=NodeType.PARSER,
                plugin_id="unstructured",
                config={"strategy": "auto"},
            ),
            PipelineNode(
                id="text_parser",
                type=NodeType.PARSER,
                plugin_id="text",
                config={},
            ),
            PipelineNode(
                id="chunker",
                type=NodeType.CHUNKER,
                plugin_id=chunking_strategy,
                config=chunker_config,
            ),
            PipelineNode(
                id="embedder",
                type=NodeType.EMBEDDER,
                plugin_id="dense_local",
                config={"model": embedding_model},
            ),
        ],
        edges=[
            # Route PDFs and Office docs to unstructured parser
            PipelineEdge(
                from_node="_source",
                to_node="unstructured_parser",
                when={"mime_type": ["application/pdf", "application/vnd.*"]},
            ),
            # Catch-all for other files (required by validation)
            PipelineEdge(
                from_node="_source",
                to_node="text_parser",
                when=None,
            ),
            # Both parsers feed into chunker
            PipelineEdge(from_node="unstructured_parser", to_node="chunker"),
            PipelineEdge(from_node="text_parser", to_node="chunker"),
            # Chunker feeds into embedder (terminal)
            PipelineEdge(from_node="chunker", to_node="embedder"),
        ],
    )

    # Validate before returning
    errors = dag.validate()
    if errors:
        error_msgs = "; ".join(e.message for e in errors)
        raise ValueError(f"Generated invalid pipeline DAG: {error_msgs}")

    return dag


__all__ = [
    "get_default_pipeline",
]
