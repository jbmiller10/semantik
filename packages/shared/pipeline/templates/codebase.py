"""Pipeline template for codebases and technical documentation.

This template is optimized for:
- Source code files in various programming languages
- Technical documentation (markdown, RST)
- Code repositories with mixed content types
"""

from __future__ import annotations

from shared.pipeline.templates.types import PipelineTemplate, TunableParameter
from shared.pipeline.types import NodeType, PipelineDAG, PipelineEdge, PipelineNode

TEMPLATE = PipelineTemplate(
    id="codebase",
    name="Codebase",
    description=(
        "Designed for source code repositories and technical documentation. Uses "
        "recursive chunking for code files to preserve logical structure, and "
        "markdown-aware chunking for documentation files."
    ),
    suggested_for=["code", "programming", "documentation", "git", "repository", "technical"],
    pipeline=PipelineDAG(
        id="codebase-dag",
        version="1.0",
        nodes=[
            # Parser: Text parser for code files
            PipelineNode(
                id="text-parser",
                type=NodeType.PARSER,
                plugin_id="text",
                config={
                    "encoding": "utf-8",
                    "fallback_encodings": ["latin-1", "cp1252"],
                },
            ),
            # Chunker for code: Recursive with syntax-aware separators
            PipelineNode(
                id="code-chunker",
                type=NodeType.CHUNKER,
                plugin_id="recursive",
                config={
                    "chunk_size": 1500,
                    "chunk_overlap": 200,
                    "separators": ["\n\nclass ", "\n\ndef ", "\n\nfunction ", "\n\n", "\n", " "],
                },
            ),
            # Chunker for docs: Markdown-aware
            PipelineNode(
                id="doc-chunker",
                type=NodeType.CHUNKER,
                plugin_id="markdown",
                config={
                    "chunk_size": 1000,
                    "chunk_overlap": 100,
                },
            ),
            # Embedder: Dense local embeddings
            PipelineNode(
                id="embedder",
                type=NodeType.EMBEDDER,
                plugin_id="dense_local",
                config={},
            ),
        ],
        edges=[
            # Markdown/RST docs go through doc-chunker
            PipelineEdge(
                from_node="_source",
                to_node="text-parser",
                when={"extension": [".md", ".rst", ".txt"]},
            ),
            # Code files and catch-all go through text-parser
            PipelineEdge(
                from_node="_source",
                to_node="text-parser",
                when=None,  # Catch-all for code files
            ),
            # Route markdown files to doc chunker
            PipelineEdge(
                from_node="text-parser",
                to_node="doc-chunker",
                when={"extension": [".md", ".rst"]},
            ),
            # Route code files to code chunker
            PipelineEdge(
                from_node="text-parser",
                to_node="code-chunker",
                when=None,  # Catch-all
            ),
            # Both chunkers feed to embedder
            PipelineEdge(from_node="code-chunker", to_node="embedder"),
            PipelineEdge(from_node="doc-chunker", to_node="embedder"),
        ],
    ),
    tunable=[
        TunableParameter(
            path="nodes.code-chunker.config.chunk_size",
            description="Target chunk size in characters for code files. Larger chunks preserve more context.",
            default=1500,
            range=(500, 3000),
        ),
        TunableParameter(
            path="nodes.code-chunker.config.chunk_overlap",
            description="Overlap between code chunks to maintain context across boundaries.",
            default=200,
            range=(0, 500),
        ),
        TunableParameter(
            path="nodes.doc-chunker.config.chunk_size",
            description="Target chunk size in characters for documentation files.",
            default=1000,
            range=(500, 2000),
        ),
    ],
)
