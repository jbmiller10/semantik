"""Pipeline template for documentation and knowledge bases.

This template is optimized for:
- Technical documentation (Markdown, RST, HTML)
- Knowledge base articles
- User guides and manuals
- Wiki content
"""

from __future__ import annotations

from shared.pipeline.templates.types import PipelineTemplate, TunableParameter
from shared.pipeline.types import NodeType, PipelineDAG, PipelineEdge, PipelineNode

TEMPLATE = PipelineTemplate(
    id="documentation",
    name="Documentation",
    description=(
        "Optimized for technical documentation and knowledge bases. Uses markdown-aware "
        "chunking to respect document structure like headers and code blocks, with "
        "keyword extraction for improved topic-based search."
    ),
    suggested_for=["documentation", "wiki", "knowledge base", "markdown", "guides", "manuals"],
    pipeline=PipelineDAG(
        id="documentation-dag",
        version="1.0",
        nodes=[
            # Parser: Text parser for text-based docs
            PipelineNode(
                id="text-parser",
                type=NodeType.PARSER,
                plugin_id="text",
                config={
                    "encoding": "utf-8",
                },
            ),
            # Parser: Unstructured for PDF manuals
            PipelineNode(
                id="pdf-parser",
                type=NodeType.PARSER,
                plugin_id="unstructured",
                config={
                    "strategy": "fast",
                },
            ),
            # Chunker: Markdown-aware chunking
            PipelineNode(
                id="chunker",
                type=NodeType.CHUNKER,
                plugin_id="markdown",
                config={
                    "chunk_size": 800,
                    "chunk_overlap": 100,
                },
            ),
            # Extractor: Keywords for topic search
            PipelineNode(
                id="extractor",
                type=NodeType.EXTRACTOR,
                plugin_id="keyword-extractor",
                config={
                    "max_keywords": 15,
                    "include_phrases": True,
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
            # PDF documents go through PDF parser
            PipelineEdge(
                from_node="_source",
                to_node="pdf-parser",
                when={"extension": [".pdf"]},
            ),
            # Text-based docs go through text parser
            PipelineEdge(
                from_node="_source",
                to_node="text-parser",
                when=None,  # Catch-all
            ),
            # Both parsers feed to markdown chunker
            PipelineEdge(from_node="text-parser", to_node="chunker"),
            PipelineEdge(from_node="pdf-parser", to_node="chunker"),
            # Chunker -> Extractor -> Embedder
            PipelineEdge(from_node="chunker", to_node="extractor"),
            PipelineEdge(from_node="extractor", to_node="embedder"),
        ],
    ),
    tunable=[
        TunableParameter(
            path="nodes.chunker.config.chunk_size",
            description="Target chunk size in characters. Smaller chunks give more precise results.",
            default=800,
            range=(400, 1500),
        ),
        TunableParameter(
            path="nodes.chunker.config.chunk_overlap",
            description="Overlap between chunks to maintain context across boundaries.",
            default=100,
            range=(0, 300),
        ),
        TunableParameter(
            path="nodes.extractor.config.max_keywords",
            description="Maximum keywords to extract per chunk for search enrichment.",
            default=15,
            range=(5, 30),
        ),
    ],
)
