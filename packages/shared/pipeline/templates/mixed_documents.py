"""Pipeline template for mixed document collections.

This template provides balanced defaults for:
- Collections with varied file types (PDF, Office docs, text)
- General-purpose document search
- When content types are unknown or mixed
"""

from __future__ import annotations

from shared.pipeline.templates.types import PipelineTemplate, TunableParameter
from shared.pipeline.types import NodeType, PipelineDAG, PipelineEdge, PipelineNode

TEMPLATE = PipelineTemplate(
    id="mixed-documents",
    name="Mixed Documents",
    description=(
        "Balanced defaults for collections with varied document types. Routes PDFs "
        "through high-quality extraction and text files through fast parsing, with "
        "recursive chunking that works well across content types."
    ),
    suggested_for=["mixed", "general", "varied", "office", "documents", "files"],
    pipeline=PipelineDAG(
        id="mixed-documents-dag",
        version="1.0",
        nodes=[
            # Parser: Unstructured for complex documents
            PipelineNode(
                id="rich-parser",
                type=NodeType.PARSER,
                plugin_id="unstructured",
                config={
                    "strategy": "auto",  # Let unstructured decide
                },
            ),
            # Parser: Text parser for simple text files
            PipelineNode(
                id="text-parser",
                type=NodeType.PARSER,
                plugin_id="text",
                config={
                    "encoding": "utf-8",
                    "fallback_encodings": ["latin-1", "cp1252"],
                },
            ),
            # Chunker: Recursive with balanced settings
            PipelineNode(
                id="chunker",
                type=NodeType.CHUNKER,
                plugin_id="recursive",
                config={
                    "chunk_size": 1000,
                    "chunk_overlap": 150,
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
            # Rich documents (PDF, DOCX, etc.) go through unstructured
            PipelineEdge(
                from_node="_source",
                to_node="rich-parser",
                when={"extension": [".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"]},
            ),
            # Plain text files go through text parser
            PipelineEdge(
                from_node="_source",
                to_node="text-parser",
                when=None,  # Catch-all
            ),
            # Both parsers feed to chunker
            PipelineEdge(from_node="rich-parser", to_node="chunker"),
            PipelineEdge(from_node="text-parser", to_node="chunker"),
            # Chunker -> Embedder
            PipelineEdge(from_node="chunker", to_node="embedder"),
        ],
    ),
    tunable=[
        TunableParameter(
            path="nodes.chunker.config.chunk_size",
            description="Target chunk size in characters. Balanced default works for most document types.",
            default=1000,
            range=(500, 2000),
        ),
        TunableParameter(
            path="nodes.chunker.config.chunk_overlap",
            description="Overlap between chunks to maintain context across boundaries.",
            default=150,
            range=(0, 400),
        ),
    ],
)
