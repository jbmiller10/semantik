"""Pipeline template for academic papers and research documents.

This template is optimized for:
- PDF documents with complex layouts (multi-column, figures, tables)
- Research papers with citations, abstracts, and structured sections
- Documents requiring high-quality text extraction and semantic chunking
"""

from __future__ import annotations

from shared.pipeline.templates.types import PipelineTemplate, TunableParameter
from shared.pipeline.types import NodeType, PipelineDAG, PipelineEdge, PipelineNode

TEMPLATE = PipelineTemplate(
    id="academic-papers",
    name="Academic Papers",
    description=(
        "Optimized for research papers and academic documents. Uses high-resolution "
        "PDF parsing to preserve document structure, semantic chunking to maintain "
        "coherent passages, and keyword extraction for improved search relevance."
    ),
    suggested_for=["PDF", "research", "citations", "papers", "journal", "academic"],
    pipeline=PipelineDAG(
        id="academic-papers-dag",
        version="1.0",
        nodes=[
            # Parser: Use unstructured for hi-res PDF parsing
            PipelineNode(
                id="parser",
                type=NodeType.PARSER,
                plugin_id="unstructured",
                config={
                    "strategy": "hi_res",
                    "infer_table_structure": True,
                    "extract_images_in_pdf": False,
                },
            ),
            # Chunker: Semantic chunking to preserve logical sections
            PipelineNode(
                id="chunker",
                type=NodeType.CHUNKER,
                plugin_id="semantic",
                config={
                    "max_tokens": 512,
                    "min_tokens": 100,
                    "breakpoint_threshold": 0.4,
                },
            ),
            # Extractor: Keywords for enhanced search
            PipelineNode(
                id="extractor",
                type=NodeType.EXTRACTOR,
                plugin_id="keyword-extractor",
                config={
                    "max_keywords": 20,
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
            # PDF files go through unstructured parser
            PipelineEdge(
                from_node="_source",
                to_node="parser",
                when={"extension": [".pdf"]},
            ),
            # Catch-all for other document types
            PipelineEdge(
                from_node="_source",
                to_node="parser",
                when=None,  # Catch-all
            ),
            # Parser -> Chunker -> Extractor -> Embedder
            PipelineEdge(from_node="parser", to_node="chunker"),
            PipelineEdge(from_node="chunker", to_node="extractor"),
            PipelineEdge(from_node="extractor", to_node="embedder"),
        ],
    ),
    tunable=[
        TunableParameter(
            path="nodes.chunker.config.max_tokens",
            description="Maximum tokens per chunk. Higher values preserve more context but may dilute relevance.",
            default=512,
            range=(256, 1024),
        ),
        TunableParameter(
            path="nodes.chunker.config.breakpoint_threshold",
            description="Semantic similarity threshold for chunk boundaries. Lower values create more chunks.",
            default=0.4,
            range=(0, 100),  # Scaled 0-100 for UI, divided by 100 for actual value
        ),
        TunableParameter(
            path="nodes.extractor.config.max_keywords",
            description="Maximum keywords to extract per chunk for search enrichment.",
            default=20,
            range=(5, 50),
        ),
    ],
)
