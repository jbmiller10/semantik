"""Pipeline template for email archives and message collections.

This template is optimized for:
- Email archives (IMAP, mbox)
- Message threads and conversations
- Email attachments (primarily PDFs)
"""

from __future__ import annotations

from shared.pipeline.templates.types import PipelineTemplate, TunableParameter
from shared.pipeline.types import NodeType, PipelineDAG, PipelineEdge, PipelineNode

TEMPLATE = PipelineTemplate(
    id="email-archive",
    name="Email Archive",
    description=(
        "Designed for email collections and message archives. Uses recursive chunking "
        "for email bodies to preserve thread context, with separate handling for PDF "
        "attachments using high-quality extraction."
    ),
    suggested_for=["email", "IMAP", "messages", "archive", "attachments", "correspondence"],
    pipeline=PipelineDAG(
        id="email-archive-dag",
        version="1.0",
        nodes=[
            # Parser: Text parser for email bodies
            PipelineNode(
                id="email-parser",
                type=NodeType.PARSER,
                plugin_id="text",
                config={
                    "encoding": "utf-8",
                    "fallback_encodings": ["latin-1", "cp1252", "iso-8859-1"],
                },
            ),
            # Parser: Unstructured for PDF attachments
            PipelineNode(
                id="attachment-parser",
                type=NodeType.PARSER,
                plugin_id="unstructured",
                config={
                    "strategy": "fast",
                },
            ),
            # Chunker for emails: Recursive with email-specific separators
            PipelineNode(
                id="email-chunker",
                type=NodeType.CHUNKER,
                plugin_id="recursive",
                config={
                    "chunk_size": 600,
                    "chunk_overlap": 100,
                    "separators": ["\n\n---", "\n\nFrom:", "\n\n>", "\n\n", "\n", " "],
                },
            ),
            # Chunker for attachments: Standard recursive
            PipelineNode(
                id="attachment-chunker",
                type=NodeType.CHUNKER,
                plugin_id="recursive",
                config={
                    "chunk_size": 1000,
                    "chunk_overlap": 150,
                },
            ),
            # Extractor: Keywords for search
            PipelineNode(
                id="extractor",
                type=NodeType.EXTRACTOR,
                plugin_id="keyword-extractor",
                config={
                    "max_keywords": 10,
                    "include_phrases": False,
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
            # PDF attachments go through attachment parser
            PipelineEdge(
                from_node="_source",
                to_node="attachment-parser",
                when={"extension": [".pdf"]},
            ),
            # Email bodies and other text go through email parser
            PipelineEdge(
                from_node="_source",
                to_node="email-parser",
                when=None,  # Catch-all for emails
            ),
            # Route parsers to appropriate chunkers
            PipelineEdge(from_node="email-parser", to_node="email-chunker"),
            PipelineEdge(from_node="attachment-parser", to_node="attachment-chunker"),
            # Both chunkers go to extractor then embedder
            PipelineEdge(from_node="email-chunker", to_node="extractor"),
            PipelineEdge(from_node="attachment-chunker", to_node="extractor"),
            PipelineEdge(from_node="extractor", to_node="embedder"),
        ],
    ),
    tunable=[
        TunableParameter(
            path="nodes.email-chunker.config.chunk_size",
            description="Target chunk size in characters for email bodies. Smaller keeps individual messages intact.",
            default=600,
            range=(300, 1200),
        ),
        TunableParameter(
            path="nodes.attachment-chunker.config.chunk_size",
            description="Target chunk size in characters for PDF attachments.",
            default=1000,
            range=(500, 2000),
        ),
        TunableParameter(
            path="nodes.extractor.config.max_keywords",
            description="Maximum keywords to extract per chunk.",
            default=10,
            range=(5, 25),
        ),
    ],
)
