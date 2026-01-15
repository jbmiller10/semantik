"""MCP tool definitions for Semantik."""

from __future__ import annotations

from typing import Any

from mcp import Tool


def build_search_tool(*, name: str, description: str, profile: dict[str, Any]) -> Tool:
    """Build a search tool for a profile."""
    return Tool(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (natural language).",
                },
                "k": {
                    "type": "integer",
                    "description": f"Number of results to return (default: {profile.get('result_count', 10)}).",
                    "minimum": 1,
                    "maximum": 100,
                },
                "search_type": {
                    "type": "string",
                    "enum": ["semantic", "question", "code"],
                    "description": f"Search type (default: {profile.get('search_type', 'semantic')}).",
                },
                "search_mode": {
                    "type": "string",
                    "enum": ["dense", "sparse", "hybrid"],
                    "description": (
                        "Search mode: 'dense' (vector only), 'sparse' (BM25/SPLADE only), "
                        "'hybrid' (dense + sparse with RRF fusion). Default: 'dense'."
                    ),
                },
                "rrf_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "description": "RRF constant k for hybrid mode ranking (default: 60).",
                },
                "use_reranker": {
                    "type": "boolean",
                    "description": f"Enable reranking (default: {profile.get('use_reranker', True)}).",
                },
                "score_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Minimum score threshold (overrides profile default if set).",
                },
                "hyde_enabled": {
                    "type": "boolean",
                    "description": f"Enable HyDE (default: {profile.get('hyde_enabled', False)}).",
                },
            },
            "required": ["query"],
        },
    )


def build_get_document_tool() -> Tool:
    return Tool(
        name="get_document",
        description="Retrieve document metadata by ID from a specific collection.",
        inputSchema={
            "type": "object",
            "properties": {
                "collection_id": {"type": "string", "description": "The collection ID (UUID)."},
                "document_id": {"type": "string", "description": "The document ID (UUID)."},
            },
            "required": ["collection_id", "document_id"],
        },
    )


def build_get_document_content_tool() -> Tool:
    return Tool(
        name="get_document_content",
        description=(
            "Retrieve the raw content of a document (may be binary for PDFs, etc.). "
            "Prefer get_chunk for extracted text."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "collection_id": {"type": "string", "description": "The collection ID (UUID)."},
                "document_id": {"type": "string", "description": "The document ID (UUID)."},
            },
            "required": ["collection_id", "document_id"],
        },
    )


def build_get_chunk_tool() -> Tool:
    return Tool(
        name="get_chunk",
        description="Retrieve the full content of a chunk by ID from a specific collection.",
        inputSchema={
            "type": "object",
            "properties": {
                "collection_id": {
                    "type": "string",
                    "description": "The collection ID (UUID). Required for partition pruning.",
                },
                "chunk_id": {"type": "string", "description": "The chunk identifier from search results."},
            },
            "required": ["collection_id", "chunk_id"],
        },
    )


def build_list_documents_tool() -> Tool:
    return Tool(
        name="list_documents",
        description="List documents in a collection (paginated).",
        inputSchema={
            "type": "object",
            "properties": {
                "collection_id": {"type": "string", "description": "The collection ID (UUID)."},
                "page": {"type": "integer", "minimum": 1, "default": 1},
                "per_page": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
                "status": {
                    "type": "string",
                    "description": "Optional document status filter (e.g., completed, failed).",
                },
            },
            "required": ["collection_id"],
        },
    )


def build_diagnostics_tool() -> Tool:
    """Build the diagnostics tool for debugging MCP server status."""
    return Tool(
        name="diagnostics",
        description=(
            "Show MCP server diagnostics including available profiles, "
            "connection status, and configuration. Useful for debugging."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )
