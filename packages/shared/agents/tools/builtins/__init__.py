"""Built-in agent tools.

This package contains Semantik-specific tools for agent use:
- SemanticSearchTool: Search documents by semantic similarity
- DocumentRetrieveTool: Retrieve document metadata and content
- ListCollectionsTool: List available document collections
- GetChunkTool: Retrieve specific text chunks

These tools are registered automatically at application startup.
"""

from shared.agents.tools.builtins.chunks import GetChunkTool
from shared.agents.tools.builtins.collections import ListCollectionsTool
from shared.agents.tools.builtins.retrieve import DocumentRetrieveTool
from shared.agents.tools.builtins.search import SemanticSearchTool

__all__ = [
    "SemanticSearchTool",
    "DocumentRetrieveTool",
    "ListCollectionsTool",
    "GetChunkTool",
]
