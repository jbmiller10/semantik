#!/usr/bin/env python3
"""
Compatibility wrapper for HybridChunker.

This module provides backward compatibility for tests that import HybridChunker directly.
"""

from enum import Enum
from packages.shared.text_processing.chunking_factory import ChunkingFactory


class ChunkingStrategy(str, Enum):
    """Enum for chunking strategies (for backward compatibility)."""
    CHARACTER = "character"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    MARKDOWN = "markdown"
    HYBRID = "hybrid"


class HybridChunker:
    """Wrapper class for backward compatibility."""
    
    def __init__(self, strategies=None, weights=None, embed_model=None, **kwargs):
        """Initialize using the factory."""
        params = {"embed_model": embed_model}
        if strategies:
            params["strategies"] = strategies
        if weights:
            params["weights"] = weights
        params.update(kwargs)
        
        self._chunker = ChunkingFactory.create_chunker({
            "strategy": "hybrid",
            "params": params
        })
        
        # Add mock attributes for test compatibility
        self._compiled_patterns = self._compile_test_patterns()
        
    def _compile_test_patterns(self):
        """Compile regex patterns for test compatibility."""
        import re
        patterns = {
            r"^#{1,6}\s+\S.*$": (re.compile(r"^#{1,6}\s+\S.*$", re.MULTILINE), 2.0),  # Headers
            r"^[\*\-\+]\s+\S.*$": (re.compile(r"^[\*\-\+]\s+\S.*$", re.MULTILINE), 1.5),  # Unordered lists
            r"^\d+\.\s+\S.*$": (re.compile(r"^\d+\.\s+\S.*$", re.MULTILINE), 1.5),  # Ordered lists
            r"\[([^\]]+)\]\(([^)]+)\)": (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), 1.0),  # Links
            r"!\[([^\]]*)\]\(([^)]+)\)": (re.compile(r"!\[([^\]]*)\]\(([^)]+)\)"), 1.5),  # Images
            r"`([^`]+)`": (re.compile(r"`([^`]+)`"), 0.5),  # Inline code
            r"^>\s*\S.*$": (re.compile(r"^>\s*\S.*$", re.MULTILINE), 1.0),  # Blockquotes
            r"\*\*([^*]+)\*\*": (re.compile(r"\*\*([^*]+)\*\*"), 0.5),  # Bold
            r"\*([^*]+)\*": (re.compile(r"\*([^*]+)\*"), 0.5),  # Italic
            r"^\s*\|[^|]+\|": (re.compile(r"^\s*\|[^|]+\|", re.MULTILINE), 2.0),  # Tables
            r"^(?:---|\\*\\*\\*|___)$": (re.compile(r"^(?:---|\\*\\*\\*|___)$", re.MULTILINE), 1.0),  # Horizontal rules
        }
        return patterns
        
    def __getattr__(self, name):
        """Delegate all attributes to the actual chunker."""
        return getattr(self._chunker, name)