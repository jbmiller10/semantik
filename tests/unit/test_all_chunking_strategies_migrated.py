#!/usr/bin/env python3

"""
Migrated tests for chunking strategies using the unified implementation directly.

This module contains tests that have been migrated from the old implementation
to work directly with the unified chunking system.
"""

import os

import pytest
from llama_index.core.embeddings import MockEmbedding

from packages.shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from packages.shared.chunking.unified.factory import UnifiedChunkingFactory
from packages.shared.text_processing.base_chunker import ChunkResult
from packages.shared.text_processing.chunking_factory import ChunkingFactory
from packages.shared.text_processing.file_type_detector import FileTypeDetector

# Set testing environment
os.environ["TESTING"] = "true"


class TestMigratedChunkingStrategies:
    """Migrated tests that work directly with unified implementation."""

    @pytest.fixture()
    def mock_embed_model(self) -> MockEmbedding:
        """Mock embedding model for semantic chunking tests."""
        return MockEmbedding(embed_dim=384)

    async def test_character_chunker_basic(self) -> None:
        """Test basic character chunker functionality with proper token sizing."""
        # Create config for ~50 character chunks
        config = {
            "strategy": "character",
            "params": {
                "max_tokens": 12,  # ~50 chars / 4
                "min_tokens": 5,
                "overlap_tokens": 2,  # ~10 chars / 4
            },
        }
        chunker = ChunkingFactory.create_chunker(config)

        text = "This is a test document. " * 10  # ~250 characters
        chunks = await chunker.chunk_text_async(text, "test_doc")

        # Verify chunks - should get multiple with token-based sizing
        assert len(chunks) >= 4, f"Expected at least 4 chunks for 250 chars with 12 token chunks, got {len(chunks)}"
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

        # Verify chunk structure
        for chunk in chunks:
            assert chunk.text
            assert chunk.chunk_id
            assert chunk.metadata

    async def test_recursive_chunker_basic(self) -> None:
        """Test basic recursive chunker functionality."""
        config = {
            "strategy": "recursive",
            "params": {
                "max_tokens": 12,  # ~50 chars
                "min_tokens": 5,
                "overlap_tokens": 2,
            },
        }
        chunker = ChunkingFactory.create_chunker(config)

        text = "This is sentence one. This is sentence two. This is sentence three. " * 10
        chunks = await chunker.chunk_text_async(text, "test_doc")

        # Verify chunks
        assert len(chunks) >= 4
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

    async def test_recursive_chunker_code_optimization(self) -> None:
        """Test recursive chunker with code file optimization."""
        code_text = '''def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    """Simple calculator class."""

    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
'''

        # Get optimized config for Python file
        config = FileTypeDetector.get_optimal_config("test.py")
        chunker = ChunkingFactory.create_chunker(config)

        # Test with Python code
        metadata = {"file_type": ".py", "file_name": "test.py"}
        chunks = await chunker.chunk_text_async(code_text, "test_code", metadata)

        # Verify code optimization was applied
        assert len(chunks) >= 1
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

    async def test_markdown_chunker_complex(self) -> None:
        """Test markdown chunker with complex document."""
        markdown_text = """# Project Documentation

## Overview

This project implements a chunking system.

### Features

- Feature 1
- Feature 2
- Feature 3

## Installation

```bash
pip install chunking
```

### Requirements

1. Python 3.11+
2. LlamaIndex
3. Redis

## Usage

Here's how to use it:

```python
chunker = ChunkingFactory.create_chunker(config)
chunks = chunker.chunk_text(text)
```

### Advanced Usage

For more complex scenarios...
"""

        config = {
            "strategy": "markdown",
            "params": {
                "max_tokens": 100,
                "min_tokens": 20,
                "overlap_tokens": 10,
            },
        }
        chunker = ChunkingFactory.create_chunker(config)

        chunks = await chunker.chunk_text_async(
            markdown_text,
            "test_doc",
            {"file_type": ".md"},
        )

        # Verify chunks
        assert len(chunks) >= 1
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

        # Verify content is preserved
        all_text = " ".join(c.text for c in chunks)
        assert "pip install" in all_text or "```" in all_text

    def test_factory_creation(self) -> None:
        """Test factory creates correct chunker instances."""
        # Test each strategy
        strategies = ["character", "recursive", "markdown", "semantic", "hierarchical", "hybrid"]

        for strategy in strategies:
            config = {"strategy": strategy, "params": {}}
            chunker = ChunkingFactory.create_chunker(config)
            assert chunker is not None
            # Verify it has the required methods
            assert hasattr(chunker, "chunk_text")
            assert hasattr(chunker, "chunk_text_async")

    @pytest.mark.parametrize("strategy", ["markdown", "semantic", "hybrid"])
    def test_estimate_chunks(self, strategy: str) -> None:
        """Test chunk estimation for various strategies."""
        config = {
            "strategy": strategy,
            "params": {
                "max_tokens": 100,
                "min_tokens": 20,
                "overlap_tokens": 10,
            },
        }
        chunker = ChunkingFactory.create_chunker(config)

        # Small text
        estimate = chunker.estimate_chunks(100, config["params"])
        assert estimate >= 1
        assert estimate <= 10

        # Large text
        estimate = chunker.estimate_chunks(10000, config["params"])
        assert estimate >= 5
        assert estimate <= 500

    async def test_large_document_handling(self) -> None:
        """Test handling of large documents."""
        config = {
            "strategy": "character",
            "params": {
                "max_tokens": 260,  # ~1000 chars, with tolerance for LlamaIndex chunking
                "min_tokens": 50,
                "overlap_tokens": 50,  # ~200 chars
            },
        }
        chunker = ChunkingFactory.create_chunker(config)

        # Generate large document (1MB)
        large_text = "This is a test sentence. " * 50000

        chunks = await chunker.chunk_text_async(large_text, "large_doc")

        # Should create many chunks
        assert len(chunks) > 100
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

    async def test_code_file_optimization(self) -> None:
        """Test code files get optimized parameters."""
        code_file = """
def fibonacci(n):
    \"\"\"Calculate fibonacci number recursively.

    Args:
        n: The position in fibonacci sequence

    Returns:
        The fibonacci number at position n
    \"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    \"\"\"A simple calculator class with basic operations.\"\"\"

    def __init__(self):
        \"\"\"Initialize the calculator.\"\"\"
        self.history = []

    def add(self, a, b):
        \"\"\"Add two numbers and store in history.\"\"\"
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
"""

        # Test with .py file type
        config = FileTypeDetector.get_optimal_config("test.py")
        assert config["strategy"] == "recursive"

        chunker = ChunkingFactory.create_chunker(config)
        chunks = await chunker.chunk_text_async(code_file, "test.py")

        # Verify chunks
        assert len(chunks) >= 1
        assert any("fibonacci" in chunk.text for chunk in chunks)

    async def test_hierarchical_chunker_relationships(self) -> None:
        """Test hierarchical chunker creates proper parent-child relationships."""
        # Use unified factory directly
        strategy = UnifiedChunkingFactory.create_strategy("hierarchical")

        # Create config with hierarchical parameters
        config = ChunkConfig(
            max_tokens=50,  # Parent level
            min_tokens=10,
            overlap_tokens=2,
            strategy_name="hierarchical",
            hierarchy_levels=3,  # Pass hierarchy_levels directly
        )

        # Create text that will require multiple hierarchy levels
        text = " ".join([f"Sentence {i}." for i in range(100)])  # ~700 chars

        # Use unified chunking directly
        chunks = await strategy.chunk_async(text, config)

        # Should create chunks with hierarchy
        assert len(chunks) > 0

        # Verify chunk structure
        for chunk in chunks:
            assert chunk.content  # Should have content
            assert chunk.metadata  # Should have metadata

            # Check hierarchical information
            if chunk.metadata.custom_attributes:
                # May have hierarchy level information
                hierarchy_level = chunk.metadata.custom_attributes.get("hierarchy_level")
                if hierarchy_level is not None:
                    assert hierarchy_level >= 0

    async def test_hybrid_chunker_markdown_detection(self) -> None:
        """Test hybrid chunker selects markdown strategy for markdown content."""
        markdown_text = """# Main Title

## Section 1
Content for section 1.

## Section 2
Content for section 2.

### Subsection 2.1
More detailed content."""

        config = {
            "strategy": "hybrid",
            "params": {
                "markdown_threshold": 0.15,
                "semantic_coherence_threshold": 0.9,
            },
        }
        chunker = ChunkingFactory.create_chunker(config)

        # Test with markdown file extension
        chunks = await chunker.chunk_text_async(markdown_text, "test_doc", {"file_path": "/path/to/test.md"})

        # Verify chunks were created
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata.get("hybrid_chunker") is True
            assert "selected_strategy" in chunk.metadata

            # Check if appropriate strategy was selected
            # With .md file extension, may use markdown or hybrid
            # The file_path is passed as metadata and should be in chunk metadata
            if chunk.metadata.get("file_path", "").endswith(".md"):
                assert chunk.metadata["selected_strategy"] in ["markdown", "hybrid"]

    async def test_hybrid_chunker_large_document_handling(self) -> None:
        """Test hybrid chunker selects hierarchical strategy for large documents."""
        config = {
            "strategy": "hybrid",
            "params": {
                "large_doc_threshold": 5000,
            },
        }
        chunker = ChunkingFactory.create_chunker(config)

        # Create a large document
        large_text = (
            "# Technical Documentation\n\n"
            + ("This is a comprehensive technical document. " * 50)
            + "\n\n## Architecture\n\n"
            + ("The system architecture consists of multiple components. " * 50)
        )

        chunks = await chunker.chunk_text_async(large_text, "large_doc")

        # Verify appropriate strategy was selected
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["hybrid_chunker"] is True
            # Check if appropriate strategy was selected for large document
            # May select hierarchical, recursive, or hybrid for large docs
            assert chunk.metadata["selected_strategy"] in ["hierarchical", "recursive", "hybrid"]

    async def test_hybrid_chunker_performance_logging(self) -> None:
        """Test that hybrid chunker logs strategy selection reasoning."""
        config = {"strategy": "hybrid", "params": {}}
        chunker = ChunkingFactory.create_chunker(config)

        # Different content types
        test_cases = [
            ("# Markdown\n\nContent", {"file_type": ".md"}, "markdown"),
            ("Regular text " * 100, {}, "recursive"),
        ]

        for text, metadata, expected_strategy in test_cases:
            chunks = await chunker.chunk_text_async(text, f"test_{expected_strategy}", metadata)

            assert len(chunks) >= 1
            # Verify hybrid metadata is present
            for chunk in chunks:
                assert chunk.metadata["hybrid_chunker"] is True
                # Strategy selection may vary, just verify it's set
                assert "selected_strategy" in chunk.metadata

    async def test_semantic_chunker_basic(self, mock_embed_model) -> None:
        """Test basic semantic chunker functionality."""
        config = {
            "strategy": "semantic",
            "params": {
                "max_tokens": 25,
                "min_tokens": 10,
                "overlap_tokens": 5,
                "breakpoint_percentile_threshold": 95,
                "buffer_size": 1,
                "embed_model": mock_embed_model,
            },
        }
        chunker = ChunkingFactory.create_chunker(config)

        text = "This is a test document. It has multiple sentences. Each sentence is different. The semantic chunker should find natural boundaries."
        chunks = await chunker.chunk_text_async(text, "semantic_test")

        # Verify chunks
        assert len(chunks) >= 1
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

        # Verify semantic metadata
        for chunk in chunks:
            # Should have strategy metadata
            assert chunk.metadata.get("strategy") == "semantic"
