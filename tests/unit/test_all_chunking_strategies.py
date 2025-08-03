#!/usr/bin/env python3
"""
Comprehensive unit tests for all chunking strategies.

This module tests all implemented chunking strategies with various
edge cases, configurations, and document types.
"""

import os
from typing import Any

import pytest
from llama_index.core.embeddings import MockEmbedding

from packages.shared.text_processing.base_chunker import ChunkResult
from packages.shared.text_processing.chunking_factory import ChunkingFactory
from packages.shared.text_processing.strategies.character_chunker import CharacterChunker
from packages.shared.text_processing.strategies.markdown_chunker import MarkdownChunker
from packages.shared.text_processing.strategies.recursive_chunker import RecursiveChunker

# Week 2: Advanced strategies
from packages.shared.text_processing.strategies.semantic_chunker import SemanticChunker
from packages.shared.text_processing.strategies.hierarchical_chunker import HierarchicalChunker
from packages.shared.text_processing.strategies.hybrid_chunker import HybridChunker

# Set testing environment
os.environ["TESTING"] = "true"


class TestChunkingStrategies:
    """Comprehensive tests for all strategies."""

    @pytest.fixture()
    def mock_embed_model(self) -> MockEmbedding:
        """Mock embedding model for semantic chunking tests."""
        return MockEmbedding(embed_dim=384)

    # Test data fixtures
    EDGE_CASES = {
        "empty": "",
        "single_char": "A",
        "unicode": "Hello 世界! 🌍 → €£¥",
        "very_long_line": "a" * 50000,
        "null_bytes": "Hello\x00World",
        "mixed_encoding": "Hello World with special chars: é à ñ",
        "only_whitespace": "   \n\n\t\t  ",
        "html_injection": "<script>alert('xss')</script>",
        "sql_injection": "'; DROP TABLE chunks; --",
        "repeated_newlines": "Line1\n\n\n\n\nLine2",
        "mixed_line_endings": "Line1\r\nLine2\nLine3\r",
        "emoji_heavy": "👍 Test 😊 with 🚀 many 🌟 emojis 🎉",
        "special_markdown": "# Header\n\n```python\ncode\n```\n\n> Quote",
    }

    CODE_SAMPLES = {
        "python": '''def fibonacci(n):
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
''',
        "javascript": """function greet(name) {
    console.log(`Hello, ${name}!`);
}

const calculator = {
    add: (a, b) => a + b,
    multiply: (a, b) => a * b
};

export default calculator;
""",
    }

    MARKDOWN_SAMPLES = {
        "simple": """# Main Header

This is a paragraph with some text.

## Subheader

More content here.
""",
        "complex": """# Project Documentation

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
""",
    }

    ALL_STRATEGIES = ["character", "recursive", "markdown", "semantic", "hierarchical", "hybrid"]

    def get_default_config(self, strategy: str) -> dict[str, Any]:
        """Get default configuration for a strategy."""
        configs = {
            "character": {
                "strategy": "character",
                "params": {"chunk_size": 100, "chunk_overlap": 20},
            },
            "recursive": {
                "strategy": "recursive",
                "params": {"chunk_size": 100, "chunk_overlap": 20},
            },
            "markdown": {
                "strategy": "markdown",
                "params": {},
            },
            "semantic": {
                "strategy": "semantic",
                "params": {"breakpoint_percentile_threshold": 90, "buffer_size": 1, "max_chunk_size": 1000},
            },
            "hierarchical": {
                "strategy": "hierarchical",
                "params": {"chunk_sizes": [500, 200, 100], "chunk_overlap": 20},
            },
            "hybrid": {
                "strategy": "hybrid",
                "params": {"markdown_density_threshold": 0.1, "topic_diversity_threshold": 0.7},
            },
        }
        return configs.get(strategy, configs["recursive"])

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    @pytest.mark.parametrize(("edge_case_name", "text"), EDGE_CASES.items())
    async def test_edge_cases(
        self,
        strategy: str,
        edge_case_name: str,
        text: str,
    ) -> None:
        """Test all strategies handle edge cases gracefully."""
        config = self.get_default_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)

        # Should not raise exception
        try:
            chunks = await chunker.chunk_text_async(text, "test")

            # Validate chunks
            if text.strip():  # Non-empty input
                assert isinstance(chunks, list)
                for chunk in chunks:
                    assert isinstance(chunk, ChunkResult)
                    if chunk.text:  # Allow empty chunks for edge cases
                        assert chunk.text  # Non-empty chunk
            else:  # Empty input
                assert chunks == []

        except Exception as e:
            pytest.fail(f"{strategy} failed on {edge_case_name}: {e}")

    async def test_character_chunker_basic(self) -> None:
        """Test basic character chunker functionality."""
        chunker = CharacterChunker(chunk_size=50, chunk_overlap=10)

        text = "This is a test document. " * 10  # ~250 characters
        chunks = await chunker.chunk_text_async(text, "test_doc")

        # Verify chunks
        assert len(chunks) > 1
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
        assert all(len(chunk.text) <= 250 for chunk in chunks)  # Token size ~4x char size

        # Verify overlap
        for i in range(len(chunks) - 1):
            # Some overlap should exist between consecutive chunks
            assert chunks[i].end_offset > chunks[i + 1].start_offset

    async def test_recursive_chunker_basic(self) -> None:
        """Test basic recursive chunker functionality."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)

        text = "This is sentence one. This is sentence two. This is sentence three. " * 10
        chunks = await chunker.chunk_text_async(text, "test_doc")

        # Verify chunks
        assert len(chunks) > 1
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

        # Verify sentence boundaries are respected
        for chunk in chunks:
            # Most chunks should end with sentence punctuation
            stripped = chunk.text.strip()
            if stripped and len(stripped) > 20:  # Skip very small chunks
                assert stripped[-1] in ".!?" or "..." in stripped

    async def test_recursive_chunker_code_optimization(self) -> None:
        """Test recursive chunker with code file optimization."""
        chunker = RecursiveChunker()

        # Test with Python code
        metadata = {"file_type": ".py", "file_name": "test.py"}
        chunks = await chunker.chunk_text_async(
            self.CODE_SAMPLES["python"],
            "test_code",
            metadata,
        )

        # Verify code optimization was applied
        assert len(chunks) >= 1
        assert all(chunk.metadata.get("is_code_file") is True for chunk in chunks)

        # Code chunks should be smaller
        assert all(len(chunk.text) < 2000 for chunk in chunks)

    async def test_markdown_chunker_basic(self) -> None:
        """Test basic markdown chunker functionality."""
        chunker = MarkdownChunker()

        chunks = await chunker.chunk_text_async(
            self.MARKDOWN_SAMPLES["simple"],
            "test_doc",
        )

        # Verify chunks
        assert len(chunks) >= 1
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

        # Verify markdown structure is preserved
        # Each major section should be its own chunk
        chunk_texts = [chunk.text for chunk in chunks]
        assert any("Main Header" in text for text in chunk_texts)
        assert any("Subheader" in text for text in chunk_texts)

    async def test_markdown_chunker_complex(self) -> None:
        """Test markdown chunker with complex document."""
        chunker = MarkdownChunker()

        chunks = await chunker.chunk_text_async(
            self.MARKDOWN_SAMPLES["complex"],
            "test_doc",
            {"file_type": ".md"},
        )

        # Verify chunks
        assert len(chunks) > 3  # Should have multiple sections
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

        # Code blocks should be preserved within chunks
        code_chunks = [c for c in chunks if "```" in c.text]
        assert len(code_chunks) >= 2  # At least bash and python examples

    async def test_factory_creation(self) -> None:
        """Test factory creates correct chunker instances."""
        # Character chunker
        config = {"strategy": "character", "params": {"chunk_size": 100}}
        chunker = ChunkingFactory.create_chunker(config)
        assert isinstance(chunker, CharacterChunker)
        assert chunker.chunk_size == 100

        # Recursive chunker
        config = {"strategy": "recursive", "params": {"chunk_size": 200}}
        chunker = ChunkingFactory.create_chunker(config)
        assert isinstance(chunker, RecursiveChunker)
        assert chunker.chunk_size == 200

        # Markdown chunker
        config = {"strategy": "markdown", "params": {}}
        chunker = ChunkingFactory.create_chunker(config)
        assert isinstance(chunker, MarkdownChunker)

    async def test_factory_invalid_strategy(self) -> None:
        """Test factory handles invalid strategy."""
        config = {"strategy": "invalid_strategy", "params": {}}

        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            ChunkingFactory.create_chunker(config)

    def test_factory_available_strategies(self) -> None:
        """Test factory returns available strategies."""
        strategies = ChunkingFactory.get_available_strategies()

        assert "character" in strategies
        assert "recursive" in strategies
        assert "markdown" in strategies
        # Week 2: Advanced strategies
        assert "semantic" in strategies
        assert "hierarchical" in strategies
        assert "hybrid" in strategies
        assert len(strategies) >= 6

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_validate_config(self, strategy: str) -> None:
        """Test configuration validation."""
        config = self.get_default_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)

        # Valid config
        assert chunker.validate_config(config["params"]) is True

        # Invalid configs
        if strategy in ["character", "recursive"]:
            # Invalid chunk size
            assert chunker.validate_config({"chunk_size": -100}) is False
            assert chunker.validate_config({"chunk_size": "not_a_number"}) is False

            # Invalid overlap
            assert (
                chunker.validate_config(
                    {
                        "chunk_size": 100,
                        "chunk_overlap": 200,  # Greater than chunk size
                    }
                )
                is False
            )
        elif strategy == "semantic":
            # Invalid threshold
            assert chunker.validate_config({"breakpoint_percentile_threshold": -10}) is False
            assert chunker.validate_config({"breakpoint_percentile_threshold": 150}) is False
            
            # Invalid buffer size
            assert chunker.validate_config({"buffer_size": -1}) is False
            
        elif strategy == "hierarchical":
            # Invalid chunk sizes
            assert chunker.validate_config({"chunk_sizes": "not_a_list"}) is False
            assert chunker.validate_config({"chunk_sizes": [100]}) is False  # Too few levels
            assert chunker.validate_config({"chunk_sizes": [-100, 50]}) is False  # Negative size
            
        elif strategy == "hybrid":
            # Invalid thresholds
            assert chunker.validate_config({"markdown_density_threshold": -0.1}) is False
            assert chunker.validate_config({"topic_diversity_threshold": 1.5}) is False

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    def test_estimate_chunks(self, strategy: str) -> None:
        """Test chunk estimation."""
        config = self.get_default_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)

        # Small text
        estimate = chunker.estimate_chunks(100, config["params"])
        assert estimate >= 1
        assert estimate <= 5

        # Large text
        estimate = chunker.estimate_chunks(10000, config["params"])
        assert estimate > 10
        assert estimate < 1000

    async def test_metadata_preservation(self) -> None:
        """Test metadata is preserved in chunks."""
        chunker = RecursiveChunker(chunk_size=50)

        metadata = {
            "source": "test_file.txt",
            "author": "Test Author",
            "custom_field": 123,
        }

        text = "This is a test. " * 10
        chunks = await chunker.chunk_text_async(text, "test_doc", metadata)

        # All chunks should have metadata
        assert all(chunk.metadata for chunk in chunks)

        # Strategy should be added to metadata
        assert all(chunk.metadata["strategy"] == "recursive" for chunk in chunks)

        # Original metadata should be preserved
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test_file.txt"
            assert chunk.metadata.get("author") == "Test Author"
            assert chunk.metadata.get("custom_field") == 123

    async def test_chunk_ids_unique(self) -> None:
        """Test chunk IDs are unique."""
        chunker = CharacterChunker(chunk_size=50)

        text = "Test text. " * 20
        chunks = await chunker.chunk_text_async(text, "doc123")

        # All chunk IDs should be unique
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

        # Chunk IDs should follow pattern
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"doc123_{i:04d}"

    async def test_semantic_chunker_basic(self) -> None:
        """Test basic semantic chunker functionality."""
        chunker = SemanticChunker(
            breakpoint_percentile_threshold=80,
            buffer_size=1,
            max_chunk_size=1000,
        )

        # Use topic-diverse text that should create semantic boundaries
        text = (
            "Machine learning is a subset of artificial intelligence that focuses on algorithms. "
            "These algorithms can learn from data without being explicitly programmed. "
            "In the field of cooking, French cuisine is known for its sophisticated techniques. "
            "Sauce preparation requires careful attention to temperature and timing. "
            "Quantum physics deals with the behavior of matter at the atomic level. "
            "Particles can exist in multiple states simultaneously through superposition."
        )
        
        chunks = await chunker.chunk_text_async(text, "test_semantic")

        # Verify chunks
        assert len(chunks) >= 1
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
        assert all(chunk.metadata.get("strategy") == "semantic" for chunk in chunks)
        assert all("breakpoint_threshold" in chunk.metadata for chunk in chunks)

        # Verify text is preserved
        reassembled = "".join(chunk.text for chunk in chunks)
        assert len(reassembled) <= len(text) + 100  # Allow for some overlap

    async def test_semantic_chunker_fallback(self) -> None:
        """Test semantic chunker fallback to recursive on failure."""
        # Create chunker with invalid embedding model to trigger fallback
        chunker = SemanticChunker(
            embed_model=None,  # This might cause issues
            breakpoint_percentile_threshold=95,
        )

        text = "This is a test text. " * 50
        chunks = await chunker.chunk_text_async(text, "test_fallback")

        # Should still get chunks (from fallback)
        assert len(chunks) >= 1
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

    async def test_hierarchical_chunker_basic(self) -> None:
        """Test basic hierarchical chunker functionality."""
        chunker = HierarchicalChunker(
            chunk_sizes=[400, 200, 100],
            chunk_overlap=20,
        )

        text = "This is a test document. " * 100  # ~2500 characters
        chunks = await chunker.chunk_text_async(text, "test_hierarchical")

        # Verify chunks
        assert len(chunks) >= 1
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
        assert all(chunk.metadata.get("strategy") == "hierarchical" for chunk in chunks)

        # Check for both leaf and parent chunks
        leaf_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "leaf"]
        parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]

        assert len(leaf_chunks) >= 1
        # Parent chunks are optional depending on hierarchy
        
        # Verify hierarchy metadata
        for chunk in chunks:
            assert "chunk_level" in chunk.metadata
            assert "chunk_size_target" in chunk.metadata
            assert isinstance(chunk.metadata["chunk_level"], int)

    async def test_hierarchical_chunker_relationships(self) -> None:
        """Test hierarchical chunker parent-child relationships."""
        chunker = HierarchicalChunker(
            chunk_sizes=[300, 150],  # Two levels for simpler testing
            chunk_overlap=10,
        )

        text = "This is paragraph one with some content. " * 10
        text += "This is paragraph two with different content. " * 10
        
        chunks = await chunker.chunk_text_async(text, "test_hierarchy")

        # Verify hierarchy structure exists
        assert len(chunks) >= 2
        
        # Check that hierarchy paths are present
        for chunk in chunks:
            assert "hierarchy_path" in chunk.metadata
            assert isinstance(chunk.metadata["hierarchy_path"], list)

    async def test_hybrid_chunker_markdown_selection(self) -> None:
        """Test hybrid chunker selects markdown strategy for markdown content."""
        chunker = HybridChunker(
            markdown_density_threshold=0.1,
            topic_diversity_threshold=0.7,
        )

        markdown_text = """# Main Header

This is a paragraph with some text.

## Subheader  

More content here with `code` and **bold** text.

### Another Section

- List item 1
- List item 2

```python
def example():
    return "code block"
```
"""

        chunks = await chunker.chunk_text_async(markdown_text, "test_md", {"file_type": ".md"})

        # Verify chunks and strategy selection
        assert len(chunks) >= 1
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)
        assert all(chunk.metadata.get("strategy") == "hybrid" for chunk in chunks)
        assert all(chunk.metadata.get("sub_strategy") == "markdown" for chunk in chunks)
        assert all("file type indicates markdown" in chunk.metadata.get("selection_reason", "") for chunk in chunks)

    async def test_hybrid_chunker_semantic_selection(self) -> None:
        """Test hybrid chunker selects semantic strategy for topic-diverse content."""
        chunker = HybridChunker(
            markdown_density_threshold=0.5,  # High threshold to avoid markdown
            topic_diversity_threshold=0.3,   # Lower threshold for testing
            semantic_min_length=100,         # Low threshold for testing
        )

        # Create content with high topic diversity
        diverse_text = (
            "Machine learning algorithms require extensive training data sets. "
            "Neural networks process information through interconnected nodes. "
            "In culinary arts, French techniques emphasize precise temperature control. "
            "Molecular gastronomy combines science with creative presentation methods. "
            "Quantum mechanics describes particle behavior at subatomic scales. "
            "Superposition allows particles to exist in multiple states simultaneously. "
            "Astrophysics studies celestial bodies and cosmic phenomena extensively. "
            "Dark matter comprises approximately twenty-seven percent of the universe."
        )

        chunks = await chunker.chunk_text_async(diverse_text, "test_diverse")

        # Verify strategy selection
        assert len(chunks) >= 1
        assert all(chunk.metadata.get("strategy") == "hybrid" for chunk in chunks)
        # Should select semantic due to topic diversity
        expected_strategies = ["semantic", "recursive"]  # Fallback might occur
        assert all(chunk.metadata.get("sub_strategy") in expected_strategies for chunk in chunks)

    async def test_hybrid_chunker_recursive_default(self) -> None:
        """Test hybrid chunker defaults to recursive for general text."""
        chunker = HybridChunker(
            markdown_density_threshold=0.5,  # High threshold
            topic_diversity_threshold=0.8,   # High threshold
        )

        # Simple, low-diversity text
        simple_text = "This is a simple test document. " * 20

        chunks = await chunker.chunk_text_async(simple_text, "test_simple")

        # Verify chunks and default strategy
        assert len(chunks) >= 1
        assert all(chunk.metadata.get("strategy") == "hybrid" for chunk in chunks)
        assert all(chunk.metadata.get("sub_strategy") == "recursive" for chunk in chunks)
        assert all("default choice" in chunk.metadata.get("selection_reason", "") for chunk in chunks)

    async def test_hybrid_chunker_analytics(self) -> None:
        """Test hybrid chunker analytics tracking."""
        chunker = HybridChunker(enable_analytics=True)

        # Process different types of content
        texts = [
            ("Simple text. " * 10, "simple"),
            ("# Header\n\nMarkdown content.", "markdown"),
            ("Machine learning uses algorithms. Cooking requires techniques. Physics studies matter.", "diverse"),
        ]

        for text, doc_id in texts:
            await chunker.chunk_text_async(text, doc_id)

        # Check analytics
        analytics = chunker.get_selection_analytics()
        assert "selection_stats" in analytics
        assert "analytics_summary" in analytics
        assert analytics["analytics_summary"]["total_selections"] == 3

    async def test_unicode_handling(self) -> None:
        """Test proper Unicode handling."""
        chunker = RecursiveChunker(chunk_size=50)

        # Various Unicode texts
        texts = [
            "Hello 世界! Testing 中文 characters.",
            "Émojis: 🚀 🌟 🎉 and symbols: → € £ ¥",
            "Mixed: Café, naïve, résumé, Zürich",
            "RTL: العربية and עברית text",
        ]

        for text in texts:
            chunks = await chunker.chunk_text_async(text, "unicode_test")

            # Should handle without errors
            assert len(chunks) >= 1

            # Reassembled text should match original (minus whitespace changes)
            reassembled = " ".join(chunk.text.strip() for chunk in chunks)
            assert text.strip() in reassembled or reassembled in text.strip()

    async def test_large_document_handling(self) -> None:
        """Test handling of large documents."""
        chunker = CharacterChunker(chunk_size=1000, chunk_overlap=200)

        # Generate large document (1MB)
        large_text = "This is a test sentence. " * 50000

        chunks = await chunker.chunk_text_async(large_text, "large_doc")

        # Should create many chunks
        assert len(chunks) > 100

        # All chunks should be within size limits
        assert all(len(chunk.text) <= 5000 for chunk in chunks)

        # Verify offsets are correct
        for i in range(len(chunks) - 1):
            assert chunks[i].end_offset <= chunks[i + 1].end_offset

    @pytest.mark.integration()
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

    def multiply(self, a, b):
        \"\"\"Multiply two numbers and store in history.\"\"\"
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def subtract(self, a, b):
        \"\"\"Subtract b from a and store in history.\"\"\"
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def divide(self, a, b):
        \"\"\"Divide a by b and store in history.\"\"\"
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result

    def get_history(self):
        \"\"\"Get the calculation history.\"\"\"
        return self.history.copy()

    def clear_history(self):
        \"\"\"Clear the calculation history.\"\"\"
        self.history.clear()

# Example usage
if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(5, 3))
    print(calc.multiply(4, 7))
    print(calc.subtract(10, 4))
    print(calc.divide(20, 5))
    print("History:", calc.get_history())
"""

        # Test with .py file type
        from packages.shared.text_processing.file_type_detector import FileTypeDetector

        config = FileTypeDetector.get_optimal_config("test.py")
        assert config["strategy"] == "recursive"
        assert config["params"]["chunk_size"] == 400  # Optimized for code
        assert config["params"]["chunk_overlap"] == 50

        # Test chunking preserves code structure reasonably
        chunker = ChunkingFactory.create_chunker(config)
        chunks = await chunker.chunk_text_async(code_file, "test.py")

        # Verify chunks
        assert len(chunks) >= 2
        assert any("fibonacci" in chunk.text for chunk in chunks)
        assert any("Calculator" in chunk.text for chunk in chunks)
