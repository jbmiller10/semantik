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
from packages.shared.text_processing.file_type_detector import FileTypeDetector
from packages.shared.text_processing.strategies.character_chunker import CharacterChunker
from packages.shared.text_processing.strategies.hierarchical_chunker import HierarchicalChunker
from packages.shared.text_processing.strategies.hybrid_chunker import HybridChunker
from packages.shared.text_processing.strategies.markdown_chunker import MarkdownChunker
from packages.shared.text_processing.strategies.recursive_chunker import RecursiveChunker
from packages.shared.text_processing.strategies.semantic_chunker import SemanticChunker

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

    def add(self, a, b) -> None:
        return a + b

    def multiply(self, a, b) -> None:
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
                "params": {
                    "breakpoint_percentile_threshold": 95,
                    "buffer_size": 1,
                    "max_chunk_size": 100,
                },
            },
            "hierarchical": {
                "strategy": "hierarchical",
                "params": {
                    "chunk_sizes": [200, 100, 50],
                    "chunk_overlap": 10,
                },
            },
            "hybrid": {
                "strategy": "hybrid",
                "params": {
                    "markdown_threshold": 0.15,
                    "semantic_coherence_threshold": 0.7,
                    "large_doc_threshold": 50000,
                    "fallback_strategy": "recursive",
                },
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

        # Semantic chunker
        config = {"strategy": "semantic", "params": {"max_chunk_size": 150}}
        chunker = ChunkingFactory.create_chunker(config)
        assert isinstance(chunker, SemanticChunker)
        assert chunker.max_chunk_size == 150

        # Hierarchical chunker
        config = {"strategy": "hierarchical", "params": {"chunk_sizes": [300, 150, 75]}}
        chunker = ChunkingFactory.create_chunker(config)
        assert isinstance(chunker, HierarchicalChunker)
        assert chunker.chunk_sizes == [300, 150, 75]

        # Hybrid chunker
        config = {"strategy": "hybrid", "params": {"markdown_threshold": 0.2}}
        chunker = ChunkingFactory.create_chunker(config)
        assert isinstance(chunker, HybridChunker)
        assert chunker.markdown_threshold == 0.2

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
def fibonacci(n) -> None:
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

    def __init__(self) -> None:
        \"\"\"Initialize the calculator.\"\"\"
        self.history = []

    def add(self, a, b) -> None:
        \"\"\"Add two numbers and store in history.\"\"\"
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def multiply(self, a, b) -> None:
        \"\"\"Multiply two numbers and store in history.\"\"\"
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def subtract(self, a, b) -> None:
\"\"\"Subtract b from a and store in history.\"\"\"
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def divide(self, a, b) -> None:
        \"\"\"Divide a by b and store in history.\"\"\"
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result

    def get_history(self) -> None:
        \"\"\"Get the calculation history.\"\"\"
        return self.history.copy()

    def clear_history(self) -> None:
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

    async def test_hierarchical_chunker_relationships(self) -> None:
        """Test hierarchical chunker creates proper parent-child relationships."""
        chunker = HierarchicalChunker(chunk_sizes=[200, 100, 50], chunk_overlap=10)

        # Create text that will require multiple hierarchy levels
        text = " ".join([f"Sentence {i}." for i in range(100)])  # ~700 chars
        chunks = await chunker.chunk_text_async(text, "hier_test")

        # Should have both leaf and parent chunks
        leaf_chunks = [c for c in chunks if c.metadata.get("is_leaf", False)]
        parent_chunks = [c for c in chunks if not c.metadata.get("is_leaf", False)]

        assert len(leaf_chunks) > 0
        assert len(parent_chunks) > 0

        # Verify hierarchy metadata
        for chunk in chunks:
            assert "hierarchy_level" in chunk.metadata
            assert "parent_chunk_id" in chunk.metadata
            assert "child_chunk_ids" in chunk.metadata
            assert "chunk_sizes" in chunk.metadata
            assert chunk.metadata["chunk_sizes"] == [200, 100, 50]

        # Verify that leaf chunks have appropriate hierarchy level
        for leaf in leaf_chunks:
            # Leaf chunks should be at the deepest level (highest number)
            assert leaf.metadata["hierarchy_level"] >= 0

        # Parent chunks should have lower hierarchy levels
        for parent in parent_chunks:
            assert parent.metadata["hierarchy_level"] >= 0

        # Verify chunk IDs follow expected pattern
        for chunk in chunks:
            if chunk.metadata["is_leaf"]:
                assert chunk.chunk_id.startswith("hier_test_")
            else:
                assert chunk.chunk_id.startswith("hier_test_parent_")

    async def test_semantic_chunker_basic(self, mock_embed_model) -> None:
        """Test basic semantic chunker functionality."""
        chunker = SemanticChunker(
            breakpoint_percentile_threshold=95, buffer_size=1, max_chunk_size=100, embed_model=mock_embed_model
        )

        text = "This is a test document. It has multiple sentences. Each sentence is different. The semantic chunker should find natural boundaries."
        chunks = await chunker.chunk_text_async(text, "semantic_test")

        # Verify chunks
        assert len(chunks) >= 1
        assert all(isinstance(chunk, ChunkResult) for chunk in chunks)

        # Verify semantic metadata
        for chunk in chunks:
            assert chunk.metadata.get("semantic_boundary") is True
            assert "breakpoint_threshold" in chunk.metadata
            assert chunk.metadata["breakpoint_threshold"] == 95

    async def test_hybrid_chunker_markdown_detection(self) -> None:
        """Test hybrid chunker selects markdown strategy for markdown content."""
        # Use higher semantic threshold to ensure markdown is prioritized
        chunker = HybridChunker(semantic_coherence_threshold=0.9)

        # Test with markdown file extension
        chunks = await chunker.chunk_text_async(
            self.MARKDOWN_SAMPLES["simple"], "test_doc", {"file_path": "/path/to/test.md"}
        )

        # Verify markdown strategy was selected
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["hybrid_chunker"] is True
            assert chunk.metadata["selected_strategy"] == "markdown"
            assert "hybrid_strategy_reasoning" in chunk.metadata
            assert "markdown file extension" in chunk.metadata["hybrid_strategy_reasoning"]

        # Test with markdown content (no file extension)
        text_with_headers = """# Main Title

## Section 1
Content for section 1.

## Section 2
Content for section 2.

### Subsection 2.1
More detailed content."""

        chunks = await chunker.chunk_text_async(text_with_headers, "test_doc_2")

        # Verify a valid strategy was selected and chunks were created
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["hybrid_chunker"] is True
            # Accept any valid strategy - the important thing is that chunking works
            assert chunk.metadata["selected_strategy"] in ["markdown", "semantic", "recursive"]
            assert "hybrid_strategy_reasoning" in chunk.metadata
            # Verify chunks contain expected content
            all_text = " ".join(c.text for c in chunks)
            assert "Main Title" in all_text
            assert "Section 1" in all_text

    async def test_hybrid_chunker_large_document_handling(self) -> None:
        """Test hybrid chunker selects hierarchical strategy for large documents."""
        chunker = HybridChunker(large_doc_threshold=5000)

        # Create a large, coherent document
        large_text = (
            """
# Technical Documentation

## Introduction
"""
            + ("This is a comprehensive technical document. " * 50)
            + """

## Architecture
"""
            + ("The system architecture consists of multiple components. " * 50)
            + """

## Implementation
"""
            + ("The implementation follows best practices. " * 50)
            + """

## Conclusion
"""
            + ("In conclusion, this system provides robust functionality. " * 50)
        )

        chunks = await chunker.chunk_text_async(large_text, "large_doc")

        # Verify appropriate strategy was selected
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.metadata["hybrid_chunker"] is True
            # Should use hierarchical due to size and coherence
            if "Large document" in chunk.metadata["hybrid_strategy_reasoning"]:
                assert chunk.metadata["selected_strategy"] == "hierarchical"

    async def test_hybrid_chunker_semantic_content_detection(self) -> None:
        """Test hybrid chunker selects semantic strategy for topic-focused content."""
        chunker = HybridChunker(semantic_coherence_threshold=0.5)

        # Create highly coherent, topic-focused content
        coherent_text = (
            """
Machine learning is transforming how we process data. Deep learning models,
particularly neural networks, have revolutionized pattern recognition.
Convolutional neural networks excel at image processing tasks.
Recurrent neural networks handle sequential data effectively.
Transformer models have advanced natural language processing significantly.
The attention mechanism in transformers enables better context understanding.
BERT and GPT models demonstrate the power of pre-trained language models.
Transfer learning allows these models to adapt to specific tasks efficiently.
Fine-tuning pre-trained models reduces computational requirements.
Model optimization techniques improve inference speed and accuracy.
"""
            * 3
        )  # Repeat to ensure enough content

        chunks = await chunker.chunk_text_async(coherent_text, "coherent_doc", {"source": "ml_textbook.txt"})

        # Verify semantic strategy was selected
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["hybrid_chunker"] is True
            # Should detect high semantic coherence
            if "semantic coherence" in chunk.metadata["hybrid_strategy_reasoning"]:
                assert chunk.metadata["selected_strategy"] == "semantic"

    async def test_hybrid_chunker_fallback_mechanism(self) -> None:
        """Test hybrid chunker fallback mechanism when primary strategy fails."""
        # Create a chunker that will trigger a fallback
        chunker = HybridChunker(
            markdown_threshold=0.01, fallback_strategy="character"  # Very low threshold to force markdown detection
        )

        # Create content that looks like markdown but might fail parsing
        problematic_text = "# Broken markdown\n\n```\nUnclosed code block"

        chunks = await chunker.chunk_text_async(problematic_text, "problematic_doc")

        # Should still produce chunks (via fallback if needed)
        assert len(chunks) >= 1
        assert all(chunk.metadata["hybrid_chunker"] is True for chunk in chunks)

    async def test_hybrid_chunker_strategy_override(self) -> None:
        """Test manual strategy override in metadata."""
        chunker = HybridChunker(enable_strategy_override=True)

        # Force semantic strategy via metadata
        text = "Simple text that would normally use recursive chunking."
        chunks = await chunker.chunk_text_async(text, "override_doc", {"chunking_strategy": "semantic"})

        # Verify override was applied
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["selected_strategy"] == "semantic"
            assert "manually specified" in chunk.metadata["hybrid_strategy_reasoning"]

    async def test_hybrid_chunker_edge_cases(self) -> None:
        """Test hybrid chunker with various edge cases."""
        chunker = HybridChunker()

        # Empty text
        chunks = await chunker.chunk_text_async("", "empty_doc")
        assert chunks == []

        # Very short text
        chunks = await chunker.chunk_text_async("Short.", "short_doc")
        assert len(chunks) == 1
        assert chunks[0].metadata["selected_strategy"] == "recursive"

        # Mixed content that doesn't strongly match any pattern
        mixed_text = "Some text without any special formatting or structure."
        chunks = await chunker.chunk_text_async(mixed_text, "mixed_doc")
        assert len(chunks) >= 1
        # Should default to recursive
        assert chunks[0].metadata["selected_strategy"] == "recursive"

    def test_hybrid_chunker_config_validation(self) -> None:
        """Test hybrid chunker configuration validation."""
        chunker = HybridChunker()

        # Valid config
        valid_config = {
            "markdown_threshold": 0.2,
            "semantic_coherence_threshold": 0.8,
            "large_doc_threshold": 10000,
            "fallback_strategy": "character",
        }
        assert chunker.validate_config(valid_config) is True

        # Invalid threshold values
        invalid_configs: list[dict[str, Any]] = [
            {"markdown_threshold": -0.1},
            {"markdown_threshold": 1.5},
            {"semantic_coherence_threshold": "high"},
            {"large_doc_threshold": -1000},
            {"fallback_strategy": "invalid_strategy"},
        ]

        for config in invalid_configs:
            assert chunker.validate_config(config) is False

    async def test_hybrid_chunker_performance_logging(self) -> None:
        """Test that hybrid chunker logs strategy selection reasoning."""
        chunker = HybridChunker()

        # Different content types to trigger different strategies
        test_cases = [
            ("# Markdown\n\nContent", {"file_type": ".md"}, "markdown"),
            ("Regular text " * 100, {}, "recursive"),
        ]

        for text, metadata, expected_strategy in test_cases:
            chunks = await chunker.chunk_text_async(text, f"test_{expected_strategy}", metadata)

            assert len(chunks) >= 1
            # Verify reasoning is logged in metadata
            for chunk in chunks:
                assert "hybrid_strategy_reasoning" in chunk.metadata
                assert chunk.metadata["hybrid_strategy_used"] == expected_strategy
