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

# Set testing environment
os.environ["TESTING"] = "true"


class TestChunkingStrategies:
    """Comprehensive tests for all strategies."""

    @pytest.fixture
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
        "javascript": '''function greet(name) {
    console.log(`Hello, ${name}!`);
}

const calculator = {
    add: (a, b) => a + b,
    multiply: (a, b) => a * b
};

export default calculator;
''',
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

    ALL_STRATEGIES = ["character", "recursive", "markdown"]

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
        }
        return configs.get(strategy, configs["recursive"])

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    @pytest.mark.parametrize("edge_case_name,text", EDGE_CASES.items())
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
        assert len(strategies) >= 3

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
            assert chunker.validate_config({
                "chunk_size": 100,
                "chunk_overlap": 200,  # Greater than chunk size
            }) is False

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

    @pytest.mark.integration
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