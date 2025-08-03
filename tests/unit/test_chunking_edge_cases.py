#!/usr/bin/env python3
"""
Edge case tests for all chunking strategies.

This module provides comprehensive edge case testing to ensure
robustness across all chunking implementations.
"""

import asyncio
import os
from typing import List, Tuple

import pytest

from packages.shared.text_processing.chunking_factory import ChunkingFactory
from packages.shared.text_processing.base_chunker import ChunkResult

# Set testing environment
os.environ["TESTING"] = "true"


class TestChunkingEdgeCases:
    """Edge case tests for all chunking strategies."""
    
    # Document edge cases
    EDGE_CASE_DOCUMENTS = {
        "empty_variations": [
            "",                    # Completely empty
            " ",                   # Single space
            "\n",                  # Single newline
            "\t",                  # Single tab
            "   \n\t  ",          # Mixed whitespace
            "\n\n\n\n\n",         # Multiple newlines
            "\r\n\r\n",           # Windows line endings
            "\u200b",             # Zero-width space
            "\ufeff",             # Byte order mark
        ],
        "boundary_cases": [
            "a",                   # Single character
            "ab",                  # Two characters
            "a b",                 # Two chars with space
            "word",                # Single word
            "Two words",           # Two words
            "Short sentence.",     # Single sentence
            "One. Two.",          # Two sentences
            "One.\nTwo.",         # Sentences with newline
        ],
        "special_characters": [
            "Hello\x00World",      # Null byte
            "Test\x01\x02\x03",   # Control characters
            "Unicode: 你好世界",    # Chinese
            "Emoji: 😀🚀🌟",       # Emojis
            "RTL: مرحبا",          # Right-to-left
            "Mixed: Hello שלום",   # Mixed direction
            "Symbols: ∑∫∂∆",      # Mathematical symbols
            "Currency: $€£¥₹",    # Currency symbols
        ],
        "repetitive_patterns": [
            "a" * 1000,           # Single character repeated
            "ab" * 500,           # Pattern repeated
            "word " * 200,        # Word repeated
            "line\n" * 100,       # Line repeated
            "." * 500,            # Punctuation repeated
            " " * 1000,           # Spaces repeated
        ],
        "nested_structures": [
            "((()))",             # Nested parentheses
            "[[[text]]]",         # Nested brackets
            "{{{{{data}}}}}",     # Nested braces
            "```code```" * 10,    # Code blocks
            "# H1\n## H2\n### H3",# Nested headers
            "<div><p>text</p></div>", # HTML-like
        ],
    }
    
    ALL_STRATEGIES = ["character", "recursive", "markdown", "semantic", "hierarchical", "hybrid"]
    
    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    @pytest.mark.parametrize("doc", EDGE_CASE_DOCUMENTS["empty_variations"])
    async def test_empty_document_variations(self, strategy: str, doc: str) -> None:
        """Test strategies handle various empty/whitespace documents."""
        config = self._get_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)
        
        chunks = await chunker.chunk_text_async(doc, "empty_test")
        
        # Should handle gracefully
        assert isinstance(chunks, list)
        
        # Empty or whitespace-only should produce no chunks
        if not doc.strip():
            assert len(chunks) == 0
        else:
            # If chunks are produced, they should be valid
            for chunk in chunks:
                assert isinstance(chunk, ChunkResult)
                assert chunk.chunk_id
                assert isinstance(chunk.metadata, dict)

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    @pytest.mark.parametrize("doc", EDGE_CASE_DOCUMENTS["boundary_cases"])
    async def test_boundary_size_documents(self, strategy: str, doc: str) -> None:
        """Test strategies handle documents at size boundaries."""
        config = self._get_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)
        
        chunks = await chunker.chunk_text_async(doc, "boundary_test")
        
        # Should produce at least one chunk for non-empty input
        if doc.strip():
            assert len(chunks) >= 1
            
            # First chunk should contain the content
            assert doc in chunks[0].text or chunks[0].text in doc
            
            # Verify chunk boundaries
            for chunk in chunks:
                assert chunk.start_offset >= 0
                assert chunk.end_offset >= chunk.start_offset
                assert chunk.end_offset <= len(doc)

    @pytest.mark.parametrize("strategy", ["semantic", "hierarchical", "hybrid"])
    async def test_advanced_strategy_minimum_content(self, strategy: str) -> None:
        """Test advanced strategies with minimal content."""
        minimal_docs = [
            "Word",
            "Two words",
            "Short sentence here.",
            "One. Two. Three.",
            "Line one\nLine two",
        ]
        
        config = self._get_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)
        
        for doc in minimal_docs:
            chunks = await chunker.chunk_text_async(doc, "minimal_test")
            
            # Should handle small documents
            assert len(chunks) >= 1
            
            # Content should be preserved
            combined = " ".join(chunk.text for chunk in chunks)
            assert doc in combined or combined in doc

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    async def test_exact_chunk_size_boundary(self, strategy: str) -> None:
        """Test documents exactly at chunk size boundaries."""
        config = self._get_config(strategy)
        
        # Get chunk size for strategy
        chunk_size = 100  # Default
        if strategy in ["character", "recursive"]:
            chunk_size = config["params"].get("chunk_size", 100)
        elif strategy == "hierarchical":
            chunk_size = config["params"]["chunk_sizes"][0]
        
        # Create document exactly at chunk size
        doc = "a" * chunk_size
        
        chunker = ChunkingFactory.create_chunker(config)
        chunks = await chunker.chunk_text_async(doc, "exact_boundary")
        
        # Should handle exact boundaries
        assert len(chunks) >= 1
        
        # For size-based strategies, should create predictable chunks
        if strategy in ["character", "recursive"]:
            # Might be 1 or 2 chunks depending on overlap
            assert len(chunks) <= 2

    @pytest.mark.parametrize("strategy", ALL_STRATEGIES)
    async def test_unicode_boundary_splitting(self, strategy: str) -> None:
        """Test strategies don't split unicode characters."""
        # Documents with multi-byte characters
        unicode_docs = [
            "Hello 世界 World",  # Mixed ASCII and Chinese
            "Test 🚀 Emoji 🌟",  # Emojis
            "Café résumé naïve", # Accented characters
            "Привет мир",        # Cyrillic
            "مرحبا بالعالم",     # Arabic
        ]
        
        config = self._get_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)
        
        for doc in unicode_docs:
            chunks = await chunker.chunk_text_async(doc, "unicode_boundary")
            
            # Verify no character corruption
            for chunk in chunks:
                # Should be valid UTF-8
                chunk.text.encode('utf-8')
                
                # Should not contain replacement characters
                assert '\ufffd' not in chunk.text

    @pytest.mark.parametrize("strategy", ["semantic", "hierarchical", "hybrid"])
    async def test_advanced_strategy_with_noise(self, strategy: str) -> None:
        """Test advanced strategies handle noisy/corrupted input."""
        noisy_docs = [
            "Normal text with \x00\x01\x02 binary data",
            "Text with lots........ of...... dots......",
            "CAPS TEXT WITH RANDOM case ChAnGeS",
            "Text\n\n\n\nwith\n\n\n\nexcessive\n\n\n\nnewlines",
            "Text     with     excessive     spaces",
            "!!!Lots!!! of!!! punctuation!!!",
        ]
        
        config = self._get_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)
        
        for doc in noisy_docs:
            try:
                chunks = await chunker.chunk_text_async(doc, "noisy_test")
                
                # Should handle noisy input
                assert isinstance(chunks, list)
                
                # Should produce some chunks
                if doc.strip():
                    assert len(chunks) >= 1
                    
            except Exception as e:
                # Should not crash on noisy input
                pytest.fail(f"Strategy {strategy} failed on noisy input: {str(e)}")

    async def test_cross_strategy_consistency(self) -> None:
        """Test all strategies produce consistent results for same input."""
        test_doc = "This is a test document. " * 50  # ~1250 chars
        
        results = {}
        for strategy in self.ALL_STRATEGIES:
            config = self._get_config(strategy)
            chunker = ChunkingFactory.create_chunker(config)
            
            chunks = await chunker.chunk_text_async(test_doc, "consistency_test")
            
            results[strategy] = {
                "num_chunks": len(chunks),
                "total_length": sum(len(c.text) for c in chunks),
                "chunks": chunks,
            }
        
        # All strategies should preserve content (allowing for overlap)
        doc_length = len(test_doc)
        for strategy, result in results.items():
            # Total length should be close to original (with some overlap allowed)
            assert result["total_length"] <= doc_length * 2  # Max 2x for overlap
            assert result["total_length"] >= doc_length * 0.9  # At least 90%
            
            # All chunks should have required metadata
            for chunk in result["chunks"]:
                assert chunk.metadata.get("strategy") == strategy
                assert "chunk_index" in chunk.metadata

    async def test_concurrent_edge_case_processing(self) -> None:
        """Test concurrent processing of edge cases doesn't cause issues."""
        # Mix of edge cases
        edge_cases = [
            "",                          # Empty
            "a",                        # Single char
            "Test " * 1000,             # Large
            "Unicode: 你好 🚀",          # Unicode
            "\n\n\n",                   # Whitespace
            "((()))",                   # Nested
        ]
        
        # Process with all strategies concurrently
        tasks = []
        for strategy in self.ALL_STRATEGIES:
            config = self._get_config(strategy)
            chunker = ChunkingFactory.create_chunker(config)
            
            for i, doc in enumerate(edge_cases):
                task = chunker.chunk_text_async(doc, f"{strategy}_{i}")
                tasks.append((strategy, doc, task))
        
        # Gather results
        results = []
        for strategy, doc, task in tasks:
            try:
                chunks = await task
                results.append((strategy, doc, len(chunks)))
            except Exception as e:
                pytest.fail(f"Failed on {strategy} with '{doc[:20]}...': {e}")
        
        # Verify all completed
        assert len(results) == len(tasks)

    async def test_performance_edge_cases(self) -> None:
        """Test performance doesn't degrade catastrophically on edge cases."""
        import time
        
        # Pathological cases that might cause performance issues
        pathological_cases = [
            ("deep_nesting", "(" * 1000 + "text" + ")" * 1000),
            ("repetitive", "a" * 10000),
            ("many_newlines", "\n" * 5000),
            ("unicode_heavy", "🚀" * 1000),
            ("mixed_boundaries", "a b " * 2000),
        ]
        
        for case_name, doc in pathological_cases:
            for strategy in ["semantic", "hierarchical", "hybrid"]:
                config = self._get_config(strategy)
                chunker = ChunkingFactory.create_chunker(config)
                
                start = time.time()
                chunks = await chunker.chunk_text_async(doc[:5000], f"{case_name}_{strategy}")
                duration = time.time() - start
                
                # Should complete in reasonable time (5 seconds)
                assert duration < 5.0, f"{strategy} took {duration:.1f}s on {case_name}"
                
                # Should produce chunks
                assert isinstance(chunks, list)

    def _get_config(self, strategy: str) -> dict:
        """Get test configuration for strategy."""
        configs = {
            "character": {
                "strategy": "character",
                "params": {"chunk_size": 100, "chunk_overlap": 20}
            },
            "recursive": {
                "strategy": "recursive", 
                "params": {"chunk_size": 100, "chunk_overlap": 20}
            },
            "markdown": {
                "strategy": "markdown",
                "params": {}
            },
            "semantic": {
                "strategy": "semantic",
                "params": {
                    "breakpoint_percentile_threshold": 90,
                    "buffer_size": 1,
                    "max_chunk_size": 1000
                }
            },
            "hierarchical": {
                "strategy": "hierarchical",
                "params": {
                    "chunk_sizes": [500, 200, 100],
                    "chunk_overlap": 20
                }
            },
            "hybrid": {
                "strategy": "hybrid",
                "params": {
                    "markdown_density_threshold": 0.1,
                    "topic_diversity_threshold": 0.7
                }
            }
        }
        return configs[strategy]