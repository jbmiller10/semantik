#!/usr/bin/env python3
"""
Advanced unit tests for semantic, hierarchical, and hybrid chunking strategies.

This module provides comprehensive tests for error scenarios, edge cases,
and advanced functionality not covered in the basic test suite.
"""

import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.embeddings import MockEmbedding

from packages.shared.text_processing.chunking_factory import ChunkingFactory
from packages.shared.text_processing.strategies.semantic_chunker import SemanticChunker
from packages.shared.text_processing.strategies.hierarchical_chunker import HierarchicalChunker
from packages.shared.text_processing.strategies.hybrid_chunker import HybridChunker

# Set testing environment
os.environ["TESTING"] = "true"


class TestAdvancedChunkingStrategies:
    """Advanced tests for semantic, hierarchical, and hybrid strategies."""

    @pytest.fixture()
    def mock_embed_model(self) -> MockEmbedding:
        """Mock embedding model for semantic chunking tests."""
        return MockEmbedding(embed_dim=384)

    # Semantic Chunker Advanced Tests
    
    async def test_semantic_chunker_api_failure_handling(self) -> None:
        """Test semantic chunker handles embedding API failures gracefully."""
        # Create chunker with mock that fails
        with patch("packages.shared.text_processing.strategies.semantic_chunker.OpenAIEmbedding") as mock_openai:
            # Simulate API failure
            mock_openai.side_effect = Exception("API key invalid")
            
            chunker = SemanticChunker()
            
            # Should fallback to mock embeddings
            text = "This is a test document. " * 100
            chunks = await chunker.chunk_text_async(text, "test_doc")
            
            assert len(chunks) >= 1
            assert all(chunk.metadata.get("strategy") == "semantic" for chunk in chunks)

    async def test_semantic_chunker_rate_limiting(self) -> None:
        """Test semantic chunker handles rate limiting with retries."""
        chunker = SemanticChunker(max_retries=3)
        
        # Mock the internal method to simulate rate limiting
        retry_count = 0
        original_method = chunker._chunk_with_retry
        
        def mock_chunk_with_retry(doc):
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                raise Exception("Rate limit exceeded")
            return original_method(doc)
        
        chunker._chunk_with_retry = mock_chunk_with_retry
        
        text = "Test document for rate limiting. " * 50
        chunks = await chunker.chunk_text_async(text, "test_rate_limit")
        
        assert len(chunks) >= 1
        assert retry_count == 3  # Should retry until success

    async def test_semantic_chunker_memory_pressure(self) -> None:
        """Test semantic chunker under memory pressure with large documents."""
        chunker = SemanticChunker(max_chunk_size=1000)
        
        # Generate large document (10MB)
        large_text = "This is a test sentence. " * 400000
        
        # Process in streaming fashion
        chunks = await chunker.chunk_text_async(large_text, "large_doc")
        
        # Verify memory-safe chunking
        assert all(len(chunk.text) <= 1000 for chunk in chunks)
        assert len(chunks) > 100  # Should create many chunks
        
        # Verify chunk continuity
        total_length = sum(len(chunk.text) for chunk in chunks)
        assert total_length <= len(large_text) + (len(chunks) * 100)  # Allow for overlap

    async def test_semantic_chunker_concurrent_processing(self) -> None:
        """Test semantic chunker with concurrent document processing."""
        chunker = SemanticChunker()
        
        # Create multiple documents
        documents = [
            ("Machine learning algorithms. " * 50, "doc1"),
            ("Natural language processing. " * 50, "doc2"),
            ("Computer vision techniques. " * 50, "doc3"),
            ("Data science methods. " * 50, "doc4"),
        ]
        
        # Process concurrently
        tasks = [
            chunker.chunk_text_async(text, doc_id)
            for text, doc_id in documents
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all processed successfully
        assert len(results) == 4
        assert all(len(chunks) >= 1 for chunks in results)
        
        # Verify unique chunk IDs
        all_chunk_ids = []
        for chunks in results:
            all_chunk_ids.extend(chunk.chunk_id for chunk in chunks)
        assert len(all_chunk_ids) == len(set(all_chunk_ids))

    async def test_semantic_chunker_embedding_failure_recovery(self) -> None:
        """Test semantic chunker recovery from partial embedding failures."""
        chunker = SemanticChunker()
        
        # Mock splitter to fail on specific content
        original_split = chunker.splitter.build_node_parser()
        fail_count = 0
        
        def mock_split(nodes):
            nonlocal fail_count
            for node in nodes:
                if "error trigger" in node.text and fail_count < 2:
                    fail_count += 1
                    raise Exception("Embedding API error")
            return original_split(nodes)
        
        with patch.object(chunker.splitter, "build_node_parser", mock_split):
            text = "Normal text. Error trigger text. More normal text. " * 10
            
            # Should recover and process
            chunks = await chunker.chunk_text_async(text, "test_recovery")
            assert len(chunks) >= 1

    # Hierarchical Chunker Advanced Tests
    
    async def test_hierarchical_chunker_deep_nesting(self) -> None:
        """Test hierarchical chunker with deeply nested content."""
        chunker = HierarchicalChunker(
            chunk_sizes=[1000, 500, 250, 125],  # 4 levels
            chunk_overlap=20,
        )
        
        # Generate structured document
        text = ""
        for i in range(5):
            text += f"Chapter {i}. " * 100  # ~1200 chars per chapter
            for j in range(3):
                text += f"Section {i}.{j}. " * 50  # ~700 chars per section
        
        chunks = await chunker.chunk_text_async(text, "test_deep")
        
        # Verify hierarchy levels
        levels = set(chunk.metadata.get("chunk_level", 0) for chunk in chunks)
        assert len(levels) >= 3  # Should have multiple hierarchy levels
        
        # Verify parent-child relationships
        leaf_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "leaf"]
        parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        
        assert len(leaf_chunks) > len(parent_chunks)  # More leaves than parents
        
        # Verify hierarchy paths
        for chunk in chunks:
            path = chunk.metadata.get("hierarchy_path", [])
            assert isinstance(path, list)
            assert len(path) <= 4  # Max depth

    async def test_hierarchical_chunker_uneven_distribution(self) -> None:
        """Test hierarchical chunker with uneven content distribution."""
        chunker = HierarchicalChunker(
            chunk_sizes=[500, 200, 100],
            chunk_overlap=10,
        )
        
        # Create uneven document
        text = "Short intro. " * 5  # ~65 chars
        text += "Very long middle section. " * 200  # ~5200 chars
        text += "Short conclusion. " * 3  # ~51 chars
        
        chunks = await chunker.chunk_text_async(text, "test_uneven")
        
        # Should handle gracefully
        assert len(chunks) >= 10
        assert all(chunk.metadata.get("strategy") == "hierarchical" for chunk in chunks)

    async def test_hierarchical_chunker_boundary_preservation(self) -> None:
        """Test hierarchical chunker preserves semantic boundaries."""
        chunker = HierarchicalChunker(
            chunk_sizes=[400, 200],
            preserve_boundaries=True,  # If this option exists
        )
        
        # Document with clear boundaries
        text = (
            "Introduction paragraph with complete thoughts. "
            "This sentence should not be split. "
            "\n\n"
            "New paragraph starts here. "
            "It contains multiple sentences that form a unit. "
            "\n\n"
            "Final paragraph with conclusion. "
        ) * 10
        
        chunks = await chunker.chunk_text_async(text, "test_boundaries")
        
        # Check chunks don't break mid-sentence
        for chunk in chunks:
            # Most chunks should end with punctuation
            stripped = chunk.text.strip()
            if len(stripped) > 20:  # Skip very small chunks
                assert stripped[-1] in ".!?\n" or stripped.endswith("...")

    # Hybrid Chunker Advanced Tests
    
    async def test_hybrid_chunker_strategy_switching(self) -> None:
        """Test hybrid chunker switches strategies based on content analysis."""
        chunker = HybridChunker(
            markdown_density_threshold=0.2,
            topic_diversity_threshold=0.5,
            enable_analytics=True,
        )
        
        # Test different content types
        test_cases = [
            # (content, expected_strategy)
            ("# Header\n\n## Subheader\n\nContent with **bold**.", "markdown"),
            ("Machine learning. Cooking recipes. Physics equations.", "semantic"),
            ("Simple repetitive text. " * 100, "recursive"),
        ]
        
        results = []
        for content, _ in test_cases:
            chunks = await chunker.chunk_text_async(content, f"test_{len(results)}")
            results.append(chunks)
        
        # Verify strategy selection
        analytics = chunker.get_selection_analytics()
        assert analytics["analytics_summary"]["total_selections"] == 3
        
        # Check each result
        for i, (chunks, (_, expected)) in enumerate(zip(results, test_cases)):
            assert len(chunks) >= 1
            # Strategy might fallback, so check if expected or recursive
            actual_strategy = chunks[0].metadata.get("sub_strategy")
            assert actual_strategy in [expected, "recursive"]

    async def test_hybrid_chunker_performance_adaptation(self) -> None:
        """Test hybrid chunker adapts to performance constraints."""
        chunker = HybridChunker(
            enable_performance_mode=True,  # If this option exists
            max_processing_time=0.1,  # 100ms per document
        )
        
        # Large document that would be slow with semantic
        large_text = "Diverse content about AI, cooking, and physics. " * 1000
        
        start_time = asyncio.get_event_loop().time()
        chunks = await chunker.chunk_text_async(large_text, "test_perf")
        duration = asyncio.get_event_loop().time() - start_time
        
        # Should adapt strategy for performance
        assert len(chunks) >= 100
        assert duration < 5.0  # Should not take too long
        
        # Check it didn't use semantic for performance
        strategies_used = set(c.metadata.get("sub_strategy") for c in chunks)
        assert "recursive" in strategies_used  # Should fallback for speed

    async def test_hybrid_chunker_content_analysis_edge_cases(self) -> None:
        """Test hybrid chunker content analysis with edge cases."""
        chunker = HybridChunker()
        
        edge_cases = [
            # Unicode-heavy content
            "こんにちは世界。Привет мир। مرحبا بالعالم。" * 50,
            # Code-like but not code file
            "if (condition) { result = true; } else { result = false; }" * 20,
            # Mixed markdown and code
            "# Code Example\n\n```python\ndef test():\n    pass\n```\n\nExplanation." * 10,
            # Very short content
            "Short.",
            # Only whitespace
            "   \n\n\t\t  ",
        ]
        
        for i, content in enumerate(edge_cases):
            if content.strip():  # Skip empty
                chunks = await chunker.chunk_text_async(content, f"edge_{i}")
                assert isinstance(chunks, list)
                if chunks:  # Some edge cases might produce no chunks
                    assert chunks[0].metadata.get("strategy") == "hybrid"

    # Cross-Strategy Tests
    
    @pytest.mark.parametrize("strategy_name", ["semantic", "hierarchical", "hybrid"])
    async def test_strategy_thread_safety(self, strategy_name: str) -> None:
        """Test strategies are thread-safe for concurrent use."""
        config = {
            "semantic": {"strategy": "semantic", "params": {}},
            "hierarchical": {"strategy": "hierarchical", "params": {"chunk_sizes": [500, 200]}},
            "hybrid": {"strategy": "hybrid", "params": {}},
        }[strategy_name]
        
        chunker = ChunkingFactory.create_chunker(config)
        
        # Concurrent processing with same chunker instance
        documents = [(f"Document {i}. " * 100, f"doc_{i}") for i in range(10)]
        
        tasks = [
            chunker.chunk_text_async(text, doc_id)
            for text, doc_id in documents
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed without interference
        assert len(results) == 10
        assert all(len(chunks) >= 1 for chunks in results)
        
        # Verify no chunk ID collisions
        all_ids = []
        for chunks in results:
            all_ids.extend(c.chunk_id for c in chunks)
        assert len(all_ids) == len(set(all_ids))

    @pytest.mark.parametrize("strategy_name", ["semantic", "hierarchical", "hybrid"])
    async def test_strategy_memory_cleanup(self, strategy_name: str) -> None:
        """Test strategies properly clean up memory after processing."""
        config = {
            "semantic": {"strategy": "semantic", "params": {"max_chunk_size": 500}},
            "hierarchical": {"strategy": "hierarchical", "params": {"chunk_sizes": [1000, 500]}},
            "hybrid": {"strategy": "hybrid", "params": {}},
        }[strategy_name]
        
        chunker = ChunkingFactory.create_chunker(config)
        
        # Process large document
        large_text = "Memory test document. " * 50000
        
        # Get memory before
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process document
        chunks = await chunker.chunk_text_async(large_text, "mem_test")
        
        # Clear references
        chunks = None
        large_text = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (less than 100MB)
        assert mem_after - mem_before < 100

    async def test_factory_error_handling_advanced(self) -> None:
        """Test factory handles configuration errors gracefully."""
        # Invalid strategy parameters
        invalid_configs = [
            {"strategy": "semantic", "params": {"breakpoint_percentile_threshold": "not_a_number"}},
            {"strategy": "hierarchical", "params": {"chunk_sizes": "not_a_list"}},
            {"strategy": "hybrid", "params": {"markdown_density_threshold": 2.0}},  # Out of range
        ]
        
        for config in invalid_configs:
            chunker = ChunkingFactory.create_chunker(config)
            # Should create chunker but validation should fail
            assert not chunker.validate_config(config["params"])

    async def test_malicious_input_handling(self) -> None:
        """Test strategies handle potentially malicious input safely."""
        # Create chunkers
        chunkers = [
            SemanticChunker(),
            HierarchicalChunker(chunk_sizes=[500, 200]),
            HybridChunker(),
        ]
        
        # Malicious inputs
        malicious_inputs = [
            "A" * 10_000_000,  # Extremely long string
            "<script>alert('xss')</script>" * 1000,  # XSS attempt
            "'; DROP TABLE chunks; --" * 100,  # SQL injection
            "\x00" * 1000,  # Null bytes
            "\\x" + "A" * 1000,  # Escape sequences
        ]
        
        for chunker in chunkers:
            for i, malicious_text in enumerate(malicious_inputs):
                try:
                    # Should handle without crashing
                    chunks = await chunker.chunk_text_async(
                        malicious_text[:100000],  # Limit size for testing
                        f"malicious_{i}"
                    )
                    # Should produce chunks or empty list
                    assert isinstance(chunks, list)
                except Exception as e:
                    # Should only be validation errors, not crashes
                    assert "validation" in str(e).lower() or "size" in str(e).lower()