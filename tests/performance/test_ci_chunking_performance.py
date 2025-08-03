#!/usr/bin/env python3
"""
CI-friendly performance tests for chunking strategies.

This module provides lightweight performance tests that can run reliably in CI
environments without GPU support or heavy resource requirements.
"""

# Import CI wrapper first to ensure proper mocking
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_test_wrapper import *  # noqa: F401, F403

import asyncio
import time
from typing import Dict, Any, List

import pytest

from packages.shared.text_processing.chunking_factory import ChunkingFactory
from packages.shared.text_processing.base_chunker import ChunkResult

# Ensure testing environment
os.environ["TESTING"] = "true"


class TestCIChunkingPerformance:
    """Lightweight performance tests suitable for CI environments."""
    
    @pytest.fixture(autouse=True)
    def setup_test_env(self):
        """Ensure proper test environment for each test."""
        os.environ["TESTING"] = "true"
        os.environ["USE_MOCK_EMBEDDINGS"] = "true"
        yield
    
    def generate_test_document(self, size: str = "small") -> str:
        """Generate simple test documents for CI."""
        sizes = {
            "small": 500,    # 500 chars
            "medium": 2000,  # 2KB
            "large": 5000,   # 5KB
        }
        
        char_count = sizes.get(size, 500)
        sentences = []
        
        # Generate simple sentences
        base_sentence = "This is a test sentence for chunking performance evaluation. "
        while len("".join(sentences)) < char_count:
            sentences.append(base_sentence)
        
        return "".join(sentences)[:char_count]
    
    @pytest.mark.timeout(10)  # 10 second timeout
    async def test_basic_semantic_chunking(self) -> None:
        """Test basic semantic chunking with mock embeddings."""
        config = {
            "strategy": "semantic",
            "params": {
                "breakpoint_percentile_threshold": 90,
                "buffer_size": 1,
                "max_chunk_size": 500,
                "max_retries": 1,
                "embed_batch_size": 4,
            }
        }
        
        chunker = ChunkingFactory.create_chunker(config)
        doc = self.generate_test_document("small")
        
        start_time = time.time()
        chunks = await chunker.chunk_text_async(doc, "test_doc_1")
        duration = time.time() - start_time
        
        assert len(chunks) > 0, "Should produce at least one chunk"
        assert duration < 2.0, f"Small document should chunk in under 2s, took {duration:.2f}s"
        
        # Verify chunk structure
        for chunk in chunks:
            assert isinstance(chunk, ChunkResult)
            assert chunk.text
            assert chunk.chunk_id
            assert chunk.metadata.get("strategy") == "semantic"
    
    @pytest.mark.timeout(10)
    async def test_hierarchical_chunking_performance(self) -> None:
        """Test hierarchical chunking performance."""
        config = {
            "strategy": "hierarchical",
            "params": {
                "chunk_sizes": [500, 250],  # Just 2 levels for CI
                "chunk_overlap": 50,
            }
        }
        
        chunker = ChunkingFactory.create_chunker(config)
        doc = self.generate_test_document("medium")
        
        start_time = time.time()
        chunks = await chunker.chunk_text_async(doc, "test_doc_2")
        duration = time.time() - start_time
        
        assert len(chunks) > 0
        assert duration < 1.0, f"Hierarchical chunking should be fast, took {duration:.2f}s"
        
        # Check hierarchy
        levels = set(chunk.metadata.get("level", 0) for chunk in chunks)
        assert len(levels) <= 2, "Should have at most 2 hierarchy levels"
    
    @pytest.mark.timeout(10)
    async def test_hybrid_chunking_performance(self) -> None:
        """Test hybrid chunking performance."""
        config = {
            "strategy": "hybrid",
            "params": {
                "enable_analytics": False,  # Disable for CI
                "markdown_density_threshold": 0.1,
                "topic_diversity_threshold": 0.7,
            }
        }
        
        chunker = ChunkingFactory.create_chunker(config)
        
        # Create mixed content
        doc = """
# Test Document

This is a regular paragraph with some content.

## Section 1
More content here with different topics.

```python
def example():
    return "code block"
```

## Section 2
Final section with conclusion.
"""
        
        start_time = time.time()
        chunks = await chunker.chunk_text_async(doc, "test_doc_3")
        duration = time.time() - start_time
        
        assert len(chunks) > 0
        assert duration < 1.0, f"Hybrid chunking should be fast, took {duration:.2f}s"
    
    @pytest.mark.timeout(15)
    async def test_concurrent_chunking(self) -> None:
        """Test concurrent chunking with multiple strategies."""
        configs = [
            {"strategy": "recursive", "params": {"chunk_size": 200, "chunk_overlap": 50}},
            {"strategy": "character", "params": {"chunk_size": 200, "chunk_overlap": 50}},
            {"strategy": "token", "params": {"chunk_size": 50, "chunk_overlap": 10}},
        ]
        
        # Create small test documents
        documents = [self.generate_test_document("small") for _ in range(3)]
        
        # Process concurrently
        start_time = time.time()
        tasks = []
        
        for i, (config, doc) in enumerate(zip(configs, documents)):
            chunker = ChunkingFactory.create_chunker(config)
            task = chunker.chunk_text_async(doc, f"concurrent_{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time
        
        assert all(len(chunks) > 0 for chunks in results)
        assert duration < 3.0, f"Concurrent processing should complete quickly, took {duration:.2f}s"
    
    @pytest.mark.timeout(10)
    def test_sync_chunking_performance(self) -> None:
        """Test synchronous chunking performance."""
        config = {
            "strategy": "recursive",
            "params": {
                "chunk_size": 300,
                "chunk_overlap": 50,
            }
        }
        
        chunker = ChunkingFactory.create_chunker(config)
        doc = self.generate_test_document("medium")
        
        start_time = time.time()
        chunks = chunker.chunk_text(doc, "sync_test")
        duration = time.time() - start_time
        
        assert len(chunks) > 0
        assert duration < 0.5, f"Sync recursive chunking should be very fast, took {duration:.2f}s"
    
    @pytest.mark.timeout(10)
    async def test_empty_document_handling(self) -> None:
        """Test performance with empty or minimal documents."""
        strategies = ["semantic", "hierarchical", "hybrid", "recursive"]
        
        for strategy in strategies:
            if strategy == "semantic":
                config = {"strategy": strategy, "params": {"max_retries": 1}}
            else:
                config = {"strategy": strategy, "params": {}}
            
            chunker = ChunkingFactory.create_chunker(config)
            
            # Test empty document
            chunks = await chunker.chunk_text_async("", f"empty_{strategy}")
            assert len(chunks) == 0, f"{strategy} should return no chunks for empty doc"
            
            # Test whitespace-only document
            chunks = await chunker.chunk_text_async("   \n\n   ", f"whitespace_{strategy}")
            assert len(chunks) == 0, f"{strategy} should return no chunks for whitespace"
    
    @pytest.mark.timeout(20)
    async def test_batch_processing_performance(self) -> None:
        """Test batch processing performance for CI."""
        config = {
            "strategy": "recursive",
            "params": {
                "chunk_size": 200,
                "chunk_overlap": 20,
            }
        }
        
        chunker = ChunkingFactory.create_chunker(config)
        
        # Create 10 small documents
        documents = [(self.generate_test_document("small"), f"batch_{i}") for i in range(10)]
        
        start_time = time.time()
        all_chunks = []
        
        # Process in batches of 5
        for i in range(0, len(documents), 5):
            batch = documents[i:i+5]
            tasks = [chunker.chunk_text_async(doc, doc_id) for doc, doc_id in batch]
            batch_results = await asyncio.gather(*tasks)
            all_chunks.extend(batch_results)
        
        duration = time.time() - start_time
        
        assert len(all_chunks) == 10, "Should process all documents"
        assert all(len(chunks) > 0 for chunks in all_chunks)
        assert duration < 2.0, f"Batch processing should be efficient, took {duration:.2f}s"
        
        # Calculate throughput
        total_chars = sum(len(doc) for doc, _ in documents)
        chars_per_sec = total_chars / duration
        print(f"CI Batch throughput: {chars_per_sec:.0f} chars/sec")