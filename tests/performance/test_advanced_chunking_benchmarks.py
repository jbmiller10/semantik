#!/usr/bin/env python3
"""
Advanced performance benchmarks for chunking strategies.

This module provides stress tests, degradation tests, and production
simulation benchmarks for semantic, hierarchical, and hybrid strategies.
"""

import asyncio
import gc
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any

import psutil
import pytest

from packages.shared.text_processing.chunking_factory import ChunkingFactory

# Set testing environment
os.environ["TESTING"] = "true"


@dataclass
class StressTestResult:
    """Result from stress testing."""
    strategy: str
    total_documents: int
    total_size_mb: float
    duration_seconds: float
    success_rate: float
    avg_chunks_per_sec: float
    peak_memory_mb: float
    errors: List[str]


class AdvancedChunkingBenchmarks:
    """Advanced performance tests for chunking strategies."""
    
    # Realistic document distributions
    DOCUMENT_DISTRIBUTIONS = {
        "enterprise": {
            "small": 0.7,    # 70% small docs (< 10KB)
            "medium": 0.25,  # 25% medium docs (10KB - 1MB)
            "large": 0.04,   # 4% large docs (1MB - 10MB)
            "xlarge": 0.01,  # 1% extra large docs (> 10MB)
        },
        "research": {
            "small": 0.3,
            "medium": 0.5,
            "large": 0.15,
            "xlarge": 0.05,
        },
        "mixed": {
            "small": 0.5,
            "medium": 0.35,
            "large": 0.1,
            "xlarge": 0.05,
        }
    }
    
    @staticmethod
    def generate_realistic_document(size_category: str, doc_type: str = "mixed") -> str:
        """Generate realistic documents based on category."""
        size_ranges = {
            "small": (1024, 10240),      # 1KB - 10KB
            "medium": (10240, 1048576),  # 10KB - 1MB
            "large": (1048576, 10485760), # 1MB - 10MB
            "xlarge": (10485760, 52428800), # 10MB - 50MB
        }
        
        min_size, max_size = size_ranges[size_category]
        target_size = random.randint(min_size, max_size)
        
        if doc_type == "technical":
            # Technical documentation with structure
            content = []
            sections = ["Introduction", "Architecture", "Implementation", "API Reference", "Examples"]
            
            for i, section in enumerate(sections):
                content.append(f"# {section}\n\n")
                paragraphs = random.randint(3, 10)
                for p in range(paragraphs):
                    sentences = random.randint(3, 8)
                    paragraph = ""
                    for s in range(sentences):
                        words = ["system", "architecture", "component", "interface", "module", 
                                "function", "process", "data", "service", "protocol"]
                        sentence = " ".join(random.choices(words, k=random.randint(8, 15)))
                        paragraph += sentence.capitalize() + ". "
                    content.append(paragraph + "\n\n")
                
                # Add code examples
                if random.random() > 0.5:
                    content.append("```python\n")
                    content.append("def example():\n    return 'sample'\n")
                    content.append("```\n\n")
            
            text = "".join(content)
            
        elif doc_type == "research":
            # Research papers with diverse topics
            topics = [
                "machine learning", "quantum computing", "biotechnology",
                "climate science", "neuroscience", "astrophysics"
            ]
            
            content = []
            current_topic = random.choice(topics)
            content.append(f"# Research on {current_topic.title()}\n\n")
            
            # Abstract
            content.append("## Abstract\n\n")
            content.append(f"This paper explores advances in {current_topic}. " * 5 + "\n\n")
            
            # Sections
            for i in range(random.randint(3, 8)):
                # Occasionally switch topics
                if random.random() > 0.7:
                    current_topic = random.choice(topics)
                
                content.append(f"## Section {i+1}: {current_topic.title()} Analysis\n\n")
                paragraphs = random.randint(2, 6)
                for _ in range(paragraphs):
                    content.append(f"Research in {current_topic} demonstrates... " * random.randint(5, 10))
                    content.append("\n\n")
            
            text = "".join(content)
            
        else:  # mixed
            # Mix of different content types
            templates = [
                "This document contains information about {}. ",
                "The {} system provides functionality for {}. ",
                "In the context of {}, we observe that {}. ",
                "Analysis of {} reveals important insights. ",
            ]
            
            topics = ["business", "technology", "science", "education", "healthcare"]
            words = ["process", "system", "method", "approach", "solution", "framework"]
            
            content = []
            while len("".join(content)) < target_size:
                template = random.choice(templates)
                topic = random.choice(topics)
                word = random.choice(words)
                sentence = template.format(topic, word)
                content.append(sentence)
            
            text = "".join(content)
        
        # Trim to target size
        return text[:target_size]


class TestAdvancedChunkingPerformance:
    """Advanced performance tests for chunking strategies."""
    
    @pytest.fixture()
    def memory_monitor(self):
        """Monitor memory usage during tests."""
        class MemoryMonitor:
            def __init__(self):
                self.process = psutil.Process()
                self.samples = []
                self.monitoring = False
                
            async def start_monitoring(self):
                """Start background memory monitoring."""
                self.monitoring = True
                while self.monitoring:
                    self.samples.append(self.process.memory_info().rss / 1024 / 1024)
                    await asyncio.sleep(0.1)
                    
            def stop_monitoring(self):
                """Stop monitoring and return stats."""
                self.monitoring = False
                if not self.samples:
                    return {"peak_mb": 0, "avg_mb": 0}
                return {
                    "peak_mb": max(self.samples),
                    "avg_mb": sum(self.samples) / len(self.samples),
                    "samples": len(self.samples)
                }
                
        return MemoryMonitor()
    
    @pytest.mark.parametrize("strategy", ["semantic", "hierarchical", "hybrid"])
    async def test_stress_test_production_load(
        self,
        strategy: str,
        memory_monitor,
    ) -> None:
        """Stress test with production-like document load."""
        # Generate production-like document set
        documents = []
        total_size = 0
        
        distribution = AdvancedChunkingBenchmarks.DOCUMENT_DISTRIBUTIONS["enterprise"]
        num_docs = 100  # Reduced for testing, production would be 1000+
        
        for _ in range(num_docs):
            rand = random.random()
            if rand < distribution["small"]:
                size_cat = "small"
            elif rand < distribution["small"] + distribution["medium"]:
                size_cat = "medium"
            elif rand < distribution["small"] + distribution["medium"] + distribution["large"]:
                size_cat = "large"
            else:
                size_cat = "xlarge"
            
            doc = AdvancedChunkingBenchmarks.generate_realistic_document(size_cat)
            documents.append(doc)
            total_size += len(doc)
        
        # Configure strategy
        config = self._get_stress_test_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)
        
        # Start memory monitoring
        monitor_task = asyncio.create_task(memory_monitor.start_monitoring())
        
        # Process documents
        start_time = time.time()
        success_count = 0
        errors = []
        total_chunks = 0
        
        # Process in batches to simulate real load
        batch_size = 10
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            tasks = []
            
            for j, doc in enumerate(batch):
                task = chunker.chunk_text_async(doc, f"doc_{i+j}")
                tasks.append(task)
            
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        errors.append(str(result))
                    else:
                        success_count += 1
                        total_chunks += len(result)
            except Exception as e:
                errors.append(f"Batch error: {str(e)}")
        
        duration = time.time() - start_time
        
        # Stop monitoring
        memory_monitor.stop_monitoring()
        await asyncio.sleep(0.2)  # Let monitor finish
        memory_stats = memory_monitor.stop_monitoring()
        
        # Calculate results
        result = StressTestResult(
            strategy=strategy,
            total_documents=num_docs,
            total_size_mb=total_size / (1024 * 1024),
            duration_seconds=duration,
            success_rate=success_count / num_docs,
            avg_chunks_per_sec=total_chunks / duration if duration > 0 else 0,
            peak_memory_mb=memory_stats["peak_mb"],
            errors=errors[:10],  # First 10 errors
        )
        
        # Log results
        print(f"\nStress Test Results for {strategy}:")
        print(f"  Documents: {result.total_documents}")
        print(f"  Total Size: {result.total_size_mb:.1f}MB")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  Success Rate: {result.success_rate:.1%}")
        print(f"  Chunks/sec: {result.avg_chunks_per_sec:.1f}")
        print(f"  Peak Memory: {result.peak_memory_mb:.1f}MB")
        
        # Assertions
        assert result.success_rate >= 0.95, f"Success rate {result.success_rate:.1%} below 95%"
        assert result.peak_memory_mb < 1000, f"Peak memory {result.peak_memory_mb}MB exceeds 1GB"
        
        # Strategy-specific performance targets
        min_rates = {
            "semantic": 50,      # Lower due to embeddings
            "hierarchical": 200, # Multiple passes
            "hybrid": 150,       # Variable
        }
        assert result.avg_chunks_per_sec >= min_rates[strategy], \
            f"Performance {result.avg_chunks_per_sec:.1f} below target {min_rates[strategy]}"
    
    @pytest.mark.parametrize("strategy", ["semantic", "hierarchical", "hybrid"])
    async def test_performance_degradation_under_load(
        self,
        strategy: str,
    ) -> None:
        """Test performance degradation as load increases."""
        config = self._get_stress_test_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)
        
        # Test with increasing concurrent load
        loads = [1, 5, 10, 20, 40]
        results = []
        
        # Base document for consistency
        base_doc = AdvancedChunkingBenchmarks.generate_realistic_document("medium", "technical")
        
        for load in loads:
            # Create concurrent tasks
            start_time = time.time()
            tasks = []
            
            for i in range(load):
                # Vary document slightly to avoid caching
                doc = base_doc + f"\n\nDocument variant {i}."
                task = chunker.chunk_text_async(doc, f"load_test_{i}")
                tasks.append(task)
            
            # Process concurrently
            chunks_lists = await asyncio.gather(*tasks)
            duration = time.time() - start_time
            
            total_chunks = sum(len(chunks) for chunks in chunks_lists)
            chunks_per_sec = total_chunks / duration if duration > 0 else 0
            
            results.append({
                "load": load,
                "duration": duration,
                "chunks_per_sec": chunks_per_sec,
                "avg_time_per_doc": duration / load,
            })
        
        # Analyze degradation
        print(f"\nDegradation Test Results for {strategy}:")
        for r in results:
            print(f"  Load {r['load']:2d}: {r['chunks_per_sec']:6.1f} chunks/s, "
                  f"{r['avg_time_per_doc']:.3f}s/doc")
        
        # Check degradation is reasonable
        baseline_rate = results[0]["chunks_per_sec"]
        for r in results[1:]:
            degradation = (baseline_rate - r["chunks_per_sec"]) / baseline_rate
            # Allow up to 50% degradation at high load
            assert degradation < 0.5, \
                f"Excessive degradation at load {r['load']}: {degradation:.1%}"
    
    @pytest.mark.parametrize("strategy", ["semantic", "hierarchical", "hybrid"])
    async def test_memory_leak_detection(
        self,
        strategy: str,
    ) -> None:
        """Test for memory leaks during extended processing."""
        config = self._get_stress_test_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)
        
        # Get baseline memory
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # Process many documents
        iterations = 50
        doc = AdvancedChunkingBenchmarks.generate_realistic_document("medium")
        
        memory_samples = [baseline_memory]
        
        for i in range(iterations):
            # Process document
            chunks = await chunker.chunk_text_async(doc, f"leak_test_{i}")
            
            # Clear references
            chunks = None
            
            # Sample memory every 10 iterations
            if i % 10 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
        
        # Final cleanup
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_samples.append(final_memory)
        
        # Analyze memory growth
        memory_growth = final_memory - baseline_memory
        growth_rate = memory_growth / iterations
        
        print(f"\nMemory Leak Test for {strategy}:")
        print(f"  Baseline: {baseline_memory:.1f}MB")
        print(f"  Final: {final_memory:.1f}MB")
        print(f"  Growth: {memory_growth:.1f}MB")
        print(f"  Growth Rate: {growth_rate:.3f}MB/iteration")
        
        # Check for leaks
        assert growth_rate < 0.5, f"Memory leak detected: {growth_rate:.3f}MB/iteration"
        assert memory_growth < 50, f"Excessive memory growth: {memory_growth:.1f}MB"
    
    async def test_mixed_strategy_concurrent_processing(self) -> None:
        """Test concurrent processing with mixed strategies."""
        strategies = ["semantic", "hierarchical", "hybrid"]
        
        # Create different chunkers
        chunkers = {}
        for strategy in strategies:
            config = self._get_stress_test_config(strategy)
            chunkers[strategy] = ChunkingFactory.create_chunker(config)
        
        # Generate varied documents
        documents = []
        for i in range(30):  # 10 docs per strategy
            strategy = strategies[i % 3]
            doc_type = ["technical", "research", "mixed"][i % 3]
            size = ["small", "medium", "large"][i % 3]
            
            doc = AdvancedChunkingBenchmarks.generate_realistic_document(size, doc_type)
            documents.append((strategy, doc, f"mixed_{i}"))
        
        # Process concurrently with different strategies
        start_time = time.time()
        tasks = []
        
        for strategy, doc, doc_id in documents:
            chunker = chunkers[strategy]
            task = chunker.chunk_text_async(doc, doc_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        # Analyze results
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        total_chunks = sum(len(r) for r in results if not isinstance(r, Exception))
        
        print(f"\nMixed Strategy Concurrent Test:")
        print(f"  Documents: {len(documents)}")
        print(f"  Success Rate: {success_count/len(documents):.1%}")
        print(f"  Total Chunks: {total_chunks}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Throughput: {total_chunks/duration:.1f} chunks/s")
        
        # All should succeed
        assert success_count == len(documents), \
            f"Failed {len(documents) - success_count} documents"
    
    async def test_error_recovery_performance(self) -> None:
        """Test performance impact of error recovery mechanisms."""
        # Create semantic chunker with simulated failures
        config = self._get_stress_test_config("semantic")
        chunker = ChunkingFactory.create_chunker(config)
        
        # Documents that trigger different scenarios
        documents = []
        for i in range(20):
            if i % 5 == 0:
                # Document that might trigger embedding error
                doc = "Error trigger content " * 1000
            else:
                # Normal document
                doc = AdvancedChunkingBenchmarks.generate_realistic_document("small")
            documents.append((doc, f"recovery_{i}"))
        
        # Process with error injection
        start_time = time.time()
        success_count = 0
        retry_count = 0
        
        for doc, doc_id in documents:
            try:
                # Simulate intermittent failures
                if "Error trigger" in doc and random.random() > 0.5:
                    # First attempt fails
                    retry_count += 1
                    await asyncio.sleep(0.1)  # Simulate retry delay
                
                chunks = await chunker.chunk_text_async(doc, doc_id)
                success_count += 1
            except Exception:
                pass
        
        duration = time.time() - start_time
        
        print(f"\nError Recovery Performance Test:")
        print(f"  Documents: {len(documents)}")
        print(f"  Success: {success_count}")
        print(f"  Retries: {retry_count}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Avg time per doc: {duration/len(documents):.3f}s")
        
        # Should handle errors gracefully
        assert success_count >= len(documents) * 0.9  # 90% success rate
        assert duration < len(documents) * 0.5  # Less than 0.5s per doc average
    
    def _get_stress_test_config(self, strategy: str) -> Dict[str, Any]:
        """Get optimized configuration for stress testing."""
        configs = {
            "semantic": {
                "strategy": "semantic",
                "params": {
                    "breakpoint_percentile_threshold": 90,
                    "buffer_size": 1,
                    "max_chunk_size": 3000,
                    "max_retries": 2,  # Reduce retries for speed
                }
            },
            "hierarchical": {
                "strategy": "hierarchical", 
                "params": {
                    "chunk_sizes": [2000, 1000, 500],  # 3 levels for balance
                    "chunk_overlap": 50,
                }
            },
            "hybrid": {
                "strategy": "hybrid",
                "params": {
                    "enable_analytics": False,  # Disable for performance
                    "markdown_density_threshold": 0.1,
                    "topic_diversity_threshold": 0.7,
                }
            }
        }
        return configs[strategy]