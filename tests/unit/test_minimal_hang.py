#!/usr/bin/env python3
"""Minimal test to isolate the hanging issue."""

def test_hierarchical_chunker_malicious():
    """Test the malicious input handling."""
    from packages.shared.text_processing.strategies.hierarchical_chunker import HierarchicalChunker
    
    chunker = HierarchicalChunker()
    
    # Test null bytes
    text_with_null = "Normal text\x00with null bytes"
    chunks = chunker.chunk_text(text_with_null, "test_doc")
    assert len(chunks) > 0
    
    # Test very long lines
    long_line = "a" * 10000
    chunks = chunker.chunk_text(long_line, "test_doc")
    assert len(chunks) > 0
    
    # Test deeply nested structure
    nested_text = "Start\n" + "\n".join(["  " * i + f"Level {i}" for i in range(50)])
    chunks = chunker.chunk_text(nested_text, "test_doc")
    assert len(chunks) > 0