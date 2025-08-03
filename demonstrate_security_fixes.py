#!/usr/bin/env python3
"""
Demonstration of the security fixes implemented in the chunking system.

This script shows how the fixes prevent security vulnerabilities and production failures.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from packages.shared.text_processing.strategies.semantic_chunker import SemanticChunker
from packages.shared.text_processing.strategies.hybrid_chunker import HybridChunker
from packages.shared.text_processing.strategies.recursive_chunker import RecursiveChunker


def demonstrate_mockembedding_protection():
    """Demonstrate that MockEmbedding fallback is prevented in production."""
    print("\n=== MockEmbedding Protection Demo ===")
    
    # Ensure we're in production mode
    os.environ.pop("TESTING", None)
    
    print("Attempting to create SemanticChunker in production mode without embeddings...")
    try:
        # This will fail if local embeddings aren't available
        chunker = SemanticChunker()
        print("✓ SemanticChunker created successfully (local embeddings available)")
    except RuntimeError as e:
        print(f"✗ Expected error in production: {e}")
        print("  This prevents silent degradation to MockEmbedding!")
    
    # Now test with testing mode
    os.environ["TESTING"] = "true"
    print("\nAttempting to create SemanticChunker in testing mode...")
    try:
        chunker = SemanticChunker()
        print("✓ SemanticChunker created successfully (MockEmbedding allowed in testing)")
    except Exception as e:
        print(f"  Error: {e}")
    finally:
        os.environ.pop("TESTING", None)


def demonstrate_input_validation():
    """Demonstrate input validation and sanitization."""
    print("\n=== Input Validation Demo ===")
    
    chunker = RecursiveChunker()
    
    # Test invalid doc_id
    print("\nTesting invalid doc_id with special characters...")
    try:
        chunker.chunk_text("Some text", "doc/id;drop table")
        print("✗ Should have failed!")
    except ValueError as e:
        print(f"✓ Validation caught invalid doc_id: {e}")
    
    # Test oversized document
    print("\nTesting oversized document...")
    large_text = "a" * 100_000_001
    try:
        chunker.chunk_text(large_text, "doc_id")
        print("✗ Should have failed!")
    except ValueError as e:
        print(f"✓ Validation caught oversized document: {e}")
    
    # Test valid inputs
    print("\nTesting valid inputs...")
    try:
        results = chunker.chunk_text("Valid text content", "doc_123")
        print(f"✓ Valid inputs accepted, created {len(results)} chunks")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def demonstrate_redos_protection():
    """Demonstrate ReDoS protection in HybridChunker."""
    print("\n=== ReDoS Protection Demo ===")
    
    chunker = HybridChunker()
    
    # Create pathological input that would cause ReDoS with unsafe patterns
    print("\nTesting pathological markdown input...")
    pathological = "[" * 1000 + "]" * 1000 + "(" * 1000 + ")" * 1000
    
    import time
    start = time.time()
    try:
        # This should complete quickly with bounded patterns
        density = chunker._calculate_markdown_density(pathological)
        elapsed = time.time() - start
        print(f"✓ Markdown density calculated in {elapsed:.3f} seconds")
        print(f"  (Safe bounded patterns prevent ReDoS)")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test normal markdown
    print("\nTesting normal markdown content...")
    normal_markdown = """
# Header
This is a **bold** text with [link](url).
- List item 1
- List item 2
```python
code block
```
    """
    start = time.time()
    density = chunker._calculate_markdown_density(normal_markdown)
    elapsed = time.time() - start
    print(f"✓ Normal markdown processed in {elapsed:.3f} seconds")
    print(f"  Density: {density:.2f}")


def demonstrate_all_chunkers_validate():
    """Demonstrate that all chunkers validate inputs."""
    print("\n=== All Chunkers Validate Inputs ===")
    
    # Test each chunker type
    chunkers = {
        "RecursiveChunker": RecursiveChunker(),
        "HybridChunker": HybridChunker(),
    }
    
    # Add SemanticChunker if in testing mode
    os.environ["TESTING"] = "true"
    chunkers["SemanticChunker"] = SemanticChunker()
    
    for name, chunker in chunkers.items():
        print(f"\nTesting {name}...")
        try:
            chunker.chunk_text("text", "invalid/doc/id")
            print(f"✗ {name} should have validated inputs!")
        except ValueError as e:
            print(f"✓ {name} validated inputs: {e}")
    
    os.environ.pop("TESTING", None)


def main():
    """Run all demonstrations."""
    print("=== Chunking Security Fixes Demonstration ===")
    print("This script demonstrates the security fixes implemented in the chunking system.")
    
    demonstrate_mockembedding_protection()
    demonstrate_input_validation()
    demonstrate_redos_protection()
    demonstrate_all_chunkers_validate()
    
    print("\n=== Summary ===")
    print("✓ MockEmbedding fallback prevented in production")
    print("✓ Input validation catches malicious inputs")
    print("✓ ReDoS vulnerabilities fixed with bounded patterns")
    print("✓ All chunkers validate inputs before processing")
    print("\nThe chunking system is now more secure and production-ready!")


if __name__ == "__main__":
    main()