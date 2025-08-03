#!/usr/bin/env python3
"""Demonstrate that all 6 chunking strategies are working correctly."""

import asyncio
from typing import List
from packages.shared.text_processing.chunking_factory import ChunkingFactory
from packages.shared.text_processing.chunking_config import ChunkingConfig, ChunkingStrategyEnum


async def demonstrate_all_strategies():
    """Demonstrate all 6 chunking strategies with sample text."""
    
    # Sample text with different content types
    sample_text = """# Introduction to Python Programming

Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.

## Key Features

### Simple and Readable Syntax
Python's syntax is designed to be intuitive and mirrors natural language to some extent. This makes it an excellent choice for beginners.

### Versatile and Powerful
Despite its simplicity, Python is incredibly powerful and can be used for:
- Web development (Django, Flask)
- Data science (NumPy, Pandas, Scikit-learn)
- Machine learning (TensorFlow, PyTorch)
- Automation and scripting

## Getting Started

To get started with Python, you'll need to install it on your system. Python is available for Windows, macOS, and Linux.

```python
# Your first Python program
print("Hello, World!")

# Variables and data types
name = "Alice"
age = 25
height = 5.6

# Basic operations
result = age * 2
print(f"{name} will be {result} in {age} years")
```

## Conclusion

Python's combination of simplicity and power makes it one of the most popular programming languages in the world. Whether you're building web applications, analyzing data, or creating machine learning models, Python has the tools and libraries to help you succeed.
"""

    strategies = [
        {
            "name": "Character-based",
            "strategy": ChunkingStrategyEnum.CHARACTER,
            "params": {
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        },
        {
            "name": "Recursive",
            "strategy": ChunkingStrategyEnum.RECURSIVE,
            "params": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "separators": ["\n\n", "\n", ". ", " ", ""]
            }
        },
        {
            "name": "Markdown",
            "strategy": ChunkingStrategyEnum.MARKDOWN,
            "params": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "headers_to_split_on": ["#", "##", "###"]
            }
        },
        {
            "name": "Semantic",
            "strategy": ChunkingStrategyEnum.SEMANTIC,
            "params": {
                "breakpoint_percentile_threshold": 90,
                "buffer_size": 1,
                "max_chunk_size": 500
            }
        },
        {
            "name": "Hierarchical",
            "strategy": ChunkingStrategyEnum.HIERARCHICAL,
            "params": {
                "target_chunk_sizes": [1000, 500, 250],
                "overlap_sizes": [100, 50, 25]
            }
        },
        {
            "name": "Hybrid",
            "strategy": ChunkingStrategyEnum.HYBRID,
            "params": {
                "primary_strategy": "markdown",
                "fallback_strategy": "recursive",
                "use_adaptive_selection": True
            }
        }
    ]

    print("=" * 80)
    print("CHUNKING STRATEGIES DEMONSTRATION")
    print("=" * 80)
    print(f"\nSample text length: {len(sample_text)} characters\n")

    for strategy_config in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_config['name']} ({strategy_config['strategy'].value})")
        print(f"{'='*60}")
        
        try:
            # Create configuration
            config = ChunkingConfig(
                strategy=strategy_config["strategy"],
                chunk_size=strategy_config["params"].get("chunk_size", 500),
                chunk_overlap=strategy_config["params"].get("chunk_overlap", 50),
                chunking_params=strategy_config["params"]
            )
            
            # Create chunker
            chunker = ChunkingFactory.create_chunker(config)
            
            # Process text
            chunks = await chunker.chunk_text_async(sample_text, "demo")
            
            print(f"✓ Successfully created {len(chunks)} chunks")
            print(f"  Average chunk size: {sum(len(c.text) for c in chunks) / len(chunks):.0f} chars")
            
            # Show first chunk as example
            if chunks:
                first_chunk = chunks[0]
                preview = first_chunk.text[:150] + "..." if len(first_chunk.text) > 150 else first_chunk.text
                print(f"  First chunk preview: {preview}")
                print(f"  Metadata: {first_chunk.metadata}")
                
        except Exception as e:
            print(f"✗ Failed: {str(e)}")
            # For strategies that need embeddings, show fallback behavior
            if "embedding" in str(e).lower() or "semantic" in strategy_config['name'].lower():
                print("  Note: This strategy requires embedding models which are not available in test mode")

    print(f"\n{'='*80}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*80}\n")

    # Additional test: Verify factory can handle all strategies
    print("\nVerifying ChunkingFactory supports all strategies:")
    for strategy in ChunkingStrategyEnum:
        try:
            config = ChunkingConfig(strategy=strategy)
            chunker = ChunkingFactory.create_chunker(config)
            print(f"  ✓ {strategy.value}: {chunker.__class__.__name__}")
        except Exception as e:
            print(f"  ✗ {strategy.value}: {str(e)}")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_all_strategies())