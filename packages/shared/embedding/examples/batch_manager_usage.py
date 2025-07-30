#!/usr/bin/env python3
"""
Example usage of AdaptiveBatchSizeManager.

This script demonstrates how to use the AdaptiveBatchSizeManager to dynamically
manage batch sizes for embedding operations based on available GPU memory.
"""

import logging
from shared.embedding import AdaptiveBatchSizeManager

# Set up logging to see informative messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main():
    """Demonstrate AdaptiveBatchSizeManager usage."""

    # Initialize the manager with default 20% safety margin
    manager = AdaptiveBatchSizeManager(default_safety_margin=0.2)

    print("=== AdaptiveBatchSizeManager Demo ===\n")

    # Example 1: Calculate initial batch size for different models
    print("1. Calculating initial batch sizes for different models:")
    models = [
        ("sentence-transformers/all-MiniLM-L6-v2", "float32"),
        ("BAAI/bge-large-en-v1.5", "float32"),
        ("Qwen/Qwen3-Embedding-8B", "float32"),
        ("Qwen/Qwen3-Embedding-8B", "int8"),
    ]

    for model, quantization in models:
        batch_size = manager.calculate_initial_batch_size(model_name=model, quantization=quantization, text_length=512)
        print(f"  {model} ({quantization}): batch_size = {batch_size}")

    print("\n2. Storing and retrieving batch sizes:")

    # Store some batch sizes
    manager.update_batch_size("BAAI/bge-large-en-v1.5", "float32", 32)
    manager.update_batch_size("BAAI/bge-large-en-v1.5", "float16", 64)

    # Retrieve them
    size_32 = manager.get_current_batch_size("BAAI/bge-large-en-v1.5", "float32")
    size_16 = manager.get_current_batch_size("BAAI/bge-large-en-v1.5", "float16")
    size_unknown = manager.get_current_batch_size("BAAI/bge-large-en-v1.5", "int8")

    print(f"  BAAI/bge-large-en-v1.5 (float32): {size_32}")
    print(f"  BAAI/bge-large-en-v1.5 (float16): {size_16}")
    print(f"  BAAI/bge-large-en-v1.5 (int8): {size_unknown}")

    print("\n3. Using get_or_calculate_batch_size (convenience method):")

    # First call calculates and stores
    batch1 = manager.get_or_calculate_batch_size("sentence-transformers/all-mpnet-base-v2", "float32")
    print(f"  First call (calculated): {batch1}")

    # Second call retrieves stored value
    batch2 = manager.get_or_calculate_batch_size("sentence-transformers/all-mpnet-base-v2", "float32")
    print(f"  Second call (retrieved): {batch2}")

    print("\n4. Viewing all stored batch sizes:")
    all_sizes = manager.get_all_batch_sizes()
    for key, size in all_sizes.items():
        model_quant = key.split(":")
        print(f"  {model_quant[0]} ({model_quant[1]}): {size}")

    print("\n5. Resetting and clearing batch sizes:")

    # Reset specific batch size
    manager.reset_batch_size("BAAI/bge-large-en-v1.5", "float32")
    print("  Reset BAAI/bge-large-en-v1.5 (float32)")

    # Clear all
    manager.clear_all()
    print("  Cleared all batch sizes")

    print("\n6. Example integration with embedding service:")
    print(
        """
    # In your embedding service:
    batch_manager = AdaptiveBatchSizeManager()
    
    def embed_documents(texts, model_name, quantization):
        # Get optimal batch size
        batch_size = batch_manager.get_or_calculate_batch_size(
            model_name, 
            quantization,
            text_length=sum(len(t) for t in texts) // len(texts)
        )
        
        # Process in batches
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = model.encode(batch)
                embeddings.extend(batch_embeddings)
            except torch.cuda.OutOfMemoryError:
                # Reduce batch size on OOM
                new_size = max(1, batch_size // 2)
                batch_manager.update_batch_size(model_name, quantization, new_size)
                # Retry with smaller batch...
    """
    )


if __name__ == "__main__":
    main()
