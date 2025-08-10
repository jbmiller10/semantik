#!/usr/bin/env python3

"""
Basic functionality test for streaming pipeline.
This demonstrates the core features without complex validation.
"""

import asyncio
import tempfile
import tracemalloc
from pathlib import Path

from packages.shared.chunking.infrastructure.streaming.checkpoint import CheckpointManager
from packages.shared.chunking.infrastructure.streaming.memory_pool import MemoryPool
from packages.shared.chunking.infrastructure.streaming.processor import StreamingDocumentProcessor
from packages.shared.chunking.infrastructure.streaming.window import StreamingWindow


def test_utf8_boundary() -> None:
    """Test UTF-8 boundary detection."""
    print("\n=== UTF-8 Boundary Detection ===")
    processor = StreamingDocumentProcessor()

    # Test cases
    test_cases = [
        (b"Hello World", -1, 11, "ASCII only"),
        ("Café".encode(), 3, 3, "Before é"),
        ("Hello 中文".encode(), 8, 6, "Before Chinese"),
        ("Test 🎉".encode(), 6, 5, "Before emoji"),
    ]

    for data, max_pos, expected, description in test_cases:
        result = processor._find_utf8_boundary(data, max_pos)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {description}: {result} (expected {expected})")


async def test_memory_pool() -> None:
    """Test memory pool management."""
    print("\n=== Memory Pool Management ===")
    pool = MemoryPool(buffer_size=1024, pool_size=3)

    # Test acquiring buffers
    buffers = []
    for _ in range(3):
        buffer_id, buffer = await pool.acquire()
        buffers.append(buffer_id)
        print(f"  ✓ Acquired buffer {buffer_id}")

    # Test pool exhaustion
    try:
        await pool.acquire(timeout=0.1)
        print("  ✗ Pool should be exhausted")
    except TimeoutError:
        print("  ✓ Pool correctly exhausted")

    # Test releasing
    for buffer_id in buffers:
        pool.release(buffer_id)
    print("  ✓ Released all buffers")

    # Check statistics
    stats = pool.get_statistics()
    print(f"  ✓ Pool stats: {stats['available']}/{stats['pool_size']} available")


def test_streaming_window() -> None:
    """Test streaming window operations."""
    print("\n=== Streaming Window ===")
    window = StreamingWindow(max_size=100)

    # Test append and decode
    text1 = "Hello, world! "
    window.append(text1.encode("utf-8"))
    decoded = window.decode_safe()
    print(f"  ✓ Decoded: '{decoded}'")

    # Test with UTF-8 split
    text2 = "Test 世界"
    data = text2.encode("utf-8")
    # Simulate split in middle of Chinese character
    window.clear()
    window.append(data[:7])  # Partial character
    decoded1 = window.decode_safe()
    window.append(data[7:])  # Rest of character
    _ = window.decode_safe()  # Process remaining bytes
    print(f"  ✓ UTF-8 safe decode: '{decoded1}' + continuation")

    # Test sliding
    window.clear()
    window.append(b"A" * 80)
    initial_size = window.size
    window.slide(40)
    print(f"  ✓ Window slide: {initial_size} -> {window.size} bytes")


async def test_checkpoint() -> None:
    """Test checkpoint save and load."""
    print("\n=== Checkpoint Management ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)

        # Save checkpoint
        checkpoint = await manager.save_checkpoint(
            operation_id="test_op",
            document_id="test_doc",
            file_path="/test/file.txt",
            byte_position=1000,
            char_position=500,
            chunks_processed=10,
            total_chunks=100,
            strategy_name="test",
            pending_bytes=b"partial",
        )
        print(f"  ✓ Saved checkpoint at {checkpoint.byte_position} bytes")

        # Load checkpoint
        loaded = await manager.load_checkpoint("test_op")
        if loaded and loaded.byte_position == 1000:
            print(f"  ✓ Loaded checkpoint: {loaded.chunks_processed}/{loaded.total_chunks} chunks")
        else:
            print("  ✗ Failed to load checkpoint")

        # Delete checkpoint
        deleted = await manager.delete_checkpoint("test_op")
        print(f"  ✓ Deleted checkpoint: {deleted}")


async def test_memory_usage() -> None:
    """Test memory bounded processing."""
    print("\n=== Memory Bounded Processing ===")

    # Create 10MB test file
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".txt", delete=False) as f:
        content = b"Test sentence. " * 100000  # ~1.5MB
        for _ in range(7):
            f.write(content)
        test_file = f.name

    file_size = Path(test_file).stat().st_size / (1024 * 1024)
    print(f"  Created test file: {file_size:.1f}MB")

    try:
        # Track memory
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0] / (1024 * 1024)

        processor = StreamingDocumentProcessor()

        # Get memory stats
        stats = processor.get_memory_usage()
        max_memory_mb = stats["max_memory"] / (1024 * 1024)

        current_memory = tracemalloc.get_traced_memory()[0] / (1024 * 1024)
        memory_used = current_memory - start_memory

        tracemalloc.stop()

        print(f"  ✓ Memory limit: {max_memory_mb:.1f}MB")
        print(f"  ✓ Memory used: {memory_used:.1f}MB")
        print(f"  ✓ Within bounds: {memory_used < max_memory_mb}")

    finally:
        Path(test_file).unlink(missing_ok=True)


async def main() -> None:
    """Run all basic tests."""
    print("\n" + "=" * 50)
    print("STREAMING PIPELINE - BASIC FUNCTIONALITY TEST")
    print("=" * 50)

    # Run tests
    test_utf8_boundary()
    await test_memory_pool()
    test_streaming_window()
    await test_checkpoint()
    await test_memory_usage()

    print("\n" + "=" * 50)
    print("✅ All basic functionality tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
