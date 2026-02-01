#!/usr/bin/env python3

"""
Integration tests for GPU memory management with real GPU operations.

These tests are designed to run on systems with CUDA GPUs and will be skipped
if no GPU is available. They test real GPU memory allocation, monitoring,
and cleanup operations to ensure proper memory management in production.

The tests use real models when possible (with smaller models for faster testing)
and verify that:
1. GPU memory is properly tracked
2. Adaptive batch sizing works with real memory constraints
3. Memory is properly freed after operations
4. Concurrent operations are thread-safe
"""

import argparse
import asyncio
import gc
import os
import sys
import unittest

import torch

# Mock metrics module before importing
sys.modules.setdefault("shared.metrics.prometheus", unittest.mock.MagicMock())

# Skip all tests if CUDA is not available
GPU_AVAILABLE = torch.cuda.is_available()
SKIP_GPU_TESTS = not GPU_AVAILABLE or os.environ.get("SKIP_GPU_TESTS", "").lower() == "true"

if GPU_AVAILABLE:
    # Import actual modules only if GPU is available
    try:
        from shared.embedding import EmbeddingService
        from vecpipe.memory_utils import get_gpu_memory_info, get_model_memory_requirement
        from vecpipe.model_manager import ModelManager
    except ImportError as e:
        print(f"Warning: Could not import required modules: {e}")
        SKIP_GPU_TESTS = True


def get_memory_usage() -> tuple[float, float]:
    """Get current process GPU *reserved* memory usage in MB.

    We intentionally use per-process stats (`torch.cuda.memory_reserved`) instead
    of device-wide free/used (`torch.cuda.mem_get_info`) to avoid flakiness when
    other processes are using the GPU.
    """
    if not GPU_AVAILABLE:
        return 0.0, 0.0

    reserved_bytes = torch.cuda.memory_reserved()
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    return reserved_bytes / (1024 * 1024), total_bytes / (1024 * 1024)


def get_allocated_mb() -> float:
    """Get current process GPU *allocated* memory usage in MB."""
    if not GPU_AVAILABLE:
        return 0.0
    return torch.cuda.memory_allocated() / (1024 * 1024)


def force_gpu_cleanup() -> None:
    """Force GPU memory cleanup"""
    gc.collect()
    if GPU_AVAILABLE:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@unittest.skipIf(SKIP_GPU_TESTS, "GPU not available or GPU tests disabled")
class TestEmbeddingGPUMemory(unittest.TestCase):
    """Integration tests for GPU memory management with real GPU operations"""

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test class with GPU info"""
        if GPU_AVAILABLE:
            cls.gpu_name = torch.cuda.get_device_name(0)
            cls.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\nRunning GPU tests on: {cls.gpu_name} ({cls.gpu_memory_gb:.1f} GB)")

    def setUp(self) -> None:
        """Set up each test"""
        # Force cleanup before each test
        force_gpu_cleanup()
        self.initial_memory_used, self.total_memory = get_memory_usage()
        self.initial_allocated = get_allocated_mb()
        print(
            f"\nInitial GPU reserved: {self.initial_memory_used:.1f}/{self.total_memory:.1f} MB "
            f"(allocated: {self.initial_allocated:.1f} MB)"
        )

    def tearDown(self) -> None:
        """Clean up after each test"""
        # Force cleanup after each test
        force_gpu_cleanup()
        final_memory_used, _ = get_memory_usage()
        final_allocated = get_allocated_mb()
        memory_leaked = final_memory_used - self.initial_memory_used
        allocated_leaked = final_allocated - self.initial_allocated
        print(f"Reserved leaked: {memory_leaked:.1f} MB (allocated leaked: {allocated_leaked:.1f} MB)")

        # Warn if significant memory leak detected (>100MB)
        if memory_leaked > 100:
            print(f"WARNING: Significant memory leak detected: {memory_leaked:.1f} MB")

    def test_memory_monitoring(self) -> None:
        """Test real GPU memory tracking during embedding operations"""
        print("\n=== Testing GPU Memory Monitoring ===")

        # Create embedding service
        service = EmbeddingService(mock_mode=False)

        # Use a small model for testing
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Record memory before loading model
        reserved_before_load, _ = get_memory_usage()
        allocated_before_load = get_allocated_mb()
        print(f"Memory before model load: {reserved_before_load:.1f} MB (allocated: {allocated_before_load:.1f} MB)")

        # Load model
        success = service.load_model(model_name, "float32")
        assert success, "Failed to load model"

        # Record memory after loading model
        reserved_after_load, _ = get_memory_usage()
        allocated_after_load = get_allocated_mb()
        model_reserved = reserved_after_load - reserved_before_load
        model_allocated = allocated_after_load - allocated_before_load
        print(
            f"Memory after model load: {reserved_after_load:.1f} MB "
            f"(allocated: {allocated_after_load:.1f} MB; model reserved {model_reserved:.1f} MB, "
            f"model allocated {model_allocated:.1f} MB)"
        )

        # Model should use some memory
        assert model_allocated > 10, "Model should allocate at least 10MB of GPU memory"

        # Generate embeddings with different batch sizes
        test_texts = ["This is a test sentence for GPU memory monitoring."] * 100

        for batch_size in [8, 16, 32]:
            memory_before_embed = get_allocated_mb()

            embeddings = service.generate_embeddings(test_texts, model_name=model_name, batch_size=batch_size)

            memory_after_embed = get_allocated_mb()
            embedding_memory = memory_after_embed - memory_before_embed

            print(f"Batch size {batch_size}: Embedding memory spike: {embedding_memory:.1f} MB")

            assert embeddings is not None
            assert len(embeddings) == len(test_texts)

            # Clean up embeddings to free memory
            del embeddings
            torch.cuda.empty_cache()

        # Unload model
        service.unload_model()
        force_gpu_cleanup()

        # Memory should be mostly freed
        reserved_after_unload, _ = get_memory_usage()
        allocated_after_unload = get_allocated_mb()
        reserved_freed = reserved_after_load - reserved_after_unload
        allocated_freed = allocated_after_load - allocated_after_unload
        print(
            f"Memory after unload: {reserved_after_unload:.1f} MB (allocated: {allocated_after_unload:.1f} MB; "
            f"freed reserved {reserved_freed:.1f} MB, freed allocated {allocated_freed:.1f} MB)"
        )

        # Reserved memory includes caching; allow more retention but still expect most of the model footprint to drop.
        assert reserved_freed > model_reserved * 0.6, (
            f"Should free at least 60% of reserved model memory, but only freed {reserved_freed:.1f}/"
            f"{model_reserved:.1f} MB"
        )

        # Allocated memory reflects this process' live tensors; this should be mostly freed after unload.
        assert allocated_freed > model_allocated * 0.9, (
            f"Should free at least 90% of allocated model memory, but only freed {allocated_freed:.1f}/"
            f"{model_allocated:.1f} MB"
        )

    def test_adaptive_sizing_with_models(self) -> None:
        """Test adaptive batch sizing with different real models if GPU available"""
        print("\n=== Testing Adaptive Batch Sizing ===")

        # Skip if less than 4GB GPU memory
        if self.total_memory < 4000:
            self.skipTest(f"GPU has only {self.total_memory:.1f} MB memory, need at least 4GB for this test")

        # Use the synchronous wrapper, but tweak the underlying async service.
        service = EmbeddingService(mock_mode=False)
        service._service.enable_adaptive_batch_size = True  # type: ignore[attr-defined]
        service._service.min_batch_size = 1  # type: ignore[attr-defined]

        # Test with progressively larger models
        test_models = [
            ("sentence-transformers/all-MiniLM-L6-v2", 384),  # ~90MB model
            ("sentence-transformers/all-mpnet-base-v2", 768),  # ~420MB model
        ]

        for model_name, expected_dim in test_models:
            print(f"\nTesting model: {model_name}")

            # Load model
            success = service.load_model(model_name, "float32")
            if not success:
                print(f"Skipping {model_name} - failed to load")
                continue

            # Test with large batch that might cause OOM
            large_text = "This is a long test sentence " * 50  # ~250 tokens
            test_texts = [large_text] * 200

            # Record initial batch size
            initial_batch_size = 64
            service.current_batch_size = initial_batch_size
            service.original_batch_size = initial_batch_size

            # Generate embeddings - this might trigger batch size reduction
            try:
                embeddings = service.generate_embeddings(test_texts, model_name=model_name, batch_size=initial_batch_size)

                assert embeddings is not None
                assert len(embeddings) == len(test_texts)
                assert len(embeddings[0]) == expected_dim

                # Check if batch size was adapted
                final_batch_size = service._service.current_batch_size or initial_batch_size  # type: ignore[attr-defined]
                if final_batch_size < initial_batch_size:
                    print(f"Batch size adapted: {initial_batch_size} -> {final_batch_size}")
                else:
                    print(f"Batch size unchanged: {final_batch_size}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Model {model_name} caused OOM even with adaptive sizing")
                else:
                    raise

            # Clean up
            service.unload_model()
            force_gpu_cleanup()

    def test_stress_with_varying_lengths(self) -> None:
        """Test GPU memory handling with texts of varying lengths"""
        print("\n=== Testing Variable Length Text Processing ===")

        service = EmbeddingService(mock_mode=False)
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Load model
        success = service.load_model(model_name, "float32")
        assert success

        # Create texts of varying lengths
        base_sentence = "The quick brown fox jumps over the lazy dog. "
        test_texts = []
        for i in range(1, 21):  # 1 to 20 repetitions
            test_texts.extend([base_sentence * i] * 5)  # 5 texts of each length

        print(
            f"Processing {len(test_texts)} texts with lengths from {len(test_texts[0])} to {len(test_texts[-1])} chars"
        )

        # Process in batches and monitor memory
        batch_size = 16
        max_memory_spike = 0

        for i in range(0, len(test_texts), batch_size):
            batch = test_texts[i : i + batch_size]
            memory_before = get_allocated_mb()

            embeddings = service.generate_embeddings(batch, model_name=model_name, batch_size=batch_size)

            memory_after = get_allocated_mb()
            memory_spike = memory_after - memory_before
            max_memory_spike = max(max_memory_spike, memory_spike)

            assert len(embeddings) == len(batch)

            # Clean up batch embeddings
            del embeddings

        print(f"Maximum memory spike during processing: {max_memory_spike:.1f} MB")

        # Clean up
        service.unload_model()
        force_gpu_cleanup()

    def test_memory_cleanup(self) -> None:
        """Verify memory is properly freed after embeddings"""
        print("\n=== Testing Memory Cleanup ===")

        service = EmbeddingService(mock_mode=False)
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Load model
        service.load_model(model_name, "float32")
        reserved_with_model, _ = get_memory_usage()
        allocated_with_model = get_allocated_mb()

        # Generate many embeddings
        test_texts = ["Test sentence for memory cleanup verification."] * 1000

        # Process and immediately delete embeddings
        for i in range(5):
            print(f"\nIteration {i+1}/5")
            memory_before = get_allocated_mb()

            embeddings = service.generate_embeddings(test_texts, model_name=model_name, batch_size=32)

            memory_with_embeddings = get_allocated_mb()
            embeddings_memory = memory_with_embeddings - memory_before
            print(f"Embeddings use {embeddings_memory:.1f} MB")

            # Delete embeddings
            del embeddings
            force_gpu_cleanup()

            memory_after_cleanup = get_allocated_mb()
            reserved_after_cleanup, _ = get_memory_usage()
            print(
                f"After cleanup: allocated={memory_after_cleanup:.1f} MB, reserved={reserved_after_cleanup:.1f} MB"
            )

            # `sentence-transformers` + PyTorch CUDA allocator may keep some memory
            # reserved/allocated for reuse. Assert we don't grow unbounded.
            assert memory_after_cleanup <= allocated_with_model + 100.0
            assert reserved_after_cleanup <= reserved_with_model + 200.0

        # Clean up
        service.unload_model()
        force_gpu_cleanup()

    def test_concurrent_gpu_operations(self) -> None:
        """Test thread safety with real GPU operations"""
        print("\n=== Testing Concurrent GPU Operations ===")

        # Use ModelManager for thread-safe operations
        manager = ModelManager(unload_after_seconds=300)
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Test texts
        test_texts = [
            "Thread safety test sentence one.",
            "Thread safety test sentence two.",
            "Thread safety test sentence three.",
            "Thread safety test sentence four.",
            "Thread safety test sentence five.",
        ]

        async def run_concurrent() -> list[list[float] | None]:
            return await asyncio.gather(
                *[manager.generate_embedding_async(text, model_name, "float32") for text in test_texts]
            )

        print(f"Running {len(test_texts)} concurrent embedding operations (async gather)...")
        allocated_before = get_allocated_mb()
        embeddings = asyncio.run(run_concurrent())
        allocated_after = get_allocated_mb()

        assert all(e is not None for e in embeddings)
        embedding_sizes = [len(e) for e in embeddings if e is not None]
        assert all(size == embedding_sizes[0] for size in embedding_sizes), "All embeddings should have same dimension"
        print(f"Allocated delta: {allocated_after - allocated_before:.1f} MB")

        asyncio.run(manager.unload_model_async())
        manager.executor.shutdown(wait=True)
        force_gpu_cleanup()


@unittest.skipIf(SKIP_GPU_TESTS, "GPU not available or GPU tests disabled")
class TestGPUMemoryUtils(unittest.TestCase):
    """Test GPU memory utility functions with real GPU"""

    def test_get_gpu_memory_info(self) -> None:
        """Test GPU memory info retrieval"""
        free_mb, total_mb = get_gpu_memory_info()

        print(f"\nGPU Memory Info: {free_mb} MB free / {total_mb} MB total")

        assert total_mb > 0, "Total GPU memory should be greater than 0"
        assert free_mb > 0, "Free GPU memory should be greater than 0"
        assert free_mb <= total_mb, "Free memory should not exceed total memory"

    def test_model_memory_estimation(self) -> None:
        """Test model memory requirement estimation"""
        test_cases = [
            ("sentence-transformers/all-MiniLM-L6-v2", "float32", 200),  # ~90MB model + overhead
            ("sentence-transformers/all-mpnet-base-v2", "float32", 600),  # ~420MB model + overhead
            ("Qwen/Qwen3-Embedding-0.6B", "float16", 1200),  # From memory_utils.py
            ("Qwen/Qwen3-Embedding-4B", "float32", 16000),  # From memory_utils.py
        ]

        for model_name, quantization, expected_min_mb in test_cases:
            estimated_mb = get_model_memory_requirement(model_name, quantization)
            print(f"\n{model_name} ({quantization}): {estimated_mb} MB estimated")

            # For known models, should match or exceed expected minimum
            if "Qwen" in model_name:
                assert (
                    estimated_mb >= expected_min_mb
                ), f"Estimation for {model_name} should be at least {expected_min_mb} MB"


if __name__ == "__main__":
    # Add command line option to skip GPU tests

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU tests")
    args, remaining = parser.parse_known_args()

    if args.skip_gpu:
        os.environ["SKIP_GPU_TESTS"] = "true"

    # Remove parsed args and run unittest with remaining args
    sys.argv = [sys.argv[0]] + remaining
    unittest.main(verbosity=2)
