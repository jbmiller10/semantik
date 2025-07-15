#!/usr/bin/env python3
"""
Integration tests for the embedding service async/sync interaction
"""
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

# Mock metrics before importing
sys.modules["shared.metrics.prometheus"] = MagicMock()


class TestEmbeddingIntegration(unittest.TestCase):
    """Integration tests for embedding service"""

    @patch("torch.cuda.is_available")
    def test_async_sync_wrapper_interaction(self, mock_cuda):
        """Test that sync wrapper properly calls async implementation"""
        mock_cuda.return_value = False

        from shared.embedding import EmbeddingService
        from shared.embedding.dense import DenseEmbeddingService

        # Create service
        service = EmbeddingService()

        # Verify the internal service is the async one
        assert isinstance(service._service, DenseEmbeddingService)

        # The sync wrapper should create its own event loop
        assert service._loop is None

        # After a sync operation, it should have created a loop
        service.get_model_info("dummy", "float32")
        # Note: The loop is closed after each operation in the current implementation

    def test_concurrent_embedding_requests(self):
        """Test handling concurrent embedding requests"""
        # This tests thread safety of the sync wrapper
        from shared.embedding import get_embedding_service_sync

        def make_request(i):
            try:
                service = get_embedding_service_sync()
                # Just verify we can get the service
                return f"Request {i}: {service.device}"
            except Exception as e:
                return f"Request {i} failed: {e}"

        # Run multiple requests concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [f.result() for f in futures]

        # All requests should succeed
        for result in results:
            assert "failed" not in result
            # Just check that device is available (either cpu or cuda)
            assert "cpu" in result.lower() or "cuda" in result.lower()

    def test_performance_baseline(self):
        """Establish performance baseline for embedding generation"""
        from shared.embedding import EmbeddingService

        service = EmbeddingService(mock_mode=True)

        # Test texts
        texts = ["test text"] * 100

        # Measure time for mock embeddings
        start = time.time()
        embeddings = service.generate_embeddings(texts, "test-model", batch_size=32)
        duration = time.time() - start

        # Mock mode should be very fast
        assert duration < 1.0  # Should complete in under 1 second
        assert embeddings is not None
        assert len(embeddings) == 100

    @patch("torch.cuda.is_available")
    def test_async_service_lifecycle(self, mock_cuda):
        """Test async service lifecycle management"""
        mock_cuda.return_value = False

        import asyncio

        async def async_test():
            from shared.embedding import cleanup, get_embedding_service

            # Get service
            service1 = await get_embedding_service()
            service2 = await get_embedding_service()

            # Should be singleton
            assert service1 is service2

            # Cleanup
            await cleanup()

            # After cleanup, should get new instance
            service3 = await get_embedding_service()
            assert service3 is not service1

        # Run the async test
        asyncio.run(async_test())

    def test_backwards_compatibility(self):
        """Test that old API still works"""
        from shared.embedding import embedding_service, enhanced_embedding_service

        # These should exist for backwards compatibility
        assert embedding_service is not None
        assert enhanced_embedding_service is not None
        assert embedding_service is enhanced_embedding_service

        # Should have expected methods
        assert hasattr(embedding_service, "load_model")
        assert hasattr(embedding_service, "generate_embeddings")
        assert hasattr(embedding_service, "generate_single_embedding")


if __name__ == "__main__":
    # For async tests
    unittest.main()
