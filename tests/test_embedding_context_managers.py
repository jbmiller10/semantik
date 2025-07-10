"""Tests for embedding service context managers."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.embedding.base import BaseEmbeddingService
from shared.embedding.context import ManagedEmbeddingService, embedding_service_context, temporary_embedding_service


class MockEmbeddingService(BaseEmbeddingService):
    """Mock embedding service for testing."""

    def __init__(self, **kwargs):
        self._initialized = False
        self.cleanup_called = False
        self.initialize_called = False
        self.model_name = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self, model_name: str, **kwargs):
        self.initialize_called = True
        self.model_name = model_name
        self._initialized = True

    async def embed_texts(self, texts: list[str], batch_size: int = 32, **kwargs):
        return [[0.1] * 384 for _ in texts]

    async def embed_single(self, text: str, **kwargs):
        return [0.1] * 384

    def get_dimension(self) -> int:
        return 384

    def get_model_info(self) -> dict:
        return {"model_name": self.model_name, "dimension": 384, "device": "cpu", "max_sequence_length": 512}

    async def cleanup(self):
        self.cleanup_called = True
        self._initialized = False


class TestEmbeddingServiceContext:
    """Test embedding_service_context function."""

    @pytest.mark.asyncio
    async def test_context_manager_basic_usage(self):
        """Test basic context manager usage."""
        mock_service = MockEmbeddingService()

        with patch("shared.embedding.context.get_embedding_service", return_value=mock_service):
            async with embedding_service_context() as service:
                assert service is mock_service
                embeddings = await service.embed_texts(["test"])
                assert len(embeddings) == 1

            # Cleanup should be called after exiting context
            assert mock_service.cleanup_called

    @pytest.mark.asyncio
    async def test_context_manager_with_exception(self):
        """Test context manager cleans up even with exceptions."""
        mock_service = MockEmbeddingService()

        with patch("shared.embedding.context.get_embedding_service", return_value=mock_service):
            with pytest.raises(ValueError):
                async with embedding_service_context() as service:
                    assert service is mock_service
                    raise ValueError("Test error")

            # Cleanup should still be called
            assert mock_service.cleanup_called

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_error(self):
        """Test context manager handles cleanup errors gracefully."""
        mock_service = MockEmbeddingService()
        mock_service.cleanup = AsyncMock(side_effect=Exception("Cleanup failed"))

        with patch("shared.embedding.context.get_embedding_service", return_value=mock_service):
            # Should not raise even if cleanup fails
            async with embedding_service_context() as service:
                assert service is mock_service

            # Cleanup was attempted
            mock_service.cleanup.assert_called_once()


class TestTemporaryEmbeddingService:
    """Test temporary_embedding_service context manager."""

    @pytest.mark.asyncio
    async def test_temporary_service_creation(self):
        """Test creating a temporary service with specific model."""
        async with temporary_embedding_service("test-model", service_class=MockEmbeddingService) as service:
            assert service.initialize_called
            assert service.model_name == "test-model"
            assert service.is_initialized

        # Should be cleaned up after context
        assert service.cleanup_called

    @pytest.mark.asyncio
    async def test_temporary_service_with_kwargs(self):
        """Test passing kwargs to temporary service."""
        async with temporary_embedding_service(
            "test-model", service_class=MockEmbeddingService, device="cuda", quantization="int8"
        ) as service:
            assert service.model_name == "test-model"

    @pytest.mark.asyncio
    async def test_temporary_service_exception_handling(self):
        """Test temporary service handles exceptions properly."""
        with pytest.raises(RuntimeError):
            async with temporary_embedding_service("test-model", service_class=MockEmbeddingService) as service:
                raise RuntimeError("Test error")

        # Cleanup should still happen
        assert service.cleanup_called


class TestManagedEmbeddingService:
    """Test ManagedEmbeddingService class."""

    @pytest.mark.asyncio
    async def test_managed_service_async_context(self):
        """Test using ManagedEmbeddingService as async context manager."""
        mock_service = MockEmbeddingService()

        with patch("shared.embedding.context.get_embedding_service", return_value=mock_service):
            managed = ManagedEmbeddingService(mock_mode=True)

            async with managed as service:
                assert service is mock_service
                embeddings = await service.embed_texts(["test"])
                assert len(embeddings) == 1

            assert mock_service.cleanup_called

    def test_managed_service_sync_context_not_supported(self):
        """Test that sync context manager raises error."""
        managed = ManagedEmbeddingService()

        with pytest.raises(RuntimeError, match="Synchronous context manager not supported"):
            with managed:
                pass


class TestBaseEmbeddingServiceContextManager:
    """Test context manager implementation in BaseEmbeddingService."""

    @pytest.mark.asyncio
    async def test_base_service_context_manager(self):
        """Test using service directly as context manager."""
        service = MockEmbeddingService()
        service._initialized = True

        async with service as s:
            assert s is service
            embeddings = await s.embed_texts(["test"])
            assert len(embeddings) == 1

        assert service.cleanup_called

    @pytest.mark.asyncio
    async def test_base_service_context_with_exception(self):
        """Test base service context manager with exception."""
        service = MockEmbeddingService()
        service._initialized = True

        with pytest.raises(ValueError):
            async with service as s:
                raise ValueError("Test error")

        assert service.cleanup_called

    @pytest.mark.asyncio
    async def test_base_service_cleanup_error_suppressed(self):
        """Test that cleanup errors don't mask original exception."""
        service = MockEmbeddingService()
        service.cleanup = AsyncMock(side_effect=Exception("Cleanup failed"))

        # The ValueError should propagate, not the cleanup error
        with pytest.raises(ValueError, match="Original error"):
            async with service:
                raise ValueError("Original error")


class TestConcurrentContextManagers:
    """Test concurrent usage of context managers."""

    @pytest.mark.asyncio
    async def test_multiple_temporary_services(self):
        """Test creating multiple temporary services concurrently."""

        async def create_and_use_service(model_name: str):
            async with temporary_embedding_service(model_name, service_class=MockEmbeddingService) as service:
                embeddings = await service.embed_texts([f"test from {model_name}"])
                return model_name, len(embeddings)

        # Create multiple services concurrently
        tasks = [create_and_use_service(f"model-{i}") for i in range(5)]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 5
        for model_name, embed_count in results:
            assert embed_count == 1

    @pytest.mark.asyncio
    async def test_nested_context_managers(self):
        """Test nested context manager usage."""
        outer_service = MockEmbeddingService()
        inner_service = MockEmbeddingService()

        with patch("shared.embedding.context.get_embedding_service", return_value=outer_service):
            async with embedding_service_context() as outer:
                assert outer is outer_service

                async with temporary_embedding_service("inner-model", service_class=MockEmbeddingService) as inner:
                    # Both services should be active
                    assert outer is not inner
                    assert inner.model_name == "inner-model"

                # Inner should be cleaned up
                assert inner.cleanup_called
                assert not outer.cleanup_called

            # Now outer should be cleaned up
            assert outer.cleanup_called
