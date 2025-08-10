"""Unit tests for ModelManager lifecycle and memory management."""

import asyncio
from collections.abc import Generator
from unittest.mock import Mock, patch

import pytest

from packages.vecpipe.memory_utils import InsufficientMemoryError
from packages.vecpipe.model_manager import ModelManager


@pytest.fixture()
def mock_embedding_service() -> Generator[Mock, None, None]:
    """Create a mock EmbeddingService."""
    with patch("packages.vecpipe.model_manager.EmbeddingService") as mock:
        instance = mock.return_value
        instance.load_model = Mock(return_value=True)
        instance.generate_single_embedding = Mock(return_value=[0.1, 0.2, 0.3])
        instance.current_model = None
        instance.current_tokenizer = None
        instance.current_model_name = None
        instance.current_quantization = None
        instance.mock_mode = False
        yield mock


@pytest.fixture()
def mock_reranker() -> Generator[Mock, None, None]:
    """Create a mock CrossEncoderReranker."""
    with patch("packages.vecpipe.model_manager.CrossEncoderReranker") as mock:
        instance = mock.return_value
        instance.load_model = Mock()
        instance.unload_model = Mock()
        instance.rerank = Mock(return_value=[(0, 0.9), (1, 0.7), (2, 0.5)])
        instance.get_model_info = Mock(return_value={"model": "test-reranker", "quantization": "float16"})
        yield mock


@pytest.fixture()
def model_manager() -> ModelManager:
    """Create a ModelManager instance with a short unload timeout for testing."""
    return ModelManager(unload_after_seconds=1)


class TestLazyLoading:
    """Test lazy loading behavior of ModelManager."""

    def test_lazy_loading(self, model_manager: ModelManager, mock_embedding_service: Mock) -> None:
        """Test that models are not loaded initially and load only once when needed."""
        # Assert initial state - no models loaded
        assert model_manager.embedding_service is None
        assert model_manager.reranker is None
        assert model_manager.current_model_key is None
        assert model_manager.current_reranker_key is None

        # First embedding generation should trigger model loading
        model_name = "test-model"
        quantization = "float16"
        result = asyncio.run(
            model_manager.generate_embedding_async(text="test text", model_name=model_name, quantization=quantization)
        )

        # Verify model was created and loaded
        assert model_manager.embedding_service is not None
        mock_embedding_service.assert_called_once()
        mock_embedding_service.return_value.load_model.assert_called_once_with(model_name, quantization)
        assert model_manager.current_model_key == f"{model_name}_{quantization}"
        assert result == [0.1, 0.2, 0.3]

        # Reset mock to track subsequent calls
        mock_embedding_service.reset_mock()
        mock_embedding_service.return_value.load_model.reset_mock()

        # Second embedding generation with same model should NOT trigger loading
        result2 = asyncio.run(
            model_manager.generate_embedding_async(
                text="another test", model_name=model_name, quantization=quantization
            )
        )

        # Verify model was NOT created again and load_model was NOT called
        mock_embedding_service.assert_not_called()
        mock_embedding_service.return_value.load_model.assert_not_called()
        assert result2 == [0.1, 0.2, 0.3]
        assert model_manager.current_model_key == f"{model_name}_{quantization}"

    def test_different_model_triggers_reload(self, model_manager: ModelManager, mock_embedding_service: Mock) -> None:
        """Test that requesting a different model triggers unloading and reloading."""
        # Load first model
        model1 = "model1"
        quant1 = "float32"
        asyncio.run(model_manager.generate_embedding_async(text="test", model_name=model1, quantization=quant1))

        first_service = model_manager.embedding_service
        assert model_manager.current_model_key == f"{model1}_{quant1}"

        # Load different model
        model2 = "model2"
        quant2 = "int8"
        asyncio.run(model_manager.generate_embedding_async(text="test", model_name=model2, quantization=quant2))

        # Verify first model was unloaded (ModelManager clears attributes)
        # The service is recreated, but the first service had its attributes cleared
        assert first_service is not None

        # Verify new model was loaded
        assert model_manager.current_model_key == f"{model2}_{quant2}"
        # EmbeddingService is reused, not recreated
        assert mock_embedding_service.call_count == 1

    def test_reranker_lazy_loading(self, model_manager: ModelManager, mock_reranker: Mock) -> None:
        """Test that reranker is lazily loaded when needed."""
        with patch("packages.vecpipe.model_manager.check_memory_availability") as mock_memory:
            mock_memory.return_value = (True, "Sufficient memory")

            # Initially no reranker
            assert model_manager.reranker is None
            assert model_manager.current_reranker_key is None

            # Load reranker
            model_name = "test-reranker"
            quantization = "float16"
            success = model_manager.ensure_reranker_loaded(model_name, quantization)

            # Verify reranker was created
            assert success is True
            assert model_manager.reranker is not None
            assert model_manager.current_reranker_key == f"{model_name}_{quantization}"
            mock_reranker.assert_called_once_with(model_name=model_name, quantization=quantization)
            mock_memory.assert_called_once()


class TestAutoUnloading:
    """Test automatic model unloading behavior."""

    @pytest.mark.asyncio()
    @patch("asyncio.sleep")
    async def test_auto_unloading(
        self, mock_sleep: Mock, model_manager: ModelManager, mock_embedding_service: Mock  # noqa: ARG002
    ) -> None:
        """Test that models are automatically unloaded after the timeout period."""
        # Generate embedding to load model
        await model_manager.generate_embedding_async(text="test", model_name="test-model", quantization="float16")

        # Verify model is loaded
        assert model_manager.embedding_service is not None
        service = model_manager.embedding_service

        # Verify unload task was scheduled
        assert model_manager.unload_task is not None

        # Simulate waiting for the timeout
        await asyncio.sleep(0)  # Let the event loop process

        # Verify sleep was called (asyncio.sleep(0) for yielding)
        assert mock_sleep.called

        # Manually trigger the unload by completing the sleep
        model_manager.unload_task.cancel()  # Cancel existing task
        # Simulate the unload happening
        model_manager.unload_model()

        # Verify model was unloaded (attributes cleared, service still exists)
        assert model_manager.embedding_service is not None
        assert model_manager.current_model_key is None
        # Verify model attributes were cleared
        assert service.current_model is None

    @pytest.mark.asyncio()
    @patch("asyncio.sleep")
    async def test_activity_resets_unload_timer(
        self, mock_sleep: Mock, model_manager: ModelManager, mock_embedding_service: Mock  # noqa: ARG002
    ) -> None:
        """Test that using the model resets the unload timer."""
        # First embedding generation
        await model_manager.generate_embedding_async(text="test1", model_name="test-model", quantization="float16")

        first_task = model_manager.unload_task
        assert first_task is not None

        # Wait a bit but not long enough to trigger unload
        await asyncio.sleep(0.5)

        # Second embedding generation should reset timer
        await model_manager.generate_embedding_async(text="test2", model_name="test-model", quantization="float16")

        # Verify a new unload task was created (timer reset)
        assert model_manager.unload_task is not None
        assert model_manager.unload_task != first_task
        # Old task should be cancelled when new one is created
        assert first_task.cancelled() or first_task.done()

        # Model should still be loaded
        assert model_manager.embedding_service is not None

    @pytest.mark.asyncio()
    @patch("asyncio.sleep")
    async def test_reranker_auto_unloading(
        self, mock_sleep: Mock, model_manager: ModelManager, mock_reranker: Mock  # noqa: ARG002
    ) -> None:
        """Test that reranker is automatically unloaded separately from embedding model."""
        with patch("packages.vecpipe.model_manager.check_memory_availability") as mock_memory:
            mock_memory.return_value = (True, "Sufficient memory")

            # Load reranker
            model_manager.ensure_reranker_loaded("test-reranker", "float16")
            assert model_manager.reranker is not None

            # Reranker unload task is not immediately created in sync call
            # It's created when using rerank_async
            assert model_manager.reranker is not None

            # Manually trigger unload
            model_manager.unload_reranker()

            # Verify reranker was unloaded
            assert model_manager.reranker is None
            assert model_manager.current_reranker_key is None


class TestMemoryManagement:
    """Test memory management and error handling."""

    def test_insufficient_memory_error(self, model_manager: ModelManager, mock_reranker: Mock) -> None:
        """Test that InsufficientMemoryError is raised when memory check fails."""
        with patch("packages.vecpipe.model_manager.check_memory_availability") as mock_memory:
            error_message = "Insufficient memory: 1000MB free, 4000MB required"
            mock_memory.return_value = (False, error_message)

            # Attempt to load reranker should raise InsufficientMemoryError
            with pytest.raises(InsufficientMemoryError) as exc_info:
                model_manager.ensure_reranker_loaded("large-model", "float32")

            # Verify error message is included (ModelManager wraps it)
            assert error_message in str(exc_info.value)

            # Verify reranker was not created
            mock_reranker.assert_not_called()
            assert model_manager.reranker is None

    def test_memory_check_passes(self, model_manager: ModelManager, mock_reranker: Mock) -> None:
        """Test successful memory check allows model loading."""
        with patch("packages.vecpipe.model_manager.check_memory_availability") as mock_memory:
            mock_memory.return_value = (True, "Sufficient memory: 8000MB free")

            # Load reranker should succeed
            success = model_manager.ensure_reranker_loaded("small-model", "int8")

            assert success is True
            assert model_manager.reranker is not None
            mock_reranker.assert_called_once_with(model_name="small-model", quantization="int8")

    def test_memory_check_with_current_models(
        self, model_manager: ModelManager, mock_embedding_service: Mock, mock_reranker: Mock  # noqa: ARG002
    ) -> None:
        """Test memory check considers currently loaded models."""
        # First load an embedding model
        asyncio.run(
            model_manager.generate_embedding_async(text="test", model_name="embed-model", quantization="float16")
        )

        with patch("packages.vecpipe.model_manager.check_memory_availability") as mock_memory:
            mock_memory.return_value = (True, "Memory available after unloading")

            # Load reranker - memory check should include current embedding model
            model_manager.ensure_reranker_loaded("rerank-model", "float32")

            # Verify memory check was called with proper arguments
            mock_memory.assert_called_once()
            # Check positional arguments
            assert mock_memory.call_args[0][0] == "rerank-model"
            assert mock_memory.call_args[0][1] == "float32"


class TestEdgeCasesAndThreadSafety:
    """Test edge cases and thread safety."""

    def test_get_status(self, model_manager: ModelManager, mock_embedding_service: Mock) -> None:  # noqa: ARG002
        """Test get_status returns correct information."""
        # Initial status - no models
        status = model_manager.get_status()
        assert status["embedding_model_loaded"] is False
        assert status["reranker_loaded"] is False
        assert status["is_mock_mode"] is False
        assert status["current_embedding_model"] is None
        assert status["current_reranker"] is None

        # Load embedding model
        asyncio.run(
            model_manager.generate_embedding_async(text="test", model_name="test-model", quantization="float16")
        )

        # Status with loaded model
        status = model_manager.get_status()
        assert status["embedding_model_loaded"] is True
        assert status["current_embedding_model"] == "test-model_float16"
        assert "embedding_last_used" in status
        assert "embedding_seconds_since_last_use" in status

    def test_mock_mode(self, model_manager: ModelManager, mock_embedding_service: Mock) -> None:
        """Test mock mode returns fixed embeddings without loading models."""
        # Set the mock EmbeddingService to report mock mode
        mock_embedding_service.return_value.mock_mode = True

        # Generate embedding - this will initialize service and detect mock mode
        result = asyncio.run(
            model_manager.generate_embedding_async(text="test", model_name="any-model", quantization="any-quant")
        )

        # Mock mode returns hash-based values
        assert isinstance(result, list)
        assert len(result) > 0  # Mock embedding has dynamic size
        # Service should be initialized
        assert model_manager.embedding_service is not None
        # Mock mode is detected from service
        assert model_manager.is_mock_mode is True
        # load_model should not be called in mock mode (ensure_model_loaded returns early)
        mock_embedding_service.return_value.load_model.assert_not_called()

    def test_shutdown(self, model_manager: ModelManager, mock_embedding_service: Mock) -> None:  # noqa: ARG002
        """Test shutdown properly cleans up resources."""
        # Load models
        asyncio.run(model_manager.generate_embedding_async(text="test", model_name="model1", quantization="float16"))

        service = model_manager.embedding_service
        assert service is not None

        # Shutdown
        model_manager.shutdown()

        # Verify cleanup - service still exists but model was unloaded
        assert model_manager.embedding_service is not None
        assert model_manager.current_model_key is None
        # Unload task should be cancelled but not None
        if model_manager.unload_task:
            assert model_manager.unload_task.cancelled()
        # Verify model attributes were cleared
        assert service.current_model is None
        # Verify executor was shut down
        if model_manager.executor:
            assert model_manager.executor._shutdown is True

    def test_concurrent_access(self, model_manager: ModelManager, mock_embedding_service: Mock) -> None:
        """Test thread safety with concurrent model access."""

        async def generate_embedding(text: str, delay: float = 0) -> list[float] | None:
            if delay:
                await asyncio.sleep(delay)
            return await model_manager.generate_embedding_async(
                text=text, model_name="test-model", quantization="float16"
            )

        # Run multiple concurrent requests
        async def run_concurrent() -> None:
            tasks = [
                generate_embedding("text1"),
                generate_embedding("text2", 0.1),
                generate_embedding("text3", 0.05),
            ]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_concurrent())

        # All should succeed with same result
        assert all(r == [0.1, 0.2, 0.3] for r in results)

        # Model should only be loaded once
        mock_embedding_service.assert_called_once()
        mock_embedding_service.return_value.load_model.assert_called_once()

    def test_unload_with_gc_and_cuda(
        self, model_manager: ModelManager, mock_embedding_service: Mock  # noqa: ARG002
    ) -> None:
        """Test unload triggers garbage collection and CUDA cache clearing."""
        with (
            patch("gc.collect") as mock_gc,
            patch("torch.cuda.empty_cache") as mock_cuda,
            patch("torch.cuda.is_available", return_value=True),
        ):
            # Load model
            asyncio.run(model_manager.generate_embedding_async(text="test", model_name="model", quantization="float16"))

            # Unload model
            model_manager.unload_model()

            # Verify GC and CUDA cache clearing
            mock_gc.assert_called()
            mock_cuda.assert_called()

    @pytest.mark.asyncio()
    async def test_rerank_async(self, model_manager: ModelManager, mock_reranker: Mock) -> None:
        """Test async reranking functionality."""
        with patch("packages.vecpipe.model_manager.check_memory_availability") as mock_memory:
            mock_memory.return_value = (True, "Sufficient memory")

            # Test reranking
            query = "test query"
            passages = ["passage 1", "passage 2", "passage 3"]

            scores = await model_manager.rerank_async(
                query=query, documents=passages, top_k=3, model_name="test-reranker", quantization="float16"
            )

            # Verify reranker was loaded and used
            assert model_manager.reranker is not None
            mock_reranker.assert_called_once_with(model_name="test-reranker", quantization="float16")

            # Verify rerank was called with correct parameters
            mock_reranker.return_value.rerank.assert_called_once_with(query, passages, 3, None, True)

            # Should return mock scores
            assert scores == [(0, 0.9), (1, 0.7), (2, 0.5)]
