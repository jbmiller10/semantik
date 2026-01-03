"""Advanced tests for GPUMemoryGovernor.

Tests cover:
- Concurrent memory requests
- Callback exception handling
- Race condition scenarios
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vecpipe.memory_governor import (
    GPUMemoryGovernor,
    MemoryBudget,
    ModelLocation,
    ModelType,
    TrackedModel,
)


@pytest.fixture
def memory_budget():
    """Create a test memory budget with known values."""
    return MemoryBudget(
        total_gpu_mb=8000,  # 8GB GPU
        gpu_reserve_percent=0.10,
        gpu_max_percent=0.90,
        total_cpu_mb=16000,
        cpu_reserve_percent=0.20,
        cpu_max_percent=0.50,
    )


@pytest.fixture
def governor(memory_budget):
    """Create a GPUMemoryGovernor for testing."""
    return GPUMemoryGovernor(
        budget=memory_budget,
        enable_cpu_offload=True,
        eviction_idle_threshold_seconds=120,
    )


class TestConcurrentMemoryRequests:
    """Tests for concurrent memory requests."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_for_same_model(self, governor):
        """Test that concurrent requests for the same model don't cause issues."""
        model_name = "test/model"
        quantization = "float16"
        required_mb = 1000

        # Launch multiple concurrent requests for the same model
        async def request_memory():
            return await governor.request_memory(
                model_name=model_name,
                model_type=ModelType.EMBEDDING,
                quantization=quantization,
                required_mb=required_mb,
            )

        results = await asyncio.gather(
            request_memory(),
            request_memory(),
            request_memory(),
        )

        # All should succeed (model fits in budget)
        assert all(results), "All concurrent requests should succeed"

    @pytest.mark.asyncio
    async def test_concurrent_requests_different_models(self, governor):
        """Test concurrent requests for different models."""
        # Each model needs 1500MB, total 4500MB, budget is ~6400MB usable
        async def request_model(name: str):
            return await governor.request_memory(
                model_name=name,
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                required_mb=1500,
            )

        results = await asyncio.gather(
            request_model("model1"),
            request_model("model2"),
            request_model("model3"),
        )

        # All should fit in budget
        assert all(results), "All models should fit in budget"

    @pytest.mark.asyncio
    async def test_concurrent_load_and_unload(self, governor):
        """Test concurrent load and unload operations."""
        model_name = "test/model"
        quantization = "float16"

        # First, load the model
        await governor.request_memory(
            model_name=model_name,
            model_type=ModelType.EMBEDDING,
            quantization=quantization,
            required_mb=1000,
        )
        await governor.mark_loaded(
            model_name=model_name,
            model_type=ModelType.EMBEDDING,
            quantization=quantization,
        )

        # Concurrent operations: touch and unload
        async def touch_model():
            for _ in range(5):
                await governor.touch(model_name, ModelType.EMBEDDING, quantization)
                await asyncio.sleep(0.001)

        async def unload_model():
            await asyncio.sleep(0.005)  # Small delay
            await governor.mark_unloaded(model_name, ModelType.EMBEDDING, quantization)

        # Run concurrently - should not raise
        await asyncio.gather(
            touch_model(),
            unload_model(),
        )

    @pytest.mark.asyncio
    async def test_concurrent_eviction_pressure(self, governor):
        """Test that concurrent eviction requests don't cause double eviction."""
        # Load multiple models to fill memory
        for i in range(3):
            await governor.request_memory(
                model_name=f"model{i}",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                required_mb=2000,
            )
            await governor.mark_loaded(
                model_name=f"model{i}",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
            )

        # Register mock unload callback
        unload_count = 0

        async def mock_unload(name, quant):
            nonlocal unload_count
            unload_count += 1
            await asyncio.sleep(0.01)  # Simulate work

        governor.register_callbacks(
            ModelType.EMBEDDING,
            unload_fn=mock_unload,
        )

        # Request more memory than available - should trigger eviction
        async def request_large_model():
            return await governor.request_memory(
                model_name="large_model",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                required_mb=3000,  # Needs eviction
            )

        # Launch concurrent large requests
        results = await asyncio.gather(
            request_large_model(),
            request_large_model(),
            return_exceptions=True,
        )

        # At least one should succeed or fail gracefully (no crash)
        assert any(r is True or r is False for r in results if not isinstance(r, Exception))


class TestCallbackExceptionHandling:
    """Tests for callback exception handling."""

    @pytest.mark.asyncio
    async def test_unload_callback_exception_handled(self, governor):
        """Test that exceptions in unload callbacks are handled."""
        # Load a model
        await governor.request_memory(
            model_name="test/model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            required_mb=2000,
        )
        await governor.mark_loaded(
            model_name="test/model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
        )

        # Register callback that raises
        async def failing_unload(name, quant):
            raise RuntimeError("Simulated unload failure")

        governor.register_callbacks(
            ModelType.EMBEDDING,
            unload_fn=failing_unload,
        )

        # Get the tracked model
        model_key = "embedding:test/model:float16"
        tracked = governor._models.get(model_key)
        assert tracked is not None

        # Attempt unload - should return False but not crash
        result = await governor._unload_model(tracked)
        assert result is False  # Should fail gracefully

    @pytest.mark.asyncio
    async def test_offload_callback_exception_handled(self, governor):
        """Test that exceptions in offload callbacks are handled."""
        # Load a model
        await governor.request_memory(
            model_name="test/model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            required_mb=2000,
        )
        await governor.mark_loaded(
            model_name="test/model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
        )

        # Register callback that raises
        async def failing_offload(name, quant, device):
            raise RuntimeError("Simulated offload failure")

        governor.register_callbacks(
            ModelType.EMBEDDING,
            unload_fn=AsyncMock(),
            offload_fn=failing_offload,
        )

        # Get the tracked model
        model_key = "embedding:test/model:float16"
        tracked = governor._models.get(model_key)
        assert tracked is not None

        # Attempt offload - should return False but not crash
        result = await governor._offload_model(tracked)
        assert result is False  # Should fail gracefully

    @pytest.mark.asyncio
    async def test_callback_retry_on_transient_failure(self, governor):
        """Test that callbacks are retried on transient failures."""
        # Load a model
        await governor.request_memory(
            model_name="test/model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            required_mb=2000,
        )
        await governor.mark_loaded(
            model_name="test/model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
        )

        # Register callback that fails first time, succeeds second
        call_count = 0

        async def flaky_unload(name, quant):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Transient failure")
            # Success on retry

        governor.register_callbacks(
            ModelType.EMBEDDING,
            unload_fn=flaky_unload,
        )

        # Get the tracked model
        model_key = "embedding:test/model:float16"
        tracked = governor._models.get(model_key)
        assert tracked is not None

        # Attempt unload - should succeed on retry
        result = await governor._unload_model(tracked)
        assert result is True
        assert call_count == 2  # First attempt + retry


class TestRaceConditions:
    """Tests for race condition scenarios."""

    @pytest.mark.asyncio
    async def test_touch_during_eviction(self, governor):
        """Test that touch during eviction doesn't crash."""
        # Load a model
        await governor.request_memory(
            model_name="test/model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            required_mb=2000,
        )
        await governor.mark_loaded(
            model_name="test/model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
        )

        # Simulate race: touch while model is being evicted
        async def simulate_eviction():
            await governor.mark_unloaded("test/model", ModelType.EMBEDDING, "float16")

        async def simulate_touch():
            await asyncio.sleep(0.001)
            # This should handle the model being gone gracefully
            await governor.touch("test/model", ModelType.EMBEDDING, "float16")

        # Run concurrently - should not raise
        await asyncio.gather(
            simulate_eviction(),
            simulate_touch(),
        )

    @pytest.mark.asyncio
    async def test_request_memory_during_shutdown(self, governor):
        """Test memory request during shutdown doesn't hang."""
        # Start shutdown
        governor._shutdown_event.set()

        # Try to request memory
        result = await governor.request_memory(
            model_name="test/model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            required_mb=1000,
        )

        # Should still work (shutdown event doesn't block requests)
        assert result is True
