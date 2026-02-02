"""Unit tests for GPUMemoryGovernor.

Tests cover:
- Budget calculations
- LRU eviction ordering
- Pressure level responses
- Callback execution
- Model tracking lifecycle
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from vecpipe.memory_governor import (
    EvictionRecord,
    GPUMemoryGovernor,
    MemoryBudget,
    ModelLocation,
    ModelType,
    PressureLevel,
    TrackedModel,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def memory_budget() -> MemoryBudget:
    """Create a test memory budget with known values."""
    return MemoryBudget(
        total_gpu_mb=16000,  # 16GB GPU
        gpu_max_percent=0.90,
        total_cpu_mb=32000,  # 32GB CPU
        cpu_max_percent=0.50,
    )


@pytest.fixture()
def small_memory_budget() -> MemoryBudget:
    """Create a constrained memory budget for small GPU tests."""
    return MemoryBudget(
        total_gpu_mb=8000,  # 8GB GPU
        gpu_max_percent=0.90,
        total_cpu_mb=16000,  # 16GB CPU
        cpu_max_percent=0.50,
    )


@pytest_asyncio.fixture
async def governor(memory_budget: MemoryBudget) -> AsyncGenerator[GPUMemoryGovernor, None]:
    """Create a GPUMemoryGovernor instance for testing."""
    gov = GPUMemoryGovernor(
        budget=memory_budget,
        enable_cpu_offload=True,
        eviction_idle_threshold_seconds=120,
        pressure_check_interval_seconds=15,
    )
    yield gov
    await gov.shutdown()


@pytest_asyncio.fixture
async def governor_no_offload(memory_budget: MemoryBudget) -> AsyncGenerator[GPUMemoryGovernor, None]:
    """Create a GPUMemoryGovernor with CPU offloading disabled."""
    gov = GPUMemoryGovernor(
        budget=memory_budget,
        enable_cpu_offload=False,
    )
    yield gov
    await gov.shutdown()


# =============================================================================
# MemoryBudget Tests
# =============================================================================


class TestMemoryBudget:
    """Tests for MemoryBudget calculations."""

    def test_usable_gpu_mb_calculation(self, memory_budget: MemoryBudget) -> None:
        """Test GPU usable memory calculation."""
        # With 16000MB total and 90% max
        # usable = 16000 * 0.90 = 14400
        expected = int(16000 * 0.90)  # 14400
        assert memory_budget.usable_gpu_mb == expected

    def test_usable_cpu_mb_calculation(self, memory_budget: MemoryBudget) -> None:
        """Test CPU usable memory calculation."""
        # With 32000MB total and 50% max
        # usable = 32000 * 0.50 = 16000
        expected = int(32000 * 0.50)  # 16000
        assert memory_budget.usable_cpu_mb == expected

    def test_auto_detect_cpu_memory(self) -> None:
        """Test that CPU memory is auto-detected via factory function."""
        from vecpipe.memory_governor import create_memory_budget

        budget = create_memory_budget(total_gpu_mb=16000)
        # Should auto-detect from system
        assert budget.total_cpu_mb > 0


# =============================================================================
# TrackedModel Tests
# =============================================================================


class TestTrackedModel:
    """Tests for TrackedModel data class."""

    def test_model_key_generation(self) -> None:
        """Test unique model key is generated correctly."""
        model = TrackedModel(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.GPU,
            memory_mb=2000,
        )
        expected_key = "embedding:Qwen/Qwen3-Embedding-0.6B:float16"
        assert model.model_key == expected_key

    def test_idle_seconds_calculation(self) -> None:
        """Test idle time is calculated correctly."""
        past_time = time.time() - 60  # 60 seconds ago
        model = TrackedModel(
            model_name="test-model",
            model_type=ModelType.RERANKER,
            quantization="int8",
            location=ModelLocation.GPU,
            memory_mb=1000,
            last_used=past_time,
        )
        # Should be approximately 60 seconds
        assert 59 <= model.idle_seconds <= 61


# =============================================================================
# GPUMemoryGovernor Core Tests
# =============================================================================


class TestGPUMemoryGovernorCore:
    """Tests for core GPUMemoryGovernor functionality."""

    @pytest.mark.asyncio()
    async def test_request_memory_fits_in_budget(self, governor: GPUMemoryGovernor) -> None:
        """Test memory request that fits within budget."""
        result = await governor.request_memory(
            model_name="test-embedding",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            required_mb=2000,
        )
        assert result is True

    @pytest.mark.asyncio()
    async def test_mark_loaded_registers_model(self, governor: GPUMemoryGovernor) -> None:
        """Test that mark_loaded properly registers a model."""
        with patch.object(governor, "_get_model_memory", return_value=2000):
            await governor.mark_loaded(
                model_name="test-embedding",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                model_ref=MagicMock(),
            )

        models = governor.get_loaded_models()
        assert len(models) == 1
        assert models[0]["model_name"] == "test-embedding"
        assert models[0]["location"] == "gpu"

    @pytest.mark.asyncio()
    async def test_mark_unloaded_removes_model(self, governor: GPUMemoryGovernor) -> None:
        """Test that mark_unloaded removes a model from tracking."""
        with patch.object(governor, "_get_model_memory", return_value=2000):
            await governor.mark_loaded(
                model_name="test-embedding",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
            )
            assert len(governor.get_loaded_models()) == 1

            await governor.mark_unloaded(
                model_name="test-embedding",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
            )

        assert len(governor.get_loaded_models()) == 0

    @pytest.mark.asyncio()
    async def test_touch_updates_last_used(self, governor: GPUMemoryGovernor) -> None:
        """Test that touch updates the last_used timestamp."""
        with patch.object(governor, "_get_model_memory", return_value=2000):
            await governor.mark_loaded(
                model_name="test-embedding",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
            )

        # Get initial last_used
        model_key = governor._make_key("test-embedding", ModelType.EMBEDDING, "float16")
        initial_last_used = governor._models[model_key].last_used

        await asyncio.sleep(0.01)  # Small delay

        await governor.touch(
            model_name="test-embedding",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
        )

        new_last_used = governor._models[model_key].last_used
        assert new_last_used > initial_last_used

    @pytest.mark.asyncio()
    async def test_request_memory_for_already_loaded_model(self, governor: GPUMemoryGovernor) -> None:
        """Test requesting memory for an already loaded model returns True."""
        with patch.object(governor, "_get_model_memory", return_value=2000):
            await governor.mark_loaded(
                model_name="test-embedding",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
            )

        result = await governor.request_memory(
            model_name="test-embedding",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            required_mb=2000,
        )
        assert result is True

    @pytest.mark.asyncio()
    async def test_request_memory_enforces_reserve_for_already_loaded_model(self, governor: GPUMemoryGovernor) -> None:
        """Already-loaded models should still trigger eviction to restore reserve headroom."""
        # Arrange: two large models resident, leaving less than reserve free
        with patch.object(governor, "_get_model_memory", return_value=7000):
            await governor.mark_loaded("embed", ModelType.EMBEDDING, "float16")
            await governor.mark_loaded("rerank", ModelType.RERANKER, "float16")

        # Register callbacks so eviction can proceed
        embed_offload = AsyncMock()
        embed_unload = AsyncMock()
        governor.register_callbacks(ModelType.EMBEDDING, unload_fn=embed_unload, offload_fn=embed_offload)
        governor.register_callbacks(ModelType.RERANKER, unload_fn=AsyncMock(), offload_fn=AsyncMock())

        # Make free-memory calculation deterministic (independent of CUDA availability)
        def _free_mb(_mode=None) -> int:
            return governor._budget.usable_gpu_mb - governor._get_gpu_usage()

        with patch.object(governor, "_probe_free_gpu_mb", side_effect=_free_mb):
            result = await governor.request_memory("rerank", ModelType.RERANKER, "float16", required_mb=7000)

        assert result is True
        embed_offload.assert_awaited_once_with("embed", "float16", "cpu")
        embed_unload.assert_not_awaited()

        embed_key = governor._make_key("embed", ModelType.EMBEDDING, "float16")
        rerank_key = governor._make_key("rerank", ModelType.RERANKER, "float16")
        assert governor._models[embed_key].location == ModelLocation.CPU
        assert governor._models[rerank_key].location == ModelLocation.GPU

    @pytest.mark.asyncio()
    async def test_request_memory_reserve_enforcement_without_callbacks_returns_false(
        self, governor: GPUMemoryGovernor
    ) -> None:
        """Reserve enforcement should fail gracefully when eviction callbacks are missing."""
        with patch.object(governor, "_get_model_memory", return_value=7000):
            await governor.mark_loaded("embed", ModelType.EMBEDDING, "float16")
            await governor.mark_loaded("rerank", ModelType.RERANKER, "float16")

        def _free_mb(_mode=None) -> int:
            return governor._budget.usable_gpu_mb - governor._get_gpu_usage()

        with patch.object(governor, "_probe_free_gpu_mb", side_effect=_free_mb):
            result = await governor.request_memory("rerank", ModelType.RERANKER, "float16", required_mb=7000)

        assert result is False


# =============================================================================
# LRU Eviction Tests
# =============================================================================


class TestLRUEviction:
    """Tests for LRU-based eviction behavior."""

    @pytest.mark.asyncio()
    async def test_eviction_candidates_sorted_by_last_used(self, governor: GPUMemoryGovernor) -> None:
        """Test that eviction candidates are sorted oldest first (LRU)."""
        with patch.object(governor, "_get_model_memory", return_value=1000):
            # Load models with different last_used times
            await governor.mark_loaded("model-a", ModelType.EMBEDDING, "float16")
            await asyncio.sleep(0.01)
            await governor.mark_loaded("model-b", ModelType.EMBEDDING, "float16")
            await asyncio.sleep(0.01)
            await governor.mark_loaded("model-c", ModelType.EMBEDDING, "float16")

        candidates = governor._get_eviction_candidates(exclude_key=None)

        # Should be sorted oldest first
        assert len(candidates) == 3
        assert candidates[0].model_name == "model-a"
        assert candidates[1].model_name == "model-b"
        assert candidates[2].model_name == "model-c"

    @pytest.mark.asyncio()
    async def test_touch_moves_model_to_end_of_lru(self, governor: GPUMemoryGovernor) -> None:
        """Test that touching a model moves it to end of LRU (most recently used)."""
        with patch.object(governor, "_get_model_memory", return_value=1000):
            await governor.mark_loaded("model-a", ModelType.EMBEDDING, "float16")
            await asyncio.sleep(0.01)
            await governor.mark_loaded("model-b", ModelType.EMBEDDING, "float16")
            await asyncio.sleep(0.01)
            await governor.mark_loaded("model-c", ModelType.EMBEDDING, "float16")

        # Touch model-a (oldest) to make it most recently used
        await governor.touch("model-a", ModelType.EMBEDDING, "float16")

        candidates = governor._get_eviction_candidates(exclude_key=None)

        # model-a should now be last (most recently used)
        assert candidates[0].model_name == "model-b"
        assert candidates[1].model_name == "model-c"
        assert candidates[2].model_name == "model-a"

    @pytest.mark.asyncio()
    async def test_exclude_key_skips_model_in_eviction(self, governor: GPUMemoryGovernor) -> None:
        """Test that excluded model is not in eviction candidates."""
        with patch.object(governor, "_get_model_memory", return_value=1000):
            await governor.mark_loaded("model-a", ModelType.EMBEDDING, "float16")
            await governor.mark_loaded("model-b", ModelType.EMBEDDING, "float16")

        exclude_key = governor._make_key("model-a", ModelType.EMBEDDING, "float16")
        candidates = governor._get_eviction_candidates(exclude_key=exclude_key)

        assert len(candidates) == 1
        assert candidates[0].model_name == "model-b"


# =============================================================================
# Memory Pressure Tests
# =============================================================================


class TestMemoryPressure:
    """Tests for memory pressure level calculation and handling."""

    def test_pressure_level_low(self, governor: GPUMemoryGovernor) -> None:
        """Test LOW pressure level when usage < 60%."""
        # No models loaded, usage is 0%
        pressure = governor._calculate_pressure_level()
        assert pressure == PressureLevel.LOW

    @pytest.mark.asyncio()
    async def test_pressure_level_moderate(self, governor: GPUMemoryGovernor) -> None:
        """Test MODERATE pressure level when usage 60-80%."""
        with patch.object(governor, "_get_model_memory", return_value=9000):
            # Load models to reach ~62.5% of 14400MB budget
            await governor.mark_loaded("model-a", ModelType.EMBEDDING, "float16")

        pressure = governor._calculate_pressure_level()
        assert pressure == PressureLevel.MODERATE

    @pytest.mark.asyncio()
    async def test_pressure_level_high(self, governor: GPUMemoryGovernor) -> None:
        """Test HIGH pressure level when usage 80-90%."""
        with patch.object(governor, "_get_model_memory", return_value=12000):
            # Load model to reach ~83% of 14400MB budget
            await governor.mark_loaded("model-a", ModelType.EMBEDDING, "float16")

        pressure = governor._calculate_pressure_level()
        assert pressure == PressureLevel.HIGH

    @pytest.mark.asyncio()
    async def test_pressure_level_critical(self, governor: GPUMemoryGovernor) -> None:
        """Test CRITICAL pressure level when usage > 90%."""
        with patch.object(governor, "_get_model_memory", return_value=13500):
            # Load model to reach ~94% of 14400MB budget
            await governor.mark_loaded("model-a", ModelType.EMBEDDING, "float16")

        pressure = governor._calculate_pressure_level()
        assert pressure == PressureLevel.CRITICAL


# =============================================================================
# Callback Tests
# =============================================================================


class TestCallbacks:
    """Tests for callback registration and execution."""

    def test_register_callbacks(self, governor: GPUMemoryGovernor) -> None:
        """Test callback registration."""
        unload_fn = AsyncMock()
        offload_fn = AsyncMock()

        governor.register_callbacks(
            model_type=ModelType.EMBEDDING,
            unload_fn=unload_fn,
            offload_fn=offload_fn,
        )

        assert governor._callbacks[ModelType.EMBEDDING]["unload"] is unload_fn
        assert governor._callbacks[ModelType.EMBEDDING]["offload"] is offload_fn

    @pytest.mark.asyncio()
    async def test_offload_callback_invoked(self, governor: GPUMemoryGovernor) -> None:
        """Test that offload callback is invoked during eviction."""
        offload_fn = AsyncMock()
        unload_fn = AsyncMock()

        governor.register_callbacks(
            model_type=ModelType.EMBEDDING,
            unload_fn=unload_fn,
            offload_fn=offload_fn,
        )

        # Create tracked model with old last_used to avoid grace period
        tracked = TrackedModel(
            model_name="test-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.GPU,
            memory_mb=2000,
            last_used=time.time() - 120,  # 2 minutes ago
        )
        model_key = tracked.model_key
        governor._models[model_key] = tracked

        await governor._offload_model(tracked)

        offload_fn.assert_called_once_with("test-model", "float16", "cpu")

    @pytest.mark.asyncio()
    async def test_unload_callback_invoked(self, governor: GPUMemoryGovernor) -> None:
        """Test that unload callback is invoked during eviction."""
        unload_fn = AsyncMock()

        governor.register_callbacks(
            model_type=ModelType.EMBEDDING,
            unload_fn=unload_fn,
        )

        tracked = TrackedModel(
            model_name="test-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.GPU,
            memory_mb=2000,
            last_used=time.time() - 120,
        )
        model_key = tracked.model_key
        governor._models[model_key] = tracked

        await governor._unload_model(tracked)

        unload_fn.assert_called_once_with("test-model", "float16")


# =============================================================================
# CPU Offload Tests
# =============================================================================


class TestCPUOffload:
    """Tests for CPU offloading behavior."""

    def test_can_offload_to_cpu_within_budget(self, governor: GPUMemoryGovernor) -> None:
        """Test can_offload_to_cpu when there's room."""
        # CPU usable is 16000MB, no models offloaded
        can_offload = governor._can_offload_to_cpu(5000)
        assert can_offload is True

    def test_can_offload_to_cpu_exceeds_budget(self, governor: GPUMemoryGovernor) -> None:
        """Test can_offload_to_cpu when no room."""
        # Try to offload more than usable CPU budget
        can_offload = governor._can_offload_to_cpu(20000)
        assert can_offload is False

    @pytest.mark.asyncio()
    async def test_offload_disabled_skips_cpu(self, governor_no_offload: GPUMemoryGovernor) -> None:
        """Test that offloading is skipped when disabled."""
        unload_fn = AsyncMock()
        offload_fn = AsyncMock()

        governor_no_offload.register_callbacks(
            model_type=ModelType.EMBEDDING,
            unload_fn=unload_fn,
            offload_fn=offload_fn,
        )

        # Load a model
        with patch.object(governor_no_offload, "_get_model_memory", return_value=2000):
            await governor_no_offload.mark_loaded("test-model", ModelType.EMBEDDING, "float16")

        # Set last_used to old time
        model_key = governor_no_offload._make_key("test-model", ModelType.EMBEDDING, "float16")
        governor_no_offload._models[model_key].last_used = time.time() - 120

        # Try to make room
        await governor_no_offload._make_room(2000, exclude_key=None)

        # Offload should not be called, but unload should be
        offload_fn.assert_not_called()
        unload_fn.assert_called_once()


# =============================================================================
# Memory Stats Tests
# =============================================================================


class TestMemoryStats:
    """Tests for memory statistics reporting."""

    @pytest.mark.asyncio()
    async def test_get_memory_stats_structure(self, governor: GPUMemoryGovernor) -> None:
        """Test get_memory_stats returns expected structure."""
        stats = governor.get_memory_stats()

        expected_keys = [
            "cuda_available",
            "total_mb",
            "free_mb",
            "used_mb",
            "used_percent",
            "allocated_mb",
            "budget_total_mb",
            "budget_usable_mb",
            "cpu_budget_total_mb",
            "cpu_budget_usable_mb",
            "cpu_used_mb",
            "models_loaded",
            "models_offloaded",
            "pressure_level",
            "total_evictions",
            "total_offloads",
            "total_restorations",
            "total_unloads",
        ]

        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    @pytest.mark.asyncio()
    async def test_get_loaded_models_returns_list(self, governor: GPUMemoryGovernor) -> None:
        """Test get_loaded_models returns model info."""
        with patch.object(governor, "_get_model_memory", return_value=2000):
            await governor.mark_loaded("test-model", ModelType.EMBEDDING, "float16")

        models = governor.get_loaded_models()
        assert len(models) == 1
        assert "model_name" in models[0]
        assert "model_type" in models[0]
        assert "location" in models[0]
        assert "memory_mb" in models[0]
        assert "idle_seconds" in models[0]
        assert "use_count" in models[0]


# =============================================================================
# Eviction Record Tests
# =============================================================================


class TestEvictionRecord:
    """Tests for eviction history tracking."""

    @pytest.mark.asyncio()
    async def test_eviction_recorded(self, governor: GPUMemoryGovernor) -> None:
        """Test that evictions are recorded in history."""
        unload_fn = AsyncMock()
        governor.register_callbacks(ModelType.EMBEDDING, unload_fn=unload_fn)

        tracked = TrackedModel(
            model_name="test-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.GPU,
            memory_mb=2000,
            last_used=time.time() - 120,
        )
        model_key = tracked.model_key
        governor._models[model_key] = tracked

        await governor._unload_model(tracked)

        history = governor.get_eviction_history()
        assert len(history) == 1
        assert history[0]["model_name"] == "test-model"
        assert history[0]["action"] == "unloaded"
        assert history[0]["reason"] == "memory_pressure"

    @pytest.mark.asyncio()
    async def test_eviction_history_bounded(self, governor: GPUMemoryGovernor) -> None:
        """Test that eviction history is bounded to max size."""
        from vecpipe.memory_governor import EvictionAction

        # Set small max for testing
        governor._max_history_size = 5

        for i in range(10):
            record = EvictionRecord(
                model_name=f"model-{i}",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                reason="test",
                action=EvictionAction.UNLOADED,
                memory_freed_mb=100,
            )
            governor._eviction_history.append(record)
            governor._total_evictions += 1

            # Trim if over limit
            if len(governor._eviction_history) > governor._max_history_size:
                governor._eviction_history = governor._eviction_history[-governor._max_history_size :]

        history = governor.get_eviction_history()
        assert len(history) == 5


# =============================================================================
# Monitor Tests
# =============================================================================


class TestMonitor:
    """Tests for background monitoring."""

    @pytest.mark.asyncio()
    async def test_start_monitor_creates_task(self, governor: GPUMemoryGovernor) -> None:
        """Test that start_monitor creates a background task."""
        await governor.start_monitor()
        assert governor._monitor_task is not None
        assert not governor._monitor_task.done()

    @pytest.mark.asyncio()
    async def test_shutdown_stops_monitor(self, governor: GPUMemoryGovernor) -> None:
        """Test that shutdown stops the monitor task."""
        await governor.start_monitor()
        assert governor._monitor_task is not None

        await governor.shutdown()
        assert governor._monitor_task is None or governor._monitor_task.done()

    @pytest.mark.asyncio()
    async def test_multiple_start_monitor_idempotent(self, governor: GPUMemoryGovernor) -> None:
        """Test that calling start_monitor multiple times is safe."""
        await governor.start_monitor()
        task1 = governor._monitor_task

        await governor.start_monitor()
        task2 = governor._monitor_task

        assert task1 is task2  # Same task, not replaced


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestIntegration:
    """Integration-style tests for realistic scenarios."""

    @pytest.mark.asyncio()
    async def test_eviction_on_memory_request(self, small_memory_budget: MemoryBudget) -> None:
        """Test that models are evicted when memory request exceeds budget."""
        governor = GPUMemoryGovernor(
            budget=small_memory_budget,
            enable_cpu_offload=False,  # Disable offload for simpler test
        )

        unload_fn = AsyncMock()
        governor.register_callbacks(ModelType.EMBEDDING, unload_fn=unload_fn)

        try:
            # Usable GPU budget: 8000 * 0.90 = 7200MB
            with patch.object(governor, "_get_model_memory", return_value=3000):
                # Load first model (3000MB, leaves 4200MB)
                await governor.mark_loaded("model-a", ModelType.EMBEDDING, "float16")

            # Set model-a to be old enough to evict
            model_key_a = governor._make_key("model-a", ModelType.EMBEDDING, "float16")
            governor._models[model_key_a].last_used = time.time() - 120

            # Request memory for large model that requires eviction
            # Required with overhead: 5000 * 1.2 = 6000MB
            # Current usage: 3000MB + 6000MB = 9000MB > 7200MB budget
            # Need to free: 9000 - 7200 = 1800MB (but model-a has 3000MB)
            result = await governor.request_memory(
                model_name="model-b",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                required_mb=5000,
            )

            # Eviction should have occurred
            assert result is True
            unload_fn.assert_called()

        finally:
            await governor.shutdown()

    @pytest.mark.asyncio()
    async def test_offload_preferred_over_unload(self, memory_budget: MemoryBudget) -> None:
        """Test that CPU offload is preferred over full unload."""
        governor = GPUMemoryGovernor(
            budget=memory_budget,
            enable_cpu_offload=True,
        )

        offload_fn = AsyncMock()
        unload_fn = AsyncMock()
        governor.register_callbacks(
            ModelType.EMBEDDING,
            unload_fn=unload_fn,
            offload_fn=offload_fn,
        )

        try:
            with patch.object(governor, "_get_model_memory", return_value=3000):
                await governor.mark_loaded("model-a", ModelType.EMBEDDING, "float16")

            model_key = governor._make_key("model-a", ModelType.EMBEDDING, "float16")
            governor._models[model_key].last_used = time.time() - 120

            # Make room - should prefer offload
            await governor._make_room(3000, exclude_key=None)

            # Offload should be called, not unload
            offload_fn.assert_called_once()
            unload_fn.assert_not_called()

        finally:
            await governor.shutdown()

    @pytest.mark.asyncio()
    async def test_model_restore_from_cpu(self, memory_budget: MemoryBudget) -> None:
        """Test restoring an offloaded model from CPU."""
        governor = GPUMemoryGovernor(
            budget=memory_budget,
            enable_cpu_offload=True,
        )

        offload_fn = AsyncMock()
        unload_fn = AsyncMock()
        governor.register_callbacks(
            ModelType.EMBEDDING,
            unload_fn=unload_fn,
            offload_fn=offload_fn,
        )

        try:
            # Manually create an offloaded model
            tracked = TrackedModel(
                model_name="offloaded-model",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                location=ModelLocation.CPU,
                memory_mb=2000,
            )
            model_key = tracked.model_key
            governor._models[model_key] = tracked

            # Request memory for the offloaded model
            result = await governor.request_memory(
                model_name="offloaded-model",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                required_mb=2000,
            )

            assert result is True
            # Offload callback should be called with "cuda" to restore
            offload_fn.assert_called_with("offloaded-model", "float16", "cuda")

        finally:
            await governor.shutdown()


# =============================================================================
# _restore_from_cpu Failure Tests
# =============================================================================


class TestRestoreFromCPUFailures:
    """Tests for _restore_from_cpu error paths."""

    @pytest.mark.asyncio()
    async def test_restore_model_not_tracked(self, governor: GPUMemoryGovernor) -> None:
        """Restore returns False if model not tracked."""
        result = await governor._restore_from_cpu("nonexistent:model:key", required_mb=2000)
        assert result is False

    @pytest.mark.asyncio()
    async def test_restore_model_not_on_cpu(self, governor: GPUMemoryGovernor) -> None:
        """Restore returns False if model is on GPU, not CPU."""
        # Register model on GPU (not CPU)
        tracked = TrackedModel(
            model_name="gpu-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.GPU,  # On GPU, not CPU
            memory_mb=2000,
        )
        model_key = tracked.model_key
        governor._models[model_key] = tracked

        result = await governor._restore_from_cpu(model_key, required_mb=2000)
        assert result is False

    @pytest.mark.asyncio()
    async def test_restore_insufficient_memory(self, small_memory_budget: MemoryBudget) -> None:
        """Restore returns False if GPU memory insufficient after eviction."""
        governor = GPUMemoryGovernor(
            budget=small_memory_budget,  # 8GB GPU, 7200MB usable
            enable_cpu_offload=True,
        )

        # Register callbacks - unload raises exception to simulate failure to free memory
        async def failing_unload(_model_name: str, _quantization: str) -> None:
            raise RuntimeError("Unload failed")  # Simulates failure

        async def noop_offload(_model_name: str, _quantization: str, _target_device: str) -> None:
            pass  # Offload does nothing (used for restore callback)

        async def noop_unload(_model_name: str, _quantization: str) -> None:
            pass

        governor.register_callbacks(
            ModelType.RERANKER,
            unload_fn=failing_unload,
        )
        governor.register_callbacks(
            ModelType.EMBEDDING,
            unload_fn=noop_unload,
            offload_fn=noop_offload,  # Required for restore
        )

        try:
            # Fill GPU with a model
            with patch.object(governor, "_get_model_memory", return_value=6000):
                await governor.mark_loaded("blocker-model", ModelType.RERANKER, "float16")

            # Create an offloaded model that needs to restore but won't fit
            tracked = TrackedModel(
                model_name="offloaded-model",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                location=ModelLocation.CPU,
                memory_mb=3000,
            )
            model_key = tracked.model_key
            governor._models[model_key] = tracked

            # Try to restore - should fail because 6000 + 3000 > 7200 usable
            # and blocker-model's unload callback returns False (fails to free)
            blocker_key = governor._make_key("blocker-model", ModelType.RERANKER, "float16")
            # Make the blocker old enough to be evicted
            governor._models[blocker_key].last_used = time.time() - 120

            result = await governor._restore_from_cpu(model_key, required_mb=3000)
            assert result is False

        finally:
            await governor.shutdown()

    @pytest.mark.asyncio()
    async def test_restore_missing_callback(self, governor: GPUMemoryGovernor) -> None:
        """Restore raises RuntimeError if offload callback not registered."""
        # Clear any existing callbacks
        governor._callbacks.clear()

        # Create an offloaded model
        tracked = TrackedModel(
            model_name="offloaded-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.CPU,
            memory_mb=2000,
        )
        model_key = tracked.model_key
        governor._models[model_key] = tracked

        with pytest.raises(RuntimeError, match="No offload callback registered"):
            await governor._restore_from_cpu(model_key, required_mb=2000)

    @pytest.mark.asyncio()
    async def test_restore_callback_exception(self, governor: GPUMemoryGovernor) -> None:
        """Restore returns False and logs if callback raises exception."""
        offload_fn = AsyncMock(side_effect=RuntimeError("CUDA out of memory"))
        unload_fn = AsyncMock()
        governor.register_callbacks(
            ModelType.EMBEDDING,
            unload_fn=unload_fn,
            offload_fn=offload_fn,
        )

        # Create an offloaded model
        tracked = TrackedModel(
            model_name="offloaded-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.CPU,
            memory_mb=2000,
        )
        model_key = tracked.model_key
        governor._models[model_key] = tracked

        result = await governor._restore_from_cpu(model_key, required_mb=2000)
        assert result is False

        # Model should still be on CPU (not corrupted)
        assert governor._models[model_key].location == ModelLocation.CPU

    @pytest.mark.asyncio()
    async def test_restore_unloaded_model_fails(self, governor: GPUMemoryGovernor) -> None:
        """Restore returns False if model is already unloaded."""
        tracked = TrackedModel(
            model_name="unloaded-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.UNLOADED,  # Already unloaded
            memory_mb=2000,
        )
        model_key = tracked.model_key
        governor._models[model_key] = tracked

        result = await governor._restore_from_cpu(model_key, required_mb=2000)
        assert result is False


# =============================================================================
# _get_model_memory RuntimeError Tests
# =============================================================================


class TestGetModelMemoryRuntimeError:
    """Tests for _get_model_memory RuntimeError when import fails."""

    def test_get_model_memory_docstring_documents_runtime_error(self, memory_budget: MemoryBudget) -> None:
        """Verify the method docstring documents RuntimeError for import failures."""
        governor = GPUMemoryGovernor(memory_budget)
        docstring = governor._get_model_memory.__doc__

        # Verify docstring documents the RuntimeError
        assert "RuntimeError" in docstring
        assert "memory_utils" in docstring

    def test_get_model_memory_returns_valid_int_for_known_model(self, memory_budget: MemoryBudget) -> None:
        """Test _get_model_memory returns valid memory value when working correctly."""
        governor = GPUMemoryGovernor(memory_budget)

        # This should work and return an int
        result = governor._get_model_memory("Qwen/Qwen3-Embedding-0.6B", "int8")

        assert isinstance(result, int)
        assert result > 0


# =============================================================================
# CPU-Only Mode Tests (total_gpu_mb=0)
# =============================================================================


class TestCPUOnlyMode:
    """Tests for CPU-only mode when total_gpu_mb=0."""

    @pytest.fixture()
    def cpu_only_budget(self) -> MemoryBudget:
        """Create a budget with no GPU memory (CPU-only mode)."""
        return MemoryBudget(
            total_gpu_mb=0,  # CPU-only mode
            gpu_max_percent=0.90,
            total_cpu_mb=32000,
            cpu_max_percent=0.50,
        )

    @pytest_asyncio.fixture
    async def cpu_only_governor(self, cpu_only_budget: MemoryBudget) -> AsyncGenerator[GPUMemoryGovernor, None]:
        """Create a CPU-only governor for testing."""
        governor = GPUMemoryGovernor(cpu_only_budget)
        yield governor
        await governor.shutdown()

    @pytest.mark.asyncio()
    async def test_request_memory_succeeds_without_budget_check(self, cpu_only_governor: GPUMemoryGovernor) -> None:
        """Memory requests always succeed in CPU-only mode."""
        # Register a callback to avoid RuntimeError
        cpu_only_governor.register_callbacks(
            model_type=ModelType.EMBEDDING,
            unload_fn=AsyncMock(),
        )

        # CPU-only mode should always return True without needing to calculate memory
        result = await cpu_only_governor.request_memory(
            model_name="any-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            required_mb=2000,  # required_mb is a parameter
        )

        # Should succeed even without GPU budget
        assert result is True

    @pytest.mark.asyncio()
    async def test_cpu_only_mode_logs_debug_message(self, cpu_only_governor: GPUMemoryGovernor) -> None:
        """CPU-only mode logs debug message about bypassing GPU budget check."""
        cpu_only_governor.register_callbacks(
            model_type=ModelType.EMBEDDING,
            unload_fn=AsyncMock(),
        )

        with patch("vecpipe.memory_governor.logger") as mock_logger:
            await cpu_only_governor.request_memory(
                model_name="test-model",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                required_mb=2000,
            )

            # Verify debug log was called
            mock_logger.debug.assert_any_call(
                "CPU-only mode: allowing %s load without GPU budget check",
                "test-model",
            )

    def test_cpu_only_budget_has_zero_usable_gpu(self, cpu_only_budget: MemoryBudget) -> None:
        """Verify CPU-only budget reports zero usable GPU memory."""
        assert cpu_only_budget.usable_gpu_mb == 0
        assert cpu_only_budget.total_gpu_mb == 0


# =============================================================================
# TrackedModel Validation Tests
# =============================================================================


class TestTrackedModelValidation:
    """Tests for TrackedModel validation in __post_init__."""

    def test_negative_memory_mb_raises_error(self) -> None:
        """memory_mb must be non-negative."""
        with pytest.raises(ValueError, match="memory_mb must be non-negative"):
            TrackedModel(
                model_name="test-model",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                location=ModelLocation.GPU,
                memory_mb=-100,
            )

    def test_negative_use_count_raises_error(self) -> None:
        """use_count must be non-negative."""
        with pytest.raises(ValueError, match="use_count must be non-negative"):
            TrackedModel(
                model_name="test-model",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                location=ModelLocation.GPU,
                memory_mb=100,
                use_count=-1,
            )

    def test_empty_model_name_raises_error(self) -> None:
        """model_name cannot be empty."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            TrackedModel(
                model_name="",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                location=ModelLocation.GPU,
                memory_mb=100,
            )

    def test_empty_quantization_raises_error(self) -> None:
        """quantization cannot be empty."""
        with pytest.raises(ValueError, match="quantization cannot be empty"):
            TrackedModel(
                model_name="test-model",
                model_type=ModelType.EMBEDDING,
                quantization="",
                location=ModelLocation.GPU,
                memory_mb=100,
            )

    def test_zero_memory_mb_is_valid(self) -> None:
        """memory_mb=0 is valid (edge case)."""
        model = TrackedModel(
            model_name="test-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.GPU,
            memory_mb=0,
        )
        assert model.memory_mb == 0

    def test_zero_use_count_is_valid(self) -> None:
        """use_count=0 is valid (default)."""
        model = TrackedModel(
            model_name="test-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.GPU,
            memory_mb=100,
            use_count=0,
        )
        assert model.use_count == 0


# =============================================================================
# MemoryBudget Validation Tests
# =============================================================================


class TestMemoryBudgetValidation:
    """Tests for MemoryBudget validation edge cases."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("gpu_max_percent", -0.1),
            ("gpu_max_percent", 1.1),
            ("cpu_max_percent", -0.1),
            ("cpu_max_percent", 1.1),
        ],
    )
    def test_invalid_percentage_raises_error(self, field: str, value: float) -> None:
        """Percentages must be between 0.0 and 1.0."""
        kwargs = {"total_gpu_mb": 8000, field: value}
        with pytest.raises(ValueError, match=f"{field} must be between 0.0 and 1.0"):
            MemoryBudget(**kwargs)

    def test_negative_gpu_mb_raises_error(self) -> None:
        """total_gpu_mb must be non-negative."""
        with pytest.raises(ValueError, match="total_gpu_mb must be non-negative"):
            MemoryBudget(total_gpu_mb=-1)

    def test_negative_cpu_mb_raises_error(self) -> None:
        """total_cpu_mb must be non-negative."""
        with pytest.raises(ValueError, match="total_cpu_mb must be non-negative"):
            MemoryBudget(total_gpu_mb=8000, total_cpu_mb=-1)

    def test_edge_case_percentages_zero_valid(self) -> None:
        """0.0 is a valid percentage value for max_percent fields."""
        budget = MemoryBudget(
            total_gpu_mb=8000,
            gpu_max_percent=0.0,
            cpu_max_percent=0.0,
        )
        assert budget.gpu_max_percent == 0.0
        assert budget.cpu_max_percent == 0.0

    def test_edge_case_percentages_one_valid(self) -> None:
        """1.0 is a valid percentage value."""
        budget = MemoryBudget(
            total_gpu_mb=8000,
            gpu_max_percent=1.0,
            cpu_max_percent=1.0,
        )
        assert budget.gpu_max_percent == 1.0
        assert budget.cpu_max_percent == 1.0

    def test_zero_gpu_mb_valid(self) -> None:
        """total_gpu_mb=0 is valid (CPU-only mode)."""
        budget = MemoryBudget(total_gpu_mb=0)
        assert budget.total_gpu_mb == 0
        assert budget.usable_gpu_mb == 0

    def test_zero_cpu_mb_valid(self) -> None:
        """total_cpu_mb=0 is valid (no CPU offloading)."""
        budget = MemoryBudget(total_gpu_mb=8000, total_cpu_mb=0)
        assert budget.total_cpu_mb == 0
        assert budget.usable_cpu_mb == 0


# =============================================================================
# Monitor Loop Circuit Breaker Tests
# =============================================================================


class TestMonitorLoopCircuitBreaker:
    """Tests for circuit breaker and degraded state in _monitor_loop."""

    @pytest.mark.asyncio()
    async def test_circuit_breaker_triggers_after_max_failures(self, memory_budget: MemoryBudget) -> None:
        """Circuit breaker triggers after 5 consecutive failures."""
        gov = GPUMemoryGovernor(memory_budget)

        # Make _calculate_pressure_level raise an error
        call_count = 0

        def failing_pressure_level() -> PressureLevel:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("GPU error")

        gov._calculate_pressure_level = failing_pressure_level

        # Track sleep calls and stop after circuit breaker triggers
        sleep_count = 0

        async def mock_sleep(seconds: float) -> None:
            nonlocal sleep_count
            sleep_count += 1
            # Stop after first backoff sleep (which happens after 5 failures)
            if seconds >= 30:  # Backoff sleep is 30s, interval is much smaller
                gov._shutdown_event.set()

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await gov._monitor_loop()

        assert gov._circuit_breaker_triggers >= 1
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_degraded_state_after_repeated_triggers(self, memory_budget: MemoryBudget) -> None:
        """Degraded state set after 3 circuit breaker triggers."""
        gov = GPUMemoryGovernor(memory_budget)
        gov._max_circuit_breaker_triggers = 3

        # Make _calculate_pressure_level raise an error
        def failing_pressure_level() -> PressureLevel:
            raise RuntimeError("GPU error")

        gov._calculate_pressure_level = failing_pressure_level

        # Track backoff sleeps and stop after 3 triggers
        backoff_count = 0

        async def mock_sleep(seconds: float) -> None:
            nonlocal backoff_count
            # Backoff sleeps are >= 30s, regular interval is much smaller
            if seconds >= 30:
                backoff_count += 1
                if backoff_count >= 3:  # Stop after 3 circuit breaker triggers
                    gov._shutdown_event.set()

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await gov._monitor_loop()

        assert gov._degraded_state is True
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_backoff_resets_after_successful_iterations(self, memory_budget: MemoryBudget) -> None:
        """Successful iterations reset the backoff counter."""
        gov = GPUMemoryGovernor(memory_budget)

        # Return LOW pressure (no errors) - this counts as success
        iteration_count = 0

        def success_pressure_level() -> PressureLevel:
            nonlocal iteration_count
            iteration_count += 1
            return PressureLevel.LOW

        gov._calculate_pressure_level = success_pressure_level

        async def mock_sleep(_seconds: float) -> None:
            # Stop after enough iterations
            if iteration_count >= 12:
                gov._shutdown_event.set()

        with patch("asyncio.sleep", side_effect=mock_sleep):
            await gov._monitor_loop()

        # After 10+ successes, degraded state should be cleared
        assert gov._degraded_state is False
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_exponential_backoff_increases(self, memory_budget: MemoryBudget) -> None:
        """Backoff doubles after each circuit breaker trigger (up to max)."""
        gov = GPUMemoryGovernor(memory_budget)

        # Make _calculate_pressure_level raise an error
        def failing_pressure_level() -> PressureLevel:
            raise RuntimeError("GPU error")

        gov._calculate_pressure_level = failing_pressure_level

        sleep_calls: list[float] = []

        async def record_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            # Stop after collecting backoff values (30s, 60s)
            # Count backoff sleeps (>= 30s) and stop after 2
            backoff_sleeps = [s for s in sleep_calls if s >= 30]
            if len(backoff_sleeps) >= 2:
                gov._shutdown_event.set()

        with patch("asyncio.sleep", side_effect=record_sleep):
            await gov._monitor_loop()

        # Filter for backoff sleeps (>= 30s)
        backoff_sleeps = [s for s in sleep_calls if s >= 30]
        assert len(backoff_sleeps) >= 2
        # First backoff is 30s, second should be 60s
        assert backoff_sleeps[0] == 30
        assert backoff_sleeps[1] == 60
        await gov.shutdown()


# =============================================================================
# Pressure Handler Continuation Tests
# =============================================================================


class TestPressureHandlerContinuation:
    """Tests for pressure handlers continuing after partial failures."""

    @pytest.fixture()
    def governor_with_models(self, memory_budget: MemoryBudget) -> GPUMemoryGovernor:
        """Governor with multiple tracked models."""
        gov = GPUMemoryGovernor(memory_budget)
        # Add 3 models
        for i in range(3):
            gov._models[f"embedding:model{i}:float16"] = TrackedModel(
                model_name=f"model{i}",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                location=ModelLocation.GPU,
                memory_mb=1000,
                last_used=time.time() - 600,  # Idle for 10 minutes
            )
        return gov

    @pytest.mark.asyncio()
    async def test_critical_pressure_continues_after_callback_failure(
        self, governor_with_models: GPUMemoryGovernor
    ) -> None:
        """Critical pressure handler continues processing after callback failure."""
        gov = governor_with_models
        models_attempted: set[str] = set()

        async def failing_unload(name: str, _quantization: str) -> bool:
            models_attempted.add(name)
            if name == "model1":
                raise RuntimeError("Unload failed")
            return True

        gov.register_callbacks(ModelType.EMBEDDING, unload_fn=failing_unload)

        await gov._handle_critical_pressure()

        # All 3 unique models should have been attempted despite model1 failure
        # (there may be retries for model1, but all 3 models should be attempted)
        assert models_attempted == {"model0", "model1", "model2"}
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_high_pressure_continues_after_callback_returns_false(
        self, governor_with_models: GPUMemoryGovernor
    ) -> None:
        """High pressure handler continues when callback returns False."""
        gov = governor_with_models
        # Disable CPU offload so it uses unload directly
        gov._enable_cpu_offload = False
        models_attempted: set[str] = set()

        async def partial_failure_unload(name: str, _quantization: str) -> bool:
            models_attempted.add(name)
            return name != "model1"  # Return False for model1

        gov.register_callbacks(ModelType.EMBEDDING, unload_fn=partial_failure_unload)

        await gov._handle_high_pressure()

        # All models should be attempted
        assert models_attempted == {"model0", "model1", "model2"}
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_moderate_pressure_continues_on_offload_exception(self, memory_budget: MemoryBudget) -> None:
        """Moderate pressure handler continues after offload exception."""
        gov = GPUMemoryGovernor(memory_budget, enable_cpu_offload=True)
        # Add models that are idle beyond threshold
        for i in range(3):
            gov._models[f"embedding:model{i}:float16"] = TrackedModel(
                model_name=f"model{i}",
                model_type=ModelType.EMBEDDING,
                quantization="float16",
                location=ModelLocation.GPU,
                memory_mb=1000,
                last_used=time.time() - 200,  # Idle beyond eviction threshold (120s)
            )

        offload_calls: list[str] = []

        async def failing_offload(name: str, _quantization: str, _target: str) -> bool:
            offload_calls.append(name)
            if name == "model0":
                raise RuntimeError("Offload failed")
            return True

        async def noop_unload(_name: str, _quantization: str) -> bool:
            return True

        gov.register_callbacks(ModelType.EMBEDDING, unload_fn=noop_unload, offload_fn=failing_offload)

        await gov._handle_moderate_pressure()

        # All eligible models should be attempted
        assert len(offload_calls) >= 1
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_failed_models_list_populated(self, governor_with_models: GPUMemoryGovernor) -> None:
        """Failed models are tracked correctly in handlers."""
        gov = governor_with_models

        async def always_fail(_name: str, _quantization: str) -> bool:
            raise RuntimeError("Always fails")

        gov.register_callbacks(ModelType.EMBEDDING, unload_fn=always_fail)

        # Run handler - should not raise, just log failures
        await gov._handle_critical_pressure()

        # Handler should complete without raising
        await gov.shutdown()


# =============================================================================
# Probe Mode Tests
# =============================================================================


class TestProbeMode:
    """Tests for GPU probe mode configuration."""

    @pytest.mark.asyncio()
    async def test_fast_probe_does_not_call_empty_cache(self, memory_budget: MemoryBudget) -> None:
        """Fast probe mode should not call torch.cuda.empty_cache()."""
        from vecpipe.memory_governor import ProbeMode

        gov = GPUMemoryGovernor(memory_budget, probe_mode="fast")

        with patch("torch.cuda.empty_cache") as mock_empty:
            with patch("torch.cuda.synchronize") as mock_sync:
                with patch("torch.cuda.mem_get_info", return_value=(8_000_000_000, 16_000_000_000)):
                    with patch("torch.cuda.is_available", return_value=True):
                        gov._probe_free_gpu_mb(ProbeMode.FAST)

        mock_empty.assert_not_called()
        mock_sync.assert_not_called()
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_safe_probe_calls_synchronize_only(self, memory_budget: MemoryBudget) -> None:
        """Safe probe mode should call synchronize but not empty_cache."""
        from vecpipe.memory_governor import ProbeMode

        gov = GPUMemoryGovernor(memory_budget, probe_mode="safe")

        with patch("torch.cuda.empty_cache") as mock_empty:
            with patch("torch.cuda.synchronize") as mock_sync:
                with patch("torch.cuda.mem_get_info", return_value=(8_000_000_000, 16_000_000_000)):
                    with patch("torch.cuda.is_available", return_value=True):
                        gov._probe_free_gpu_mb(ProbeMode.SAFE)

        mock_empty.assert_not_called()
        mock_sync.assert_called_once()
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_aggressive_probe_calls_gc_sync_and_empty_cache(self, memory_budget: MemoryBudget) -> None:
        """Aggressive probe mode should call gc, synchronize, and empty_cache."""
        from vecpipe.memory_governor import ProbeMode

        gov = GPUMemoryGovernor(memory_budget, probe_mode="aggressive")

        with patch("torch.cuda.empty_cache") as mock_empty:
            with patch("torch.cuda.synchronize") as mock_sync:
                with patch("torch.cuda.mem_get_info", return_value=(8_000_000_000, 16_000_000_000)):
                    with patch("torch.cuda.is_available", return_value=True):
                        with patch("gc.collect") as mock_gc:
                            gov._probe_free_gpu_mb(ProbeMode.AGGRESSIVE)

        mock_gc.assert_called_once()
        mock_sync.assert_called_once()
        mock_empty.assert_called_once()
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_probe_mode_upgrade_under_pressure(self, memory_budget: MemoryBudget) -> None:
        """Should upgrade to SAFE mode when usage exceeds threshold."""
        from vecpipe.memory_governor import ProbeMode

        gov = GPUMemoryGovernor(memory_budget, probe_mode="fast", probe_safe_threshold=0.85)

        # Load model to push usage above 85%
        # Budget is 14400 usable (16000 * 0.9), so 13000 is ~90%
        with patch.object(gov, "_get_model_memory", return_value=13000):
            await gov.mark_loaded("big-model", ModelType.EMBEDDING, "float16")

        mode = gov._select_probe_mode()
        assert mode == ProbeMode.SAFE
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_probe_mode_stays_fast_under_threshold(self, memory_budget: MemoryBudget) -> None:
        """Should stay in FAST mode when usage is below threshold."""
        from vecpipe.memory_governor import ProbeMode

        gov = GPUMemoryGovernor(memory_budget, probe_mode="fast", probe_safe_threshold=0.85)

        # Load small model - usage well below 85%
        with patch.object(gov, "_get_model_memory", return_value=2000):
            await gov.mark_loaded("small-model", ModelType.EMBEDDING, "float16")

        mode = gov._select_probe_mode()
        assert mode == ProbeMode.FAST
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_force_accurate_returns_aggressive(self, memory_budget: MemoryBudget) -> None:
        """force_accurate=True should always return AGGRESSIVE mode."""
        from vecpipe.memory_governor import ProbeMode

        gov = GPUMemoryGovernor(memory_budget, probe_mode="fast")

        # Even with no models loaded, force_accurate should return AGGRESSIVE
        mode = gov._select_probe_mode(force_accurate=True)
        assert mode == ProbeMode.AGGRESSIVE
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_default_probe_mode_from_init(self, memory_budget: MemoryBudget) -> None:
        """Probe mode set in __init__ should be used by default."""
        from vecpipe.memory_governor import ProbeMode

        gov = GPUMemoryGovernor(memory_budget, probe_mode="safe")

        assert gov._probe_mode == ProbeMode.SAFE
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_probe_override_mode_parameter(self, memory_budget: MemoryBudget) -> None:
        """Explicit mode parameter should override default."""
        from vecpipe.memory_governor import ProbeMode

        gov = GPUMemoryGovernor(memory_budget, probe_mode="fast")

        # Even though default is fast, passing aggressive should use aggressive
        with patch("torch.cuda.empty_cache") as mock_empty:
            with patch("torch.cuda.synchronize") as mock_sync:
                with patch("torch.cuda.mem_get_info", return_value=(8_000_000_000, 16_000_000_000)):
                    with patch("torch.cuda.is_available", return_value=True):
                        with patch("gc.collect"):
                            gov._probe_free_gpu_mb(ProbeMode.AGGRESSIVE)

        mock_empty.assert_called_once()
        mock_sync.assert_called_once()
        await gov.shutdown()


# =============================================================================
# CUDA Cache Clearing Tests
# =============================================================================


class TestCUDACacheClearing:
    """Tests for CUDA cache clearing after evictions."""

    @pytest.mark.asyncio()
    async def test_pressure_level_uses_max_of_tracked_and_actual(self, memory_budget: MemoryBudget) -> None:
        """Pressure level should use MAX of tracked and actual GPU usage.

        This catches cases where CUDA cache holds fragmented memory (e.g., after
        LLM unload) that isn't tracked by the governor. If tracked=40% but
        actual=91%, pressure should be CRITICAL.
        """
        gov = GPUMemoryGovernor(memory_budget)

        # Load a small model - 40% of 14400MB budget = 5760MB
        with patch.object(gov, "_get_model_memory", return_value=5760):
            await gov.mark_loaded("small-model", ModelType.EMBEDDING, "float16")

        # Tracked usage is ~40%, but simulate actual GPU showing 91% used
        # 91% of 16000MB total = ~14560MB used, so free = ~1440MB
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.mem_get_info", return_value=(1_440_000_000, 16_000_000_000)):
                pressure = gov._calculate_pressure_level()

        # Should be CRITICAL (>90%) based on actual usage, not MODERATE (~40% tracked)
        assert pressure == PressureLevel.CRITICAL
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_pressure_level_uses_tracked_when_higher(self, memory_budget: MemoryBudget) -> None:
        """Pressure level uses tracked when tracked > actual (conservative)."""
        gov = GPUMemoryGovernor(memory_budget)

        # Load a large model - 85% of 14400MB budget = 12240MB
        with patch.object(gov, "_get_model_memory", return_value=12240):
            await gov.mark_loaded("large-model", ModelType.EMBEDDING, "float16")

        # Simulate actual GPU showing only 50% used (less than tracked)
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.mem_get_info", return_value=(8_000_000_000, 16_000_000_000)):
                pressure = gov._calculate_pressure_level()

        # Should be HIGH (85% tracked > 50% actual)
        assert pressure == PressureLevel.HIGH
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_make_room_clears_cuda_cache_after_evictions(self, memory_budget: MemoryBudget) -> None:
        """_make_room should clear CUDA cache before and after evictions."""
        gov = GPUMemoryGovernor(memory_budget, enable_cpu_offload=False)

        unload_fn = AsyncMock()
        gov.register_callbacks(ModelType.EMBEDDING, unload_fn=unload_fn)

        # Add an idle model
        tracked = TrackedModel(
            model_name="test-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.GPU,
            memory_mb=2000,
            last_used=time.time() - 120,  # Idle
        )
        gov._models[tracked.model_key] = tracked

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize") as mock_sync:
                with patch("torch.cuda.empty_cache") as mock_empty:
                    await gov._make_room(2000)

        # Cache should be cleared TWICE: once at start to reclaim stale memory,
        # and once after eviction to reclaim newly freed memory
        assert mock_sync.call_count == 2
        assert mock_empty.call_count == 2
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_make_room_clears_cache_even_when_no_evictions(self, memory_budget: MemoryBudget) -> None:
        """_make_room should clear CUDA cache even if no evictions occurred.

        This is critical when models_loaded=0 but PyTorch's cache still holds
        memory from previously unloaded models. Without the initial cache clear,
        the allocation would fail with no candidates to evict.
        """
        gov = GPUMemoryGovernor(memory_budget, enable_cpu_offload=False)

        unload_fn = AsyncMock()
        gov.register_callbacks(ModelType.EMBEDDING, unload_fn=unload_fn)

        # Add a model that's too recently used to evict (within grace period)
        tracked = TrackedModel(
            model_name="active-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.GPU,
            memory_mb=2000,
            last_used=time.time(),  # Just used - will be skipped
        )
        gov._models[tracked.model_key] = tracked

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize") as mock_sync:
                with patch("torch.cuda.empty_cache") as mock_empty:
                    # Without force=True, the grace period prevents eviction
                    await gov._make_room(2000, force=False)

        # Cache should still be cleared once at the start (to reclaim stale memory),
        # even though no evictions occurred
        mock_sync.assert_called_once()
        mock_empty.assert_called_once()
        unload_fn.assert_not_called()
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_critical_pressure_clears_cuda_cache(self, memory_budget: MemoryBudget) -> None:
        """Critical pressure handler should clear CUDA cache after evictions."""
        gov = GPUMemoryGovernor(memory_budget)

        unload_fn = AsyncMock()
        gov.register_callbacks(ModelType.EMBEDDING, unload_fn=unload_fn)

        # Add an idle model (idle > 5 seconds for critical pressure)
        tracked = TrackedModel(
            model_name="idle-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.GPU,
            memory_mb=2000,
            last_used=time.time() - 60,  # Idle for 60 seconds
        )
        gov._models[tracked.model_key] = tracked

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize") as mock_sync:
                with patch("torch.cuda.empty_cache") as mock_empty:
                    await gov._handle_critical_pressure()

        # Cache should be cleared after critical pressure eviction
        mock_sync.assert_called_once()
        mock_empty.assert_called_once()
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_high_pressure_clears_cuda_cache(self, memory_budget: MemoryBudget) -> None:
        """High pressure handler should clear CUDA cache after evictions."""
        gov = GPUMemoryGovernor(memory_budget, enable_cpu_offload=False)

        unload_fn = AsyncMock()
        gov.register_callbacks(ModelType.EMBEDDING, unload_fn=unload_fn)

        # Add model idle > 30 seconds for high pressure
        tracked = TrackedModel(
            model_name="idle-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.GPU,
            memory_mb=2000,
            last_used=time.time() - 60,  # Idle for 60 seconds
        )
        gov._models[tracked.model_key] = tracked

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize") as mock_sync:
                with patch("torch.cuda.empty_cache") as mock_empty:
                    await gov._handle_high_pressure()

        # Cache should be cleared after high pressure eviction
        mock_sync.assert_called_once()
        mock_empty.assert_called_once()
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_moderate_pressure_clears_cuda_cache(self, memory_budget: MemoryBudget) -> None:
        """Moderate pressure handler should clear CUDA cache after evictions."""
        gov = GPUMemoryGovernor(
            memory_budget,
            enable_cpu_offload=True,
            eviction_idle_threshold_seconds=60,
        )

        offload_fn = AsyncMock()
        unload_fn = AsyncMock()
        gov.register_callbacks(ModelType.EMBEDDING, unload_fn=unload_fn, offload_fn=offload_fn)

        # Add model idle beyond threshold for moderate pressure offload
        tracked = TrackedModel(
            model_name="idle-model",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            location=ModelLocation.GPU,
            memory_mb=2000,
            last_used=time.time() - 120,  # Idle for 120 seconds
        )
        gov._models[tracked.model_key] = tracked

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize") as mock_sync:
                with patch("torch.cuda.empty_cache") as mock_empty:
                    await gov._handle_moderate_pressure()

        # Cache should be cleared after moderate pressure offload
        mock_sync.assert_called_once()
        mock_empty.assert_called_once()
        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_clear_cuda_cache_handles_no_cuda(self, memory_budget: MemoryBudget) -> None:
        """_clear_cuda_cache should handle missing CUDA gracefully."""
        gov = GPUMemoryGovernor(memory_budget)

        with patch("torch.cuda.is_available", return_value=False):
            # Should not raise
            gov._clear_cuda_cache()

        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_clear_cuda_cache_handles_exception(self, memory_budget: MemoryBudget) -> None:
        """_clear_cuda_cache should log warning on exception."""
        gov = GPUMemoryGovernor(memory_budget)

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.synchronize", side_effect=RuntimeError("CUDA error")):
                # Should not raise, just log warning
                gov._clear_cuda_cache()

        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_get_actual_usage_percent_cpu_only_mode(self, memory_budget: MemoryBudget) -> None:
        """_get_actual_usage_percent returns 0 in CPU-only mode."""
        cpu_only_budget = MemoryBudget(total_gpu_mb=0, total_cpu_mb=16000)
        gov = GPUMemoryGovernor(cpu_only_budget)

        result = gov._get_actual_usage_percent()
        assert result == 0.0

        await gov.shutdown()

    @pytest.mark.asyncio()
    async def test_get_actual_usage_percent_with_cuda(self, memory_budget: MemoryBudget) -> None:
        """_get_actual_usage_percent returns correct percentage from CUDA."""
        gov = GPUMemoryGovernor(memory_budget)

        # 10GB used out of 16GB = 62.5%
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.mem_get_info", return_value=(6_000_000_000, 16_000_000_000)):
                result = gov._get_actual_usage_percent()

        assert 62.0 <= result <= 63.0  # ~62.5%
        await gov.shutdown()
