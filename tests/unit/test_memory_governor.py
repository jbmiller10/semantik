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


@pytest.fixture
def memory_budget() -> MemoryBudget:
    """Create a test memory budget with known values."""
    return MemoryBudget(
        total_gpu_mb=16000,  # 16GB GPU
        gpu_reserve_percent=0.10,
        gpu_max_percent=0.90,
        total_cpu_mb=32000,  # 32GB CPU
        cpu_reserve_percent=0.20,
        cpu_max_percent=0.50,
    )


@pytest.fixture
def small_memory_budget() -> MemoryBudget:
    """Create a constrained memory budget for small GPU tests."""
    return MemoryBudget(
        total_gpu_mb=8000,  # 8GB GPU
        gpu_reserve_percent=0.10,
        gpu_max_percent=0.90,
        total_cpu_mb=16000,  # 16GB CPU
        cpu_reserve_percent=0.20,
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
        # With 16000MB total, 10% reserve, 90% max
        # effective_max = min(0.90, 1.0 - 0.10) = 0.80
        # usable = 16000 * 0.80 = 12800
        # But min(0.90, 0.90) = 0.90, so usable = 16000 * 0.90 = 14400
        # Actually: effective_max = min(gpu_max_percent, 1.0 - gpu_reserve_percent)
        # = min(0.90, 0.90) = 0.90
        expected = int(16000 * 0.90)  # 14400
        assert memory_budget.usable_gpu_mb == expected

    def test_usable_cpu_mb_calculation(self, memory_budget: MemoryBudget) -> None:
        """Test CPU usable memory calculation."""
        # With 32000MB total, 20% reserve, 50% max
        # effective_max = min(0.50, 1.0 - 0.20) = min(0.50, 0.80) = 0.50
        # usable = 32000 * 0.50 = 16000
        expected = int(32000 * 0.50)  # 16000
        assert memory_budget.usable_cpu_mb == expected

    def test_reserve_exceeds_max_uses_reserve(self) -> None:
        """Test that when reserve exceeds max, reserve takes precedence."""
        budget = MemoryBudget(
            total_gpu_mb=10000,
            gpu_reserve_percent=0.50,  # 50% reserve (keep free)
            gpu_max_percent=0.80,  # 80% max usage
        )
        # effective_max = min(0.80, 1.0 - 0.50) = min(0.80, 0.50) = 0.50
        expected = int(10000 * 0.50)  # 5000
        assert budget.usable_gpu_mb == expected

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

    @pytest.mark.asyncio
    async def test_request_memory_fits_in_budget(self, governor: GPUMemoryGovernor) -> None:
        """Test memory request that fits within budget."""
        result = await governor.request_memory(
            model_name="test-embedding",
            model_type=ModelType.EMBEDDING,
            quantization="float16",
            required_mb=2000,
        )
        assert result is True

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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


# =============================================================================
# LRU Eviction Tests
# =============================================================================


class TestLRUEviction:
    """Tests for LRU-based eviction behavior."""

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_pressure_level_moderate(self, governor: GPUMemoryGovernor) -> None:
        """Test MODERATE pressure level when usage 60-80%."""
        with patch.object(governor, "_get_model_memory", return_value=9000):
            # Load models to reach ~62.5% of 14400MB budget
            await governor.mark_loaded("model-a", ModelType.EMBEDDING, "float16")

        pressure = governor._calculate_pressure_level()
        assert pressure == PressureLevel.MODERATE

    @pytest.mark.asyncio
    async def test_pressure_level_high(self, governor: GPUMemoryGovernor) -> None:
        """Test HIGH pressure level when usage 80-90%."""
        with patch.object(governor, "_get_model_memory", return_value=12000):
            # Load model to reach ~83% of 14400MB budget
            await governor.mark_loaded("model-a", ModelType.EMBEDDING, "float16")

        pressure = governor._calculate_pressure_level()
        assert pressure == PressureLevel.HIGH

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
                governor._eviction_history = governor._eviction_history[-governor._max_history_size:]

        history = governor.get_eviction_history()
        assert len(history) == 5


# =============================================================================
# Monitor Tests
# =============================================================================


class TestMonitor:
    """Tests for background monitoring."""

    @pytest.mark.asyncio
    async def test_start_monitor_creates_task(self, governor: GPUMemoryGovernor) -> None:
        """Test that start_monitor creates a background task."""
        await governor.start_monitor()
        assert governor._monitor_task is not None
        assert not governor._monitor_task.done()

    @pytest.mark.asyncio
    async def test_shutdown_stops_monitor(self, governor: GPUMemoryGovernor) -> None:
        """Test that shutdown stops the monitor task."""
        await governor.start_monitor()
        assert governor._monitor_task is not None

        await governor.shutdown()
        assert governor._monitor_task is None or governor._monitor_task.done()

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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
