"""Tests for GovernedModelManager.

Tests the integration between ModelManager, GPUMemoryGovernor, and CPUOffloader.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import vecpipe.governed_model_manager as gmm
from vecpipe.governed_model_manager import GovernedModelManager
from vecpipe.memory_governor import MemoryBudget, ModelType


@pytest.fixture()
def mock_settings():
    """Mock settings to prevent actual model loading."""
    with patch("vecpipe.model_manager.settings") as mock:
        mock.USE_MOCK_EMBEDDINGS = True
        mock.MODEL_UNLOAD_AFTER_SECONDS = 300
        yield mock


@pytest.fixture()
def memory_budget():
    """Create a test memory budget."""
    return MemoryBudget(
        total_gpu_mb=8000,
        total_cpu_mb=16000,
        gpu_reserve_percent=0.10,
        gpu_max_percent=0.90,
        cpu_reserve_percent=0.20,
        cpu_max_percent=0.50,
    )


@pytest.fixture()
def governed_manager(mock_settings, memory_budget):  # noqa: ARG001 - mock_settings needed to patch settings
    """Create a GovernedModelManager for testing."""
    return GovernedModelManager(
        unload_after_seconds=300,
        budget=memory_budget,
        enable_cpu_offload=True,
        enable_preemptive_eviction=False,  # Disable for unit tests
    )


class TestGovernedModelManagerInit:
    """Tests for GovernedModelManager initialization."""

    def test_init_creates_governor(self, governed_manager):
        """Test that initialization creates a governor instance."""
        assert governed_manager._governor is not None
        assert governed_manager._governor_initialized is False

    def test_init_creates_offloader(self, governed_manager):
        """Test that initialization creates an offloader instance."""
        assert governed_manager._offloader is not None

    def test_init_registers_callbacks(self, governed_manager):
        """Test that callbacks are registered for both model types."""
        callbacks = governed_manager._governor._callbacks
        # Check callbacks exist for both model types (use values due to enum identity)
        assert len(callbacks) >= 2
        # Find embedding callbacks by matching enum value
        embedding_callbacks = None
        for model_type, cbs in callbacks.items():
            if model_type.value == ModelType.EMBEDDING.value:
                embedding_callbacks = cbs
                break
        assert embedding_callbacks is not None, "Embedding callbacks not registered"
        assert "unload" in embedding_callbacks
        assert "offload" in embedding_callbacks

    def test_init_with_custom_budget(self, mock_settings):
        """Test initialization with custom memory budget."""
        budget = MemoryBudget(
            total_gpu_mb=4000,
            total_cpu_mb=8000,
        )
        manager = GovernedModelManager(budget=budget)
        assert manager._governor._budget.total_gpu_mb == 4000
        assert manager._governor._budget.total_cpu_mb == 8000

    def test_init_with_preemptive_eviction_disabled(self, mock_settings, memory_budget):
        """Test initialization with preemptive eviction disabled."""
        manager = GovernedModelManager(
            budget=memory_budget,
            enable_preemptive_eviction=False,
        )
        assert manager._enable_preemptive_eviction is False


class TestGovernedModelManagerStart:
    """Tests for GovernedModelManager start/shutdown."""

    @pytest.mark.asyncio()
    async def test_start_initializes_governor(self, governed_manager):
        """Test that start() marks governor as initialized."""
        await governed_manager.start()
        assert governed_manager._governor_initialized is True

    @pytest.mark.asyncio()
    async def test_start_with_preemptive_eviction(self, mock_settings, memory_budget):
        """Test that start() starts monitor when preemptive eviction enabled."""
        manager = GovernedModelManager(
            budget=memory_budget,
            enable_preemptive_eviction=True,
        )
        await manager.start()
        assert manager._governor._monitor_task is not None
        # Clean up
        await manager.shutdown_async()

    @pytest.mark.asyncio()
    async def test_shutdown_async_stops_governor(self, governed_manager):
        """Test that shutdown_async() properly shuts down governor."""
        await governed_manager.start()
        await governed_manager.shutdown_async()
        # Governor should be shut down (monitor task stopped if any)
        assert governed_manager._governor._shutdown_event.is_set()


class TestGovernedModelManagerGetStatus:
    """Tests for get_status method."""

    def test_get_status_includes_governor_info(self, governed_manager):
        """Test that get_status includes governor information."""
        status = governed_manager.get_status()
        assert "governor" in status
        assert "memory_stats" in status["governor"]
        assert "loaded_models" in status["governor"]
        assert "eviction_history_count" in status["governor"]

    def test_get_status_includes_offloaded_models(self, governed_manager):
        """Test that get_status includes offloaded model information."""
        status = governed_manager.get_status()
        assert "offloaded_models" in status
        assert isinstance(status["offloaded_models"], list)


class TestGovernedModelManagerModelKeyParsing:
    """Tests for model key parsing."""

    def test_parse_valid_model_key(self, governed_manager):
        """Test parsing a valid model key."""
        key = governed_manager._get_model_key("Qwen/Qwen3-Embedding-0.6B", "float16")
        parsed = governed_manager._parse_model_key(key)
        assert parsed is not None
        assert parsed[0] == "Qwen/Qwen3-Embedding-0.6B"
        assert parsed[1] == "float16"

    def test_parse_empty_model_key(self, governed_manager):
        """Test parsing an empty key returns None."""
        parsed = governed_manager._parse_model_key("")
        assert parsed is None

    def test_parse_invalid_model_key_format(self, governed_manager):
        """Test parsing key without separator returns None."""
        parsed = governed_manager._parse_model_key("no-separator-here")
        assert parsed is None

    def test_get_model_key_rejects_underscore_in_quantization(self, governed_manager):
        """Test that quantization with underscore is rejected."""
        with pytest.raises(ValueError, match="cannot contain underscore"):
            governed_manager._get_model_key("model", "float_16")


class TestModelRestoreErrorStateMismatch:
    """Tests for ModelRestoreError when governor and offloader state diverges."""

    @pytest.mark.asyncio()
    async def test_restore_raises_error_on_state_mismatch_embedding(self, governed_manager):
        """ModelRestoreError raised when governor thinks model is offloaded but offloader doesn't have it."""
        # Mock provider to exist (so we don't get the "provider is None" error)
        governed_manager._provider = type("MockProvider", (), {"model": object()})()

        # Call the offload callback with target_device="cuda" without actually offloading first
        # This simulates state mismatch where governor thinks model is offloaded but offloader doesn't have it
        # Note: Use Exception match since ModelRestoreError class identity varies by import path
        with pytest.raises(Exception, match="state mismatch between governor and offloader") as exc_info:
            await governed_manager._governor_offload_embedding(
                model_name="Qwen/test-model",
                quantization="float16",
                target_device="cuda",
            )

        assert "ModelRestoreError" in type(exc_info.value).__name__
        assert "embedding:Qwen/test-model:float16" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_restore_raises_error_on_state_mismatch_reranker(self, governed_manager):
        """ModelRestoreError raised for reranker when state diverges."""
        # Mock reranker to exist
        governed_manager.reranker = type("MockReranker", (), {"model": object()})()

        # Note: Use Exception match since ModelRestoreError class identity varies by import path
        with pytest.raises(Exception, match="state mismatch between governor and offloader") as exc_info:
            await governed_manager._governor_offload_reranker(
                model_name="Qwen/test-reranker",
                quantization="float16",
                target_device="cuda",
            )

        assert "ModelRestoreError" in type(exc_info.value).__name__
        assert "reranker:Qwen/test-reranker:float16" in str(exc_info.value)


class TestGovernedModelManagerCriticalFailures:
    """Tests for critical failure tracking and scheduling helpers."""

    def test_schedule_governor_coro_records_critical_failure(self, governed_manager):
        """Critical scheduling errors should be recorded and raised."""

        async def _boom():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            governed_manager._schedule_governor_coro(
                _boom(),
                critical=True,
                description="critical-op",
            )

        failures = governed_manager.get_critical_failures()
        assert failures
        assert failures[0]["description"] == "critical-op"

    def test_schedule_governor_coro_noncritical_does_not_raise(self, governed_manager):
        """Non-critical scheduling errors should be logged but not raised."""

        async def _boom():
            raise RuntimeError("boom")

        governed_manager._schedule_governor_coro(_boom(), critical=False, description="noncritical-op")

        assert governed_manager.has_critical_failures() is False

    def test_clear_critical_failures(self, governed_manager):
        """Clearing critical failures returns the count and empties the list."""
        governed_manager._record_critical_failure("test", ValueError("boom"))

        cleared = governed_manager.clear_critical_failures()

        assert cleared == 1
        assert governed_manager.has_critical_failures() is False

    def test_run_governor_coro_returns_value(self, governed_manager):
        """_run_governor_coro should return the coroutine result when no loop is running."""

        async def _return_value():
            return "ok"

        assert governed_manager._run_governor_coro(_return_value()) == "ok"


class TestGovernedModelManagerEnsureProviderInitialized:
    """Tests for the core embedding provider init path with governor tracking."""

    @pytest.mark.asyncio()
    async def test_denied_memory_request_raises(self, governed_manager, monkeypatch):
        monkeypatch.setattr(gmm, "get_model_memory_requirement", lambda *_args, **_kwargs: 123)
        monkeypatch.setattr(governed_manager._governor, "request_memory", AsyncMock(return_value=False))

        with pytest.raises(RuntimeError, match="Cannot allocate memory for model"):
            await governed_manager._ensure_provider_initialized("Qwen/test-model", "float16")

    @pytest.mark.asyncio()
    async def test_fast_path_touches_governor_and_skips_parent_load(self, governed_manager, monkeypatch):
        model_name = "Qwen/test-model"
        quantization = "float16"
        model_key = governed_manager._get_model_key(model_name, quantization)
        provider = object()

        governed_manager._provider = provider
        governed_manager.current_model_key = model_key

        monkeypatch.setattr(gmm, "get_model_memory_requirement", lambda *_args, **_kwargs: 1)
        monkeypatch.setattr(governed_manager._governor, "request_memory", AsyncMock(return_value=True))
        touch = AsyncMock()
        monkeypatch.setattr(governed_manager._governor, "touch", touch)

        parent_ensure = AsyncMock()
        monkeypatch.setattr(gmm.ModelManager, "_ensure_provider_initialized", parent_ensure)

        result = await governed_manager._ensure_provider_initialized(model_name, quantization)

        assert result is provider
        touch.assert_awaited_once()
        call = touch.await_args
        assert call.args[0] == model_name
        assert call.args[1].value == ModelType.EMBEDDING.value
        assert call.args[2] == quantization
        parent_ensure.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_load_success_marks_loaded(self, governed_manager, monkeypatch):
        model_name = "Qwen/test-model"
        quantization = "float16"
        provider = types.SimpleNamespace(model=object())

        monkeypatch.setattr(gmm, "get_model_memory_requirement", lambda *_args, **_kwargs: 1)
        monkeypatch.setattr(governed_manager._governor, "request_memory", AsyncMock(return_value=True))
        mark_loaded = AsyncMock()
        monkeypatch.setattr(governed_manager._governor, "mark_loaded", mark_loaded)
        monkeypatch.setattr(governed_manager._governor, "mark_unloaded", AsyncMock())

        monkeypatch.setattr(gmm.ModelManager, "_ensure_provider_initialized", AsyncMock(return_value=provider))

        result = await governed_manager._ensure_provider_initialized(model_name, quantization)

        assert result is provider
        mark_loaded.assert_awaited_once()
        _, kwargs = mark_loaded.await_args
        assert kwargs["model_name"] == model_name
        assert kwargs["model_type"].value == ModelType.EMBEDDING.value
        assert kwargs["quantization"] == quantization
        assert kwargs["model_ref"] is provider.model

    @pytest.mark.asyncio()
    async def test_load_failure_marks_unloaded_and_reraises(self, governed_manager, monkeypatch):
        model_name = "Qwen/test-model"
        quantization = "float16"

        monkeypatch.setattr(gmm, "get_model_memory_requirement", lambda *_args, **_kwargs: 1)
        monkeypatch.setattr(governed_manager._governor, "request_memory", AsyncMock(return_value=True))
        mark_unloaded = AsyncMock()
        monkeypatch.setattr(governed_manager._governor, "mark_unloaded", mark_unloaded)

        async def _boom(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(gmm.ModelManager, "_ensure_provider_initialized", _boom)

        with pytest.raises(RuntimeError, match="boom"):
            await governed_manager._ensure_provider_initialized(model_name, quantization)

        mark_unloaded.assert_awaited_once()
        call = mark_unloaded.await_args
        assert call.args[0] == model_name
        assert call.args[1].value == ModelType.EMBEDDING.value
        assert call.args[2] == quantization

    @pytest.mark.asyncio()
    async def test_switching_models_discards_offloader_and_clears_cuda_cache(self, governed_manager, monkeypatch):
        old_model = "Qwen/old-model"
        old_quantization = "float16"
        new_model = "Qwen/new-model"
        new_quantization = "float16"

        governed_manager._provider = object()
        governed_manager.current_model_key = governed_manager._get_model_key(old_model, old_quantization)

        # Avoid importing real torch in unit tests.
        torch_mod = types.ModuleType("torch")
        empty_cache = MagicMock()
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=empty_cache)
        monkeypatch.setitem(sys.modules, "torch", torch_mod)

        monkeypatch.setattr(gmm, "get_model_memory_requirement", lambda *_args, **_kwargs: 1)
        monkeypatch.setattr(governed_manager._governor, "request_memory", AsyncMock(return_value=True))
        mark_unloaded = AsyncMock()
        monkeypatch.setattr(governed_manager._governor, "mark_unloaded", mark_unloaded)
        monkeypatch.setattr(governed_manager._governor, "mark_loaded", AsyncMock())

        governed_manager._offloader.discard = MagicMock()
        monkeypatch.setattr(
            gmm.ModelManager, "_ensure_provider_initialized", AsyncMock(return_value=types.SimpleNamespace())
        )

        await governed_manager._ensure_provider_initialized(new_model, new_quantization)

        governed_manager._offloader.discard.assert_called_once_with(f"embedding:{old_model}:{old_quantization}")
        mark_unloaded.assert_awaited_once()
        call = mark_unloaded.await_args
        assert call.args[0] == old_model
        assert call.args[1].value == ModelType.EMBEDDING.value
        assert call.args[2] == old_quantization
        empty_cache.assert_called_once()
