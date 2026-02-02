"""Tests for GovernedModelManager.

Tests the integration between ModelManager, GPUMemoryGovernor, and CPUOffloader.
"""

import asyncio
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
        gpu_max_percent=0.90,
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


class TestGovernedModelManagerLocks:
    """Tests for per-model lock acquisition."""

    def test_get_embedding_lock_creates_new_lock(self, governed_manager):
        """First call creates a new asyncio.Lock."""
        lock = governed_manager._get_embedding_lock("model_key")
        assert isinstance(lock, asyncio.Lock)

    def test_get_embedding_lock_returns_same_lock(self, governed_manager):
        """Subsequent calls return the same lock instance."""
        lock1 = governed_manager._get_embedding_lock("model_key")
        lock2 = governed_manager._get_embedding_lock("model_key")
        assert lock1 is lock2

    def test_get_embedding_lock_different_keys_different_locks(self, governed_manager):
        """Different model keys get different locks."""
        lock1 = governed_manager._get_embedding_lock("key1")
        lock2 = governed_manager._get_embedding_lock("key2")
        assert lock1 is not lock2

    def test_get_reranker_lock_creates_new_lock(self, governed_manager):
        """First call creates a new asyncio.Lock for reranker."""
        lock = governed_manager._get_reranker_lock("model_key")
        assert isinstance(lock, asyncio.Lock)

    def test_get_reranker_lock_returns_same_lock(self, governed_manager):
        """Subsequent calls return the same lock instance for reranker."""
        lock1 = governed_manager._get_reranker_lock("model_key")
        lock2 = governed_manager._get_reranker_lock("model_key")
        assert lock1 is lock2


class TestGovernedModelManagerUnloadModelAsync:
    """Tests for async model unloading."""

    @pytest.mark.asyncio()
    async def test_unload_model_async_notifies_governor(self, governed_manager, monkeypatch):
        """Unloading a model notifies governor of unload."""
        model_name = "Qwen/test-model"
        quantization = "float16"
        governed_manager.current_model_key = governed_manager._get_model_key(model_name, quantization)

        mark_unloaded = AsyncMock()
        monkeypatch.setattr(governed_manager._governor, "mark_unloaded", mark_unloaded)
        monkeypatch.setattr(gmm.ModelManager, "unload_model_async", AsyncMock())

        await governed_manager.unload_model_async()

        mark_unloaded.assert_awaited_once()
        call = mark_unloaded.await_args
        assert call.args[0] == model_name
        assert call.args[1].value == ModelType.EMBEDDING.value
        assert call.args[2] == quantization

    @pytest.mark.asyncio()
    async def test_unload_model_async_skips_if_no_model(self, governed_manager, monkeypatch):
        """Unloading when no model loaded skips governor notification."""
        governed_manager.current_model_key = None

        mark_unloaded = AsyncMock()
        monkeypatch.setattr(governed_manager._governor, "mark_unloaded", mark_unloaded)
        monkeypatch.setattr(gmm.ModelManager, "unload_model_async", AsyncMock())

        await governed_manager.unload_model_async()

        mark_unloaded.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_unload_model_async_handles_invalid_key(self, governed_manager, monkeypatch):
        """Unloading with unparseable key logs warning but doesn't crash."""
        governed_manager.current_model_key = "invalid-no-underscore"

        mark_unloaded = AsyncMock()
        monkeypatch.setattr(governed_manager._governor, "mark_unloaded", mark_unloaded)
        monkeypatch.setattr(gmm.ModelManager, "unload_model_async", AsyncMock())

        await governed_manager.unload_model_async()

        mark_unloaded.assert_not_awaited()


class TestGovernedModelManagerEnsureRerankerLoaded:
    """Tests for reranker loading with governor integration."""

    def test_ensure_reranker_loaded_mock_mode_returns_true(self, governed_manager):
        """Mock mode returns True without loading."""
        governed_manager.is_mock_mode = True
        result = governed_manager.ensure_reranker_loaded("model", "float16")
        assert result is True

    def test_ensure_reranker_loaded_denied_memory_raises(self, governed_manager, monkeypatch):
        """Denied memory allocation raises InsufficientMemoryError."""
        governed_manager.is_mock_mode = False
        monkeypatch.setattr(gmm, "get_model_memory_requirement", lambda *_args, **_kwargs: 500)
        monkeypatch.setattr(governed_manager, "_run_governor_coro", MagicMock(return_value=False))

        with pytest.raises(Exception, match="Cannot allocate memory for reranker"):
            governed_manager.ensure_reranker_loaded("model", "float16")

    def test_ensure_reranker_loaded_fast_path_touches_and_returns(self, governed_manager, monkeypatch):
        """Fast path when reranker already loaded touches governor and returns."""
        model_name = "Qwen/reranker"
        quantization = "float16"
        model_key = governed_manager._get_model_key(model_name, quantization)

        governed_manager.is_mock_mode = False
        governed_manager.current_reranker_key = model_key
        governed_manager.reranker = MagicMock()
        governed_manager._current_reranker_key = model_key

        monkeypatch.setattr(gmm, "get_model_memory_requirement", lambda *_args, **_kwargs: 100)
        monkeypatch.setattr(governed_manager, "_run_governor_coro", MagicMock(return_value=True))
        schedule_coro = MagicMock()
        monkeypatch.setattr(governed_manager, "_schedule_governor_coro", schedule_coro)

        result = governed_manager.ensure_reranker_loaded(model_name, quantization)

        assert result is True
        schedule_coro.assert_called_once()

    def test_ensure_reranker_loaded_success_marks_loaded(self, governed_manager, monkeypatch):
        """Successful load marks model as loaded in governor."""
        model_name = "Qwen/reranker"
        quantization = "float16"

        governed_manager.is_mock_mode = False
        governed_manager.current_reranker_key = None
        governed_manager._current_reranker_key = None
        governed_manager.reranker = types.SimpleNamespace(model=object())

        monkeypatch.setattr(gmm, "get_model_memory_requirement", lambda *_args, **_kwargs: 100)
        run_coro = MagicMock(return_value=True)
        monkeypatch.setattr(governed_manager, "_run_governor_coro", run_coro)
        monkeypatch.setattr(governed_manager, "_schedule_governor_coro", MagicMock())
        monkeypatch.setattr(gmm.ModelManager, "ensure_reranker_loaded", MagicMock(return_value=True))

        result = governed_manager.ensure_reranker_loaded(model_name, quantization)

        assert result is True
        assert governed_manager._current_reranker_key == governed_manager._get_model_key(model_name, quantization)
        # Verify mark_loaded was called (via _run_governor_coro)
        assert run_coro.call_count >= 2  # request_memory and mark_loaded

    def test_ensure_reranker_loaded_failure_marks_unloaded(self, governed_manager, monkeypatch):
        """Load failure schedules mark_unloaded and re-raises."""
        model_name = "Qwen/reranker"
        quantization = "float16"

        governed_manager.is_mock_mode = False
        governed_manager.current_reranker_key = None
        governed_manager._current_reranker_key = None

        monkeypatch.setattr(gmm, "get_model_memory_requirement", lambda *_args, **_kwargs: 100)
        monkeypatch.setattr(governed_manager, "_run_governor_coro", MagicMock(return_value=True))
        schedule_coro = MagicMock()
        monkeypatch.setattr(governed_manager, "_schedule_governor_coro", schedule_coro)
        monkeypatch.setattr(gmm.ModelManager, "ensure_reranker_loaded", MagicMock(side_effect=RuntimeError("boom")))

        with pytest.raises(RuntimeError, match="boom"):
            governed_manager.ensure_reranker_loaded(model_name, quantization)

        # mark_unloaded should be scheduled as critical
        schedule_coro.assert_called_once()
        assert schedule_coro.call_args.kwargs.get("critical") is True


class TestGovernedModelManagerUnloadReranker:
    """Tests for reranker unloading."""

    def test_unload_reranker_notifies_governor(self, governed_manager, monkeypatch):
        """Unloading reranker schedules governor notification."""
        model_name = "Qwen/reranker"
        quantization = "float16"
        model_key = governed_manager._get_model_key(model_name, quantization)
        # Set both the parent's tracking and our internal tracking
        governed_manager.current_reranker_key = model_key
        governed_manager._current_reranker_key = model_key

        schedule_coro = MagicMock()
        monkeypatch.setattr(governed_manager, "_schedule_governor_coro", schedule_coro)
        monkeypatch.setattr(gmm.ModelManager, "unload_reranker", MagicMock())

        governed_manager.unload_reranker()

        schedule_coro.assert_called_once()
        assert schedule_coro.call_args.kwargs.get("critical") is True
        assert governed_manager._current_reranker_key is None

    def test_unload_reranker_skips_if_no_reranker(self, governed_manager, monkeypatch):
        """Unloading when no reranker loaded skips governor notification."""
        governed_manager._current_reranker_key = None

        schedule_coro = MagicMock()
        monkeypatch.setattr(governed_manager, "_schedule_governor_coro", schedule_coro)
        monkeypatch.setattr(gmm.ModelManager, "unload_reranker", MagicMock())

        governed_manager.unload_reranker()

        schedule_coro.assert_not_called()


class TestGovernedModelManagerGovernorUnloadCallbacks:
    """Tests for governor unload callback execution."""

    @pytest.mark.asyncio()
    async def test_governor_unload_embedding_success(self, governed_manager, monkeypatch):
        """Governor callback successfully unloads matching embedding model."""
        model_name = "Qwen/test-model"
        quantization = "float16"
        model_key = governed_manager._get_model_key(model_name, quantization)

        governed_manager.current_model_key = model_key
        governed_manager._provider = object()

        discard = MagicMock()
        governed_manager._offloader.discard = discard
        parent_unload = AsyncMock()
        monkeypatch.setattr(gmm.ModelManager, "unload_model_async", parent_unload)

        await governed_manager._governor_unload_embedding(model_name, quantization)

        discard.assert_called_once_with(f"embedding:{model_name}:{quantization}")
        parent_unload.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_governor_unload_embedding_skips_mismatch(self, governed_manager, monkeypatch):
        """Governor callback skips unload if model key doesn't match."""
        governed_manager.current_model_key = "other_float16"

        parent_unload = AsyncMock()
        monkeypatch.setattr(gmm.ModelManager, "unload_model_async", parent_unload)

        await governed_manager._governor_unload_embedding("different-model", "float16")

        parent_unload.assert_not_awaited()

    @pytest.mark.asyncio()
    async def test_governor_unload_reranker_success(self, governed_manager, monkeypatch):
        """Governor callback successfully unloads matching reranker."""
        model_name = "Qwen/reranker"
        quantization = "float16"
        model_key = governed_manager._get_model_key(model_name, quantization)

        governed_manager.current_reranker_key = model_key
        governed_manager.reranker = object()

        discard = MagicMock()
        governed_manager._offloader.discard = discard
        parent_unload = MagicMock()
        monkeypatch.setattr(gmm.ModelManager, "unload_reranker", parent_unload)

        await governed_manager._governor_unload_reranker(model_name, quantization)

        discard.assert_called_once_with(f"reranker:{model_name}:{quantization}")
        parent_unload.assert_called_once()

    @pytest.mark.asyncio()
    async def test_governor_unload_reranker_skips_mismatch(self, governed_manager, monkeypatch):
        """Governor callback skips unload if reranker key doesn't match."""
        governed_manager.current_reranker_key = "other_float16"

        parent_unload = MagicMock()
        monkeypatch.setattr(gmm.ModelManager, "unload_reranker", parent_unload)

        await governed_manager._governor_unload_reranker("different-model", "float16")

        parent_unload.assert_not_called()


class TestGovernedModelManagerGovernorOffloadCallbacks:
    """Tests for governor offload callback execution."""

    @pytest.mark.asyncio()
    async def test_governor_offload_embedding_to_cpu(self, governed_manager, monkeypatch):
        """Offload embedding model to CPU successfully."""
        model_name = "Qwen/test-model"
        quantization = "float16"
        mock_model = object()

        governed_manager._provider = types.SimpleNamespace(model=mock_model)

        offload_to_cpu = MagicMock()
        governed_manager._offloader.offload_to_cpu = offload_to_cpu

        await governed_manager._governor_offload_embedding(model_name, quantization, "cpu")

        offload_to_cpu.assert_called_once_with(f"embedding:{model_name}:{quantization}", mock_model)

    @pytest.mark.asyncio()
    async def test_governor_offload_embedding_to_cpu_no_provider_raises(self, governed_manager):
        """Offload to CPU raises if provider is None."""
        governed_manager._provider = None

        with pytest.raises(RuntimeError, match="provider is None"):
            await governed_manager._governor_offload_embedding("model", "float16", "cpu")

    @pytest.mark.asyncio()
    async def test_governor_offload_embedding_to_cpu_no_model_attr_raises(self, governed_manager):
        """Offload to CPU raises if provider has no model attribute."""
        governed_manager._provider = types.SimpleNamespace()

        with pytest.raises(RuntimeError, match="has no model attribute"):
            await governed_manager._governor_offload_embedding("model", "float16", "cpu")

    @pytest.mark.asyncio()
    async def test_governor_offload_embedding_to_cuda_success(self, governed_manager, monkeypatch):
        """Restore embedding model to GPU successfully."""
        model_name = "Qwen/test-model"
        quantization = "float16"
        offloader_key = f"embedding:{model_name}:{quantization}"

        governed_manager._provider = types.SimpleNamespace(model=object())

        governed_manager._offloader.is_offloaded = MagicMock(return_value=True)
        restore_to_gpu = MagicMock()
        governed_manager._offloader.restore_to_gpu = restore_to_gpu

        await governed_manager._governor_offload_embedding(model_name, quantization, "cuda")

        restore_to_gpu.assert_called_once_with(offloader_key)

    @pytest.mark.asyncio()
    async def test_governor_offload_reranker_to_cpu(self, governed_manager, monkeypatch):
        """Offload reranker to CPU successfully."""
        model_name = "Qwen/reranker"
        quantization = "float16"
        mock_model = object()

        governed_manager.reranker = types.SimpleNamespace(model=mock_model)

        offload_to_cpu = MagicMock()
        governed_manager._offloader.offload_to_cpu = offload_to_cpu

        await governed_manager._governor_offload_reranker(model_name, quantization, "cpu")

        offload_to_cpu.assert_called_once_with(f"reranker:{model_name}:{quantization}", mock_model)

    @pytest.mark.asyncio()
    async def test_governor_offload_reranker_to_cpu_no_reranker_raises(self, governed_manager):
        """Offload to CPU raises if reranker is None."""
        governed_manager.reranker = None

        with pytest.raises(RuntimeError, match="reranker is None"):
            await governed_manager._governor_offload_reranker("model", "float16", "cpu")

    @pytest.mark.asyncio()
    async def test_governor_offload_reranker_to_cuda_success(self, governed_manager, monkeypatch):
        """Restore reranker to GPU successfully."""
        model_name = "Qwen/reranker"
        quantization = "float16"
        offloader_key = f"reranker:{model_name}:{quantization}"

        governed_manager.reranker = types.SimpleNamespace(model=object())

        governed_manager._offloader.is_offloaded = MagicMock(return_value=True)
        restore_to_gpu = MagicMock()
        governed_manager._offloader.restore_to_gpu = restore_to_gpu

        await governed_manager._governor_offload_reranker(model_name, quantization, "cuda")

        restore_to_gpu.assert_called_once_with(offloader_key)


class TestGovernedModelManagerGenerateEmbeddingAsync:
    """Tests for async embedding generation."""

    @pytest.mark.asyncio()
    async def test_generate_embedding_async_success(self, governed_manager, monkeypatch):
        """Successfully generates single embedding with lock protection."""
        import numpy as np

        model_name = "Qwen/test-model"
        quantization = "float16"

        mock_embedding = np.array([0.1, 0.2, 0.3])
        mock_provider = AsyncMock()
        mock_provider.embed_single = AsyncMock(return_value=mock_embedding)

        monkeypatch.setattr(governed_manager, "_ensure_provider_initialized", AsyncMock(return_value=mock_provider))
        monkeypatch.setattr(governed_manager, "_schedule_unload", AsyncMock())

        # Mock torch to avoid GPU dependencies
        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", torch_mod)

        result = await governed_manager.generate_embedding_async(
            text="test text",
            model_name=model_name,
            quantization=quantization,
            mode="query",
        )

        assert result == [0.1, 0.2, 0.3]
        mock_provider.embed_single.assert_awaited_once()

    @pytest.mark.asyncio()
    async def test_generate_embedding_async_document_mode(self, governed_manager, monkeypatch):
        """Correctly passes document mode to provider."""
        import numpy as np

        mock_provider = AsyncMock()
        mock_provider.embed_single = AsyncMock(return_value=np.array([0.1]))

        monkeypatch.setattr(governed_manager, "_ensure_provider_initialized", AsyncMock(return_value=mock_provider))
        monkeypatch.setattr(governed_manager, "_schedule_unload", AsyncMock())

        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", torch_mod)

        from shared.embedding.types import EmbeddingMode

        await governed_manager.generate_embedding_async("text", "model", "float16", mode="document")

        call = mock_provider.embed_single.await_args
        assert call.kwargs.get("mode") == EmbeddingMode.DOCUMENT


class TestGovernedModelManagerGenerateEmbeddingsBatchAsync:
    """Tests for async batch embedding generation."""

    @pytest.mark.asyncio()
    async def test_generate_embeddings_batch_async_success(self, governed_manager, monkeypatch):
        """Successfully generates batch embeddings with lock protection."""
        import numpy as np

        model_name = "Qwen/test-model"
        quantization = "float16"

        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_provider = AsyncMock()
        mock_provider.embed_texts = AsyncMock(return_value=mock_embeddings)

        monkeypatch.setattr(governed_manager, "_ensure_provider_initialized", AsyncMock(return_value=mock_provider))
        monkeypatch.setattr(governed_manager, "_schedule_unload", AsyncMock())

        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", torch_mod)

        result = await governed_manager.generate_embeddings_batch_async(
            texts=["text1", "text2"],
            model_name=model_name,
            quantization=quantization,
            batch_size=32,
            mode="query",
        )

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_provider.embed_texts.assert_awaited_once()


class TestGovernedModelManagerRerankAsync:
    """Tests for async reranking."""

    @pytest.mark.asyncio()
    async def test_rerank_async_mock_mode(self, governed_manager, monkeypatch):
        """Mock mode returns deterministic fake scores."""
        governed_manager.is_mock_mode = True

        monkeypatch.setattr(governed_manager, "ensure_reranker_loaded", MagicMock(return_value=True))
        monkeypatch.setattr(governed_manager, "_schedule_reranker_unload", AsyncMock())

        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", torch_mod)

        result = await governed_manager.rerank_async(
            query="test query",
            documents=["doc1", "doc2", "doc3"],
            top_k=2,
            model_name="reranker",
            quantization="float16",
        )

        assert len(result) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)

    @pytest.mark.asyncio()
    async def test_rerank_async_failed_load_raises(self, governed_manager, monkeypatch):
        """Raises if reranker load fails."""
        governed_manager.is_mock_mode = False
        monkeypatch.setattr(governed_manager, "ensure_reranker_loaded", MagicMock(return_value=False))

        with pytest.raises(RuntimeError, match="Failed to load reranker"):
            await governed_manager.rerank_async(
                query="test", documents=["doc"], top_k=1, model_name="model", quantization="float16"
            )


class TestGovernedModelManagerPreloadModels:
    """Tests for model preloading."""

    @pytest.mark.asyncio()
    async def test_preload_models_embedding_success(self, governed_manager, monkeypatch):
        """Successfully preloads embedding model."""
        monkeypatch.setattr(gmm, "get_model_memory_requirement", lambda *_args, **_kwargs: 100)
        monkeypatch.setattr(governed_manager._governor, "request_memory", AsyncMock(return_value=True))
        monkeypatch.setattr(governed_manager, "_ensure_provider_initialized", AsyncMock())

        result = await governed_manager.preload_models([("Qwen/test", "embedding", "float16")])

        assert result["embedding:Qwen/test:float16"] is True

    @pytest.mark.asyncio()
    async def test_preload_models_memory_denied(self, governed_manager, monkeypatch):
        """Reports error when memory allocation denied."""
        monkeypatch.setattr(gmm, "get_model_memory_requirement", lambda *_args, **_kwargs: 10000)
        monkeypatch.setattr(governed_manager._governor, "request_memory", AsyncMock(return_value=False))
        monkeypatch.setattr(governed_manager._governor, "get_memory_stats", MagicMock(return_value={}))

        result = await governed_manager.preload_models([("Qwen/large", "embedding", "float16")])

        assert "Memory allocation failed" in result["embedding:Qwen/large:float16"]

    @pytest.mark.asyncio()
    async def test_preload_models_load_failure(self, governed_manager, monkeypatch):
        """Reports error when model load fails."""
        monkeypatch.setattr(gmm, "get_model_memory_requirement", lambda *_args, **_kwargs: 100)
        monkeypatch.setattr(governed_manager._governor, "request_memory", AsyncMock(return_value=True))
        monkeypatch.setattr(
            governed_manager, "_ensure_provider_initialized", AsyncMock(side_effect=RuntimeError("boom"))
        )

        result = await governed_manager.preload_models([("Qwen/test", "embedding", "float16")])

        assert "Load failed" in result["embedding:Qwen/test:float16"]

    @pytest.mark.asyncio()
    async def test_preload_models_reranker_reserves_only(self, governed_manager, monkeypatch):
        """Rerankers reserve memory but don't load until first use."""
        monkeypatch.setattr(gmm, "get_model_memory_requirement", lambda *_args, **_kwargs: 100)
        monkeypatch.setattr(governed_manager._governor, "request_memory", AsyncMock(return_value=True))

        result = await governed_manager.preload_models([("Qwen/reranker", "reranker", "float16")])

        assert result["reranker:Qwen/reranker:float16"] is True


class TestGovernedModelManagerSyncShutdown:
    """Tests for synchronous shutdown path."""

    def test_shutdown_clears_executor(self, governed_manager, monkeypatch):
        """Sync shutdown clears the executor."""
        # Force executor creation
        executor = governed_manager._get_sync_executor()
        assert governed_manager._sync_executor is executor

        monkeypatch.setattr(governed_manager._governor, "shutdown", AsyncMock())
        governed_manager._offloader.clear = MagicMock()
        monkeypatch.setattr(gmm.ModelManager, "shutdown", MagicMock())

        governed_manager.shutdown()

        assert governed_manager._sync_executor is None

    def test_shutdown_clears_offloader(self, governed_manager, monkeypatch):
        """Sync shutdown clears the offloader."""
        clear = MagicMock()
        governed_manager._offloader.clear = clear

        monkeypatch.setattr(governed_manager._governor, "shutdown", AsyncMock())
        monkeypatch.setattr(gmm.ModelManager, "shutdown", MagicMock())

        governed_manager.shutdown()

        clear.assert_called_once()

    def test_shutdown_calls_parent(self, governed_manager, monkeypatch):
        """Sync shutdown calls parent shutdown."""
        monkeypatch.setattr(governed_manager._governor, "shutdown", AsyncMock())
        governed_manager._offloader.clear = MagicMock()
        parent_shutdown = MagicMock()
        monkeypatch.setattr(gmm.ModelManager, "shutdown", parent_shutdown)

        governed_manager.shutdown()

        parent_shutdown.assert_called_once()


class TestGovernedModelManagerGetSyncExecutor:
    """Tests for sync executor management."""

    def test_get_sync_executor_creates_on_first_call(self, governed_manager):
        """First call creates executor."""
        assert governed_manager._sync_executor is None
        executor = governed_manager._get_sync_executor()
        assert executor is not None
        assert governed_manager._sync_executor is executor
        # Clean up
        executor.shutdown(wait=False)

    def test_get_sync_executor_returns_same_instance(self, governed_manager):
        """Subsequent calls return same executor."""
        executor1 = governed_manager._get_sync_executor()
        executor2 = governed_manager._get_sync_executor()
        assert executor1 is executor2
        # Clean up
        executor1.shutdown(wait=False)


class TestGovernedModelManagerCriticalFailuresBounded:
    """Tests for bounded critical failure tracking."""

    def test_critical_failures_bounded_to_max(self, governed_manager):
        """Critical failures list is bounded to max size."""
        max_failures = governed_manager._max_critical_failures

        # Record more than max failures
        for i in range(max_failures + 5):
            governed_manager._record_critical_failure(f"failure-{i}", ValueError(f"error-{i}"))

        failures = governed_manager.get_critical_failures()
        assert len(failures) == max_failures
        # Should keep the most recent ones
        assert failures[-1]["description"] == f"failure-{max_failures + 4}"


class TestGovernedModelManagerRerankerSwitch:
    """Tests for switching between different rerankers."""

    def test_ensure_reranker_loaded_switch_discards_old_offloader(self, governed_manager, monkeypatch):
        """Switching rerankers discards old from offloader and marks unloaded."""
        old_model = "Qwen/old-reranker"
        old_quantization = "float16"
        new_model = "Qwen/new-reranker"
        new_quantization = "float16"

        old_key = governed_manager._get_model_key(old_model, old_quantization)
        governed_manager.is_mock_mode = False
        governed_manager.current_reranker_key = old_key
        governed_manager._current_reranker_key = old_key
        governed_manager.reranker = types.SimpleNamespace(model=object())

        monkeypatch.setattr(gmm, "get_model_memory_requirement", lambda *_args, **_kwargs: 100)
        monkeypatch.setattr(governed_manager, "_run_governor_coro", MagicMock(return_value=True))
        schedule_coro = MagicMock()
        monkeypatch.setattr(governed_manager, "_schedule_governor_coro", schedule_coro)

        discard = MagicMock()
        governed_manager._offloader.discard = discard

        monkeypatch.setattr(gmm.ModelManager, "ensure_reranker_loaded", MagicMock(return_value=True))

        governed_manager.ensure_reranker_loaded(new_model, new_quantization)

        discard.assert_called_once_with(f"reranker:{old_model}:{old_quantization}")
        # Check critical mark_unloaded was scheduled
        assert any(call.kwargs.get("critical") for call in schedule_coro.call_args_list)


class TestGovernedModelManagerUnloadRerankerInvalidKey:
    """Tests for unload_reranker with invalid key."""

    def test_unload_reranker_invalid_key_logs_warning(self, governed_manager, monkeypatch):
        """Invalid reranker key logs warning but doesn't crash."""
        governed_manager.current_reranker_key = "invalid-no-underscore"
        governed_manager._current_reranker_key = "invalid-no-underscore"

        schedule_coro = MagicMock()
        monkeypatch.setattr(governed_manager, "_schedule_governor_coro", schedule_coro)
        monkeypatch.setattr(gmm.ModelManager, "unload_reranker", MagicMock())

        governed_manager.unload_reranker()

        # Should not schedule since key couldn't be parsed
        schedule_coro.assert_not_called()
        # Internal tracking should still be cleared
        assert governed_manager._current_reranker_key is None


class TestGovernedModelManagerStatusWithCriticalFailures:
    """Tests for get_status including critical failures."""

    def test_get_status_includes_critical_failures_info(self, governed_manager):
        """Status includes critical failure tracking information."""
        governed_manager._record_critical_failure("test-failure", ValueError("boom"))

        status = governed_manager.get_status()

        assert "critical_failures" in status
        assert status["critical_failures"]["has_failures"] is True
        assert len(status["critical_failures"]["failures"]) == 1


class TestGovernedModelManagerRunGovernorCoroWithLoop:
    """Tests for _run_governor_coro with running event loop."""

    @pytest.mark.asyncio()
    async def test_run_governor_coro_with_running_loop(self, governed_manager):
        """Runs coroutine in separate thread when loop already running."""
        # We're already in an async context, so there's a running loop

        async def _async_work():
            return "result"

        # This should work from within an async context
        result = governed_manager._run_governor_coro(_async_work())
        assert result == "result"

    @pytest.mark.asyncio()
    async def test_run_governor_coro_error_propagates(self, governed_manager):
        """Errors from coroutine propagate through."""

        async def _boom():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            governed_manager._run_governor_coro(_boom())


class TestGovernedModelManagerShutdownAsyncCleanup:
    """Tests for shutdown_async cleanup."""

    @pytest.mark.asyncio()
    async def test_shutdown_async_clears_executor(self, governed_manager, monkeypatch):
        """shutdown_async clears executor."""
        # Force executor creation
        _ = governed_manager._get_sync_executor()
        assert governed_manager._sync_executor is not None

        monkeypatch.setattr(governed_manager._governor, "shutdown", AsyncMock())
        governed_manager._offloader.clear = MagicMock()
        monkeypatch.setattr(gmm.ModelManager, "shutdown", MagicMock())

        await governed_manager.shutdown_async()

        assert governed_manager._sync_executor is None

    @pytest.mark.asyncio()
    async def test_shutdown_async_handles_governor_error(self, governed_manager, monkeypatch):
        """shutdown_async continues even if governor shutdown fails."""
        monkeypatch.setattr(governed_manager._governor, "shutdown", AsyncMock(side_effect=RuntimeError("boom")))
        governed_manager._offloader.clear = MagicMock()
        monkeypatch.setattr(gmm.ModelManager, "shutdown", MagicMock())

        # Should not raise
        await governed_manager.shutdown_async()

        # Offloader and parent shutdown should still be called
        governed_manager._offloader.clear.assert_called_once()


class TestGovernedModelManagerScheduleGovernorCoroRunningLoop:
    """Tests for _schedule_governor_coro with running loop scenarios."""

    @pytest.mark.asyncio()
    async def test_schedule_governor_coro_with_running_loop_critical(self, governed_manager):
        """Schedules critical coroutine in running event loop."""
        called = []

        async def _track():
            called.append(True)

        governed_manager._schedule_governor_coro(_track(), critical=True, description="test-critical")

        # Give the scheduled coroutine time to run
        await asyncio.sleep(0.1)
        assert called == [True]

    @pytest.mark.asyncio()
    async def test_schedule_governor_coro_critical_failure_recorded(self, governed_manager):
        """Critical failure in scheduled coroutine is recorded."""

        async def _boom():
            raise ValueError("scheduled boom")

        governed_manager._schedule_governor_coro(_boom(), critical=True, description="boom-op")

        # Give time for the error handling
        await asyncio.sleep(0.1)

        failures = governed_manager.get_critical_failures()
        assert any("boom-op" in f["description"] for f in failures)


class TestCreateGovernedModelManager:
    """Tests for create_governed_model_manager() factory branching."""

    def test_create_governed_model_manager_cpu_only_uses_settings_timeout(self, monkeypatch):
        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", torch_mod)

        budget = MemoryBudget(total_gpu_mb=0, total_cpu_mb=1)

        with (
            patch.object(gmm.settings, "MODEL_UNLOAD_AFTER_SECONDS", 123),
            patch("vecpipe.governed_model_manager.create_memory_budget", return_value=budget) as mk_budget,
            patch("vecpipe.governed_model_manager.GovernedModelManager") as mk_manager,
        ):
            out = gmm.create_governed_model_manager(unload_after_seconds=None, total_gpu_memory_mb=None)

        mk_budget.assert_called_once_with(total_gpu_mb=0)
        mk_manager.assert_called_once()
        assert mk_manager.call_args.kwargs["unload_after_seconds"] == 123
        assert out is mk_manager.return_value

    def test_create_governed_model_manager_respects_total_gpu_override(self, monkeypatch):
        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: True, mem_get_info=lambda: (0, 0))
        monkeypatch.setitem(sys.modules, "torch", torch_mod)

        budget = MemoryBudget(total_gpu_mb=5555, total_cpu_mb=1)

        with (
            patch("vecpipe.governed_model_manager.create_memory_budget", return_value=budget) as mk_budget,
            patch("vecpipe.governed_model_manager.GovernedModelManager") as mk_manager,
        ):
            out = gmm.create_governed_model_manager(unload_after_seconds=1, total_gpu_memory_mb=5555)

        mk_budget.assert_called_once_with(total_gpu_mb=5555)
        mk_manager.assert_called_once()
        assert out is mk_manager.return_value

    def test_create_governed_model_manager_detects_gpu_memory(self, monkeypatch):
        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = types.SimpleNamespace(
            is_available=lambda: True, mem_get_info=lambda: (0, 8 * 1024 * 1024 * 1024)
        )
        monkeypatch.setitem(sys.modules, "torch", torch_mod)

        budget = MemoryBudget(total_gpu_mb=8192, total_cpu_mb=1)

        with (
            patch("vecpipe.governed_model_manager.create_memory_budget", return_value=budget) as mk_budget,
            patch("vecpipe.governed_model_manager.GovernedModelManager") as mk_manager,
        ):
            out = gmm.create_governed_model_manager(unload_after_seconds=1, total_gpu_memory_mb=None)

        mk_budget.assert_called_once_with(total_gpu_mb=8192)
        mk_manager.assert_called_once()
        assert out is mk_manager.return_value
