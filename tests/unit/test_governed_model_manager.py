"""Tests for GovernedModelManager.

Tests the integration between ModelManager, GPUMemoryGovernor, and CPUOffloader.
"""

from unittest.mock import patch

import pytest

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
