"""Tests for CPU offloader.

Tests the GPU↔CPU model transfer operations.
"""

import time

import pytest
from torch import nn

from vecpipe.cpu_offloader import (
    ModelOffloader,
    OffloadMetadata,
    GradientCheckpointWrapper,
    MemoryEfficientInference,
    defragment_cuda_memory,
    estimate_model_memory,
    get_cuda_memory_fragmentation,
    get_offloader,
)


class SimpleModel(nn.Module):
    """Simple model for testing offload operations."""

    def __init__(self, size: int = 100):
        super().__init__()
        self.linear = nn.Linear(size, size)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture()
def offloader():
    """Create a fresh ModelOffloader for testing."""
    return ModelOffloader(pin_memory=False)  # Disable pinning for tests


@pytest.fixture()
def simple_model():
    """Create a simple model on CPU."""
    return SimpleModel(size=10)


class TestOffloadMetadata:
    """Tests for OffloadMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating valid metadata."""
        model = SimpleModel()
        metadata = OffloadMetadata(
            original_device="cuda:0",
            offload_time=time.time(),
            model_ref=model,
            keep_on_gpu=[],
        )
        assert metadata.original_device == "cuda:0"
        assert metadata.model_ref is model

    def test_metadata_seconds_offloaded(self):
        """Test seconds_offloaded property."""
        past_time = time.time() - 10  # 10 seconds ago
        metadata = OffloadMetadata(
            original_device="cuda:0",
            offload_time=past_time,
            model_ref=SimpleModel(),
        )
        assert metadata.seconds_offloaded >= 10

    def test_metadata_to_dict(self):
        """Test to_dict method."""
        metadata = OffloadMetadata(
            original_device="cuda:0",
            offload_time=time.time(),
            model_ref=SimpleModel(),
        )
        d = metadata.to_dict()
        assert "original_device" in d
        assert "offload_time" in d
        assert "seconds_offloaded" in d
        assert "model_ref" not in d  # Should not expose model reference

    def test_metadata_validation_empty_device(self):
        """Test that empty device raises ValueError."""
        with pytest.raises(ValueError, match="original_device cannot be empty"):
            OffloadMetadata(
                original_device="",
                offload_time=time.time(),
                model_ref=SimpleModel(),
            )

    def test_metadata_validation_invalid_time(self):
        """Test that invalid offload_time raises ValueError."""
        with pytest.raises(ValueError, match="offload_time must be positive"):
            OffloadMetadata(
                original_device="cuda:0",
                offload_time=-1,
                model_ref=SimpleModel(),
            )


class TestModelOffloader:
    """Tests for ModelOffloader class."""

    def test_offload_to_cpu(self, offloader, simple_model):
        """Test basic offload operation."""
        # Model starts on CPU in tests, but operation should still work
        metadata = offloader.offload_to_cpu("test_model", simple_model)

        assert offloader.is_offloaded("test_model")
        assert metadata.original_device == "cpu"
        assert metadata.model_ref is simple_model

    def test_offload_records_metadata(self, offloader, simple_model):
        """Test that offload records proper metadata."""
        before_time = time.time()
        offloader.offload_to_cpu("test_model", simple_model)
        after_time = time.time()

        info = offloader.get_offload_info("test_model")
        assert info is not None
        assert info["original_device"] == "cpu"
        assert before_time <= info["offload_time"] <= after_time

    def test_restore_to_gpu_not_found(self, offloader):
        """Test restore raises KeyError for non-existent model."""
        with pytest.raises(KeyError, match="nonexistent"):
            offloader.restore_to_gpu("nonexistent")

    def test_restore_to_cpu_removes_from_tracking(self, offloader, simple_model):
        """Test that restore removes model from offloaded tracking."""
        offloader.offload_to_cpu("test_model", simple_model)
        assert offloader.is_offloaded("test_model")

        # Restore to CPU (since we don't have GPU in tests)
        offloader.restore_to_gpu("test_model", device="cpu")
        assert not offloader.is_offloaded("test_model")

    def test_offload_restore_cycle(self, offloader, simple_model):
        """Test complete offload/restore cycle."""
        # Offload
        offloader.offload_to_cpu("test_model", simple_model)
        assert offloader.is_offloaded("test_model")
        assert len(offloader.get_offloaded_models()) == 1

        # Restore
        restored = offloader.restore_to_gpu("test_model", device="cpu")
        assert restored is simple_model
        assert not offloader.is_offloaded("test_model")
        assert len(offloader.get_offloaded_models()) == 0

    def test_get_offloaded_models_list(self, offloader, simple_model):
        """Test get_offloaded_models returns correct list."""
        assert offloader.get_offloaded_models() == []

        offloader.offload_to_cpu("model1", simple_model)
        model2 = SimpleModel()
        offloader.offload_to_cpu("model2", model2)

        offloaded = offloader.get_offloaded_models()
        assert len(offloaded) == 2
        assert "model1" in offloaded
        assert "model2" in offloaded

    def test_clear_removes_all(self, offloader, simple_model):
        """Test clear removes all offloaded models."""
        offloader.offload_to_cpu("model1", simple_model)
        offloader.offload_to_cpu("model2", SimpleModel())

        offloader.clear()
        assert len(offloader.get_offloaded_models()) == 0

    def test_get_offload_info_not_found(self, offloader):
        """Test get_offload_info returns None for non-existent model."""
        assert offloader.get_offload_info("nonexistent") is None

    def test_keep_on_gpu_warning(self, offloader, simple_model, caplog):
        """Test that keep_on_gpu parameter logs a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            offloader.offload_to_cpu(
                "test_model",
                simple_model,
                keep_on_gpu=["some_layer"],
            )
        assert "keep_on_gpu parameter is not yet implemented" in caplog.text


class TestEstimateModelMemory:
    """Tests for model memory estimation."""

    def test_estimate_simple_model(self):
        """Test estimating memory for a simple model."""
        model = SimpleModel(size=100)
        estimate = estimate_model_memory(model)

        assert "parameter_mb" in estimate
        assert "buffer_mb" in estimate
        assert "total_mb" in estimate
        assert estimate["total_mb"] >= 0


class TestGetOffloader:
    """Tests for singleton offloader."""

    def test_get_offloader_returns_same_instance(self):
        """Test that get_offloader returns the same instance."""
        offloader1 = get_offloader()
        offloader2 = get_offloader()
        assert offloader1 is offloader2


class TestModelOffloaderDiscard:
    """Tests for ModelOffloader.discard() method."""

    def test_discard_removes_offloaded_model(self, offloader: ModelOffloader, simple_model):
        """Discard removes model from offloaded dict."""
        # First offload the model
        offloader.offload_to_cpu("test_model", simple_model)
        assert offloader.is_offloaded("test_model")

        # Now discard it
        result = offloader.discard("test_model")

        assert result is True
        assert not offloader.is_offloaded("test_model")

    def test_discard_returns_false_for_unknown_model(self, offloader: ModelOffloader):
        """Discard returns False when model not found."""
        result = offloader.discard("nonexistent_model")

        assert result is False

    def test_discard_after_already_discarded(self, offloader: ModelOffloader, simple_model):
        """Discarding a model twice returns False on second attempt."""
        offloader.offload_to_cpu("test_model", simple_model)

        # First discard succeeds
        assert offloader.discard("test_model") is True

        # Second discard returns False (already discarded)
        assert offloader.discard("test_model") is False


class TestGradientCheckpointWrapper:
    """Tests for gradient checkpointing helpers."""

    def test_enable_disable_checkpointing_supported(self):
        """Enable/disable toggles when supported by model."""

        class _CheckpointModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.enabled = False

            def gradient_checkpointing_enable(self):
                self.enabled = True

            def gradient_checkpointing_disable(self):
                self.enabled = False

        model = _CheckpointModel()
        GradientCheckpointWrapper.enable_checkpointing(model)
        assert model.enabled is True
        GradientCheckpointWrapper.disable_checkpointing(model)
        assert model.enabled is False

    def test_enable_checkpointing_warns_when_missing(self, caplog):
        """Missing checkpointing methods logs a warning."""

        class _NoCheckpointModel(nn.Module):
            pass

        with caplog.at_level("WARNING"):
            GradientCheckpointWrapper.enable_checkpointing(_NoCheckpointModel())
        assert "does not support gradient_checkpointing_enable" in caplog.text


class TestMemoryEfficientInference:
    """Tests for memory-efficient inference context manager."""

    def test_context_no_cuda(self, monkeypatch):
        """Context manager should work without CUDA available."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with MemoryEfficientInference(use_amp=True) as ctx:
            assert ctx is not None


class TestCudaMemoryUtilities:
    """Tests for CUDA memory helper functions."""

    def test_get_cuda_memory_fragmentation_no_cuda(self, monkeypatch):
        """No CUDA should return cuda_available=False."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        result = get_cuda_memory_fragmentation()
        assert result == {"cuda_available": False}

    def test_defragment_cuda_memory_no_cuda(self, monkeypatch):
        """Defragment should no-op when CUDA is unavailable."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        defragment_cuda_memory()

    def test_discard_does_not_affect_other_models(self, offloader: ModelOffloader, simple_model):
        """Discarding one model doesn't affect others."""
        model2 = SimpleModel(size=50)

        offloader.offload_to_cpu("model1", simple_model)
        offloader.offload_to_cpu("model2", model2)

        # Discard model1
        offloader.discard("model1")

        # model2 should still be offloaded
        assert not offloader.is_offloaded("model1")
        assert offloader.is_offloaded("model2")
