"""Tests for AdaptiveBatchSizeManager."""

import threading
import time
from unittest.mock import Mock, patch

import pytest
from shared.embedding.batch_manager import AdaptiveBatchSizeManager


class TestAdaptiveBatchSizeManager:
    """Test suite for AdaptiveBatchSizeManager."""

    def test_initialization(self) -> None:
        """Test manager initialization."""
        # Default initialization
        manager = AdaptiveBatchSizeManager()
        assert manager._default_safety_margin == 0.2
        assert manager.get_all_batch_sizes() == {}

        # Custom safety margin
        manager = AdaptiveBatchSizeManager(default_safety_margin=0.3)
        assert manager._default_safety_margin == 0.3

        # Invalid safety margin
        with pytest.raises(ValueError, match="Safety margin must be between 0.0 and 0.5"):
            AdaptiveBatchSizeManager(default_safety_margin=0.6)
        with pytest.raises(ValueError, match="Safety margin must be between 0.0 and 0.5"):
            AdaptiveBatchSizeManager(default_safety_margin=-0.1)

    def test_batch_size_operations(self) -> None:
        """Test basic batch size operations."""
        manager = AdaptiveBatchSizeManager()

        # Initial state - no batch size set
        assert manager.get_current_batch_size("model1", "float32") is None

        # Update batch size
        manager.update_batch_size("model1", "float32", 32)
        assert manager.get_current_batch_size("model1", "float32") == 32

        # Update with different quantization
        manager.update_batch_size("model1", "float16", 64)
        assert manager.get_current_batch_size("model1", "float16") == 64
        assert manager.get_current_batch_size("model1", "float32") == 32

        # Update existing
        manager.update_batch_size("model1", "float32", 48)
        assert manager.get_current_batch_size("model1", "float32") == 48

        # Invalid batch size
        with pytest.raises(ValueError, match="Batch size must be positive"):
            manager.update_batch_size("model1", "float32", 0)
        with pytest.raises(ValueError, match="Batch size must be positive"):
            manager.update_batch_size("model1", "float32", -1)

    def test_reset_batch_size(self) -> None:
        """Test resetting batch sizes."""
        manager = AdaptiveBatchSizeManager()

        # Set some batch sizes
        manager.update_batch_size("model1", "float32", 32)
        manager.update_batch_size("model1", "float16", 64)

        # Reset one
        manager.reset_batch_size("model1", "float32")
        assert manager.get_current_batch_size("model1", "float32") is None
        assert manager.get_current_batch_size("model1", "float16") == 64

        # Reset non-existent (should not raise)
        manager.reset_batch_size("model2", "float32")

    def test_clear_all(self) -> None:
        """Test clearing all batch sizes."""
        manager = AdaptiveBatchSizeManager()

        # Set multiple batch sizes
        manager.update_batch_size("model1", "float32", 32)
        manager.update_batch_size("model1", "float16", 64)
        manager.update_batch_size("model2", "int8", 128)

        # Clear all
        manager.clear_all()
        assert manager.get_all_batch_sizes() == {}

    @patch("shared.embedding.batch_manager.torch")
    def test_calculate_initial_batch_size_no_gpu(self, mock_torch: Mock) -> None:
        """Test batch size calculation when no GPU is available."""
        mock_torch.cuda.is_available.return_value = False

        manager = AdaptiveBatchSizeManager()
        batch_size = manager.calculate_initial_batch_size("sentence-transformers/all-MiniLM-L6-v2", "float32")
        assert batch_size == 1

    @patch("shared.embedding.batch_manager.torch")
    def test_calculate_initial_batch_size_with_gpu(self, mock_torch: Mock) -> None:
        """Test batch size calculation with GPU available."""
        # Mock GPU with 8GB total, 6GB free
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (
            6 * 1024 * 1024 * 1024,  # 6GB free
            8 * 1024 * 1024 * 1024,  # 8GB total
        )

        manager = AdaptiveBatchSizeManager()

        # Small model should get larger batch size
        batch_size = manager.calculate_initial_batch_size(
            "sentence-transformers/all-MiniLM-L6-v2", "float32", text_length=256
        )
        assert batch_size > 1
        assert batch_size <= 256  # Max cap

        # Larger model should get smaller batch size
        batch_size_large = manager.calculate_initial_batch_size("Qwen/Qwen3-Embedding-8B", "float32", text_length=512)
        assert batch_size_large < batch_size

    @patch("shared.embedding.batch_manager.torch")
    def test_calculate_with_different_quantizations(self, mock_torch: Mock) -> None:
        """Test that quantization affects batch size calculation."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (
            4 * 1024 * 1024 * 1024,  # 4GB free
            8 * 1024 * 1024 * 1024,  # 8GB total
        )

        manager = AdaptiveBatchSizeManager()
        model = "BAAI/bge-large-en-v1.5"

        # float16 should allow larger batch than float32
        batch_32 = manager.calculate_initial_batch_size(model, "float32")
        batch_16 = manager.calculate_initial_batch_size(model, "float16")
        batch_8 = manager.calculate_initial_batch_size(model, "int8")

        assert batch_8 >= batch_16 >= batch_32

    def test_calculate_unknown_model(self) -> None:
        """Test batch size calculation for unknown model."""
        manager = AdaptiveBatchSizeManager()
        batch_size = manager.calculate_initial_batch_size("unknown/model", "float32")
        assert batch_size == 1

    @patch("shared.embedding.batch_manager.torch")
    def test_get_or_calculate_batch_size(self, mock_torch: Mock) -> None:
        """Test the convenience method."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (
            4 * 1024 * 1024 * 1024,  # 4GB free
            8 * 1024 * 1024 * 1024,  # 8GB total
        )

        manager = AdaptiveBatchSizeManager()
        model = "sentence-transformers/all-MiniLM-L6-v2"

        # First call should calculate and store
        batch_size1 = manager.get_or_calculate_batch_size(model, "float32")
        assert batch_size1 > 1
        assert manager.get_current_batch_size(model, "float32") == batch_size1

        # Second call should return stored value
        batch_size2 = manager.get_or_calculate_batch_size(model, "float32")
        assert batch_size2 == batch_size1

    def test_thread_safety(self) -> None:
        """Test thread-safe operations."""
        manager = AdaptiveBatchSizeManager()
        results = []
        errors = []

        def update_batch_sizes(thread_id: int) -> None:
            try:
                for i in range(10):
                    manager.update_batch_size(f"model{thread_id}", "float32", (i + 1) * 10 + thread_id)
                    time.sleep(0.001)  # Small delay to increase contention
                    size = manager.get_current_batch_size(f"model{thread_id}", "float32")
                    results.append((thread_id, size))
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_batch_sizes, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check no errors occurred
        assert len(errors) == 0

        # Check final state is consistent
        for i in range(5):
            final_size = manager.get_current_batch_size(f"model{i}", "float32")
            assert final_size == 100 + i  # Last update: (9 + 1) * 10 + i = 100 + i

    @patch("shared.embedding.batch_manager.torch")
    def test_safety_margin_effect(self, mock_torch: Mock) -> None:
        """Test that safety margin affects calculations."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (
            4 * 1024 * 1024 * 1024,  # 4GB free
            8 * 1024 * 1024 * 1024,  # 8GB total
        )

        model = "sentence-transformers/all-MiniLM-L6-v2"

        # Lower safety margin should give larger batch size
        manager_low = AdaptiveBatchSizeManager(default_safety_margin=0.1)
        batch_low = manager_low.calculate_initial_batch_size(model, "float32")

        manager_high = AdaptiveBatchSizeManager(default_safety_margin=0.4)
        batch_high = manager_high.calculate_initial_batch_size(model, "float32")

        assert batch_low > batch_high

        # Test override
        batch_override = manager_high.calculate_initial_batch_size(model, "float32", safety_margin=0.1)
        assert batch_override == batch_low
