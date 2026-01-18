"""Tests for LLM memory estimation utilities."""

from vecpipe.llm_memory import _estimate_llm_weights_memory, get_llm_memory_requirement


class TestLLMMemoryEstimation:
    """Tests for LLM memory estimation functions."""

    def test_get_memory_requirement_from_registry(self) -> None:
        """Test that registered models return their curated memory values."""
        # Qwen/Qwen2.5-1.5B-Instruct should have memory_mb in registry
        result = get_llm_memory_requirement("Qwen/Qwen2.5-1.5B-Instruct", "int8")
        assert result == 2000  # From model_registry.yaml

    def test_get_memory_requirement_int4(self) -> None:
        """Test int4 quantization memory values from registry."""
        result = get_llm_memory_requirement("Qwen/Qwen2.5-1.5B-Instruct", "int4")
        assert result == 1200  # From model_registry.yaml

    def test_get_memory_requirement_float16(self) -> None:
        """Test float16 memory values from registry."""
        result = get_llm_memory_requirement("Qwen/Qwen2.5-1.5B-Instruct", "float16")
        assert result == 3500  # From model_registry.yaml

    def test_get_memory_requirement_unknown_model(self) -> None:
        """Test fallback estimation for unknown models."""
        # Unknown model should use parameter-based estimation
        result = get_llm_memory_requirement("unknown/model-7B", "int8")
        # Should be estimated based on 7B params
        assert result > 0  # Just verify it returns something reasonable

    def test_estimate_weights_memory_pattern_matching(self) -> None:
        """Test parameter parsing from model names."""
        # 7B model in int8
        result = _estimate_llm_weights_memory("some/model-7B-instruct", "int8")
        # 7B * 2000MB base * 0.55 for int8 * 1.3 overhead ≈ 10010MB
        assert 8000 < result < 12000

    def test_estimate_weights_memory_small_model(self) -> None:
        """Test estimation for small models."""
        result = _estimate_llm_weights_memory("some/model-0.5B", "int8")
        # Should be much smaller than 7B model
        assert result < 2000

    def test_estimate_weights_memory_int4_reduction(self) -> None:
        """Test int4 quantization reduces memory estimate."""
        int8_result = _estimate_llm_weights_memory("some/model-7B", "int8")
        int4_result = _estimate_llm_weights_memory("some/model-7B", "int4")
        # int4 should be significantly smaller
        assert int4_result < int8_result * 0.8

    def test_estimate_weights_memory_float16(self) -> None:
        """Test float16 memory estimate (no quantization)."""
        int8_result = _estimate_llm_weights_memory("some/model-7B", "int8")
        float16_result = _estimate_llm_weights_memory("some/model-7B", "float16")
        # float16 should be larger than int8
        assert float16_result > int8_result

    def test_estimate_weights_memory_unknown_pattern(self) -> None:
        """Test fallback for models without size in name."""
        result = _estimate_llm_weights_memory("some-model-without-size", "int8")
        # Should return conservative default
        assert result == 4000

    def test_estimate_weights_memory_float32(self) -> None:
        """Test float32 doubles the base estimate."""
        float16_result = _estimate_llm_weights_memory("some/model-1B", "float16")
        float32_result = _estimate_llm_weights_memory("some/model-1B", "float32")
        # float32 should be about 2x float16
        assert float32_result > float16_result * 1.8
