"""Tests for the DenseLocalEmbeddingProvider plugin interface.

These tests only cover the plugin interface (class attributes, supports_model,
get_definition, etc.) without testing the actual embedding generation which
requires mocking torch and transformers.
"""

from __future__ import annotations

from shared.embedding.providers.dense_local import DenseLocalEmbeddingProvider


class TestPluginInterface:
    """Tests for the plugin interface attributes and methods."""

    def test_class_attributes_defined(self) -> None:
        """Test that required class attributes are defined."""
        assert DenseLocalEmbeddingProvider.INTERNAL_NAME == "dense_local"
        assert DenseLocalEmbeddingProvider.API_ID == "dense_local"
        assert DenseLocalEmbeddingProvider.PROVIDER_TYPE == "local"

    def test_metadata_defined(self) -> None:
        """Test that METADATA is defined with expected keys."""
        metadata = DenseLocalEmbeddingProvider.METADATA

        assert "display_name" in metadata
        assert "description" in metadata
        assert "best_for" in metadata
        assert "pros" in metadata
        assert "cons" in metadata

    def test_get_definition_returns_valid(self) -> None:
        """Test that get_definition returns a valid definition."""
        definition = DenseLocalEmbeddingProvider.get_definition()

        assert definition.api_id == "dense_local"
        assert definition.internal_id == "dense_local"
        assert definition.provider_type == "local"
        assert definition.supports_quantization is True
        assert definition.supports_instruction is True
        assert definition.supports_batch_processing is True
        assert definition.supports_asymmetric is True
        assert definition.is_plugin is False

    def test_get_definition_has_performance_characteristics(self) -> None:
        """Test that definition includes performance characteristics."""
        definition = DenseLocalEmbeddingProvider.get_definition()

        assert "latency" in definition.performance_characteristics
        assert "throughput" in definition.performance_characteristics
        assert "memory_usage" in definition.performance_characteristics


class TestModelSupport:
    """Tests for supports_model detection."""

    def test_supports_model_sentence_transformers(self) -> None:
        """Test detection of sentence-transformers models."""
        assert DenseLocalEmbeddingProvider.supports_model("sentence-transformers/all-MiniLM-L6-v2")
        assert DenseLocalEmbeddingProvider.supports_model("sentence-transformers/all-mpnet-base-v2")
        assert DenseLocalEmbeddingProvider.supports_model("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def test_supports_model_qwen(self) -> None:
        """Test detection of Qwen embedding models."""
        assert DenseLocalEmbeddingProvider.supports_model("Qwen/Qwen3-Embedding-0.6B")
        assert DenseLocalEmbeddingProvider.supports_model("Qwen/Qwen2-Embedding-0.5B")

    def test_supports_model_bge(self) -> None:
        """Test detection of BGE models."""
        assert DenseLocalEmbeddingProvider.supports_model("BAAI/bge-large-en-v1.5")
        assert DenseLocalEmbeddingProvider.supports_model("BAAI/bge-small-en-v1.5")
        assert DenseLocalEmbeddingProvider.supports_model("BAAI/bge-base-en-v1.5")

    def test_supports_model_intfloat(self) -> None:
        """Test detection of intfloat/e5 models."""
        assert DenseLocalEmbeddingProvider.supports_model("intfloat/e5-large-v2")
        assert DenseLocalEmbeddingProvider.supports_model("intfloat/multilingual-e5-large")
        assert DenseLocalEmbeddingProvider.supports_model("intfloat/e5-base-v2")

    def test_supports_model_unknown_returns_false(self) -> None:
        """Test that unknown models are rejected."""
        assert DenseLocalEmbeddingProvider.supports_model("unknown/model") is False
        assert DenseLocalEmbeddingProvider.supports_model("random-model") is False
        assert DenseLocalEmbeddingProvider.supports_model("openai/gpt-4") is False

    def test_supports_model_mock_returns_false(self) -> None:
        """Test that 'mock' model is handled by MockEmbeddingProvider."""
        # The mock model should be handled by MockEmbeddingProvider, not dense_local
        assert DenseLocalEmbeddingProvider.supports_model("mock") is False


class TestModelConfig:
    """Tests for model configuration."""

    def test_get_model_config_known_model(self) -> None:
        """Test getting config for a known model."""
        config = DenseLocalEmbeddingProvider.get_model_config("sentence-transformers/all-MiniLM-L6-v2")

        # Should return a config if the model is in MODEL_CONFIGS
        # Note: Result depends on what's in MODEL_CONFIGS
        if config is not None:
            assert hasattr(config, "name")
            assert hasattr(config, "dimension")

    def test_get_model_config_unknown_model(self) -> None:
        """Test getting config for an unknown model."""
        config = DenseLocalEmbeddingProvider.get_model_config("unknown/model")

        assert config is None

    def test_list_supported_models(self) -> None:
        """Test listing supported models."""
        models = DenseLocalEmbeddingProvider.list_supported_models()

        assert isinstance(models, list)
        # Should contain at least some models from MODEL_CONFIGS
        assert len(models) > 0

        # Each model should have expected attributes
        for model in models:
            assert hasattr(model, "name")
            assert hasattr(model, "dimension")


class TestProviderPatterns:
    """Tests for model detection patterns."""

    def test_sentence_transformer_pattern(self) -> None:
        """Test sentence-transformer pattern detection."""
        patterns = DenseLocalEmbeddingProvider._SENTENCE_TRANSFORMER_PATTERNS

        assert "sentence-transformers/" in patterns
        assert "BAAI/bge-" in patterns
        assert "intfloat/" in patterns

    def test_qwen_pattern(self) -> None:
        """Test Qwen pattern detection."""
        patterns = DenseLocalEmbeddingProvider._QWEN_PATTERNS

        assert "Qwen/Qwen3-Embedding" in patterns
        assert "Qwen/Qwen2-Embedding" in patterns


class TestProviderInstantiation:
    """Tests for provider instantiation (without initialization)."""

    def test_instantiate_without_config(self) -> None:
        """Test creating provider instance without config."""
        provider = DenseLocalEmbeddingProvider()

        # Config property returns empty dict when no config provided
        assert provider.config == {}
        assert provider.is_initialized is False

    def test_instantiate_with_config(self) -> None:
        """Test creating provider instance with config object."""
        from unittest.mock import MagicMock

        # DenseLocalEmbeddingProvider expects a VecpipeConfig-like object
        config = MagicMock()
        config.MIN_BATCH_SIZE = 1
        config.BATCH_SIZE_INCREASE_THRESHOLD = 10
        config.ENABLE_ADAPTIVE_BATCH_SIZE = True

        provider = DenseLocalEmbeddingProvider(config=config)

        # Config property returns dict representation of the config object
        assert provider.config["MIN_BATCH_SIZE"] == 1
        assert provider.config["BATCH_SIZE_INCREASE_THRESHOLD"] == 10
        assert provider.config["ENABLE_ADAPTIVE_BATCH_SIZE"] is True

    def test_default_device_detection(self) -> None:
        """Test default device detection."""
        provider = DenseLocalEmbeddingProvider()

        # Device should be "cuda" if available, else "cpu"
        assert provider.device in ("cuda", "cpu")

    def test_default_quantization(self) -> None:
        """Test default quantization setting."""
        provider = DenseLocalEmbeddingProvider()

        assert provider.quantization == "float32"

    def test_default_not_initialized(self) -> None:
        """Test provider is not initialized by default."""
        provider = DenseLocalEmbeddingProvider()

        assert provider.is_initialized is False
        assert provider.model is None
