"""Tests for the MockEmbeddingProvider."""

from __future__ import annotations

import numpy as np
import pytest

from shared.embedding.providers.mock import MockEmbeddingProvider


class TestMockProviderPluginInterface:
    """Tests for the mock provider's plugin interface."""

    def test_supports_model_mock(self) -> None:
        """Test that 'mock' model is supported."""
        assert MockEmbeddingProvider.supports_model("mock") is True

    def test_supports_model_mock_uppercase(self) -> None:
        """Test that 'MOCK' (uppercase) is supported."""
        assert MockEmbeddingProvider.supports_model("MOCK") is True

    def test_supports_model_rejects_real_models(self) -> None:
        """Test that real model names are rejected."""
        assert MockEmbeddingProvider.supports_model("sentence-transformers/all-MiniLM-L6-v2") is False
        assert MockEmbeddingProvider.supports_model("Qwen/Qwen3-Embedding-0.6B") is False
        assert MockEmbeddingProvider.supports_model("BAAI/bge-large-en-v1.5") is False

    def test_get_definition(self) -> None:
        """Test getting the provider definition."""
        definition = MockEmbeddingProvider.get_definition()

        assert definition.api_id == "mock"
        assert definition.internal_id == "mock"
        assert definition.provider_type == "local"
        assert definition.supports_quantization is False
        assert definition.supports_instruction is False
        assert definition.supports_batch_processing is True
        assert definition.is_plugin is False

    def test_get_model_config_mock(self) -> None:
        """Test getting model config for mock model."""
        config = MockEmbeddingProvider.get_model_config("mock")

        assert config is not None
        assert config.name == "mock"
        assert config.dimension == 384
        assert config.supports_quantization is False

    def test_get_model_config_unsupported(self) -> None:
        """Test getting model config for unsupported model."""
        config = MockEmbeddingProvider.get_model_config("unsupported")

        assert config is None

    def test_list_supported_models(self) -> None:
        """Test listing supported models."""
        models = MockEmbeddingProvider.list_supported_models()

        assert len(models) == 1
        assert models[0].name == "mock"

    def test_class_attributes(self) -> None:
        """Test class attributes are properly defined."""
        assert MockEmbeddingProvider.INTERNAL_NAME == "mock"
        assert MockEmbeddingProvider.API_ID == "mock"
        assert MockEmbeddingProvider.PROVIDER_TYPE == "local"
        assert MockEmbeddingProvider.MOCK_MODEL_NAME == "mock"


class TestMockProviderInitialization:
    """Tests for provider initialization."""

    @pytest.fixture()
    def provider(self) -> MockEmbeddingProvider:
        """Create a mock provider instance."""
        return MockEmbeddingProvider()

    def test_is_initialized_before_init(self, provider: MockEmbeddingProvider) -> None:
        """Test is_initialized is False before initialization."""
        assert provider.is_initialized is False

    @pytest.mark.asyncio()
    async def test_initialize(self, provider: MockEmbeddingProvider) -> None:
        """Test provider initialization."""
        await provider.initialize("mock")

        assert provider.is_initialized is True
        assert provider.model_name == "mock"
        assert provider.dimension == 384

    @pytest.mark.asyncio()
    async def test_initialize_custom_dimension(self) -> None:
        """Test initialization with custom dimension via kwargs."""
        # Dimension must be passed to initialize(), not just constructor
        provider = MockEmbeddingProvider()
        await provider.initialize("mock", dimension=512)

        assert provider.dimension == 512

    @pytest.mark.asyncio()
    async def test_initialize_dimension_kwarg(self) -> None:
        """Test initialization with dimension in kwargs."""
        provider = MockEmbeddingProvider()
        await provider.initialize("mock", dimension=256)

        assert provider.dimension == 256

    @pytest.mark.asyncio()
    async def test_cleanup(self, provider: MockEmbeddingProvider) -> None:
        """Test cleanup resets state."""
        await provider.initialize("mock")
        assert provider.is_initialized is True

        await provider.cleanup()

        assert provider.is_initialized is False
        assert provider.model_name is None


class TestMockProviderEmbedding:
    """Tests for embedding generation."""

    @pytest.fixture()
    async def initialized_provider(self) -> MockEmbeddingProvider:
        """Create an initialized mock provider."""
        provider = MockEmbeddingProvider()
        await provider.initialize("mock")
        return provider

    @pytest.mark.asyncio()
    async def test_embed_texts_shape(
        self, initialized_provider: MockEmbeddingProvider
    ) -> None:
        """Test embedding shape is correct."""
        texts = ["hello", "world", "test"]
        embeddings = await initialized_provider.embed_texts(texts)

        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    @pytest.mark.asyncio()
    async def test_embed_texts_empty(
        self, initialized_provider: MockEmbeddingProvider
    ) -> None:
        """Test embedding empty list."""
        embeddings = await initialized_provider.embed_texts([])

        assert embeddings.shape == (0, 384)

    @pytest.mark.asyncio()
    async def test_embed_single_shape(
        self, initialized_provider: MockEmbeddingProvider
    ) -> None:
        """Test single embedding shape."""
        embedding = await initialized_provider.embed_single("hello")

        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    @pytest.mark.asyncio()
    async def test_embed_texts_deterministic(
        self, initialized_provider: MockEmbeddingProvider
    ) -> None:
        """Test that embeddings are deterministic."""
        text = "hello world"

        embedding1 = await initialized_provider.embed_single(text)
        embedding2 = await initialized_provider.embed_single(text)

        np.testing.assert_array_equal(embedding1, embedding2)

    @pytest.mark.asyncio()
    async def test_same_text_same_embedding(
        self, initialized_provider: MockEmbeddingProvider
    ) -> None:
        """Test same text produces same embedding across calls."""
        text = "consistent text"

        # Call multiple times
        embeddings = []
        for _ in range(5):
            embedding = await initialized_provider.embed_single(text)
            embeddings.append(embedding)

        # All should be identical
        for i in range(1, len(embeddings)):
            np.testing.assert_array_equal(embeddings[0], embeddings[i])

    @pytest.mark.asyncio()
    async def test_different_text_different_embedding(
        self, initialized_provider: MockEmbeddingProvider
    ) -> None:
        """Test different text produces different embeddings."""
        embedding1 = await initialized_provider.embed_single("hello")
        embedding2 = await initialized_provider.embed_single("world")

        # Should not be equal
        assert not np.array_equal(embedding1, embedding2)

    @pytest.mark.asyncio()
    async def test_embed_texts_not_initialized_raises(self) -> None:
        """Test that embedding without initialization raises."""
        provider = MockEmbeddingProvider()

        with pytest.raises(RuntimeError, match="not initialized"):
            await provider.embed_texts(["hello"])

    @pytest.mark.asyncio()
    async def test_embed_single_not_initialized_raises(self) -> None:
        """Test that single embedding without initialization raises."""
        provider = MockEmbeddingProvider()

        with pytest.raises(RuntimeError, match="not initialized"):
            await provider.embed_single("hello")

    @pytest.mark.asyncio()
    async def test_embed_texts_invalid_input_raises(
        self, initialized_provider: MockEmbeddingProvider
    ) -> None:
        """Test that non-list input raises ValueError."""
        with pytest.raises(ValueError, match="must be a list"):
            await initialized_provider.embed_texts("not a list")  # type: ignore[arg-type]

    @pytest.mark.asyncio()
    async def test_embed_single_invalid_input_raises(
        self, initialized_provider: MockEmbeddingProvider
    ) -> None:
        """Test that non-string input raises ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            await initialized_provider.embed_single(123)  # type: ignore[arg-type]

    @pytest.mark.asyncio()
    async def test_embeddings_normalized(
        self, initialized_provider: MockEmbeddingProvider
    ) -> None:
        """Test that embeddings are normalized by default."""
        embedding = await initialized_provider.embed_single("test text")

        # Normalized vectors have unit length
        norm = np.linalg.norm(embedding)
        np.testing.assert_almost_equal(norm, 1.0, decimal=5)


class TestMockProviderInfo:
    """Tests for provider info methods."""

    @pytest.mark.asyncio()
    async def test_get_dimension(self) -> None:
        """Test getting embedding dimension."""
        provider = MockEmbeddingProvider()
        await provider.initialize("mock")

        assert provider.get_dimension() == 384

    def test_get_dimension_not_initialized_raises(self) -> None:
        """Test that get_dimension without initialization raises."""
        provider = MockEmbeddingProvider()

        with pytest.raises(RuntimeError, match="not initialized"):
            provider.get_dimension()

    @pytest.mark.asyncio()
    async def test_get_model_info(self) -> None:
        """Test getting model info."""
        provider = MockEmbeddingProvider()
        await provider.initialize("mock")

        info = provider.get_model_info()

        assert info["model_name"] == "mock"
        assert info["dimension"] == 384
        assert info["device"] == "cpu"
        assert info["is_mock"] is True
        assert info["provider"] == "mock"

    def test_get_model_info_not_initialized_raises(self) -> None:
        """Test that get_model_info without initialization raises."""
        provider = MockEmbeddingProvider()

        with pytest.raises(RuntimeError, match="not initialized"):
            provider.get_model_info()
