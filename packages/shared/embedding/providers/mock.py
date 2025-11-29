"""Mock embedding provider for testing.

This provider generates deterministic embeddings based on text hash,
suitable for testing without GPU or actual model loading.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from shared.embedding.models import ModelConfig, get_model_config
from shared.embedding.plugin_base import BaseEmbeddingPlugin, EmbeddingProviderDefinition

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from shared.embedding.types import EmbeddingMode

logger = logging.getLogger(__name__)


class MockEmbeddingProvider(BaseEmbeddingPlugin):
    """Mock embedding provider for testing.

    Generates deterministic embeddings based on text hash. This allows for
    consistent test results without requiring GPU or actual model loading.

    Features:
    - Fast, no GPU required
    - Deterministic output based on text content
    - Configurable dimension
    - Optional normalization
    """

    INTERNAL_NAME: ClassVar[str] = "mock"
    API_ID: ClassVar[str] = "mock"
    PROVIDER_TYPE: ClassVar[str] = "local"

    METADATA: ClassVar[dict[str, Any]] = {
        "display_name": "Mock Embeddings",
        "description": "Deterministic mock embeddings for testing",
        "best_for": ["testing", "development", "ci_cd"],
        "pros": [
            "No GPU required",
            "Fast execution",
            "Deterministic output",
            "No model downloads",
        ],
        "cons": [
            "Not suitable for production",
            "No semantic meaning",
            "Random-like but deterministic vectors",
        ],
    }

    # Special model name that triggers mock provider
    MOCK_MODEL_NAME: ClassVar[str] = "mock"

    def __init__(self, config: Any | None = None, **kwargs: Any) -> None:
        """Initialize the mock embedding provider.

        Args:
            config: Optional configuration (ignored for mock)
            **kwargs: Additional options including:
                - dimension: Embedding dimension (default: 384)
        """
        self.config = config
        self.model_name: str | None = None
        self.dimension: int = kwargs.get("dimension", 384)
        self._initialized: bool = False
        self._normalize: bool = True

    @classmethod
    def get_definition(cls) -> EmbeddingProviderDefinition:
        """Return the canonical definition for this provider."""
        return EmbeddingProviderDefinition(
            api_id=cls.API_ID,
            internal_id=cls.INTERNAL_NAME,
            display_name="Mock Embeddings",
            description="Deterministic mock embeddings for testing without GPU",
            provider_type="local",
            supports_quantization=False,
            supports_instruction=False,
            supports_batch_processing=True,
            supported_models=(cls.MOCK_MODEL_NAME,),
            default_config={
                "dimension": 384,
            },
            performance_characteristics={
                "latency": "very_low",
                "throughput": "very_high",
                "memory_usage": "very_low",
            },
            is_plugin=False,
        )

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        """Check if this provider supports the given model.

        The mock provider only supports the explicit "mock" model name.
        It does NOT match real model names - those should go to the
        DenseLocalEmbeddingProvider which has its own mock_mode.
        """
        return model_name.lower() == cls.MOCK_MODEL_NAME

    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig | None:
        """Get configuration for the mock model."""
        if cls.supports_model(model_name):
            return ModelConfig(
                name=cls.MOCK_MODEL_NAME,
                dimension=384,
                description="Mock embedding model for testing",
                max_sequence_length=512,
                supports_quantization=False,
                recommended_quantization="float32",
            )
        return None

    @classmethod
    def list_supported_models(cls) -> list[ModelConfig]:
        """List supported models."""
        config = cls.get_model_config(cls.MOCK_MODEL_NAME)
        return [config] if config else []

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized

    async def initialize(self, model_name: str, **kwargs: Any) -> None:
        """Initialize the mock embedding provider.

        Args:
            model_name: Model name (used to determine dimension from known configs)
            **kwargs: Additional options:
                - dimension: Override embedding dimension
                - normalize: Whether to normalize embeddings (default: True)
        """
        self.model_name = model_name

        # Try to get dimension from model config if it's a known model
        model_config = get_model_config(model_name)
        if model_config is not None:
            self.dimension = model_config.dimension
        else:
            # Use provided dimension or default
            self.dimension = kwargs.get("dimension", 384)

        self._normalize = kwargs.get("normalize", True)
        self._initialized = True

        logger.info(f"Mock embedding provider initialized with dimension={self.dimension}")

    def _generate_deterministic_embedding(self, text: str) -> NDArray[np.float32]:
        """Generate a deterministic embedding based on text hash.

        Uses SHA-256 hash of the text to seed a random number generator,
        ensuring the same text always produces the same embedding.

        Args:
            text: Input text

        Returns:
            Deterministic embedding vector
        """
        # Create hash of text
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Use hash as seed for reproducible random numbers
        # Convert first 8 chars of hash to integer for seed
        seed = int(text_hash[:8], 16)
        rng = np.random.Generator(np.random.PCG64(seed))

        # Generate embedding
        embedding = rng.standard_normal(self.dimension).astype(np.float32)

        # Normalize if requested
        if self._normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    async def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,  # noqa: ARG002
        *,
        mode: EmbeddingMode | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        """Generate deterministic embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Ignored for mock provider
            mode: Ignored for mock provider (no asymmetric handling)
            **kwargs: Additional options (normalize)

        Returns:
            Array of shape (n_texts, dimension)
        """
        if not self._initialized:
            raise RuntimeError("Mock provider not initialized. Call initialize() first.")

        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")

        if not texts:
            return np.array([]).reshape(0, self.dimension)

        normalize = kwargs.get("normalize", self._normalize)
        original_normalize = self._normalize
        self._normalize = normalize

        try:
            embeddings = np.array([self._generate_deterministic_embedding(text) for text in texts])
        finally:
            self._normalize = original_normalize

        logger.debug(f"Mock generated {len(texts)} embeddings with dimension {self.dimension}")
        return embeddings

    async def embed_single(
        self,
        text: str,
        *,
        mode: EmbeddingMode | None = None,
        **kwargs: Any,
    ) -> NDArray[np.float32]:
        """Generate deterministic embedding for a single text.

        Args:
            text: Text to embed
            mode: Ignored for mock provider (no asymmetric handling)
            **kwargs: Additional options

        Returns:
            Embedding vector of shape (dimension,)
        """
        if not self._initialized:
            raise RuntimeError("Mock provider not initialized. Call initialize() first.")

        if not isinstance(text, str):
            raise ValueError("text must be a string")

        embeddings = await self.embed_texts([text], mode=mode, **kwargs)
        result: NDArray[np.float32] = embeddings[0]
        return result

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        if not self._initialized:
            raise RuntimeError("Mock provider not initialized")
        return self.dimension

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        if not self._initialized:
            raise RuntimeError("Mock provider not initialized")

        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": "cpu",
            "max_sequence_length": 512,
            "quantization": "float32",
            "is_mock": True,
            "provider": self.INTERNAL_NAME,
        }

    async def cleanup(self) -> None:
        """Clean up resources (no-op for mock provider)."""
        self._initialized = False
        self.model_name = None
        logger.info("Mock embedding provider cleaned up")
