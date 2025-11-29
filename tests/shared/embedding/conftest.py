"""Shared fixtures for embedding plugin system tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from shared.embedding import plugin_loader
from shared.embedding.factory import _PROVIDER_CLASSES
from shared.embedding.plugin_base import BaseEmbeddingPlugin, EmbeddingProviderDefinition
from shared.embedding.provider_registry import _PROVIDERS, _clear_caches

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray


@pytest.fixture()
def clean_registry() -> Generator[None, None, None]:  # noqa: PT004
    """Save and restore registry state around tests.

    This fixture ensures that modifications to the provider registries
    during tests don't affect other tests.
    """
    # Save original state
    original_classes = dict(_PROVIDER_CLASSES)
    original_providers = dict(_PROVIDERS)
    _clear_caches()
    plugin_loader._reset_plugin_loader_state()

    yield

    # Restore original state
    _PROVIDER_CLASSES.clear()
    _PROVIDER_CLASSES.update(original_classes)
    _PROVIDERS.clear()
    _PROVIDERS.update(original_providers)
    _clear_caches()
    plugin_loader._reset_plugin_loader_state()


@pytest.fixture()
def empty_registry() -> Generator[None, None, None]:  # noqa: PT004
    """Provide an empty registry for testing registration from scratch.

    This fixture clears all providers before the test and restores them after.
    """
    # Save original state
    original_classes = dict(_PROVIDER_CLASSES)
    original_providers = dict(_PROVIDERS)

    # Clear registries
    _PROVIDER_CLASSES.clear()
    _PROVIDERS.clear()
    _clear_caches()
    plugin_loader._reset_plugin_loader_state()

    yield

    # Restore original state
    _PROVIDER_CLASSES.clear()
    _PROVIDER_CLASSES.update(original_classes)
    _PROVIDERS.clear()
    _PROVIDERS.update(original_providers)
    _clear_caches()
    plugin_loader._reset_plugin_loader_state()


@pytest.fixture()
def dummy_definition() -> EmbeddingProviderDefinition:
    """Create a dummy provider definition for testing."""
    return EmbeddingProviderDefinition(
        api_id="dummy",
        internal_id="dummy",
        display_name="Dummy Provider",
        description="A dummy provider for testing",
        provider_type="local",
        supports_quantization=False,
        supports_instruction=False,
        supports_batch_processing=True,
        supported_models=("dummy-model",),
        default_config={"dimension": 128},
        is_plugin=False,
    )


@pytest.fixture()
def dummy_plugin_class() -> type[BaseEmbeddingPlugin]:
    """Create a dummy plugin class for testing.

    This class implements all required methods for the plugin contract.
    """

    class DummyPlugin(BaseEmbeddingPlugin):
        """Dummy embedding plugin for testing."""

        INTERNAL_NAME = "dummy"
        API_ID = "dummy"
        PROVIDER_TYPE = "local"

        METADATA = {
            "display_name": "Dummy Plugin",
            "description": "A dummy plugin for testing",
        }

        def __init__(self, config: Any | None = None, **kwargs: Any) -> None:
            super().__init__(config=config, **kwargs)
            self._initialized = False
            self.dimension = kwargs.get("dimension", 128)
            self.model_name: str | None = None

        @classmethod
        def get_definition(cls) -> EmbeddingProviderDefinition:
            return EmbeddingProviderDefinition(
                api_id=cls.API_ID,
                internal_id=cls.INTERNAL_NAME,
                display_name="Dummy Plugin",
                description="A dummy plugin for testing",
                provider_type="local",
                supports_quantization=False,
                supports_instruction=False,
                supports_batch_processing=True,
                supported_models=("dummy-model",),
                default_config={"dimension": 128},
                is_plugin=True,
            )

        @classmethod
        def supports_model(cls, model_name: str) -> bool:
            return model_name.startswith("dummy/") or model_name == "dummy-model"

        @property
        def is_initialized(self) -> bool:
            return self._initialized

        async def initialize(self, model_name: str, **kwargs: Any) -> None:
            self.model_name = model_name
            self.dimension = kwargs.get("dimension", 128)
            self._initialized = True

        async def embed_texts(
            self, texts: list[str], batch_size: int = 32, **kwargs: Any
        ) -> NDArray[np.float32]:
            if not self._initialized:
                raise RuntimeError("Not initialized")
            # Return deterministic embeddings based on text length
            return np.array(
                [[float(len(text)) / 100.0] * self.dimension for text in texts],
                dtype=np.float32,
            )

        async def embed_single(self, text: str, **kwargs: Any) -> NDArray[np.float32]:
            if not self._initialized:
                raise RuntimeError("Not initialized")
            embeddings = await self.embed_texts([text], **kwargs)
            return embeddings[0]

        def get_dimension(self) -> int:
            if not self._initialized:
                raise RuntimeError("Not initialized")
            return self.dimension

        def get_model_info(self) -> dict[str, Any]:
            if not self._initialized:
                raise RuntimeError("Not initialized")
            return {
                "model_name": self.model_name,
                "dimension": self.dimension,
                "device": "cpu",
                "provider": self.INTERNAL_NAME,
            }

        async def cleanup(self) -> None:
            self._initialized = False
            self.model_name = None

    return DummyPlugin


@pytest.fixture()
def another_dummy_definition() -> EmbeddingProviderDefinition:
    """Create a second dummy provider definition for testing."""
    return EmbeddingProviderDefinition(
        api_id="another",
        internal_id="another",
        display_name="Another Provider",
        description="Another dummy provider for testing",
        provider_type="remote",
        supports_quantization=True,
        supports_instruction=True,
        supports_batch_processing=True,
        supported_models=("another-model",),
        default_config={"dimension": 256},
        is_plugin=True,
    )
