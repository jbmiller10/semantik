"""Integration tests for ModelManager leveraging mock embedding mode.

These tests verify that ModelManager correctly uses the plugin-aware provider system
and routes to MockEmbeddingProvider when mock mode is enabled.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest


@pytest.fixture()
def model_manager(monkeypatch):
    """Create a ModelManager with mock mode enabled."""
    monkeypatch.setenv("USE_MOCK_EMBEDDINGS", "true")

    import shared.config as shared_config

    importlib.reload(shared_config)
    shared_config.settings.USE_MOCK_EMBEDDINGS = True

    # Reset plugin loader state to ensure fresh provider registration
    from shared.embedding.plugin_loader import _reset_plugin_loader_state, ensure_providers_registered

    _reset_plugin_loader_state()
    ensure_providers_registered()

    from vecpipe import model_manager as model_manager_module

    importlib.reload(model_manager_module)
    mgr = model_manager_module.ModelManager(unload_after_seconds=0)
    mgr.is_mock_mode = True  # Ensure mock mode is set
    return mgr


@pytest.mark.asyncio()
async def test_generate_embedding_returns_mock_vector(model_manager):
    """Test that mock mode returns a valid embedding vector via MockEmbeddingProvider."""
    vector = await model_manager.generate_embedding_async("integration", "mock-model", "float16")
    # MockEmbeddingProvider uses 384 dimensions by default
    assert len(vector) == 384
    assert model_manager.is_mock_mode is True
    assert model_manager._provider_name == "mock"


@pytest.mark.timeout(2)
@pytest.mark.asyncio()
async def test_auto_unload_clears_current_model(model_manager):
    """Test that models are automatically unloaded after inactivity."""
    await model_manager.generate_embedding_async("integration", "mock-model", "float16")
    assert model_manager.unload_task is not None
    # Await completion of the scheduled unload with a timeout safeguard
    await asyncio.wait_for(model_manager.unload_task, timeout=2)
    assert model_manager.unload_task.done()
    assert model_manager.current_model_key is None


# --- Plugin Integration Tests ---


class DummyPluginProvider:
    """Dummy embedding provider for testing plugin discovery.

    This simulates a third-party plugin registered via entry points.
    """

    INTERNAL_NAME = "dummy_test"
    API_ID = "dummy-test"
    PROVIDER_TYPE = "local"
    METADATA = {"display_name": "Dummy Test Provider"}

    def __init__(self, config=None, **kwargs):
        self.config = config
        self._initialized = False
        self._dimension = 128
        self.model_name = None

    @classmethod
    def get_definition(cls):
        from shared.embedding.plugin_base import EmbeddingProviderDefinition

        return EmbeddingProviderDefinition(
            api_id=cls.API_ID,
            internal_id=cls.INTERNAL_NAME,
            display_name="Dummy Test Provider",
            description="Test plugin provider",
            provider_type="local",
            supports_quantization=False,
            supports_instruction=False,
            supports_batch_processing=True,
            supported_models=("dummy/test-model",),
            is_plugin=True,
        )

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("dummy/")

    @classmethod
    def get_model_config(cls, model_name: str):
        return None

    @classmethod
    def list_supported_models(cls):
        return []

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self, model_name: str, **kwargs) -> None:
        self.model_name = model_name
        self._initialized = True

    async def embed_texts(self, texts, batch_size=32, *, mode=None, **kwargs):
        import numpy as np

        return np.random.rand(len(texts), self._dimension).astype(np.float32)

    async def embed_single(self, text, *, mode=None, **kwargs):
        import numpy as np

        return np.random.rand(self._dimension).astype(np.float32)

    async def cleanup(self) -> None:
        self._initialized = False
        self.model_name = None

    def get_dimension(self) -> int:
        return self._dimension

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "dimension": self._dimension,
            "device": "cpu",
            "is_test": True,
        }


@pytest.fixture()
def mock_dummy_plugin():
    """Register dummy plugin directly with the factory.

    This simulates what happens when a plugin is discovered via entry points,
    but registers directly to avoid the full entry point discovery process.
    """
    from shared.embedding.factory import EmbeddingProviderFactory
    from shared.embedding.plugin_loader import _reset_plugin_loader_state, ensure_providers_registered

    # Reset plugin loader state
    _reset_plugin_loader_state()

    # Ensure built-in providers are registered first
    ensure_providers_registered()

    # Register the dummy provider directly with the factory
    EmbeddingProviderFactory.register_provider("dummy_test", DummyPluginProvider)

    yield

    # Cleanup: unregister the dummy provider and reset state
    EmbeddingProviderFactory.unregister_provider("dummy_test")
    _reset_plugin_loader_state()


@pytest.mark.asyncio()
async def test_plugin_provider_used_for_model(mock_dummy_plugin):
    """Test that a plugin provider is used when its model name is requested.

    This is the acceptance test for Ticket 1: Installing a dummy plugin via entry point
    and selecting its model name results in that provider being used.
    """
    from vecpipe.model_manager import ModelManager

    mgr = ModelManager()
    mgr.is_mock_mode = False  # Don't use mock mode - test real plugin routing

    # Generate embedding with dummy model name
    embedding = await mgr.generate_embedding_async("test text", "dummy/test-model", "float32")

    # Assert the dummy provider was used
    assert mgr._provider_name == "dummy_test"
    assert embedding is not None
    assert len(embedding) == 128  # Dummy provider dimension

    # Verify provider info in status
    status = mgr.get_status()
    assert status["embedding_provider"] == "dummy_test"

    # Cleanup
    await mgr.unload_model_async()
