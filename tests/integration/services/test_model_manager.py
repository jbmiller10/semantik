"""Integration tests for ModelManager leveraging mock embedding mode."""

from __future__ import annotations

import asyncio
import importlib

import pytest


@pytest.fixture()
def model_manager(monkeypatch):
    monkeypatch.setenv("USE_MOCK_EMBEDDINGS", "true")

    import shared.config as shared_config
    import shared.embedding.dense as dense_module
    import shared.embedding.service as embedding_service_module

    importlib.reload(shared_config)
    shared_config.settings.USE_MOCK_EMBEDDINGS = True

    importlib.reload(dense_module)
    importlib.reload(embedding_service_module)

    from vecpipe import model_manager as model_manager_module

    importlib.reload(model_manager_module)
    return model_manager_module.ModelManager(unload_after_seconds=0)


@pytest.mark.asyncio()
async def test_generate_embedding_returns_mock_vector(model_manager):
    vector = await model_manager.generate_embedding_async("integration", "mock-model", "float16")
    assert len(vector) == 256
    assert model_manager.is_mock_mode is True


@pytest.mark.timeout(2)
@pytest.mark.asyncio()
async def test_auto_unload_clears_current_model(model_manager):
    await model_manager.generate_embedding_async("integration", "mock-model", "float16")
    assert model_manager.unload_task is not None
    # Await completion of the scheduled unload with a timeout safeguard
    await asyncio.wait_for(model_manager.unload_task, timeout=2)
    assert model_manager.unload_task.done()
    assert model_manager.current_model_key is None
