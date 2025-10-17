"""Integration tests for ModelManager leveraging mock embedding mode."""

from __future__ import annotations

import importlib
import asyncio

import pytest


@pytest.fixture()
def model_manager(monkeypatch):
    monkeypatch.setenv("USE_MOCK_EMBEDDINGS", "true")
    from packages.vecpipe import model_manager as model_manager_module

    importlib.reload(model_manager_module)
    manager = model_manager_module.ModelManager(unload_after_seconds=0)
    yield manager


@pytest.mark.asyncio()
async def test_generate_embedding_returns_mock_vector(model_manager):
    vector = await model_manager.generate_embedding_async("integration", "mock-model", "float16")
    assert len(vector) == 1024
    assert model_manager.current_model_key == "mock-model_float16"


@pytest.mark.asyncio()
async def test_auto_unload_clears_current_model(model_manager):
    await model_manager.generate_embedding_async("integration", "mock-model", "float16")
    assert model_manager.current_model_key is not None
    # Allow scheduled unload to run
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    # After unload delay zero seconds, model should be cleared
    assert model_manager.current_model_key is None
