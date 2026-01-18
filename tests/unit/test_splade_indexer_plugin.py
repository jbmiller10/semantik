from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_get_recommended_batch_size_thresholds() -> None:
    from shared.plugins.builtins.splade_indexer import get_recommended_batch_size

    assert get_recommended_batch_size(24000) == 128
    assert get_recommended_batch_size(12000) == 64
    assert get_recommended_batch_size(8000) == 32
    assert get_recommended_batch_size(6000) == 16
    assert get_recommended_batch_size(4000) == 8
    assert get_recommended_batch_size(1000) == 4


@pytest.mark.asyncio()
async def test_splade_initialize_updates_config_and_loads_model_cpu_path() -> None:
    from shared.plugins.builtins.splade_indexer import SPLADESparseIndexerPlugin

    class DummyParam:
        def __init__(self) -> None:
            self.requires_grad = True

    class DummyModel:
        def __init__(self) -> None:
            self.to_device: str | None = None
            self.eval_called = False
            self._params = [DummyParam(), DummyParam()]

        def to(self, device: str) -> DummyModel:
            self.to_device = device
            return self

        def eval(self) -> None:
            self.eval_called = True

        def parameters(self):
            return list(self._params)

    dummy_tokenizer = MagicMock()
    dummy_model = DummyModel()

    with (
        patch("shared.plugins.builtins.splade_indexer.torch.cuda.is_available", return_value=False),
        patch("shared.plugins.builtins.splade_indexer.AutoTokenizer.from_pretrained", return_value=dummy_tokenizer),
        patch("shared.plugins.builtins.splade_indexer.AutoModelForMaskedLM.from_pretrained", return_value=dummy_model),
    ):
        plugin = SPLADESparseIndexerPlugin(config={"device": "auto", "quantization": "float16"})
        await plugin.initialize({"max_length": 128, "batch_size": 4, "model_name": "dummy/model"})

    assert plugin._actual_device == "cpu"
    assert plugin._model is dummy_model
    assert plugin._tokenizer is dummy_tokenizer
    assert dummy_model.to_device == "cpu"
    assert dummy_model.eval_called is True
    assert all(p.requires_grad is False for p in dummy_model.parameters())


@pytest.mark.asyncio()
async def test_splade_unload_model_clears_cuda_cache_when_available() -> None:
    from shared.plugins.builtins.splade_indexer import SPLADESparseIndexerPlugin

    plugin = SPLADESparseIndexerPlugin()
    plugin._model = object()  # type: ignore[assignment]
    plugin._tokenizer = object()  # type: ignore[assignment]

    with (
        patch("shared.plugins.builtins.splade_indexer.torch.cuda.is_available", return_value=True),
        patch("shared.plugins.builtins.splade_indexer.torch.cuda.synchronize") as mock_sync,
        patch("shared.plugins.builtins.splade_indexer.torch.cuda.empty_cache") as mock_empty,
    ):
        await plugin.cleanup()

    assert plugin._model is None
    assert plugin._tokenizer is None
    mock_sync.assert_called_once()
    mock_empty.assert_called_once()
