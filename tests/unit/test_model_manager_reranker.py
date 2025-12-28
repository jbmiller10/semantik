import pytest

from vecpipe import model_manager as model_manager_module

# Import InsufficientMemoryError from the same module that model_manager uses
# to avoid class identity mismatch due to dual-path imports (vecpipe vs packages.vecpipe)
InsufficientMemoryError = model_manager_module.InsufficientMemoryError


def test_ensure_reranker_loaded_raises_when_memory_can_be_freed(monkeypatch):
    mgr = model_manager_module.ModelManager()
    mgr.is_mock_mode = False

    monkeypatch.setattr(
        model_manager_module,
        "check_memory_availability",
        lambda *_args, **_kwargs: (False, "Insufficient memory. Can free 1024MB by unloading: embedding"),
    )

    with pytest.raises(InsufficientMemoryError) as exc_info:
        mgr.ensure_reranker_loaded("test-model", "float16")

    assert "Can free" in str(exc_info.value)


def test_ensure_reranker_loaded_raises_on_oom(monkeypatch):
    class DummyReranker:
        def __init__(self, model_name: str, quantization: str) -> None:
            self.model_name = model_name
            self.quantization = quantization

        def load_model(self) -> None:
            raise RuntimeError("CUDA out of memory")

        def unload_model(self) -> None:
            return None

        def get_model_info(self) -> dict[str, str]:
            return {"model": self.model_name, "quantization": self.quantization}

    mgr = model_manager_module.ModelManager()
    mgr.is_mock_mode = False

    monkeypatch.setattr(model_manager_module, "check_memory_availability", lambda *_a, **_k: (True, "ok"))
    monkeypatch.setattr(model_manager_module, "CrossEncoderReranker", DummyReranker)
    monkeypatch.setattr(model_manager_module, "get_gpu_memory_info", lambda: (123, 456))

    with pytest.raises(InsufficientMemoryError) as exc_info:
        mgr.ensure_reranker_loaded("test-model", "float16")

    assert "GPU out of memory" in str(exc_info.value)
    assert mgr.reranker is None
    assert mgr.current_reranker_key is None
