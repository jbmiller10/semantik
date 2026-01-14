"""Additional unit tests for GPUMemoryGovernor edge cases."""

from __future__ import annotations

import pytest
import torch

from vecpipe.memory_governor import GPUMemoryGovernor, MemoryBudget, ModelLocation, ModelType, TrackedModel


@pytest.mark.asyncio()
async def test_request_memory_cpu_only_mode() -> None:
    budget = MemoryBudget(total_gpu_mb=0, total_cpu_mb=1024)
    governor = GPUMemoryGovernor(budget=budget)

    result = await governor.request_memory(
        model_name="cpu-only",
        model_type=ModelType.EMBEDDING,
        quantization="float16",
        required_mb=9999,
    )

    assert result is True


def test_get_actual_free_gpu_mb_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = MemoryBudget(total_gpu_mb=1000, total_cpu_mb=1024, gpu_max_percent=1.0)
    governor = GPUMemoryGovernor(budget=budget)

    model = TrackedModel(
        model_name="test",
        model_type=ModelType.EMBEDDING,
        quantization="float16",
        location=ModelLocation.GPU,
        memory_mb=200,
    )
    governor._models[model.model_key] = model

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert governor._get_actual_free_gpu_mb() == 800
