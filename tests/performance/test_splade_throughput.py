"""Minimal performance tests for SPLADE sparse indexer.

Tests basic throughput and model loading time with CPU fallback for CI.
"""

from __future__ import annotations

import time

import pytest

pytestmark = [pytest.mark.performance, pytest.mark.anyio]


@pytest.fixture(scope="module")
def _check_dependencies():
    """Skip tests if torch/transformers unavailable."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def hardware_config(_check_dependencies):
    """Detect hardware and set thresholds."""
    import torch

    has_cuda = torch.cuda.is_available()
    return {
        "has_cuda": has_cuda,
        "device": "cuda" if has_cuda else "cpu",
        "throughput_min": 5.0 if has_cuda else 0.3,
        "model_load_max": 60.0 if has_cuda else 120.0,
        "query_latency_max": 1.0 if has_cuda else 5.0,
    }


class TestSPLADEPerformance:
    """Basic SPLADE performance benchmarks."""

    @pytest.fixture()
    async def splade_plugin(self, hardware_config):
        """Create and initialize SPLADE plugin."""
        from shared.plugins.builtins.splade_indexer import SPLADESparseIndexerPlugin

        plugin = SPLADESparseIndexerPlugin()
        await plugin.initialize({"device": hardware_config["device"], "quantization": "float16"})
        yield plugin
        await plugin.cleanup()

    async def test_model_loading_time(self, hardware_config):
        """Model should load within acceptable time."""
        from shared.plugins.builtins.splade_indexer import SPLADESparseIndexerPlugin

        plugin = SPLADESparseIndexerPlugin()

        start = time.time()
        await plugin.initialize({"device": hardware_config["device"], "quantization": "float16"})
        duration = time.time() - start

        await plugin.cleanup()

        max_time = hardware_config["model_load_max"]
        assert duration < max_time, f"Model load took {duration:.1f}s, expected <{max_time}s"

    async def test_document_encoding_throughput(self, splade_plugin, hardware_config):
        """Document encoding should meet minimum throughput."""
        docs = [
            {"content": f"Test document number {i} with some content.", "chunk_id": f"chunk-{i}"} for i in range(20)
        ]

        start = time.time()
        results = await splade_plugin.encode_documents(docs)
        duration = time.time() - start

        throughput = len(docs) / duration
        min_throughput = hardware_config["throughput_min"]
        assert len(results) == len(docs)
        assert throughput > min_throughput, f"Throughput {throughput:.2f} docs/sec below minimum {min_throughput}"

    async def test_query_encoding_latency(self, splade_plugin, hardware_config):
        """Single query encoding should complete quickly."""
        start = time.time()
        result = await splade_plugin.encode_query("test search query")
        duration = time.time() - start

        max_latency = hardware_config["query_latency_max"]
        assert len(result.indices) > 0
        assert duration < max_latency, f"Query took {duration:.2f}s, expected <{max_latency}s"
