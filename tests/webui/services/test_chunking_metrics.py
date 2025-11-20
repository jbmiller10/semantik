"""Lightweight coverage for chunking metrics helpers."""

from packages.webui.services.chunking.metrics import ChunkingMetrics


def test_get_metrics_by_strategy_placeholder():
    metrics = ChunkingMetrics()

    result = metrics.get_metrics_by_strategy(period_days=7)

    assert result, "Expected default metrics list"
    assert all(hasattr(item, "to_api_model") for item in result)
