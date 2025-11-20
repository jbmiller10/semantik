"""Lightweight coverage for chunking metrics helpers."""

from packages.webui.services.chunking.metrics import ChunkingMetrics


def test_get_metrics_by_strategy_returns_defaults():
    metrics = ChunkingMetrics()
    result = metrics.get_metrics_by_strategy(period_days=1)

    assert result
    assert result[0].strategy
